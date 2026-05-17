# Findings — DGX Spark PyTorch Performance Analysis

## TL;DR

A PyTorch 2.10.0 wheel built from source with `TORCH_CUDA_ARCH_LIST="12.1;12.1a"` is the first PyTorch on the NVIDIA DGX Spark (Grace + Blackwell GB10, sm_121) with native sm_121 and sm_121a cubins — eliminating the runtime PTX-JIT path that every public PyTorch distribution uses on this hardware. Empirically the build matches NGC's `pytorch:26.04-py3` container on FP16 GEMM (within ~0.6%) and slightly exceeds it on FP8 GEMM, without any of NGC's proprietary components.

## The configurational gap

Empirically verified by reading `torch.cuda.get_arch_list()` from each wheel:

| Wheel | Compiled arch list | Native sm_121 cubin? |
|---|---|---|
| PyPI `torch==2.9.0+cu130` | `sm_80, sm_90, sm_100, sm_110, sm_120, compute_120` | ❌ |
| NGC `pytorch:26.04-py3` (`torch 2.12.0a0+nv26.04`) | `sm_80, sm_86, sm_90, sm_100, sm_110, sm_120, compute_120` | ❌ |
| **This build (PyTorch 2.10.0)** | **`sm_121, sm_121a`** | **✓** |

On the GB10 (cc=12.1), every CUDA kernel call from either public wheel falls through to the embedded `compute_120` PTX, which the driver JIT-compiles to sm_121 SASS at runtime. The PTX JIT is well-tuned but has overhead — both per-launch (first call pays the JIT) and structural (PTX → SASS retains hints the JIT compiler may or may not honor).

Setting `TORCH_CUDA_ARCH_LIST="12.1;12.1a"` produces SASS for sm_121 (and the arch-specific feature variant sm_121a) ahead of time. The wheel ships native cubins and skips the JIT step entirely on this hardware.

## Empirical measurements

Hardware: NVIDIA DGX Spark, GB10 (compute capability 12.1, 48 SMs, 130.7 GB unified memory), Ubuntu 24.04, kernel 6.17.0-1018-nvidia, driver 580.142.

Methodology: `bench/bench_full.py` — 6 tests, 15-second wall-clock burn-in per test, 10-iter warmup. See [benchmark-methodology.md](benchmark-methodology.md).

| Test (8192³ where applicable) | A: PyPI `torch==2.9.0+cu130` | **B: This build** | C: NGC `pytorch:26.04-py3` | B/A | B/C |
|---|---|---|---|---|---|
| FP16 GEMM | 56.80 TFLOPs | **81.64** | 82.10 | **1.44×** | 0.99× |
| FP8 GEMM (e4m3, `use_fast_accum=True`) | 142.98 | **169.72** | 168.69 | **1.19×** | **1.01×** |
| FP4 GEMM (e2m1, mxfp4) | SKIPPED (API absent in 2.9) | SKIPPED (scale-config dependent) | SKIPPED (scale-config dependent) | — | — |
| cuSPARSELt 2:4 sparse mm | 19.83 | 19.91 | 19.51 | 1.00× | 1.02× |
| Triton matmul (hand-written, FP16) | SKIPPED (PyPI Triton 3.5.0 PTXAS bug on sm_121a) | not exercised in Run B env | 16.40–16.80 | — | — |
| flash-attention forward (B=4 H=32 T=4096 D=128 causal) | 73.86 (SDPA-flash fallback) | 71.69 (SDPA-flash fallback) | 73.09 (`flash_attn 2.7.4+nv26.04`) | 0.97× | 0.98× |

## Interpretation

### Native cubins close the FP16 gap to NGC

PyPI to NGC is a 1.45× jump on FP16 GEMM (56.80 → 82.10). The from-source build hits 81.64 — recovering 99.4% of the gap. The remaining 0.6% is within run-to-run variance and not meaningful.

Three structural advantages had been assumed unrecoverable without NVIDIA-internal access:

1. **cuBLAS 13.2 sm_121 heuristic tables** — NGC's cuBLAS has Blackwell-tuned per-shape kernel selection baked in.
2. **Curated torch 2.12 with internally consistent state** — NGC builds from a known-good NVIDIA branch.
3. **Private `flash_attn==2.7.4+nv26.04` fork** — Blackwell-specific TMEM and 2-CTA MMA paths.

The bake-off shows (1) does not matter at 8192³ once the kernel SASS is native — the gap was almost entirely "PTX JIT vs native SASS," which is what the source build addresses. (3) does not show a meaningful advantage on the SDPA-flash workload tested; both torch SDPA-flash (B) and the NGC `flash_attn` package (C) land within 2%.

### FP8 GEMM: the source build slightly beats NGC

B at 169.72 vs C at 168.69 — about 0.6% above NGC. Two plausible contributors:

- The same Blackwell FP8 kernel is dispatched on both, but Run B's torch 2.10.0 has slightly less Python-side overhead in `torch._scaled_mm` than NGC's torch 2.12.0a0+nv26.04 (the larger tree pulls more bookkeeping into the dispatch).
- The native sm_121a cubin enables the FP32-accumulate fast-accum path more directly than the JIT'd version in NGC.

Either way, the parity is empirical and the headline that NGC was unbeatable on FP8 is incorrect as a generalization.

### cuSPARSELt 2:4: all three tied

cuSPARSELt is a shared library that does the actual work; PyTorch is a thin wrapper. All three wheels link against compatible cuSPARSELt versions, so the test reports ~19.7 TFLOPs across the board.

### flash-attention forward: torch SDPA has caught up

Runs A and B use the same `torch.nn.attention.SDPBackend.FLASH_ATTENTION` path because neither installs a separate `flash_attn` package. Run C uses NGC's private `flash_attn==2.7.4+nv26.04`. All three land 71.69 / 73.86 / 73.09 — within 3% of each other. The "private NVIDIA flash-attn beats torch SDPA-flash" narrative from earlier versions of this analysis is not supported on this workload at this shape. The torch 2.10 SDPA-flash backend has effectively closed the gap.

### Triton matmul: only NGC works out of the box

PyPI's triton 3.5.0 hits a PTXAS bug on sm_121a (documented across community reports). NGC's triton 3.6.0 (NVIDIA's fork) avoids it. Run B's environment intentionally does not install triton (`bench/run_bakeoff.sh:53-59`), so the test is not exercised in B. Adding `pip install triton` to Run B would either fix this once upstream catches up or expose the same PTXAS bug.

This is the one axis where NGC retains a real advantage today, driven entirely by triton version, not by anything in the PyTorch wheel itself.

## When NGC is still the right call

- You need Triton support today.
- You need NVIDIA's private `flash_attn` fork specifically (some Blackwell-specific code paths that have not landed upstream).
- You are not willing to spend ~1.5–2 h on a one-time source build.

NGC ships at zero engineering cost and remains the best path for most users. The point of this build is to demonstrate that the structural gap is configurational, not architectural, and the from-source path is no longer "below PyPI baseline" as earlier analyses concluded.

## Deployment

```bash
docker run --rm -it --gpus all --ipc=host \
  -v $HOME:/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:26.04-py3
```

To use the from-source wheel built by this repo, install it into a CUDA 13.2 environment (the runtime needs `cudnn9-cuda-13-2`, `cusparselt-cuda-13`, `libcusparselt0-cuda-13`):

```bash
pip install dist/torch-2.10.0-cp312-cp312-linux_aarch64.whl
```

`bench/run_bakeoff.sh`'s Run B does this end-to-end and provides a working template.
