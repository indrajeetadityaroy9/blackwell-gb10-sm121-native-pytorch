# PyTorch with native sm_121/sm_121a on NVIDIA DGX Spark

A from-source PyTorch build that produces the first wheel on this hardware (NVIDIA GB10, Grace + Blackwell, sm_121) with **native sm_121 and sm_121a cubins** — eliminating the runtime PTX-JIT path that every other PyTorch distribution falls back to on this hardware.

## Why this build exists

Empirically verified via `torch.cuda.get_arch_list()`:

| Wheel | Compiled arch list |
|---|---|---|
| PyPI `torch==2.9.0+cu130` | `sm_80, sm_90, sm_100, sm_110, sm_120, compute_120` |
| NGC `pytorch:26.04-py3` (`torch 2.12.0a0+nv26.04`) | `sm_80, sm_86, sm_90, sm_100, sm_110, sm_120, compute_120` |
| **This build (PyTorch 2.10.0)** | **`sm_121, sm_121a`** |

On every wheel that doesn't ship `sm_121`, the CUDA driver JIT-compiles `compute_120` PTX to `sm_121` SASS at first kernel launch. This build emits SASS for sm_121 ahead of time and additionally compiles arch-specific feature paths (`sm_121a` — TMEM, 2-CTA MMA, async tensor-core writes) that several CUTLASS-derived kernels in PyTorch rely on.

## Metrics

Three-way comparison on the same DGX Spark (`bench/bench_full.py`, 15 s burn-in per test, 8192³ where applicable):

| Test | A: PyPI `torch==2.9.0+cu130` | **B: This build (`torch==2.10.0`, sm_121/sm_121a)** | C: NGC `pytorch:26.04-py3` | B/A | B/C |
|---|---|---|---|---|---|
| FP16 GEMM | 56.80 TFLOPs | **81.64** | 82.10 | **1.44×** | 0.99× |
| FP8 GEMM (e4m3, fast_accum) | 142.98 | **169.72** | 168.69 | **1.19×** | **1.01×** |
| cuSPARSELt 2:4 sparse mm | 19.83 | 19.91 | 19.51 | 1.00× | 1.02× |
| flash-attention forward (causal, B=4 H=32 T=4096 D=128) | 73.86 (SDPA fallback) | 71.69 (SDPA fallback) | 73.09 (`flash_attn 2.7.4+nv26.04`) | 0.97× | 0.98× |
| FP4 (mxfp4) | SKIPPED (API absent in 2.9) | SKIPPED (scale-config dependent) | SKIPPED (scale-config dependent) | — | — |
| Triton matmul (FP16) | SKIPPED (PyPI triton 3.5.0 PTXAS bug on sm_121a) | not exercised in Run B env | 16.40–16.80 | — | — |

The native-cubin build matches NGC on FP16 to within ~0.6% and slightly exceeds NGC on FP8 — without any of NGC's curated proprietary components (private `flash_attn` fork, sm_121-tuned cuBLAS 13.2 dispatch tables).

## Reproduction

### Build the wheel (~1.5–2 h fresh, ~30 min warm via ccache)

```bash
bash build/source_build.sh
```

PyTorch 2.10.0 builds inside `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04` with `TORCH_CUDA_ARCH_LIST="12.1;12.1a"`. All source, build artifacts, and the resulting wheel live in the docker named volume `dgx-spark-build-strict` — no host filesystem state outside the captured launcher log.

### Three-way bake-off (~10 min once images are cached)

```bash
bash bench/run_bakeoff.sh
```

Runs A (PyPI), B (the wheel from the volume — skipped if you haven't built it), and C (NGC) under the same `bench/bench_full.py` workload.Requires Docker with NVIDIA Container Toolkit and an NGC API key (`docker login nvcr.io`) for Run C.

### Cleanup

```bash
docker volume rm dgx-spark-build-strict   # frees ~25 GB (source tree + ccache + wheel)
```

## When NGC is still the right call

Despite the bake-off ending in B ≈ C, NGC remains the path of least resistance without needing to rebuild PyTorch or pin a specific build configuration:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v $HOME:/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:26.04-py3
```

NGC ships triton 3.6.0 (the only stack where the Triton matmul test succeeds on sm_121a) and a private NVIDIA `flash_attn==2.7.4+nv26.04` fork — neither reproducible from source.Workload bottlenecked by FP16/FP8 GEMM (the bulk of dense LLM inference), this build matches NGC without those proprietary components.

## Potential Next Step

AutoKernel ([arXiv:2603.21331](https://arxiv.org/abs/2603.21331)) agent-driven kernel-replacement methodology

## References

- NGC PyTorch container: [`nvcr.io/nvidia/pytorch:26.04-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
# blackwell_sm_121_experiment
