# PyTorch with native sm_121/sm_121a on NVIDIA DGX Spark

A from-source PyTorch build that produces the first wheel on this hardware (NVIDIA GB10, Grace + Blackwell, sm_121) with **native sm_121 and sm_121a cubins** — eliminating the runtime PTX-JIT path that every other PyTorch distribution falls back to on this hardware.

NGC PyTorch container: [`nvcr.io/nvidia/pytorch:26.04-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

## Why this build exists

Empirically verified via `torch.cuda.get_arch_list()`:
| Wheel | Compiled arch list |
|---|---|
| PyPI `torch==2.10.0+cu130` | `sm_80, sm_90, sm_100, sm_110, sm_120, compute_120` |
| NGC `pytorch:26.04-py3` (`torch 2.12.0a0+nv26.04`) | `sm_80, sm_86, sm_90, sm_100, sm_110, sm_120, compute_120` |
| **This build (PyTorch 2.10.0)** | **`sm_121, sm_121a`** |

On every wheel that doesn't ship `sm_121`, the CUDA driver JIT-compiles `compute_120` PTX to `sm_121` SASS at first kernel launch. This build emits SASS for sm_121 ahead of time and additionally compiles arch-specific feature paths (`sm_121a` — TMEM, 2-CTA MMA, async tensor-core writes) that several CUTLASS-derived kernels in PyTorch rely on.

## Reproduction

### Build the wheel (~1.5–2 h fresh, ~30 min warm via ccache)

```bash
bash build/source_build.sh
```

PyTorch 2.10.0 builds inside `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04` with `TORCH_CUDA_ARCH_LIST="12.1;12.1a"`. All source, build artifacts, and the resulting wheel live in the docker named volume `dgx-spark-build-strict` — no host filesystem state outside the captured launcher log.

### Three-way bake-off

```bash
bash bench/build/build_bench_base.sh        # one-time, ~5–10 min, image ~10 GB
bash bench/build/prefetch_hf_models.sh      # one-time, pulls Llama-3.1-8B into dgx-spark-hf-cache
bash bench/run_bakeoff.sh
```

`run_bakeoff.sh` spawns a privileged clock-lock sidecar (`dgx-bench-clocklock`, NVML lock at 2418 MHz), then runs three ephemeral containers in sequence:

- **Run A** — PyPI `torch==2.10.0+cu130` on the `dgx-spark-bench-base:cuda13.2` image
- **Run B** — source-built wheel mounted read-only from `dgx-spark-build-strict`
- **Run C** — NGC `nvcr.io/nvidia/pytorch:26.04-py3`

Each run executes `bench/run_tiers.py` inside its container. Three tiers run in deterministic order:

1. **optimum-benchmark** — Llama-3.1-8B BF16, batch=1, seq_len=512, 128 new tokens. Bandwidth-bound regime; reports per-op latency.
2. **kernel-level GEMM** — four Llama-3.1-8B projection shapes at M=512 across bf16/fp16/fp8/fp4 (each dtype as an isolated subprocess so cuBLAS Lt's incomplete FP8/FP4 sm_121 heuristic table can't take out BF16/FP16 results).
3. **NCU roofline** — `ncu --set roofline` over the BF16 kernel_bench with kernel-name filtering and a 200-launch budget; per-kernel achieved % of peak parsed via the `ncu_report` Python API.

Each run emits a JSON document to `bench/logs/run{A,B,C}.json`. `bench/_summarize.py` aggregates them into `bench/logs/SUMMARY.txt` with a per-test SOL Score column for the roofline tier (gap-closure: `(measured − baseline) / (sol − baseline)`, clamped to `[0, 1]`). The optimum and kernel tiers render '—' in the score column since no SOL bound is modeled there.

Requires Docker with the NVIDIA Container Toolkit, an NGC API key (`docker login nvcr.io`) for Run C, and an `hf` CLI token for the model prefetch.

### Cleanup

```bash
bash bench/cleanup_volumes.sh   # interactive removal of all bake-off docker volumes
```

## When NGC is still the right call

Despite the build matching NGC on the workloads this bench exercises, NGC remains the path of least resistance without needing to rebuild PyTorch or pin a specific build configuration:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v $HOME:/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:26.04-py3
```

NGC ships triton 3.6.0 (the only stack where the Triton matmul test succeeds on sm_121a) and a private NVIDIA `flash_attn==2.7.4+nv26.04` fork — neither reproducible from source.

## Potential Next Step

AutoKernel ([arXiv:2603.21331](https://arxiv.org/abs/2603.21331)) agent-driven kernel-replacement methodology
