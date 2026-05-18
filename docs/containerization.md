# Containerization

All bench build and execution lives in Docker. The host runs only `docker`
commands. This document describes the container topology, why each container
exists, and the docker volumes that persist artifacts across runs.

## Container topology

| Container | Image | Privileges | Purpose | Lifetime |
|---|---|---|---|---|
| `dgx-bench-clocklock` | `nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04` | `--privileged --gpus all` | Holds NVML GPU clock lock at 2418 MHz for the entire bake-off | Spawned at start of `run_bakeoff.sh`; torn down by EXIT trap (catches Ctrl+C, crash, normal exit) |
| `runA` (ephemeral) | `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` + apt `python3` | `--gpus all` | PyPI baseline (`torch==2.9.0+cu130 + triton`) | one-shot per bake-off |
| `runB` (ephemeral) | `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04` + apt `python3 cudnn9 cusparselt` | `--gpus all` | Source-built wheel from `dgx-spark-build-strict` volume | one-shot per bake-off |
| `runC` (ephemeral) | `nvcr.io/nvidia/pytorch:26.04-py3` | `--gpus all` | NGC vendor reference | one-shot per bake-off |
| `nvbench-build` (ephemeral, on demand) | `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04` | `--gpus all` | Build nvbench shim once for sm_121; binary cached in `dgx-spark-build-strict` volume | one-shot, on first run |
| `flashinfer-build` (ephemeral, on demand) | `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04` | `--gpus all` | Source-build FlashInfer with `TORCH_CUDA_ARCH_LIST=12.1` (Tier 4) | one-shot, on demand |
| `roofline` (ephemeral, opt-in via `BENCH_PROFILE=1`) | `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04` (Nsight Compute 2026.1.0 pre-installed in `/usr/local/cuda/bin/ncu`) | `--gpus all --cap-add=SYS_ADMIN` (for HW counters) | Run `ncu --set roofline` against one chosen test | one-shot per profile run |
| `mlperf` (ephemeral, Tier 4) | `ghcr.io/mlcommons/inference:5.1-dev` | `--gpus all --ipc=host --shm-size=8g` | MLPerf v5.1 llama3.1-8b reference runner | one-shot per Tier 4 run |

## Why `--privileged` for the clock-lock container?

Tested: `--cap-add=SYS_ADMIN` alone is **not sufficient** to call NVML
`nvidia-smi --lock-gpu-clocks` from inside a container. The NVIDIA Container
Toolkit gates clock-control NVML ops on full container privilege.

We localize the privilege to a single short-lived controller container that
runs only the lock/unlock commands. The bench containers themselves are
unprivileged. This is documented as the security trade-off in the Risk Register.

## Why is the GPU clock lock in a container, not on the host?

The plan is container-first by design. The host runs only `docker` commands;
the actual NVML clock-lock call lives inside `dgx-bench-clocklock`. The trap
in `run_bakeoff.sh` uses `docker exec` to issue the unlock, so a host-side
Ctrl+C still propagates correctly.

Phase 0 (`bench/verify_clocklock.sh`) is the gate that confirms NVML lock
works on this GB10. Verified on this hardware: locked at 2418 MHz, driver
reports 2411 MHz (0.3% within target). PASS.

## Docker volumes

| Volume | Purpose | Created by |
|---|---|---|
| `dgx-spark-build-strict` | PyTorch wheel (from `build/source_build.sh`), nvbench binary (from `bench/nvbench_shim/build_nvbench.sh`), FlashInfer wheel (from `bench/e2e/build_flashinfer.sh`) | `build/source_build.sh` initially |
| `mlperf-cache` | MLPerf v5.1 weights + dataset (~16 GB; Tier 4 only) | `bench/e2e/mlperf_llama31_8b.py` |
| `bench/cache/triton/sm121/runA,runB,runC` (bind mounts) | Triton autotune cache, per-container | `run_bakeoff.sh` |

### Why per-container Triton cache?

Triton's cache key includes `backend.hash()`, which encodes the GPU
architecture (`sm_120` ≠ `sm_121`). Different containers with potentially
different Triton versions also have different cache hashes. Cross-mounting
would risk cache invalidation or incorrect lookups. Each container gets its
own `cache/triton/sm121/run<X>/` subdir.

## Build artifact provenance

| Artifact | Built by | Stored in |
|---|---|---|
| `torch-2.10.0-*.whl` (sm_121 native) | `build/source_build.sh` | `dgx-spark-build-strict:/work/pytorch/dist/` |
| `sm121_gemm` (nvbench shim binary) | `bench/nvbench_shim/build_nvbench.sh` | `dgx-spark-build-strict:/work/nvbench_shim/build/` |
| `flashinfer-*.whl` (sm_121 native) | `bench/e2e/build_flashinfer.sh` | `dgx-spark-build-strict:/work/flashinfer/dist/` |

All three are built inside containers, never on the host. Re-runs reuse the
volume; only invalidate by deleting the volume (`docker volume rm dgx-spark-build-strict`).

## Permission notes

- `docker.sock` is owned `root:docker 660`. User must be in the `docker` group
  (`getent group docker`). New shell sessions activate the group automatically;
  this session uses `sg docker -c` because the membership postdates the shell.
- HuggingFace token at `~/.cache/huggingface/token` is mounted read-only into
  the MLPerf container (`-v ~/.cache/huggingface:/root/.cache/huggingface:ro`).
  The script also extracts it into `HF_TOKEN=$(cat ~/.cache/huggingface/token)`
  for compatibility with libraries that check the env var.
