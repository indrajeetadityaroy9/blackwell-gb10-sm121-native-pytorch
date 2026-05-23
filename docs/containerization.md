# Containerization

All bench build and execution lives in Docker. The host runs only `docker` commands.

## Container topology

| Container | Image | Privileges | Purpose | Lifetime |
|---|---|---|---|---|
| `dgx-bench-clocklock` | `nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04` | `--privileged --gpus all` | Holds the NVML GPU clock lock at 2418 MHz for the entire bake-off | Spawned at start of `run_bakeoff.sh`; torn down by the EXIT/INT/TERM trap |
| Run A (ephemeral) | `dgx-spark-bench-base:cuda13.2` | `--gpus all --cap-add=SYS_ADMIN` (for NCU HW counters) | PyPI `torch==2.10.0+cu130` baseline; torch installed via `uv pip install --no-deps --force-reinstall` at container start | one-shot per bake-off |
| Run B (ephemeral) | `dgx-spark-bench-base:cuda13.2` | `--gpus all --cap-add=SYS_ADMIN` | Source-built wheel mounted read-only from `dgx-spark-build-strict`; installed via the same `--force-reinstall` pattern | one-shot per bake-off |
| Run C (ephemeral) | `nvcr.io/nvidia/pytorch:26.04-py3` | `--gpus all --cap-add=SYS_ADMIN` | NGC vendor reference. In-line setup installs `nsight-compute-2026.1`, `uv`, `optimum-benchmark`, recent `transformers`, and applies the two GB10 compat patches from the bench-base Dockerfile | one-shot per bake-off |
| `dgx-bench-clocklock-verify` (ephemeral, gate) | `nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04` | `--privileged --gpus all` | Phase 0 gate (`verify_clocklock.sh`): confirms NVML `--lock-gpu-clocks` works on this GB10 | one-shot per verify run |
| PyTorch source build (ephemeral) | `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04` | `--gpus all --ipc=host --shm-size=8g` | Clones and builds PyTorch 2.10.0 with `TORCH_CUDA_ARCH_LIST="12.1;12.1a"`; wheel deposited in `dgx-spark-build-strict` | one-shot, on first build |
| HF prefetch (ephemeral) | `python:3.12-slim` | (no GPU) | Pulls model weights into `dgx-spark-hf-cache` via the host's `hf auth token` and `hf-transfer` | one-shot per prefetch |

## Why `--privileged` for the clock-lock container?

`--cap-add=SYS_ADMIN` alone is **not** sufficient to call NVML `nvidia-smi --lock-gpu-clocks` from inside a container. The NVIDIA Container Toolkit gates clock-control NVML ops on full container privilege.

The privilege is localized to a single short-lived controller container that runs only the lock/unlock commands. The bench containers themselves stay unprivileged (only `--cap-add=SYS_ADMIN` for NCU's HW counters). The `EXIT/INT/TERM` trap in `run_bakeoff.sh` uses `docker exec` against the controller to issue the unlock, so a host-side Ctrl+C still propagates correctly.

Phase 0 (`bench/verify_clocklock.sh`) is the gate that confirms NVML lock works on this GB10: it locks at 2418 MHz, queries `clocks.sm`, validates within ±5%, then unlocks. Exits 0 (PASS) or 2 (FAIL).

## Why the trap in `run_bakeoff.sh` omits `ERR`

With `ERR` in the trap, the first failing `docker run` would fire the trap immediately and `exit "$rc"` would terminate the script — Runs B and C would never execute. Per-wheel independence is the whole point of the bake-off; the script captures `A_RC`/`B_RC`/`C_RC` and lets the next run proceed even if its predecessor fails.

## Docker volumes

| Volume | Purpose | Created by |
|---|---|---|
| `dgx-spark-build-strict` | PyTorch source tree, ccache, and the source-built wheel (`/work/pytorch/dist/torch-*.whl`) | `build/source_build.sh` |
| `dgx-spark-hf-cache` | HuggingFace model cache (Llama-3.1-8B and any other scenario models) | `bench/build/prefetch_hf_models.sh`; mounted into bench containers as `/hf-cache` |
| `dgx-spark-uv-cache` | uv package cache, shared across runs | `bench/run_bakeoff.sh` |
| `dgx-spark-apt-cache` | apt cache (Run C reinstalls `nsight-compute-2026.1` per run; this volume keeps the .deb cached) | `bench/run_bakeoff.sh` |
| `dgx-spark-triton-cache-a` / `-b` / `-c` | Per-wheel Triton JIT cache | `bench/run_bakeoff.sh` |

`bench/cleanup_volumes.sh` removes all seven (interactively unless `-y`/`--force`).

### Why per-container Triton cache?

Triton's cache key includes `backend.hash()`, which encodes the GPU architecture. Different wheels in different containers may also ship different Triton versions, with different cache hashes. Cross-mounting would risk cache invalidation or incorrect lookups. Each wheel gets its own `dgx-spark-triton-cache-<a|b|c>` volume mounted at `/root/.triton/cache`.

## bench-base image (`dgx-spark-bench-base:cuda13.2`)

Built once via `bench/build/build_bench_base.sh`; reused by Runs A and B. Layers (from `bench/build/Dockerfile.bench-base`):

1. apt: `python3 python3-pip python3-venv ca-certificates curl libopenblas0 libnuma1 cudnn9-cuda-13-2 cusparselt-cuda-13 libcusparselt0-cuda-13 nsight-compute-2026.1`, plus removing `/usr/lib/python3.12/EXTERNALLY-MANAGED` (PEP 668 marker) so subsequent `uv pip install --system` calls don't need `--break-system-packages`.
2. `uv 0.11.15` via pip.
3. torch-independent python tools: `hf-transfer`, `ncu-report==2025.3.1`.
4. torch-dependent overlay: `transformers>=4.55,<5.0`, `accelerate>=1.0`, `bitsandbytes==0.49.2`, `optimum-benchmark==0.6.0`, `triton>=3.6`. uv pulls full transitive deps; the resolver pulls a default torch from PyPI as a side effect — the per-wheel container then replaces it via `uv pip install --no-deps --force-reinstall <wheel>`.
5. Two `sed`-applied GB10 compat patches to optimum-benchmark v0.6.0:
   - `system_utils.get_gpu_vram_mb` — `pynvml.nvmlDeviceGetMemoryInfo` raises `NotSupported` on GB10 (Grace's unified LPDDR5X has no separate FB Memory exposed via NVML). Substitute `0`; only used for metadata.
   - `backends/pytorch/backend.split_between_processes` — calls `torch.distributed.is_initialized()` unconditionally; the source-built wheel (Run B) is compiled with `USE_DISTRIBUTED=0`, removing that function. Replace with `getattr(torch.distributed, "is_initialized", lambda: False)()`.

Run C applies the same two patches in-line at container start (the NGC image installs optimum-benchmark fresh per run).

## Build artifact provenance

| Artifact | Built by | Stored in |
|---|---|---|
| `torch-2.10.0-*.whl` (sm_121 / sm_121a native) | `build/source_build.sh` | `dgx-spark-build-strict:/work/pytorch/dist/` |
| `dgx-spark-bench-base:cuda13.2` | `bench/build/build_bench_base.sh` | local docker image registry |
| HF model weights | `bench/build/prefetch_hf_models.sh` | `dgx-spark-hf-cache:/` |

All three are built inside containers, never on the host. Re-runs reuse the volumes/images; invalidate by `docker volume rm` / `docker rmi`.

## Permission notes

- `docker.sock` is owned `root:docker 660`. User must be in the `docker` group (`getent group docker`).
- The HuggingFace token is read on the host via `hf auth token` and passed into containers as the `HF_TOKEN` env var; no token files are mounted into bench containers.
- NGC API key is required on the host (`docker login nvcr.io`) for Run C to pull `nvcr.io/nvidia/pytorch:26.04-py3`.
