#!/usr/bin/env bash
# 3-way DGX Spark bake-off.
#
#   A : PyPI torch==2.10.0+cu130            (dgx-spark-bench-base image)
#   B : Source-built wheel, native sm_121   (dgx-spark-bench-base image)
#   C : NGC pytorch:26.04-py3               (NGC image + python overlay)
#
# Models for the selected scenario must be pre-fetched into
# dgx-spark-hf-cache via bench/build/prefetch_hf_models.sh (uses the
# host's hf CLI token; bench containers do not need HF_TOKEN).
#
# Env:
#   BENCH_OPTIMUM_SCENARIO   recommended (default) | wide

set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$REPO_DIR/bench/logs"; mkdir -p "$LOGS"

SCENARIO="${BENCH_OPTIMUM_SCENARIO:-recommended}"

ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }

# Verify bench-base image is built (Runs A and B need it).
docker image inspect dgx-spark-bench-base:cuda13.2 >/dev/null 2>&1 || {
  log "FATAL: dgx-spark-bench-base:cuda13.2 not built; run bash bench/build/build_bench_base.sh"
  exit 2
}

# Shared runtime env: image cache paths + arch flags + scenario.
DOCKER_ENV=(
  -e "BENCH_OPTIMUM_SCENARIO=$SCENARIO"
  -e "BENCH_LOG_DIR=/logs"
  -e "HF_HOME=/hf-cache"
  -e "UV_CACHE_DIR=/root/.cache/uv"
  -e "TRITON_CACHE_DIR=/root/.triton/cache"
  -e "CUTLASS_CACHE_DIR=/root/.cache/cutlass-jit"
  -e "PYTORCH_KERNEL_CACHE_PATH=/root/.triton/cache"
  -e "HF_HUB_ENABLE_HF_TRANSFER=1"
  -e "TORCH_CUDA_ARCH_LIST=12.1;12.1a"
  -e "NVCC_APPEND_FLAGS=-gencode arch=compute_121,code=sm_121a -ptxas-options=-O3 -D__CUDA_ARCH_FEAT_SM90_ALL"
)

for v in dgx-spark-hf-cache dgx-spark-uv-cache dgx-spark-apt-cache \
         dgx-spark-triton-cache-a dgx-spark-triton-cache-b dgx-spark-triton-cache-c \
         dgx-spark-cutlass-jit-a dgx-spark-cutlass-jit-b dgx-spark-cutlass-jit-c; do
  docker volume create "$v" >/dev/null
done

# Clock-lock controller (--privileged; bench containers stay unprivileged
# except for --cap-add=SYS_ADMIN which NCU's HW counters require).
cleanup() {
  local rc=$?
  log "cleanup: stopping clock-lock controller"
  docker exec dgx-bench-clocklock nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
  docker rm -f dgx-bench-clocklock >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT INT TERM ERR

log "spawning clock-lock controller (2418 MHz)"
docker rm -f dgx-bench-clocklock >/dev/null 2>&1 || true
docker run -d --privileged --gpus all --name dgx-bench-clocklock \
  -v "$REPO_DIR/bench:/bench:ro" \
  nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04 \
  bash -c '. /bench/_clocklock.sh; lock_clocks 2418; sleep infinity' >/dev/null

log "polling for clock-lock confirmation (30s timeout)"
LOCKED=0
for _ in $(seq 1 30); do
  if docker logs dgx-bench-clocklock 2>&1 | grep -q '\[clocklock\] locked gpu'; then
    LOCKED=1; break
  fi
  if docker logs dgx-bench-clocklock 2>&1 | grep -q '\[clocklock\] FATAL'; then
    log "FATAL: NVML rejected clock lock"
    docker logs dgx-bench-clocklock 2>&1 | sed 's/^/  /'
    exit 2
  fi
  sleep 1
done
[[ "$LOCKED" == "1" ]] || { log "FATAL: clock lock did not confirm within 30s"; exit 2; }
log "clock lock active; scenario=$SCENARIO"

# ============================ Run A : PyPI torch==2.10.0+cu130 ============================
log "Run A : PyPI torch==2.10.0+cu130"
docker run --rm --gpus all --ipc=host --cap-add=SYS_ADMIN \
  "${DOCKER_ENV[@]}" \
  -v dgx-spark-hf-cache:/hf-cache \
  -v dgx-spark-uv-cache:/root/.cache/uv \
  -v dgx-spark-triton-cache-a:/root/.triton/cache \
  -v dgx-spark-cutlass-jit-a:/root/.cache/cutlass-jit \
  -v "$REPO_DIR":/repo:ro \
  -v "$LOGS:/logs" \
  dgx-spark-bench-base:cuda13.2 \
  bash -c '
    set -uo pipefail
    uv pip install --system --no-deps --force-reinstall \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.10.0+cu130 >&2
    python /repo/bench/build/build_fa4.py >&2
    python /repo/bench/run_tiers.py
  ' > "$LOGS/runA.json" 2>>"$LOGS/runA.log"
A_RC=$?
log "Run A exit: ${A_RC}"

# ============================ Run B : source-built wheel ============================
log "Run B : source-built wheel (dgx-spark-build-strict)"
docker run --rm --gpus all --ipc=host --cap-add=SYS_ADMIN \
  "${DOCKER_ENV[@]}" \
  -v dgx-spark-build-strict:/work:ro \
  -v dgx-spark-hf-cache:/hf-cache \
  -v dgx-spark-uv-cache:/root/.cache/uv \
  -v dgx-spark-triton-cache-b:/root/.triton/cache \
  -v dgx-spark-cutlass-jit-b:/root/.cache/cutlass-jit \
  -v "$REPO_DIR":/repo:ro \
  -v "$LOGS:/logs" \
  dgx-spark-bench-base:cuda13.2 \
  bash -c '
    set -uo pipefail
    uv pip install --system --no-deps --force-reinstall \
      "$(ls /work/pytorch/dist/torch-*.whl | head -1)" >&2
    python /repo/bench/build/build_fa4.py >&2
    python /repo/bench/run_tiers.py
  ' > "$LOGS/runB.json" 2>>"$LOGS/runB.log"
B_RC=$?
log "Run B exit: ${B_RC}"

# ============================ Run C : NGC vendor reference ============================
log "Run C : NGC nvcr.io/nvidia/pytorch:26.04-py3"
docker run --rm --gpus all --ipc=host --cap-add=SYS_ADMIN \
  "${DOCKER_ENV[@]}" \
  -v dgx-spark-hf-cache:/hf-cache \
  -v dgx-spark-uv-cache:/root/.cache/uv \
  -v dgx-spark-apt-cache:/var/cache/apt \
  -v dgx-spark-triton-cache-c:/root/.triton/cache \
  -v dgx-spark-cutlass-jit-c:/root/.cache/cutlass-jit \
  -v "$REPO_DIR":/repo:ro \
  -v "$LOGS:/logs" \
  nvcr.io/nvidia/pytorch:26.04-py3 \
  bash -c '
    set -uo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >&2
    apt-get install -y -qq nsight-compute-2026.1 >&2
    python3 -m pip install --quiet uv==0.11.15 >&2
    uv pip install --system hf-transfer ncu-report==2025.3.1 >&2
    uv pip install --system --no-deps --force-reinstall \
      "triton>=3.6" "nvidia-cutlass-dsl>=4.4.2" \
      "transformers>=4.55,<5.0" "accelerate>=1.0" \
      bitsandbytes==0.49.2 \
      optimum-benchmark==0.6.0 flash-attn-4==4.0.0b13 >&2
    uv pip install --system \
      hydra-core>=1.3 omegaconf>=2.3 psutil pyyaml typing_extensions tqdm \
      safetensors huggingface_hub sentencepiece tokenizers protobuf \
      einops "apache-tvm-ffi>=0.1.5,<0.2" torch-c-dlpack-ext "quack-kernels>=0.4.0" >&2
    python /repo/bench/build/build_fa4.py >&2
    python /repo/bench/run_tiers.py
  ' > "$LOGS/runC.json" 2>>"$LOGS/runC.log"
C_RC=$?
log "Run C exit: ${C_RC}"

# ============================ aggregate ============================
log "building SUMMARY.txt"
python3 "$REPO_DIR/bench/_summarize.py" "$LOGS" || true
log "exit codes: A=${A_RC}  B=${B_RC}  C=${C_RC}"

chown -R "$(id -u):$(id -g)" "$LOGS" 2>/dev/null || true
log "Done. See $LOGS/SUMMARY.txt"
