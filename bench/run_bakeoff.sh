#!/usr/bin/env bash
# 3-way DGX Spark bake-off — reproducible NCU + optimum-benchmark + FA-4.
#
#   A : PyPI torch==2.10.0+cu130            (dgx-spark-bench-base image)
#   B : Source-built wheel, native sm_121   (dgx-spark-bench-base image)
#   C : NGC pytorch:26.04-py3               (NGC image + python overlay)
#
# Run A and Run B share a pre-built bench-base image with apt and python
# overlay baked in (see bench/build/build_bench_base.sh). Per-wheel install
# is reduced to a single `uv pip install --no-deps --force-reinstall <wheel>`.
#
# Run C uses the NGC pytorch image; the overlay is installed at runtime
# (NGC ships its own torch + transformers we cannot replace).
#
# All persistent state lives in named volumes; wipe with
# bench/cleanup_volumes.sh. HF model downloads use the Rust-backed
# hf-transfer (50% faster on first fetch).
#
# Env passthrough:
#   BENCH_OPTIMUM_SCENARIO  recommended (default) | wide
#   BENCH_WIDE_CONFIRM      1 to skip the wide-scenario confirmation prompt
#   BENCH_GPU_MHZ           clock-lock target (default 2418)
#   HF_TOKEN                gated HF Hub model access (passed through)

set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$REPO_DIR/bench/logs"; mkdir -p "$LOGS"

CONTROLLER=dgx-bench-clocklock
GPU_MHZ="${BENCH_GPU_MHZ:-2418}"
SCENARIO="${BENCH_OPTIMUM_SCENARIO:-recommended}"
BENCH_BASE_IMAGE="${BENCH_BASE_TAG:-dgx-spark-bench-base:cuda13.2}"

ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }

# ---------- gate wide scenario behind explicit confirmation ----------
if [[ "$SCENARIO" == "wide" ]]; then
  if [[ "${BENCH_WIDE_CONFIRM:-}" == "1" ]]; then
    log "BENCH_OPTIMUM_SCENARIO=wide (confirmed via BENCH_WIDE_CONFIRM=1)"
  elif [[ -t 0 ]]; then
    echo
    echo "================================================================="
    echo "  BENCH_OPTIMUM_SCENARIO=wide will run 5 model configs per wheel"
    echo "  (Llama-3-8B, Llama-3-70B FP8, Mixtral-8x7B, Qwen3-30B-A3B,"
    echo "  DeepSeek-V2-Lite) — adds ~30-40 min vs recommended scenario."
    echo "================================================================="
    read -r -p "Proceed with wide scenario? [y/N]: " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 0; }
  else
    log "FATAL: BENCH_OPTIMUM_SCENARIO=wide in non-interactive shell without BENCH_WIDE_CONFIRM=1"
    exit 2
  fi
fi

# ---------- verify bench-base image is built (Runs A and B need it) ----------
if ! docker image inspect "$BENCH_BASE_IMAGE" >/dev/null 2>&1; then
  log "FATAL: bench-base image '${BENCH_BASE_IMAGE}' not found"
  log "  build it once with: bash bench/build/build_bench_base.sh"
  exit 2
fi

# ---------- forward env into each docker run ----------
DOCKER_ENV=(-e "BENCH_OPTIMUM_SCENARIO=$SCENARIO"
            -e "HF_HUB_ENABLE_HF_TRANSFER=1")
[[ -n "${HF_TOKEN:-}" ]] && DOCKER_ENV+=(-e "HF_TOKEN=$HF_TOKEN")

# ---------- per-wheel docker volumes ----------
for v in dgx-spark-hf-cache dgx-spark-uv-cache dgx-spark-apt-cache \
         dgx-spark-triton-cache-a dgx-spark-triton-cache-b dgx-spark-triton-cache-c \
         dgx-spark-cutlass-jit-a dgx-spark-cutlass-jit-b dgx-spark-cutlass-jit-c; do
  docker volume create "$v" >/dev/null
done

# ---------- clock-lock controller (--privileged) ----------
cleanup() {
  local rc=$?
  log "cleanup: stopping clock-lock controller"
  docker exec "$CONTROLLER" nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
  docker rm -f "$CONTROLLER" >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT INT TERM ERR

log "spawning clock-lock controller (--privileged, GPU=${GPU_MHZ}MHz)"
docker rm -f "$CONTROLLER" >/dev/null 2>&1 || true
docker run -d --privileged --gpus all --name "$CONTROLLER" \
  -v "$REPO_DIR/bench:/bench:ro" \
  nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04 \
  bash -c ". /bench/_clocklock.sh; lock_clocks ${GPU_MHZ}; sleep infinity" >/dev/null

log "polling for clock-lock confirmation (30s timeout)"
LOCKED=0
for i in $(seq 1 30); do
  if docker logs "$CONTROLLER" 2>&1 | grep -q '\[clocklock\] locked gpu'; then
    LOCKED=1; break
  fi
  if docker logs "$CONTROLLER" 2>&1 | grep -q '\[clocklock\] FATAL'; then
    log "FATAL: NVML rejected clock lock"
    docker logs "$CONTROLLER" 2>&1 | sed 's/^/  /'
    exit 2
  fi
  sleep 1
done
[[ "$LOCKED" == "1" ]] || { log "FATAL: clock lock did not confirm within 30s"; exit 2; }
log "clock lock active; scenario=$SCENARIO; bench-base=$BENCH_BASE_IMAGE"

# ---------- shared per-wheel post-install steps ----------
# bench-base image already has all deps. Per-wheel we only swap torch.
# --no-deps prevents the wheel's own pinned triton from being reinstalled
# (we already have triton>=3.6 baked in to avoid the sm_121a PTXAS bug).
read -r -d '' RUN_WITH_BASE <<'SH' || true
  uv pip install --system --no-deps --force-reinstall "$TORCH_INSTALL_SPEC"
  python /repo/bench/build/build_fa4.py
  python /repo/bench/run_tiers.py --json
SH

# Run C uses NGC base; install our overlay on top of NGC's existing python env.
read -r -d '' RUN_NGC_OVERLAY <<'SH' || true
  apt-get update -qq
  apt-get install -y -qq nsight-compute-2026.1
  python3 -m pip install --quiet uv==0.11.15
  uv pip install --system --no-cache \
    hf-transfer ncu-report==2025.3.1
  uv pip install --system --no-cache --no-deps \
    "triton>=3.6" nvidia-cutlass-dsl \
    "transformers>=4.55,<5.0" "accelerate>=1.0" \
    bitsandbytes==0.49.2 \
    optimum-benchmark==0.6.0 flash-attn-4==4.0.0b13
  uv pip install --system --no-cache \
    hydra-core>=1.3 omegaconf>=2.3 psutil pyyaml typing_extensions tqdm \
    safetensors huggingface_hub sentencepiece tokenizers protobuf
  python /repo/bench/build/build_fa4.py
  python /repo/bench/run_tiers.py --json
SH

# ---------- Run A : PyPI torch==2.10.0+cu130 (bench-base + wheel swap) ----------
log "Run A : PyPI torch==2.10.0+cu130"
docker run --rm --gpus all --ipc=host --cap-add=SYS_ADMIN \
  "${DOCKER_ENV[@]}" \
  -v dgx-spark-hf-cache:/hf-cache \
  -v dgx-spark-uv-cache:/root/.cache/uv \
  -v dgx-spark-triton-cache-a:/root/.triton/cache \
  -v dgx-spark-cutlass-jit-a:/root/.cache/cutlass-jit \
  -v "$REPO_DIR":/repo:ro \
  -v "$LOGS:/logs" \
  -e HF_HOME=/hf-cache \
  -e UV_CACHE_DIR=/root/.cache/uv \
  -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e CUTLASS_CACHE_DIR=/root/.cache/cutlass-jit \
  -e PYTORCH_KERNEL_CACHE_PATH=/root/.triton/cache \
  -e BENCH_LOG_DIR=/logs \
  -e TORCH_INSTALL_SPEC="torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130" \
  "$BENCH_BASE_IMAGE" \
  bash -c "set -euo pipefail; ${RUN_WITH_BASE}" > "$LOGS/runA.json" 2>>"$LOGS/runA.log"
A_RC=$?
log "Run A exit: ${A_RC}"

# ---------- Run B : source-built wheel ----------
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
  -e HF_HOME=/hf-cache \
  -e UV_CACHE_DIR=/root/.cache/uv \
  -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e CUTLASS_CACHE_DIR=/root/.cache/cutlass-jit \
  -e PYTORCH_KERNEL_CACHE_PATH=/root/.triton/cache \
  -e BENCH_LOG_DIR=/logs \
  "$BENCH_BASE_IMAGE" \
  bash -c "
    set -euo pipefail
    export TORCH_INSTALL_SPEC=\$(ls /work/pytorch/dist/torch-*.whl | head -1)
    ${RUN_WITH_BASE}
  " > "$LOGS/runB.json" 2>>"$LOGS/runB.log"
B_RC=$?
log "Run B exit: ${B_RC}"

# ---------- Run C : NGC vendor reference (overlay install) ----------
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
  -e HF_HOME=/hf-cache \
  -e UV_CACHE_DIR=/root/.cache/uv \
  -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e CUTLASS_CACHE_DIR=/root/.cache/cutlass-jit \
  -e PYTORCH_KERNEL_CACHE_PATH=/root/.triton/cache \
  -e BENCH_LOG_DIR=/logs \
  nvcr.io/nvidia/pytorch:26.04-py3 \
  bash -c "set -euo pipefail; export DEBIAN_FRONTEND=noninteractive; ${RUN_NGC_OVERLAY}" > "$LOGS/runC.json" 2>>"$LOGS/runC.log"
C_RC=$?
log "Run C exit: ${C_RC}"

# ---------- aggregate ----------
log "building SUMMARY.txt"
python3 "$REPO_DIR/bench/_summarize.py" "$LOGS" || true
log "exit codes: A=${A_RC}  B=${B_RC}  C=${C_RC}"

chown -R "$(id -u):$(id -g)" "$LOGS" 2>/dev/null || true
log "Done. See $LOGS/SUMMARY.txt"
