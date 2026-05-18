#!/usr/bin/env bash
# 3-way DGX Spark bake-off — SOL-ExecBench methodology (arXiv 2603.19173).
#   A : PyPI torch==2.9.0+cu130             (cuda:13.0.0-devel)
#   B : Source-built wheel, native sm_121   (volume dgx-spark-build-strict)
#   C : NGC pytorch:26.04-py3
#
# All execution inside Docker:
#  - --privileged controller holds GPU clock lock; trap-based unlock on exit
#  - Subprocess-per-test isolation (SOL-ExecBench)
#  - Triton cache per-container (sm_120 ≠ sm_121 backend.hash())
#  - Per-wheel JSON aggregated by bench/_summarize.py into SUMMARY.txt
#
# Env passthrough:
#   BENCH_ONLY      same as --only (e.g. "fp4")
#   BENCH_GPU_MHZ   clock-lock target (default 2418)
#   BENCH_ITERS     timed iters per test (default 50)
#   BENCH_WARMUP    warmup iters (default 5)
#   BENCH_M         GEMM dim (default 8192)
#   BENCH_PROFILE   1 → also collect ncu roofline data (Tier 3 opt-in)

set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$REPO_DIR/bench/logs"; mkdir -p "$LOGS"
TRITON_ROOT="$REPO_DIR/bench/cache/triton/sm121"
mkdir -p "$TRITON_ROOT"/{runA,runB,runC}

CONTROLLER=dgx-bench-clocklock
GPU_MHZ="${BENCH_GPU_MHZ:-2418}"

ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }

# ---------- forward env vars into each docker run ----------
DOCKER_ENV=()
for v in BENCH_ONLY BENCH_ITERS BENCH_WARMUP BENCH_M BENCH_PROFILE; do
  [[ -n "${!v:-}" ]] && DOCKER_ENV+=(-e "$v=${!v}")
done

# ---------- clock-lock controller ----------
# --privileged so NVML can lock GPU clocks; bench containers stay unprivileged.
# Trap catches Ctrl+C / crash / normal exit.
cleanup() {
  local rc=$?
  log "cleanup: stopping clock-lock controller"
  docker exec "$CONTROLLER" nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
  docker rm -f "$CONTROLLER" >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT INT TERM ERR

log "spawning clock-lock controller container (--privileged, GPU=${GPU_MHZ}MHz)"
docker rm -f "$CONTROLLER" >/dev/null 2>&1 || true
docker run -d --privileged --gpus all --name "$CONTROLLER" \
  -v "$REPO_DIR/bench:/bench:ro" \
  nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04 \
  bash -c ". /bench/_clocklock.sh; lock_clocks ${GPU_MHZ}; sleep infinity" >/dev/null

# Poll up to 30s for lock confirmation
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
log "clock lock active"

# ---------- Run A : PyPI baseline ----------
log "Run A : PyPI torch==2.9.0+cu130 + triton"
docker run --rm --gpus all --ipc=host --shm-size=4g \
  "${DOCKER_ENV[@]}" \
  -v "$REPO_DIR":/repo:ro \
  -v "$TRITON_ROOT/runA:/root/.triton/cache" \
  -v "$LOGS:/logs" \
  nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 \
  bash -c '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null
    apt-get install -y -qq python3 python3-venv python3-pip ca-certificates >/dev/null
    python3 -m venv /tmp/v && . /tmp/v/bin/activate
    pip install -q --upgrade pip 2>&1 | tail -1
    pip install -q --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.9.0+cu130 2>&1 | tail -1
    pip install -q triton 2>&1 | tail -1 || true
    python /repo/bench/bench_full.py --json > /logs/runA.json 2>>/logs/runA.log
  ' >>"$LOGS/runA.log" 2>&1
A_RC=$?
log "Run A exit: ${A_RC}"

# ---------- Run B : source-built wheel ----------
# Requires bench/build/source_build.sh to have populated dgx-spark-build-strict
# with /work/pytorch/dist/torch-*.whl; pip install fails fast if absent.
log "Run B : source-built wheel (dgx-spark-build-strict)"
docker run --rm --gpus all --ipc=host --shm-size=4g \
  "${DOCKER_ENV[@]}" \
  -v dgx-spark-build-strict:/work:ro \
  -v "$REPO_DIR":/repo:ro \
  -v "$TRITON_ROOT/runB:/root/.triton/cache" \
  -v "$LOGS:/logs" \
  nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04 \
  bash -c '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null
    apt-get install -y -qq python3 python3-venv python3-pip ca-certificates \
      libopenblas0 libnuma1 cudnn9-cuda-13-2 cusparselt-cuda-13 libcusparselt0-cuda-13 >/dev/null
    python3 -m venv /tmp/v && . /tmp/v/bin/activate
    pip install -q --upgrade pip 2>&1 | tail -1
    WHEEL=$(ls /work/pytorch/dist/torch-*.whl | head -1)
    pip install -q "$WHEEL" triton 2>&1 | tail -1
    python /repo/bench/bench_full.py --json > /logs/runB.json 2>>/logs/runB.log
  ' >>"$LOGS/runB.log" 2>&1
B_RC=$?
log "Run B exit: ${B_RC}"

# ---------- Run C : NGC vendor reference ----------
log "Run C : NGC nvcr.io/nvidia/pytorch:26.04-py3"
docker run --rm --gpus all --ipc=host --shm-size=4g \
  "${DOCKER_ENV[@]}" \
  -v "$REPO_DIR":/repo:ro \
  -v "$TRITON_ROOT/runC:/root/.triton/cache" \
  -v "$LOGS:/logs" \
  nvcr.io/nvidia/pytorch:26.04-py3 \
  bash -c '
    python /repo/bench/bench_full.py --json > /logs/runC.json 2>>/logs/runC.log
  ' >>"$LOGS/runC.log" 2>&1
C_RC=$?
log "Run C exit: ${C_RC}"

# ---------- aggregate ----------
log "building SUMMARY.txt (3-way SOL Score table)"
python3 "$REPO_DIR/bench/_summarize.py" "$LOGS" || true
log "exit codes: A=${A_RC}  B=${B_RC}  C=${C_RC}"

# Fix log ownership (root inside container → host user)
chown -R "$(id -u):$(id -g)" "$LOGS" 2>/dev/null || true
log "Done. See $LOGS/SUMMARY.txt"
