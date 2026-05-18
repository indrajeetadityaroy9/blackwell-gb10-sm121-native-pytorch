#!/usr/bin/env bash
# Tier 4 driver — end-to-end LLM benchmarks. Inherits the clock-lock controller
# pattern from run_bakeoff.sh.
#
# Components:
#   1. FlashInfer attention serving (replaces hand-rolled FA test)
#   2. MLPerf v5.1 llama3.1-8b inference (tokens/s, TTFT, ITL p50/p99)
#
# Gated by env vars (both required for full Tier 4):
#   BENCH_DOWNLOAD_MODELS=1   accept 16 GB Llama 3.1 8B HF download
#
# Output: appends to bench/logs/SUMMARY.txt under a new "## End-to-End" heading.

set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS="$REPO_DIR/bench/logs"; mkdir -p "$LOGS/mlperf"

CONTROLLER=dgx-bench-clocklock-e2e
GPU_MHZ="${BENCH_GPU_MHZ:-2418}"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }

cleanup() {
  log "cleanup: releasing clock lock"
  docker exec "$CONTROLLER" nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
  docker rm -f "$CONTROLLER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM ERR

log "spawning clock-lock controller (--privileged, GPU=${GPU_MHZ}MHz)"
docker rm -f "$CONTROLLER" >/dev/null 2>&1 || true
docker run -d --privileged --gpus all --name "$CONTROLLER" \
  -v "$REPO_DIR/bench:/bench:ro" \
  nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04 \
  bash -c ". /bench/_clocklock.sh; lock_clocks ${GPU_MHZ}; sleep infinity" >/dev/null

# Poll for confirmation
for i in $(seq 1 30); do
  docker logs "$CONTROLLER" 2>&1 | grep -q '\[clocklock\] locked gpu' && break
  sleep 1
done

# --- FlashInfer ---
log "Tier 4 / FlashInfer: requires source-built flashinfer wheel"
if docker run --rm -v dgx-spark-build-strict:/v alpine \
     sh -c 'ls /v/flashinfer/dist/flashinfer*.whl 2>/dev/null | head -1' | grep -q '.whl'; then
  log "FlashInfer wheel present — running serve_flashinfer.py"
  docker run --rm --gpus all --ipc=host --shm-size=4g \
    -v dgx-spark-build-strict:/work:ro \
    -v "$REPO_DIR":/repo:ro \
    -v "$LOGS:/logs" \
    nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04 \
    bash -c '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq >/dev/null
      apt-get install -y -qq python3 python3-venv python3-pip ca-certificates \
        libopenblas0 libnuma1 cudnn9-cuda-13-2 cusparselt-cuda-13 libcusparselt0-cuda-13 >/dev/null
      python3 -m venv /tmp/v && . /tmp/v/bin/activate
      pip install -q --upgrade pip
      pip install -q "$(ls /work/pytorch/dist/torch-*.whl | head -1)"
      pip install -q "$(ls /work/flashinfer/dist/flashinfer*.whl | head -1)"
      pip install -q flashinfer-bench==0.1.2
      python /repo/bench/e2e/serve_flashinfer.py --json > /logs/flashinfer.json 2>>/logs/flashinfer.log
    ' >>"$LOGS/flashinfer.log" 2>&1
  log "flashinfer exit: $?"
else
  log "Tier 4 / FlashInfer: SKIPPED (run bench/e2e/build_flashinfer.sh first)"
fi

# --- MLPerf v5.1 ---
log "Tier 4 / MLPerf v5.1 llama3.1-8b"
if [[ "${BENCH_DOWNLOAD_MODELS:-0}" == "1" ]]; then
  python3 "$REPO_DIR/bench/e2e/mlperf_llama31_8b.py" \
    --scenario Offline --duration 60 --out "$LOGS/mlperf_llama31_8b.json"
  log "mlperf exit: $?"
else
  log "Tier 4 / MLPerf: SKIPPED (set BENCH_DOWNLOAD_MODELS=1 for 16 GB Llama 3.1 8B download)"
fi

# Append to SUMMARY.txt
{
  echo ""
  echo "## End-to-End (Tier 4)"
  if [[ -f "$LOGS/flashinfer.json" ]]; then
    echo ""; echo "### FlashInfer attention serving"
    python3 -c "
import json
d = json.load(open('$LOGS/flashinfer.json'))
for r in d.get('results', []):
    s = r['stats']
    print(f\"  {r['name']:60s} : {r['measured']:7.2f} {r['unit']} (med={s['median_ms']:.2f}ms)\")
" 2>/dev/null || echo "  (could not parse flashinfer.json)"
  fi
  if [[ -f "$LOGS/mlperf_llama31_8b.json" ]]; then
    echo ""; echo "### MLPerf v5.1 llama3.1-8b"
    python3 -c "
import json
d = json.load(open('$LOGS/mlperf_llama31_8b.json'))
for k in ('tokens_per_sec', 'ttft_p50_ms', 'ttft_p99_ms', 'itl_p50_ms', 'itl_p99_ms'):
    if k in d: print(f'  {k:25s}: {d[k]}')
" 2>/dev/null || echo "  (could not parse mlperf json)"
  fi
} >> "$LOGS/SUMMARY.txt"

log "Done. See $LOGS/SUMMARY.txt"
