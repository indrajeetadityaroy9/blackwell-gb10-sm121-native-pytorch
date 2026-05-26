#!/usr/bin/env bash
# Tune-with-clock-lock launcher.
#
# Unlocked, GB10's GPU clock floats and benchmark latency varies ~30% — far above the
# loop's keep threshold, so keep/revert decisions are driven by noise. GPU clock locks
# are device-global, so a privileged sidecar locks the clock once and the unprivileged
# tune container inherits it. Clocks are reset on exit.
#
# Usage:  bash run_tune.sh --definition <name> --baseline <path> [tune.py args...]
set -euo pipefail

MHZ="${GB10_LOCK_MHZ:-2418}"          # GB10 Default Applications clock
IMAGE="dgx-spark-tune:cuda13.2"
LOCK_NAME="gb10-tune-clocklock"
RUNS_DIR="$(pwd)/_exp_runs"

cleanup() {
  docker exec "$LOCK_NAME" nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
  docker rm -f "$LOCK_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$RUNS_DIR"
docker rm -f "$LOCK_NAME" >/dev/null 2>&1 || true
docker run -d --privileged --gpus all --name "$LOCK_NAME" "$IMAGE" \
  bash -c "nvidia-smi --lock-gpu-clocks=${MHZ},${MHZ}; sleep infinity" >/dev/null

# Wait for the lock to take effect (NVML can lag a beat).
for _ in $(seq 1 15); do
  cur=$(docker exec "$LOCK_NAME" nvidia-smi --query-gpu=clocks.applications.gr \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  [[ -n "$cur" && "$cur" != "[N/A]" ]] && break
  sleep 1
done
echo "[run_tune] GPU clock locked at ${MHZ} MHz (applications.gr=${cur:-unknown})" >&2

docker run --rm --gpus all \
  -e GB10_RUNS=/exp_runs \
  -e GB10_LOCKED_MHZ="${MHZ}" \
  -v gb10-swarm-cache:/root/.cache/huggingface \
  -v "${RUNS_DIR}:/exp_runs" \
  "$IMAGE" \
  python3 -m tune.tune "$@"
