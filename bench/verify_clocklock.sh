#!/usr/bin/env bash
# Phase 0 GATE: verify that NVML --lock-gpu-clocks works on this GB10.
# Spawns a --privileged container, attempts the lock at 2418 MHz, polls
# nvidia-smi to confirm the SM clock matches, unlocks, prints PASS/FAIL.
#
# Outcome:
#   exit 0 + bench/logs/phase0_verify.log "PASS" → Tier 1 may include clock locking
#   exit 2 + "FAIL: <reason>"                    → Tier 1 ships without clock lock;
#                                                  docs/sol-score.md gains a caveat.
#
# Run: bash bench/verify_clocklock.sh
# Requires: docker access (sg docker -c if shell predates docker group membership)

set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/bench/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/phase0_verify.log"
CONTAINER=dgx-bench-clocklock-verify

cleanup() {
  docker exec "$CONTAINER" nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM ERR

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

: > "$LOG"
log "Phase 0: NVML clock-lock verification on GB10"

# 1. Spawn privileged controller container (long-running so we can docker exec into it)
log "spawning ${CONTAINER} container (--privileged --gpus all)"
docker run -d --privileged --gpus all --name "$CONTAINER" \
  -v "$REPO_DIR/bench:/bench:ro" \
  nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04 \
  bash -c '. /bench/_clocklock.sh; lock_clocks 2418; sleep 300' >/dev/null

# 2. Poll for confirmation (up to 30s; driver init can take >2s)
log "polling for lock confirmation (30s timeout)"
LOCKED=0
for i in $(seq 1 30); do
  if docker logs "$CONTAINER" 2>&1 | grep -q '\[clocklock\] locked gpu'; then
    LOCKED=1; break
  fi
  if docker logs "$CONTAINER" 2>&1 | grep -q '\[clocklock\] FATAL'; then
    log "FAIL: NVML rejected clock-lock inside container"
    docker logs "$CONTAINER" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
    echo "FAIL" >> "$LOG"
    exit 2
  fi
  sleep 1
done

if [[ "$LOCKED" != "1" ]]; then
  log "FAIL: lock did not confirm within 30s"
  docker logs "$CONTAINER" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  echo "FAIL" >> "$LOG"
  exit 2
fi

# 3. Confirm the lock actually took effect — query SM clock from a separate exec
log "querying SM clock to confirm 2418 MHz"
SM_CLOCK=$(docker exec "$CONTAINER" nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
log "  driver reports clocks.sm=${SM_CLOCK} MHz (target 2418)"

# 4. Verify within 5% of target (driver may report slightly different under load)
TARGET=2418
if [[ -z "$SM_CLOCK" || "$SM_CLOCK" == "[N/A]" ]]; then
  log "FAIL: cannot read SM clock from driver"
  echo "FAIL" >> "$LOG"
  exit 2
fi
DIFF=$(( SM_CLOCK > TARGET ? SM_CLOCK - TARGET : TARGET - SM_CLOCK ))
PCT_DIFF=$(( DIFF * 100 / TARGET ))
if [[ "$PCT_DIFF" -gt 5 ]]; then
  log "FAIL: SM clock ${SM_CLOCK} differs from target ${TARGET} by ${PCT_DIFF}% (>5% tolerance)"
  echo "FAIL" >> "$LOG"
  exit 2
fi

# 5. Test unlock
log "testing unlock"
docker exec "$CONTAINER" nvidia-smi --reset-gpu-clocks >/dev/null 2>&1
sleep 1
SM_AFTER=$(docker exec "$CONTAINER" nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
log "  after unlock: clocks.sm=${SM_AFTER} MHz"

log "PASS — NVML lock-gpu-clocks works on this GB10"
echo "PASS" >> "$LOG"
exit 0
