#!/usr/bin/env bash
# NVML GPU clock-lock helper, run INSIDE a --privileged container.
# The host never executes nvidia-smi clock commands; only `docker run` calls.
#
# Container topology: a small `dgx-bench-clocklock` container holds the lock
# for the entire bake-off duration. run_bakeoff.sh spawns it before runs A/B/C
# and tears it down via EXIT trap (handles Ctrl+C, crashes, normal exit).
#
# GB10 quirks (verified on this hardware):
# - Memory clock = "N/A" (Grace LPDDR5X unified memory) — only GPU clock is lockable
# - --lock-gpu-clocks requires <min,max> pair, e.g. 2418,2418 (single value rejected)
# - --supported-clocks returns N/A; use --query-gpu=clocks.max.gr for the ceiling

set -euo pipefail

lock_clocks() {
  local mhz="${1:-2418}"   # GB10 Default Applications clock (Max boost is 3003)
  echo "[clocklock] attempting NVML lock at ${mhz} MHz (GPU only; mem clock N/A on GB10)" >&2
  if ! nvidia-smi --lock-gpu-clocks="${mhz},${mhz}" >/tmp/lock.out 2>&1; then
    echo "[clocklock] FATAL: NVML --lock-gpu-clocks failed" >&2
    cat /tmp/lock.out >&2
    exit 2
  fi
  cat /tmp/lock.out >&2 || true

  # Confirm the lock actually took effect (NVML can silently no-op on some SKUs).
  local actual
  actual=$(nvidia-smi --query-gpu=clocks.applications.gr --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ -z "${actual}" || "${actual}" == "[N/A]" ]]; then
    echo "[clocklock] WARN: cannot read applications clock to verify lock (driver reports N/A)" >&2
  fi
  echo "[clocklock] locked gpu=${mhz}MHz (driver-reported applications.gr=${actual:-unknown})" >&2
}

unlock_clocks() {
  nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
  echo "[clocklock] unlocked" >&2
}

# If executed directly (not sourced), forward to lock_clocks.
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
  lock_clocks "${1:-2418}"
fi
