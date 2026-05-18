#!/usr/bin/env bash
# NVML GPU clock-lock helper. Runs inside a --privileged container.
#
# GB10: only GPU clock is lockable (mem clock = N/A on LPDDR5X).
# --lock-gpu-clocks requires <min,max> pair.

set -euo pipefail

lock_clocks() {
  local mhz="${1:-2418}"   # GB10 Default Applications clock; Max boost 3003
  echo "[clocklock] locking at ${mhz} MHz" >&2
  if ! nvidia-smi --lock-gpu-clocks="${mhz},${mhz}" >/tmp/lock.out 2>&1; then
    echo "[clocklock] FATAL: NVML --lock-gpu-clocks failed" >&2
    cat /tmp/lock.out >&2
    exit 2
  fi
  cat /tmp/lock.out >&2 || true

  # Verify the lock took effect (NVML can silently no-op on some SKUs).
  local actual
  actual=$(nvidia-smi --query-gpu=clocks.applications.gr --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ -z "${actual}" || "${actual}" == "[N/A]" ]]; then
    echo "[clocklock] WARN: driver reports N/A for applications.gr" >&2
  fi
  echo "[clocklock] locked gpu=${mhz}MHz (applications.gr=${actual:-unknown})" >&2
}

unlock_clocks() {
  nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
  echo "[clocklock] unlocked" >&2
}

# When executed (not sourced), forward to lock_clocks.
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
  lock_clocks "${1:-2418}"
fi
