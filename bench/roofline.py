"""
NCU roofline profiler — importable from run_tiers.py and standalone CLI.

Wraps `ncu --set roofline` with kernel-name filtering and a launch budget so
profiling a model forward (~10k+ kernels, 10-30× NCU replay overhead per
kernel) completes in minutes, not hours.

The filter regex restricts capture to compute/memory-dominant kernels (GEMM,
MMA, attention, cutlass primitives) — small element-wise ops and CPU-side
launches pass through at native speed. --launch-count caps the captured set
so warm-iter measurements aren't displaced by warmup spam.

Parsing uses the ncu_report Python API (SQLite-backed .ncu-rep; stable
across NCU minor versions) — not stdout regex.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Make _harness importable from same dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize import from_ncu


# Metrics from ncu's roofline rule, verified on Blackwell sm_121.
METRICS = {
    "sol_sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sol_mem": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
}

# Heavy-kernel filter. Restrict capture to compute and memory-dominant
# kernels; pass-through for element-wise, copy, launch overhead, etc.
_NAME_FILTER = r"regex:(?i)(gemm|mma|flash|sdpa|attention|cutlass)"

# Launch budget — 50 captured kernels is enough to characterize the
# attention + MLP layers of a single Llama-3-8B forward (~2 layers × 5
# matmul-class kernels = 10 kernels per pass; budget covers 5+ passes).
_LAUNCH_COUNT = 50


def profile(command: list[str], rep_path: Path) -> list:
    """Run command under `ncu --set roofline` with kernel filtering and
    launch budget, then parse the resulting .ncu-rep via normalize.from_ncu.

    Returns: list[Result] (defined in _harness.py).
    """
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ncu",
        "--set", "roofline",
        "--filter-mode", "name",
        "--name-filter", _NAME_FILTER,
        "--launch-count", str(_LAUNCH_COUNT),
        "--target-processes", "all",
        "--force-overwrite",
        "--export", str(rep_path),
        "--",
        *command,
    ]
    print(f"[roofline] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return from_ncu(rep_path)


def _cli_main() -> int:
    """Standalone CLI for ad-hoc debugging: profile an arbitrary command and
    emit JSON. Not used by run_tiers.py (which calls profile() directly)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--rep", default="/tmp/roofline.ncu-rep",
                    help="path to write .ncu-rep")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON to stdout instead of human-readable")
    ap.add_argument("command", nargs="+",
                    help="command to profile (everything after the option flags)")
    args = ap.parse_args()

    results = profile(args.command, Path(args.rep))

    if args.json:
        from dataclasses import asdict
        print(json.dumps(
            {"results": [asdict(r) for r in results]},
            indent=2, default=str,
        ))
    else:
        print(f"\nProfiled {len(results)} kernels:")
        for r in results:
            print(f"  {r.name[:60]:60s}  {r.measured:7.3f} ms  "
                  f"sol_score={r.sol_score:.3f}  limit={r.sol_limit}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
