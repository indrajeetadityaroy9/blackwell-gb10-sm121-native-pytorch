"""
NCU roofline profiler. Wraps `ncu --set roofline` with kernel-name
filtering and a launch budget so profiling a model forward (~10k+ kernels
under NCU's 10-30× replay overhead) completes in minutes, not hours.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Result
from normalize import from_ncu


def profile(command: list[str], rep_path: Path) -> list[Result]:
    """Run `command` under `ncu --set roofline` (filtering to GEMM /
    attention / cutlass kernels, capping at 50 captures); parse the
    .ncu-rep and return one Result per profiled kernel."""
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ncu",
        "--set", "roofline",
        # NCU 2026.1 syntax: --kernel-name regex:<expr> for name filtering.
        # The older --name-filter / --filter-mode pair isn't recognized.
        "--kernel-name", r"regex:(?i)(gemm|mma|flash|sdpa|attention|cutlass)",
        # Budget 200 captures ~3-5 launches per unique GEMM kernel for
        # kernel_bench (16 shapes × multiple iters); enough for stable
        # per-kernel achieved-% values without exploding NCU replay time.
        "--launch-count", "200",
        "--target-processes", "all",
        "--force-overwrite",
        "--export", str(rep_path),
        "--",
        *command,
    ]
    print(f"[roofline] {' '.join(cmd)}", file=sys.stderr)
    # NCU writes "==PROF==" progress lines to stdout; redirect to stderr so
    # they don't corrupt the JSON document run_tiers.py emits on its stdout.
    subprocess.run(cmd, check=True, stdout=sys.stderr)
    return from_ncu(rep_path)
