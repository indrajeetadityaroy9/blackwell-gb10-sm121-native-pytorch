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
        "--filter-mode", "name",
        "--name-filter", r"regex:(?i)(gemm|mma|flash|sdpa|attention|cutlass)",
        "--launch-count", "50",
        "--target-processes", "all",
        "--force-overwrite",
        "--export", str(rep_path),
        "--",
        *command,
    ]
    print(f"[roofline] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return from_ncu(rep_path)
