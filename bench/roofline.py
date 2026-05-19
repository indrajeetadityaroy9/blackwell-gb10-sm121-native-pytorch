"""
NCU roofline profiler for run_tiers.py.

Wraps `ncu --set roofline` with kernel-name filtering and a launch budget.
Without the filter+budget, profiling a model forward (~10k+ kernels with
NCU's 10-30× replay overhead per kernel) would hang for hours. The filter
restricts capture to compute/memory-dominant kernels (GEMM, MMA, attention,
cutlass primitives); --launch-count bounds the captured set.

Parsing uses normalize.from_ncu (ncu_report Python API; SQLite-backed
.ncu-rep is stable across NCU minor versions, stdout text is not).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Result
from normalize import from_ncu


# Heavy-kernel filter — capture only compute/memory-dominant kernels.
_NAME_FILTER = r"regex:(?i)(gemm|mma|flash|sdpa|attention|cutlass)"

# 50 captured kernels covers a Llama-3-8B forward's full attention + MLP
# layer set (~10 matmul-class kernels per layer × 2 layers profiled).
_LAUNCH_COUNT = 50


def profile(command: list[str], rep_path: Path) -> list[Result]:
    """Run `command` under `ncu --set roofline` with kernel filtering and
    launch budget; parse the .ncu-rep and return one Result per profiled
    kernel."""
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
