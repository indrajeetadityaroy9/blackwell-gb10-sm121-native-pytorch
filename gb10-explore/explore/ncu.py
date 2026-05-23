"""NCU report parsing — extracts kernel_name, sol_sm_pct, sol_mem_pct, limit
for the agent diagnostic prompt. Logic mirrors bench/normalize.from_ncu.
"""

from pathlib import Path
from typing import Dict, Any


def parse_report(rep_path: Path) -> Dict[str, Any]:
    """Parse a .ncu-rep via the ncu_report Python API.

    Returns dict with the longest-duration kernel's metrics:
      kernel_name, duration_ms, sol_sm_pct, sol_mem_pct, limit ('compute' or 'bandwidth')
    """
    import ncu_report

    ctx = ncu_report.load_report(str(rep_path))
    best: Dict[str, Any] = {}
    best_duration = -1.0

    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            act = rng.action_by_idx(ai)
            duration_ms = act.metric_by_name("gpu__time_duration.sum").as_double() / 1e6
            sol_sm = act.metric_by_name(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed"
            ).as_double()
            sol_mem = act.metric_by_name(
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
            ).as_double()
            if duration_ms > best_duration:
                best_duration = duration_ms
                best = {
                    "kernel_name": act.name(),
                    "duration_ms": duration_ms,
                    "sol_sm_pct": sol_sm,
                    "sol_mem_pct": sol_mem,
                    "limit": "compute" if sol_sm >= sol_mem else "bandwidth",
                }
    return best
