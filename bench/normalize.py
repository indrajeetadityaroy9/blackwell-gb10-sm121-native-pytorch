"""
Schema adapters for the three bake-off tiers; each returns list[Result].

  from_ncu     Nsight Compute roofline report → per-kernel Result
  from_optimum optimum-benchmark Hydra run dir → per-op Result
  from_kernel  kernel_bench.py stdout JSON    → per-shape Result
"""

from __future__ import annotations

import json
from pathlib import Path

from _harness import Result, Stats


def from_ncu(rep_path: Path) -> list[Result]:
    """Parse a .ncu-rep via the ncu_report Python API.
    measured = kernel duration in ms; sol = back-derived ideal duration
    (duration × achieved_pct / 100). _summarize.py's gap-closure math
    `(m_B - m_A) / (sol_A - m_A)` then yields a score where 1.0 = at peak.
    """
    import ncu_report
    ctx = ncu_report.load_report(str(rep_path))
    results: list[Result] = []

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

            achieved_pct = max(sol_sm, sol_mem)
            results.append(Result(
                name=act.name(),
                unit="ms",
                measured=duration_ms,
                sol=duration_ms * achieved_pct / 100.0,
                stats=Stats.from_samples([duration_ms]),
                extra={
                    "tier": "roofline",
                    "achieved_pct": achieved_pct,
                    "limit": "compute" if sol_sm >= sol_mem else "bandwidth",
                    "sol_sm_pct": sol_sm,
                    "sol_mem_pct": sol_mem,
                },
            ))

    return results


def from_optimum(run_dir: Path) -> list[Result]:
    """Parse optimum-benchmark Hydra run dir → list[Result].
    benchmark_report.json is flat: {<op_name>: TargetMeasurements, ...}.
    Latency values are in seconds (TimeUnit.SECOND); converted to ms."""
    report = json.loads((run_dir / "benchmark_report.json").read_text())
    model_id = run_dir.name

    results: list[Result] = []
    for op_name, op_data in report.items():
        latency = op_data["latency"]
        stats = Stats.from_samples([v * 1000.0 for v in latency["values"]])

        extra: dict = {
            "tier": "optimum",
            "model": model_id,
            "op": op_name,
        }
        # throughput is non-None only for ops that measure it (generate,
        # forward, decode); None for ops like `load` that report latency only.
        thr = op_data.get("throughput")
        if thr is not None:
            extra["throughput"] = thr["value"]
            extra["throughput_unit"] = thr["unit"]

        results.append(Result(
            name=f"optimum/{model_id}/{op_name}",
            unit="ms",
            measured=stats.mean_ms,
            sol=None,
            stats=stats,
            extra=extra,
        ))

    return results


def from_kernel(stdout: str) -> list[Result]:
    """Parse kernel_bench.py stdout JSON; stamp tier='kernel' into each extra."""
    doc = json.loads(stdout)
    return [
        Result(
            name=r["name"],
            unit=r["unit"],
            measured=r["measured"],
            sol=r["sol"],
            stats=Stats(**r["stats"]),
            extra={**r["extra"], "tier": "kernel"},
        )
        for r in doc["results"]
    ]
