"""
Schema adapters for the three bake-off tiers.

Each function returns list[Result] (the schema in _harness.py) so the
existing _summarize.py aggregator consumes them unchanged.

  from_ncu     — Nsight Compute roofline report → per-kernel Result
  from_optimum — optimum-benchmark Hydra run dir → per-op Result
  from_fa4     — fa4_bench.py stdout JSON       → per-shape Result
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _harness import Result, Stats


_NCU_METRICS = {
    "sol_sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sol_mem": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
}


def from_ncu(rep_path: Path) -> list[Result]:
    """Parse a .ncu-rep via the ncu_report Python API.

    One Result per profiled kernel. measured = kernel duration in ms; sol =
    back-derived ideal duration (duration × achieved_pct / 100). For a
    kernel achieving 30% of peak, sol = 0.3 × measured, so _summarize.py's
    gap-closure math `(m_B - m_A) / (sol_A - m_A)` produces a score where
    1.0 = at hardware peak.
    """
    import ncu_report
    ctx = ncu_report.load_report(str(rep_path))
    results: list[Result] = []

    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            act = rng.action_by_idx(ai)

            duration_ms = act.metric_by_name("gpu__time_duration.sum").as_double() / 1e6
            sol_sm = act.metric_by_name(_NCU_METRICS["sol_sm"]).as_double()
            sol_mem = act.metric_by_name(_NCU_METRICS["sol_mem"]).as_double()

            achieved_pct = max(sol_sm, sol_mem)
            limit = "compute" if sol_sm >= sol_mem else "bandwidth"
            sol_ms = duration_ms * (achieved_pct / 100.0)

            results.append(Result(
                name=act.name(),
                unit="ms",
                measured=duration_ms,
                sol=sol_ms,
                stats=Stats.from_samples([duration_ms]),
                extra={
                    "tier": "roofline",
                    "achieved_pct": achieved_pct,
                    "limit": limit,
                    "sol_sm_pct": sol_sm,
                    "sol_mem_pct": sol_mem,
                },
            ))

    return results


def from_optimum(run_dir: Path) -> list[Result]:
    """Parse optimum-benchmark Hydra run dir → list[Result].

    Expects `benchmark_report.json` produced by optimum-benchmark v0.6.0.
    Top-level shape:
        {"report": {<op_name>: {"latency": {...}, "throughput"?: {...}}, ...}}
    Latency values are in seconds (TimeUnit.SECOND); we convert to ms.
    """
    report = json.loads((run_dir / "benchmark_report.json").read_text())["report"]
    model_id = run_dir.name

    results: list[Result] = []
    for op_name, op_data in report.items():
        latency = op_data["latency"]
        samples_ms = [v * 1000.0 for v in latency["values"]]
        stats = Stats.from_samples(samples_ms)

        extra: dict[str, Any] = {
            "tier": "optimum",
            "model": model_id,
            "op": op_name,
        }
        throughput = op_data.get("throughput")
        if throughput is not None:
            extra["throughput"] = throughput["value"]
            extra["throughput_unit"] = throughput["unit"]

        results.append(Result(
            name=f"optimum/{model_id}/{op_name}",
            unit="ms",
            measured=stats.mean_ms,
            sol=None,
            stats=stats,
            extra=extra,
        ))

    return results


def from_fa4(stdout: str) -> list[Result]:
    """Parse fa4_bench.py stdout JSON → list[Result].

    fa4_bench.py emits the harness Result schema directly; this function
    rehydrates it as Result instances and stamps tier='fa4' into extra.
    """
    doc = json.loads(stdout)
    results: list[Result] = []
    for r in doc["results"]:
        extra = dict(r["extra"])
        extra["tier"] = "fa4"
        results.append(Result(
            name=r["name"],
            unit=r["unit"],
            measured=r["measured"],
            sol=r["sol"],
            stats=Stats(**r["stats"]),
            extra=extra,
        ))
    return results
