"""
Schema adapters for the three bake-off tiers.

Each function returns list[Result] (the schema in _harness.py) so the
existing _summarize.py aggregator consumes them unchanged.

  from_ncu     — Nsight Compute roofline report → per-kernel Result
  from_optimum — optimum-benchmark Hydra run dir → per-scenario Result
  from_fa4     — fa4_bench.py stdout JSON       → per-shape Result
"""

from __future__ import annotations

import json
from pathlib import Path

import ncu_report

from _harness import Result, Stats


# NCU roofline metrics; sourced from roofline.py:37-40.
_NCU_METRICS = {
    "sol_sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sol_mem": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
}


def from_ncu(rep_path: Path) -> list[Result]:
    """Parse a .ncu-rep via the ncu_report Python API.

    One Result per profiled kernel. measured = kernel duration in ms; sol =
    back-derived ideal duration (measured × max(sol_sm, sol_mem)/100). For a
    kernel achieving 30% of peak, sol = measured × 0.3, so _summarize.py's
    gap-closure math `(m - baseline) / (sol - baseline)` produces a score in
    [0,1] where 1.0 = at hardware peak.

    Why Python API only: the SQLite-backed .ncu-rep format is stable across
    NCU minor versions; the text output is not.
    """
    ctx = ncu_report.load_report(str(rep_path))
    results: list[Result] = []

    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            act = rng.action_by_idx(ai)

            duration_ns = act.metric_by_name("gpu__time_duration.sum").as_double()
            duration_ms = duration_ns / 1e6

            sol_sm = act.metric_by_name(_NCU_METRICS["sol_sm"]).as_double()
            sol_mem = act.metric_by_name(_NCU_METRICS["sol_mem"]).as_double()

            dominant_pct = max(sol_sm, sol_mem)
            sol_limit = "compute" if sol_sm >= sol_mem else "bandwidth"
            # Back-derived ideal duration: if kernel ran at peak, it would
            # take dominant_pct% of current time (since current is dominant_pct
            # of peak throughput).
            sol_ms = duration_ms * (dominant_pct / 100.0)

            results.append(Result(
                name=act.name(),
                unit="ms",
                measured=duration_ms,
                sol=sol_ms,
                sol_score=dominant_pct / 100.0,
                sol_limit=sol_limit,
                stats=Stats.from_samples([duration_ms]),
                correctness=None,
                extra={
                    "sol_sm_pct": sol_sm,
                    "sol_mem_pct": sol_mem,
                    "tier": "roofline",
                },
            ))

    return results


def from_optimum(run_dir: Path) -> list[Result]:
    """Parse optimum-benchmark Hydra run dir → Result list.

    Reads benchmark_report.json. One Result per measured operation
    (load, first_forward, forward, decode, prefill, etc.) depending on
    what the scenario produced.

    measured = latency mean in ms; sol = None (no hardware peak inferred
    at model granularity — that's what the roofline tier is for).
    """
    report_path = run_dir / "benchmark_report.json"
    doc = json.loads(report_path.read_text())

    # optimum-benchmark report shape:
    #   {"load": {"latency": {...}, "memory": {...}, "energy": {...}},
    #    "forward": {"latency": {...}, "throughput": {...}, ...},
    #    "decode": {...}, "prefill": {...}, ...}
    report = doc.get("report", doc)  # tolerate both {"report": ...} and flat
    model_id = run_dir.name

    results: list[Result] = []
    for op_name, op_data in report.items():
        if not isinstance(op_data, dict):
            continue
        latency = op_data.get("latency")
        if not latency or not isinstance(latency, dict):
            continue

        # latency dict typically has: mean, p50, p90, p95, p99, stdev, values[]
        values = latency.get("values", [])
        if not values:
            mean_ms = float(latency.get("mean", 0.0)) * 1000.0  # optimum reports s
            stats = Stats.from_samples([mean_ms])
        else:
            samples_ms = [v * 1000.0 for v in values]  # s → ms
            stats = Stats.from_samples(samples_ms)

        # Throughput (tokens/s or samples/s) if present.
        throughput = op_data.get("throughput", {})
        thr_value = throughput.get("value") if isinstance(throughput, dict) else None

        extra = {"tier": "optimum", "model": model_id, "op": op_name}
        if thr_value is not None:
            extra["throughput"] = thr_value
            extra["throughput_unit"] = throughput.get("unit", "tokens/s")

        results.append(Result(
            name=f"optimum/{model_id}/{op_name}",
            unit="ms",
            measured=stats.mean_ms,
            sol=None,
            sol_score=None,
            sol_limit=None,
            stats=stats,
            correctness=None,
            extra=extra,
        ))

    return results


def from_fa4(stdout: str) -> list[Result]:
    """Parse fa4_bench.py stdout JSON → Result list.

    fa4_bench.py emits {"results": [{name, unit, measured, stats, extra}, ...]}
    Each entry maps to one Result with sol=None (FA-4's TFLOPs are absolute;
    the NCU tier handles the peak-comparison dimension separately).
    """
    doc = json.loads(stdout)
    results: list[Result] = []
    for r in doc["results"]:
        stats = Stats(**r["stats"])
        extra = dict(r.get("extra", {}))
        extra["tier"] = "fa4"
        results.append(Result(
            name=r["name"],
            unit=r["unit"],
            measured=r["measured"],
            sol=None,
            sol_score=None,
            sol_limit=None,
            stats=stats,
            correctness=r.get("correctness"),
            extra=extra,
        ))
    return results
