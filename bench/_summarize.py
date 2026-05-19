"""
Aggregate bench/logs/run{A,B,C}.json into SUMMARY.txt with SOL Scores.

Run A is the baseline:
  SOL Score(wheel, test) = (measured − baselineA) / (sol − baselineA)
N/A when Run A skipped.

Usage: python bench/_summarize.py [bench/logs/]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
WHEELS = [
    ("A", "PyPI torch==2.9.0+cu130 + triton"),
    ("B", "Source-built wheel (native sm_121 cubins)"),
    ("C", "NGC pytorch:26.04-py3 (vendor reference)"),
]


def load_run(path: Path) -> dict:
    return json.loads(path.read_text())


def sol_score(measured: float, baseline: float, sol: float | None) -> float | None:
    if sol is None or sol <= baseline:
        return None
    return max(0.0, min(1.0, (measured - baseline) / (sol - baseline)))


def main() -> int:
    logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "logs"
    runs = {label: load_run(logs_dir / f"run{label}.json") for label, _ in WHEELS}

    # Index per (wheel, test_name) -> result dict
    by_test: dict[str, dict[str, dict]] = {}
    for label, doc in runs.items():
        for r in doc["results"]:
            by_test.setdefault(r["name"], {})[label] = r
    test_names = list(by_test.keys())  # preserve first-seen order

    out: list[str] = []
    out.append("============= DGX Spark 3-way comparison =============")
    meta = next(iter(runs.values()))
    out.append(f"Date (run A ts): {meta['ts']}")
    out.append(f"Device: {meta['device_name']}  arch_list={meta['arch_list']}")
    out.append("")

    # Per-wheel detail blocks. Tier label (roofline/optimum/fa4) sourced from
    # Result.extra["tier"] populated by bench/normalize.py; falls back to "?".
    for label, desc in WHEELS:
        doc = runs[label]
        out.append(f"--- Run {label}: {desc} ---")
        out.append(f"  torch={doc['torch_version']}  cuda={doc['cuda_version']}")
        for name in test_names:
            r = by_test[name][label]
            stats = r["stats"]
            tier = (r.get("extra") or {}).get("tier", "?")
            out.append(
                f"  [{tier:8s}] {name:42s} : {r['measured']:7.2f} {r['unit']:6s} "
                f"(med={stats['median_ms']:.2f}ms, σ={stats['stdev_pct']:.1f}%, n={stats['n']})"
            )
        out.append("")

    # SOL Score table — Run A is baseline. The roofline tier populates sol
    # from NCU's back-derived hardware peak; optimum and fa4 tiers leave
    # sol=None which renders as "—".
    out.append("--- SOL Score vs Run A baseline ---")
    out.append(f"  {'tier':10s} {'test':<42s}   A(base)     B    score_B     C    score_C    SOL")
    for name in test_names:
        rA, rB, rC = by_test[name]["A"], by_test[name]["B"], by_test[name]["C"]
        baseline = rA["measured"]
        sol = rA["sol"]
        tier = (rA.get("extra") or {}).get("tier", "?")
        row = [f"  [{tier:8s}] {name:<42s}", f"  {baseline:7.2f}"]
        for r in (rB, rC):
            m = r["measured"]
            s = sol_score(m, baseline, sol)
            row.append(f"  {m:7.2f}")
            row.append(f"   {s:.3f}" if s is not None else "       —")
        row.append(f"  {sol:6.1f}" if sol is not None else "      —")
        out.append("  ".join(row))
    out.append("")

    summary_path = logs_dir / "SUMMARY.txt"
    summary_path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[summarize] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
