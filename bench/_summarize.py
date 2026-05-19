"""
Aggregate bench/logs/run{A,B,C}.json → SUMMARY.txt.

Score(wheel, test) = clamp((measured - baseline) / (sol - baseline), [0,1])
where sol is non-None only for the roofline tier (NCU-back-derived hardware
peak); other tiers render the score column as '—'. The gap-closure formula
is sign-correct in both directions: TFLOPs (higher-is-better) yields
positive (m - baseline) / positive (sol - baseline); latency-ms
(lower-is-better) yields negative / negative.

Usage: python bench/_summarize.py <logs_dir>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


WHEELS = [
    ("A", "PyPI torch==2.10.0+cu130"),
    ("B", "Source-built wheel (native sm_121 / sm_121a cubins)"),
    ("C", "NGC pytorch:26.04-py3"),
]


def main() -> int:
    logs_dir = Path(sys.argv[1])
    runs = {
        label: json.loads((logs_dir / f"run{label}.json").read_text())
        for label, _ in WHEELS
    }

    by_test: dict[str, dict[str, dict]] = {}
    for label, doc in runs.items():
        for r in doc["results"]:
            by_test.setdefault(r["name"], {})[label] = r

    complete_names = [
        n for n in by_test
        if all(label in by_test[n] for label, _ in WHEELS)
    ]
    skipped = len(by_test) - len(complete_names)

    out: list[str] = []
    out.append("============= DGX Spark 3-way comparison =============")
    meta = next(iter(runs.values()))
    out.append(f"Date (run A): {datetime.fromtimestamp(meta['ts'], tz=timezone.utc).isoformat()}")
    out.append(f"Device: {meta['device_name']}  arch_list={meta['arch_list']}")
    out.append("")

    for label, desc in WHEELS:
        doc = runs[label]
        out.append(f"--- Run {label}: {desc} ---")
        out.append(f"  torch={doc['torch_version']}  cuda={doc['cuda_version']}")
        for name in complete_names:
            r = by_test[name][label]
            stats = r["stats"]
            tier = r["extra"]["tier"]
            out.append(
                f"  [{tier:8s}] {name:42s} : {r['measured']:7.2f} {r['unit']:6s} "
                f"(med={stats['median_ms']:.2f}ms, σ={stats['stdev_pct']:.1f}%, n={stats['n']})"
            )
        out.append("")

    out.append("--- Score vs Run A baseline ---")
    out.append(f"  {'tier':10s} {'test':<42s}   A(base)     B    score_B     C    score_C    SOL")
    for name in complete_names:
        rA, rB, rC = by_test[name]["A"], by_test[name]["B"], by_test[name]["C"]
        baseline = rA["measured"]
        sol = rA["sol"]
        tier = rA["extra"]["tier"]
        row = [f"  [{tier:8s}] {name:<42s}", f"  {baseline:7.2f}"]
        for r in (rB, rC):
            m = r["measured"]
            row.append(f"  {m:7.2f}")
            if sol is None:
                row.append("       —")
            else:
                s = max(0.0, min(1.0, (m - baseline) / (sol - baseline)))
                row.append(f"   {s:.3f}")
        row.append(f"  {sol:6.1f}" if sol is not None else "      —")
        out.append("  ".join(row))
    if skipped:
        out.append(f"  (skipped {skipped} rows where wheel coverage was incomplete — typically NCU kernels named differently across wheels)")
    out.append("")

    summary_path = logs_dir / "SUMMARY.txt"
    summary_path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[summarize] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
