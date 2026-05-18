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
from typing import Any


WHEELS = [
    ("A", "PyPI torch==2.9.0+cu130 + triton"),
    ("B", "Source-built wheel (native sm_121 cubins)"),
    ("C", "NGC pytorch:26.04-py3 (vendor reference)"),
]


def load_run(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[summarize] WARN: failed to parse {path}: {e}", file=sys.stderr)
        return None


def sol_score(measured: float, baseline: float, sol: float | None) -> float | None:
    if sol is None or sol <= baseline:
        return None
    return max(0.0, min(1.0, (measured - baseline) / (sol - baseline)))


def main() -> int:
    logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "logs"
    runs = {label: load_run(logs_dir / f"run{label}.json") for label, _ in WHEELS}

    if not any(runs.values()):
        print(f"[summarize] no run{{A,B,C}}.json found under {logs_dir}", file=sys.stderr)
        return 1

    # Index per (wheel, test_name) -> result dict
    by_test: dict[str, dict[str, dict]] = {}
    for label, doc in runs.items():
        if doc is None:
            continue
        for r in doc.get("results", []):
            by_test.setdefault(r["name"], {})[label] = r
    test_names = list(by_test.keys())  # preserve first-seen order

    out: list[str] = []
    out.append("============= DGX Spark 3-way comparison =============")
    if any(runs.values()):
        # Top metadata from the first available doc
        meta = next(d for d in runs.values() if d is not None)
        out.append(f"Date (run A ts): {meta.get('ts')}")
        out.append(f"Device: {meta.get('device_name')}  arch_list={meta.get('arch_list')}")
    out.append("")

    # Per-wheel detail blocks
    for label, desc in WHEELS:
        doc = runs.get(label)
        out.append(f"--- Run {label}: {desc} ---")
        if doc is None:
            out.append("  (SKIPPED — no log file)")
            out.append("")
            continue
        out.append(f"  torch={doc.get('torch_version')}  cuda={doc.get('cuda_version')}")
        for name in test_names:
            r = by_test.get(name, {}).get(label)
            if r is None:
                out.append(f"  {name:42s} : (not run)")
                continue
            if r.get("note") and (r.get("measured", 0) == 0):
                out.append(f"  {name:42s} : SKIPPED ({r['note'][:80]})")
                continue
            stats = r["stats"]
            out.append(
                f"  {name:42s} : {r['measured']:7.2f} {r['unit']:6s} "
                f"(med={stats['median_ms']:.2f}ms, σ={stats['stdev_pct']:.1f}%, n={stats['n']})"
            )
        out.append("")

    # SOL Score table — Run A is baseline
    out.append("--- SOL Score vs Run A baseline ---")
    out.append(f"  {'test':<42s}   A(base)     B    score_B     C    score_C    SOL")
    for name in test_names:
        rA = by_test.get(name, {}).get("A")
        rB = by_test.get(name, {}).get("B")
        rC = by_test.get(name, {}).get("C")
        baseline = (rA.get("measured") if rA and not rA.get("note") else None)
        sol = next((r.get("sol") for r in (rA, rB, rC) if r and r.get("sol") is not None), None)
        row = [f"  {name:<42s}"]
        row.append(f"  {baseline:7.2f}" if baseline else "       —")
        for r in (rB, rC):
            if r is None or (r.get("note") and r.get("measured", 0) == 0):
                row.append("       —"); row.append("        —")
                continue
            m = r["measured"]
            row.append(f"  {m:7.2f}")
            if baseline is not None:
                s = sol_score(m, baseline, sol)
                row.append(f"   {s:.3f}" if s is not None else "       —")
            else:
                row.append("       —")
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
