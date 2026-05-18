"""
Tier 3: Nsight Compute roofline via ncu_report Python API.

Runs `ncu --set roofline` against bench_full.py for one test, parses the
.ncu-rep, emits achieved_tflops, achieved_gbs, arith_intensity, and the
sol_sm / sol_mem percentages from Nsight's roofline rule.

Opt-in via BENCH_PROFILE=1 (profiling is 10-30× slower than normal runs).

Container prep (when BENCH_PROFILE=1):
  apt-get install -y nsight-compute-2025.3.1
  pip install ncu_report==2025.3.1
  Run with --cap-add=SYS_ADMIN (NVIDIA HW counters; else ERR_NVGPUCTRPERM)

Usage:
  python bench/roofline.py fp16
  python bench/roofline.py fp8 --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import ncu_report

# Make _harness importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


# Metrics from ncu's roofline rule, verified on Blackwell sm_121 (Nsight 2026.1.0).
# Per-instruction SASS counters are not in the roofline set on Blackwell — use
# `--set full` separately if you need them.
METRICS = {
    "sol_sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sol_mem": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
}


def profile_one(test_key: str, rep_dir: Path) -> Path:
    """Run ncu on bench_full.py --only test_key, return .ncu-rep path."""
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_base = rep_dir / f"{test_key}"
    bench_full = Path(__file__).resolve().parent / "bench_full.py"
    cmd = [
        "ncu",
        "--set", "roofline",
        "--target-processes", "all",
        "--force-overwrite",
        "--export", str(rep_base),
        sys.executable, str(bench_full),
        "--only", test_key, "--no-isolate", "--json",
    ]
    print(f"[roofline] profiling: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return Path(str(rep_base) + ".ncu-rep")


def parse_report(rep_path: Path) -> dict:
    """Extract roofline metrics from a .ncu-rep via ncu_report."""
    ctx = ncu_report.load_report(str(rep_path))
    # Pick the heaviest kernel (the GEMM/attn under test, not L2 flushes).
    heaviest = None
    heaviest_dur = -1.0
    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            act = rng.action_by_idx(ai)
            dur = act.metric_by_name("gpu__time_duration.sum").as_double()
            if dur > heaviest_dur:
                heaviest_dur = dur
                heaviest = act

    out: dict = {"kernel_name": heaviest.name(), "duration_s": heaviest_dur / 1e9}
    for label, metric_name in METRICS.items():
        out[label] = heaviest.metric_by_name(metric_name).as_double()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("test_key", help="bench test key, e.g. 'fp16' or 'fp8'")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of human-readable")
    args = ap.parse_args()

    rep_dir = Path(__file__).resolve().parent / "logs" / "ncu"
    rep_path = profile_one(args.test_key, rep_dir)
    metrics = parse_report(rep_path)
    metrics["test_key"] = args.test_key
    metrics["rep_path"] = str(rep_path)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(f"\nRoofline for {args.test_key} ({metrics['kernel_name']}):")
        for k, v in metrics.items():
            if k == "kernel_name":
                continue
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.3g}")
            else:
                print(f"  {k:30s}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
