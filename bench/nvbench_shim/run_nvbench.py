"""
Tier 3: invoke the nvbench C++ shim, parse JSON, emit Result rows.
Cross-validates Tier 1 within ±3%.

The binary must already exist at
/work/nvbench_shim/build/sm121_gemm in the dgx-spark-build-strict volume.
Build it first via bench/nvbench_shim/build_nvbench.sh.

Usage (from host):
  sg docker -c 'python bench/nvbench_shim/run_nvbench.py'
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Make _harness importable from both host and container.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import Result, Stats

REPO = Path(__file__).resolve().parent.parent.parent
VOLUME = "dgx-spark-build-strict"
DEVEL_IMAGE = "nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04"
BIN_PATH = "/work/nvbench_shim/build/sm121_gemm"


def run_benchmark() -> dict:
    """Invoke nvbench binary with --json -; return parsed JSON."""
    res = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all",
         "-v", f"{VOLUME}:/work",
         "-v", f"{REPO}:/repo:ro",
         DEVEL_IMAGE,
         "bash", "-c", f"{BIN_PATH} --json -"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # nvbench JSON is on stdout; logs go to stderr.
    raw = res.stdout.decode(errors="replace")
    # Skip any leading chatter; nvbench JSON starts at the first '{'
    idx = raw.index("{")
    return json.loads(raw[idx:])


def to_results(nvbench_doc: dict) -> list[Result]:
    """Convert nvbench JSON to our Result schema."""
    out: list[Result] = []
    for b in nvbench_doc["benchmarks"]:
        name = f"nvbench_{b['name']}"
        for st in b["states"]:
            gpu_summary = next(s for s in st["summaries"]
                               if s["tag"] == "nv/cold/time/gpu/mean")
            mean_s = float(gpu_summary["data"]["value"]["value"])
            mean_ms = mean_s * 1000.0
            # FLOPs derived from element_count axis (we set it to M·N·K)
            elem = next(s for s in st["summaries"]
                        if s["tag"] == "nv/cold/bw/global/element_count")
            elements = float(elem["data"]["value"]["value"])
            tflops = 2.0 * elements / mean_s / 1e12
            # nvbench reports only mean; stats are degenerate.
            stats = Stats(
                mean_ms=mean_ms, median_ms=mean_ms,
                p10_ms=mean_ms, p90_ms=mean_ms,
                stdev_ms=0.0, stdev_pct=0.0,
                min_ms=mean_ms, max_ms=mean_ms, n=1,
            )
            out.append(Result(
                name=name, unit="TFLOPs", measured=tflops,
                sol=None, sol_score=None, sol_limit=None,
                stats=stats, correctness=None,
                extra={"nvbench_state": st["name"],
                       "axis_values": st["axis_values"]},
            ))
    return out


def main() -> int:
    doc = run_benchmark()
    results = to_results(doc)
    for r in results:
        sys.stdout.write(
            f"{r.name}: {r.measured:.2f} {r.unit} (mean={r.stats.mean_ms:.2f}ms)\n"
        )
    # Save full JSON for the summarizer to pick up
    out_path = REPO / "bench" / "logs" / "nvbench_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"\n[run_nvbench] saved raw JSON to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
