"""
Tier 3: invoke the nvbench C++ shim, parse its JSON output, emit a Result row.
Builds the binary on first run (via build_nvbench.sh inside docker), then runs
it once per invocation. Cross-validates Tier 1's Python harness within ±3%.

The build container and run container are the same (cuda:13.2.0-devel) — we
just produce the binary into the dgx-spark-build-strict volume and re-mount
that volume read-only for execution.

Standalone usage (from host):
  sg docker -c 'python bench/nvbench_shim/run_nvbench.py'
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Make _harness importable when this script is run from host or container.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import Result, Stats  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
VOLUME = "dgx-spark-build-strict"
DEVEL_IMAGE = "nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04"


def _docker(args: list[str], extra_flags: list[str] | None = None,
            capture: bool = False) -> subprocess.CompletedProcess:
    flags = ["--rm", "--gpus", "all"]
    if extra_flags:
        flags += extra_flags
    cmd = ["docker", "run", *flags,
           "-v", f"{VOLUME}:/work",
           "-v", f"{REPO}:/repo:ro",
           DEVEL_IMAGE, *args]
    return subprocess.run(cmd, check=False,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def build_if_missing() -> str:
    """Build nvbench shim if not present in the volume. Returns binary path."""
    bin_path = "/work/nvbench_shim/build/sm121_gemm"
    # Probe if the binary exists in the volume
    probe = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{VOLUME}:/work",
         "alpine", "sh", "-c", f"test -x {bin_path} && echo OK || echo MISSING"],
        capture_output=True, text=True,
    )
    if "OK" in probe.stdout:
        return bin_path

    print(f"[run_nvbench] binary {bin_path} missing — building", file=sys.stderr)
    res = _docker(
        ["bash", "/repo/bench/nvbench_shim/build_nvbench.sh"],
        capture=True,
    )
    sys.stderr.write(res.stdout.decode(errors="replace"))
    sys.stderr.write(res.stderr.decode(errors="replace"))
    if res.returncode != 0:
        raise RuntimeError(
            f"nvbench shim build failed (exit {res.returncode}) — "
            "sm_121 may not be supported by upstream nvbench. See stderr above."
        )
    return bin_path


def run_benchmark(bin_path: str) -> dict:
    """Run nvbench binary with --json -, return parsed JSON document."""
    res = _docker(
        ["bash", "-c", f"{bin_path} --json -"],
        capture=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"nvbench binary exited {res.returncode}: "
            f"{res.stderr.decode(errors='replace')[-240:]}"
        )
    # nvbench emits its JSON on stdout; some logging may be interleaved on stderr.
    raw = res.stdout.decode(errors="replace")
    # nvbench output starts with `{` after some optional warmup chatter
    idx = raw.find("{")
    if idx < 0:
        raise ValueError("no JSON found in nvbench output")
    return json.loads(raw[idx:])


def to_results(nvbench_doc: dict) -> list[Result]:
    """Convert nvbench JSON to our Result schema."""
    out: list[Result] = []
    for b in nvbench_doc.get("benchmarks", []):
        name = f"nvbench_{b['name']}"
        for st in b.get("states", []):
            # nvbench reports summaries; pick the GPU time summary.
            gpu_summary = next(
                (s for s in st.get("summaries", [])
                 if s.get("tag") == "nv/cold/time/gpu/mean"),
                None,
            )
            if gpu_summary is None:
                continue
            mean_s = float(gpu_summary["data"]["value"]["value"])
            mean_ms = mean_s * 1000.0
            # FLOPs come from element_count axis (we set M*N*K)
            elem = next(
                (s for s in st.get("summaries", [])
                 if s.get("tag") == "nv/cold/bw/global/element_count"),
                None,
            )
            elements = float(elem["data"]["value"]["value"]) if elem else None
            tflops = (2.0 * elements / mean_s / 1e12) if elements else 0.0
            # Build a Result. We don't have multiple iters here (nvbench reports
            # only mean), so stats are degenerate.
            stats = Stats(
                mean_ms=mean_ms, median_ms=mean_ms,
                p10_ms=mean_ms, p90_ms=mean_ms,
                stdev_ms=0.0, stdev_pct=0.0,
                min_ms=mean_ms, max_ms=mean_ms, n=1,
            )
            out.append(Result(
                name=name, unit="TFLOPs", measured=tflops,
                sol=None, sol_score=None, sol_limit=None,
                stats=stats, correctness=None, note=None,
                extra={"nvbench_state": st.get("name"),
                       "axis_values": st.get("axis_values")},
            ))
    return out


def main() -> int:
    bin_path = build_if_missing()
    doc = run_benchmark(bin_path)
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
