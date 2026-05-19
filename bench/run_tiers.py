"""
Three-tier bake-off orchestrator. Invoked by run_bakeoff.sh inside each
wheel's container; emits one JSON document to stdout that _summarize.py
aggregates across all three wheels.

  1. optimum-benchmark wall-clock — per YAML in
     bench/configs/optimum/${BENCH_OPTIMUM_SCENARIO}/
  2. NCU roofline — Llama-3-8B BF16 forward with warmup_runs=1 iterations=1
     so NCU's --launch-count 50 budget captures the real iteration
  3. FlashAttention-4 attention — fa4_bench.py subprocess
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roofline
import torch  # noqa: F401  proves torch is importable before tier work
from _harness import Result, emit_json
from normalize import from_fa4, from_optimum


_BENCH_DIR = Path(__file__).resolve().parent
_CONFIGS_DIR = _BENCH_DIR / "configs" / "optimum"
_LOG_DIR = Path(os.environ["BENCH_LOG_DIR"])


def _list_scenario_configs(scenario_dir: Path) -> list[Path]:
    """YAMLs in scenario_dir, deduped against symlink targets (wide/
    contains symlinks back into recommended/)."""
    seen: set[Path] = set()
    out: list[Path] = []
    for y in sorted(scenario_dir.glob("*.yaml")):
        real = y.resolve()
        if real in seen:
            continue
        seen.add(real)
        out.append(y)
    return out


def run_optimum_tier(scenario_dir: Path) -> list[Result]:
    """Invoke `optimum-benchmark` per YAML; normalize each run dir."""
    results: list[Result] = []
    for cfg in _list_scenario_configs(scenario_dir):
        name = cfg.stem
        print(f"[run_tiers] tier=optimum config={name}", file=sys.stderr)
        subprocess.run([
            "optimum-benchmark",
            "--config-dir", str(scenario_dir),
            "--config-name", name,
        ], check=True)
        results.extend(from_optimum(_LOG_DIR / "optimum" / name))
    return results


def run_roofline_tier(scenario_dir: Path) -> list[Result]:
    """Profile the Llama-3-8B BF16 config under NCU; iterations=1 and
    warmup_runs=1 ensure the launch-count budget targets the JIT-warm
    iteration, not warmup spam."""
    print("[run_tiers] tier=roofline config=llama3_8b_bf16", file=sys.stderr)
    return roofline.profile(
        [
            "optimum-benchmark",
            "--config-dir", str(scenario_dir),
            "--config-name", "llama3_8b_bf16",
            "scenario.warmup_runs=1",
            "scenario.iterations=1",
        ],
        _LOG_DIR / "ncu" / "llama3_8b_bf16.ncu-rep",
    )


def run_fa4_tier() -> list[Result]:
    print("[run_tiers] tier=fa4", file=sys.stderr)
    proc = subprocess.run(
        [sys.executable, str(_BENCH_DIR / "fa4_bench.py")],
        stdout=subprocess.PIPE, check=True, text=True,
    )
    return from_fa4(proc.stdout)


def main() -> int:
    scenario_dir = (_CONFIGS_DIR / os.environ["BENCH_OPTIMUM_SCENARIO"]).resolve(strict=True)
    results = (
        run_optimum_tier(scenario_dir)
        + run_roofline_tier(scenario_dir)
        + run_fa4_tier()
    )
    emit_json(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
