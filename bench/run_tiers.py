"""
Three-tier bake-off orchestrator. Invoked by run_bakeoff.sh inside each
wheel's container; emits one JSON document to stdout that _summarize.py
aggregates across all three wheels.

Tiers run in deterministic order; any tier raising propagates and aborts
the wheel's bake-off.

  1. optimum-benchmark wall-clock — uvx optimum-benchmark per YAML in
     bench/configs/optimum/${BENCH_OPTIMUM_SCENARIO}/
  2. NCU roofline — Llama-3-8B BF16 forward profiled with
     `scenario.warmup_runs=1 scenario.iterations=1` so NCU's launch-count
     budget captures the real iteration, not warmup spam.
  3. FlashAttention-4 attention — fa4_bench.py subprocess.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roofline
import torch  # noqa: F401  — proves torch is importable before tier work
from _harness import Result, emit_json
from normalize import from_fa4, from_optimum


_BENCH_DIR = Path(__file__).resolve().parent
_CONFIGS_DIR = _BENCH_DIR / "configs" / "optimum"
_LOG_DIR = Path(os.environ["BENCH_LOG_DIR"])


def _list_scenario_configs(scenario: str) -> list[Path]:
    """YAMLs in bench/configs/optimum/<scenario>/, resolving symlinks so
    each unique config runs once (wide/ contains symlinks into recommended/)."""
    scenario_dir = _CONFIGS_DIR / scenario
    seen: set[Path] = set()
    out: list[Path] = []
    for y in sorted(scenario_dir.glob("*.yaml")):
        real = y.resolve()
        if real in seen:
            continue
        seen.add(real)
        out.append(y)
    return out


def run_optimum_tier(scenario: str) -> list[Result]:
    """Invoke `uvx optimum-benchmark` per YAML; normalize each run dir."""
    scenario_dir = _CONFIGS_DIR / scenario
    results: list[Result] = []
    for cfg in _list_scenario_configs(scenario):
        name = cfg.stem
        run_dir = _LOG_DIR / "optimum" / name
        print(f"[run_tiers] tier=optimum config={name}", file=sys.stderr)
        subprocess.run([
            "uvx", "optimum-benchmark",
            "--config-dir", str(scenario_dir),
            "--config-name", name,
        ], check=True)
        results.extend(from_optimum(run_dir))
    return results


def run_roofline_tier(scenario: str) -> list[Result]:
    """Profile the Llama-3-8B BF16 config under NCU. warmup_runs=1
    iterations=1 are forced via Hydra CLI overrides so NCU's
    --launch-count budget captures the JIT-warm iteration, not warmup."""
    cfg_name = "llama3_8b_bf16"
    rep_path = _LOG_DIR / "ncu" / f"{cfg_name}.ncu-rep"
    command = [
        "uvx", "optimum-benchmark",
        "--config-dir", str(_CONFIGS_DIR / scenario),
        "--config-name", cfg_name,
        "scenario.warmup_runs=1",
        "scenario.iterations=1",
    ]
    print(f"[run_tiers] tier=roofline config={cfg_name}", file=sys.stderr)
    return roofline.profile(command, rep_path)


def run_fa4_tier() -> list[Result]:
    print("[run_tiers] tier=fa4", file=sys.stderr)
    proc = subprocess.run(
        [sys.executable, str(_BENCH_DIR / "fa4_bench.py")],
        stdout=subprocess.PIPE, check=True, text=True,
    )
    return from_fa4(proc.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--scenario",
        default=os.environ.get("BENCH_OPTIMUM_SCENARIO", "recommended"),
        choices=("recommended", "wide"),
    )
    ap.add_argument("--json", action="store_true", required=True,
                    help="JSON output is the only supported mode; flag retained "
                         "for run_bakeoff.sh compatibility.")
    args = ap.parse_args()

    results = (
        run_optimum_tier(args.scenario)
        + run_roofline_tier(args.scenario)
        + run_fa4_tier()
    )
    emit_json(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
