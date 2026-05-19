"""
Three-tier bake-off orchestrator. Replaces bench_full.py.

Tiers run in deterministic order; any tier raising propagates and aborts
the wheel's bake-off (no fallbacks, no skip-on-fail).

  1. optimum-benchmark wall-clock — `uvx optimum-benchmark` per YAML in
     bench/configs/optimum/${BENCH_OPTIMUM_SCENARIO}/
  2. NCU roofline — Llama-3-8B BF16 under ncu with kernel filtering and
     launch budget; warmup_runs=1 iterations=1 overridden so NCU's captures
     come from the real iteration, not warmup
  3. FlashAttention-4 attention — fa4_bench.py subprocess

All three tier outputs are normalized via bench/normalize.py into the
Result schema that _summarize.py consumes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# Make _harness, normalize, roofline importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import roofline
import torch
from _harness import Result, emit_json
from normalize import from_fa4, from_optimum

_BENCH_DIR = Path(__file__).resolve().parent
_CONFIGS_DIR = _BENCH_DIR / "configs" / "optimum"
_LOG_DIR = Path(os.environ.get("BENCH_LOG_DIR", "/logs"))


# ---------- Tier 1: optimum-benchmark ----------


def _list_scenario_configs(scenario: str) -> list[Path]:
    """Return YAML paths in bench/configs/optimum/<scenario>/ (resolving
    symlinks so each unique YAML runs once)."""
    scenario_dir = _CONFIGS_DIR / scenario
    yamls = sorted(scenario_dir.glob("*.yaml"))
    seen: set[Path] = set()
    out: list[Path] = []
    for y in yamls:
        real = y.resolve()
        if real in seen:
            continue
        seen.add(real)
        out.append(y)
    return out


def run_optimum_tier(
    scenario: str, warmup_override: int | None = None, iters_override: int | None = None
) -> list[Result]:
    """Invoke optimum-benchmark via uvx for each YAML in scenario/."""
    results: list[Result] = []
    for cfg in _list_scenario_configs(scenario):
        name = cfg.stem
        run_dir = _LOG_DIR / "optimum" / name

        cmd = [
            "uvx",
            "optimum-benchmark",
            "--config-dir",
            str(_CONFIGS_DIR),
            "--config-name",
            name,
        ]
        if warmup_override is not None:
            cmd.append(f"scenario.warmup_runs={warmup_override}")
        if iters_override is not None:
            cmd.append(f"scenario.iterations={iters_override}")

        print(f"[run_tiers] tier=optimum config={name}", file=sys.stderr)
        subprocess.run(cmd, check=True)

        results.extend(from_optimum(run_dir))
    return results


# ---------- Tier 2: NCU roofline ----------


def run_roofline_tier(scenario: str) -> list[Result]:
    """Profile the Llama-3-8B BF16 forward+decode under NCU. Hydra overrides
    force scenario.warmup_runs=1 iterations=1 so the launch-count 50 budget
    captures the real iteration, not the warmup spam."""
    cfg_name = "llama3_8b_bf16"  # always profile this canonical config
    rep_path = _LOG_DIR / "ncu" / f"{cfg_name}.ncu-rep"

    command = [
        "uvx",
        "optimum-benchmark",
        "--config-dir",
        str(_CONFIGS_DIR),
        "--config-name",
        cfg_name,
        "scenario.warmup_runs=1",
        "scenario.iterations=1",
    ]
    print(f"[run_tiers] tier=roofline config={cfg_name}", file=sys.stderr)
    return roofline.profile(command, rep_path)


# ---------- Tier 3: FA-4 ----------


def run_fa4_tier() -> list[Result]:
    fa4_script = _BENCH_DIR / "fa4_bench.py"
    print(f"[run_tiers] tier=fa4 script={fa4_script}", file=sys.stderr)
    proc = subprocess.run(
        [sys.executable, str(fa4_script)],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    )
    return from_fa4(proc.stdout)


# ---------- CLI ----------

ALL_TIERS = ("optimum", "roofline", "fa4")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--only",
        default="",
        help=f"comma-separated tiers (one of: {','.join(ALL_TIERS)}); empty = all",
    )
    ap.add_argument(
        "--scenario",
        default=os.environ.get("BENCH_OPTIMUM_SCENARIO", "recommended"),
        choices=("recommended", "wide"),
        help="optimum-benchmark config directory (overrides BENCH_OPTIMUM_SCENARIO)",
    )
    ap.add_argument(
        "--ncu-budget-iters",
        type=int,
        default=None,
        help="when running --only optimum from inside the roofline tier, "
        "forces scenario.warmup_runs and iterations to this value",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit JSON to stdout instead of human-readable",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    tiers_to_run = (
        [t.strip() for t in args.only.split(",") if t.strip()]
        if args.only
        else list(ALL_TIERS)
    )
    unknown = [t for t in tiers_to_run if t not in ALL_TIERS]
    if unknown:
        print(
            f"[ERROR] --only unknown tier: {unknown} (valid: {ALL_TIERS})",
            file=sys.stderr,
        )
        return 1

    results: list[Result] = []
    if "optimum" in tiers_to_run:
        results.extend(
            run_optimum_tier(
                args.scenario,
                warmup_override=args.ncu_budget_iters,
                iters_override=args.ncu_budget_iters,
            )
        )
    if "roofline" in tiers_to_run:
        results.extend(run_roofline_tier(args.scenario))
    if "fa4" in tiers_to_run:
        results.extend(run_fa4_tier())

    if args.json:
        emit_json(results)
    else:
        for r in results:
            sol_str = f" sol={r.sol:.3g}" if r.sol is not None else ""
            score_str = f" score={r.sol_score:.3f}" if r.sol_score is not None else ""
            print(
                f"  [{r.extra.get('tier', '?'):8s}] {r.name[:60]:60s} "
                f"{r.measured:8.3f} {r.unit:6s}{sol_str}{score_str}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
