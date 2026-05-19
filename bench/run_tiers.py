"""
Bake-off orchestrator. Invoked by run_bakeoff.sh inside each wheel's
container; emits one JSON document to stdout that _summarize.py
aggregates across all three wheels.

Tiers (deterministic order; any tier raising propagates and aborts the
wheel's bake-off):

  1. optimum-benchmark wall-clock — per YAML in
     bench/configs/optimum/${BENCH_OPTIMUM_SCENARIO}/
     Bandwidth-bound regime: tests model-level end-to-end latency.

  2. Kernel-level GEMM — kernel_bench.py
     Compute-bound regime: 4 Llama-3-8B projections × 4 dtypes (bf16/fp16/
     fp8/fp4) at M=512. Tests cuBLAS dispatch quality — where native
     sm_121 cubins (Run B) should differentiate from PTX-JIT (Runs A, C).

  3. NCU roofline — profiles kernel_bench.py under `ncu --set roofline`
     with --filter-mode name + --launch-count 200 to capture per-kernel
     achieved % of peak across the GEMM shapes.

The FA-4 attention tier is deferred — FA-4 v4.0.0b13 + quack-kernels
crash on sm_121 (`'NoneType' object has no attribute '_trait'` inside
`cute.nvgpu.cpasync.tma_partition`). Re-enable when upstream supports
sm_121 cleanly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: F401  proves torch is importable before tier work

from _harness import Result, emit_json
from normalize import from_fa4, from_kernel, from_optimum  # noqa: F401  from_fa4 reserved for re-enabling FA-4
import roofline


_BENCH_DIR = Path(__file__).resolve().parent
_CONFIGS_DIR = _BENCH_DIR / "configs" / "optimum"
_LOG_DIR = Path(os.environ["BENCH_LOG_DIR"])


def _list_scenario_configs(scenario_dir: Path) -> list[Path]:
    """YAMLs in scenario_dir, deduped against symlink targets (wide/
    contains symlinks back into recommended/). Underscore-prefixed
    names (e.g., `_base_.yaml`) are Hydra-private inheritance bases,
    not runnable configs — skipped."""
    seen: set[Path] = set()
    out: list[Path] = []
    for y in sorted(scenario_dir.glob("*.yaml")):
        if y.name.startswith("_"):
            continue
        real = y.resolve()
        if real in seen:
            continue
        seen.add(real)
        out.append(y)
    return out


def run_optimum_tier(scenario_dir: Path) -> list[Result]:
    """Invoke `optimum-benchmark` per YAML; normalize each run dir.
    Redirects optimum-benchmark's stdout (its log output) to stderr so
    run_tiers.py's stdout stays clean for the final JSON document."""
    results: list[Result] = []
    for cfg in _list_scenario_configs(scenario_dir):
        name = cfg.stem
        print(f"[run_tiers] tier=optimum config={name}", file=sys.stderr)
        subprocess.run([
            "optimum-benchmark",
            "--config-dir", str(scenario_dir),
            "--config-name", name,
        ], check=True, stdout=sys.stderr)
        results.extend(from_optimum(_LOG_DIR / "optimum" / name))
    return results


_KERNEL_DTYPES = ("bf16", "fp16", "fp8", "fp4")


def run_kernel_tier() -> list[Result]:
    """Compute-bound GEMM measurements via kernel_bench.py subprocess —
    one subprocess per dtype so cuBLAS heuristic-table gaps on sm_121 for
    FP8/FP4 (CUBLAS_STATUS_NOT_INITIALIZED) don't take out BF16/FP16
    results. Per-dtype failures are recorded as zero-length result lists;
    the wheel still produces JSON for the dtypes that succeeded."""
    results: list[Result] = []
    for dtype in _KERNEL_DTYPES:
        print(f"[run_tiers] tier=kernel dtype={dtype}", file=sys.stderr)
        proc = subprocess.run(
            [sys.executable, str(_BENCH_DIR / "kernel_bench.py"), dtype],
            stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
        )
        if proc.returncode == 0:
            results.extend(from_kernel(proc.stdout))
        else:
            print(f"[run_tiers] tier=kernel dtype={dtype} skipped (returncode={proc.returncode})",
                  file=sys.stderr)
    return results


def run_roofline_tier() -> list[Result]:
    """Profile the BF16 kernel_bench under NCU — BF16 GEMM is the most
    stable codepath across all three wheels. --launch-count 200 captures
    ~5 launches per unique GEMM kernel (4 shapes × 55 iters)."""
    print("[run_tiers] tier=roofline target=kernel_bench dtype=bf16", file=sys.stderr)
    return roofline.profile(
        [sys.executable, str(_BENCH_DIR / "kernel_bench.py"), "bf16"],
        _LOG_DIR / "ncu" / "kernel_bench_bf16.ncu-rep",
    )


def main() -> int:
    scenario_dir = (_CONFIGS_DIR / os.environ["BENCH_OPTIMUM_SCENARIO"]).resolve(strict=True)
    results = (
        run_optimum_tier(scenario_dir)
        + run_kernel_tier()
        + run_roofline_tier()
    )
    emit_json(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
