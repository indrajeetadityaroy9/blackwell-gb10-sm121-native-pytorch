"""Top-level orchestrator: Stage 0 → Stage 1 → Stage 2.

Single canonical execution path. Stage 2 always runs; pass `--max-stage2-iters 0`
to make its loop body execute zero times (no separate branch).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .attempt_memory import AttemptMemory
from .bench import bench_solution
from .bottleneck import classify
from .data import EvaluationStatus
from .loop_helpers import (
    _build_solution_for_runner,
    ncu_profile_subprocess,
    promote_to_traces,
)
from .stage1 import CandidateResult, run_stage1
from .stage2 import Stage2Best, run_stage2
from .trace import TRACES_ROOT, load_definition


@dataclass
class TuneResult:
    baseline_runtime_ms: float
    stage1_count: int
    stage1_correct: int
    stage1_best_runtime_ms: float
    stage1_best_params: Optional[tuple]
    final_runtime_ms: float
    final_source: str
    overall_speedup: float


def tune(
    definition_name: str,
    baseline_path: Path,
    llm_endpoint: str,
    llm_model: str,
    max_stage2_iters: int = 30,
) -> TuneResult:
    definition = load_definition(definition_name)
    baseline_src = baseline_path.read_text()
    memory = AttemptMemory()

    # Stage 0: baseline bench + NCU + classify.
    baseline_sol = _build_solution_for_runner(definition, baseline_src)
    baseline_ev = bench_solution(definition, baseline_sol, run_ncu=False)
    if baseline_ev.status != EvaluationStatus.PASSED:
        raise RuntimeError(f"Baseline failed: {baseline_ev.status} — {baseline_ev.extra_msg}")
    baseline_ncu = ncu_profile_subprocess(baseline_src, definition)
    report = classify(baseline_ncu, op_type=definition.op_type)

    # Stage 1: deterministic template-grid sweep.
    stage1_results = run_stage1(definition, report.allowed_action, memory)
    correct = [r for r in stage1_results if r.correct]
    if correct:
        stage1_best_cand = min(correct, key=lambda r: r.runtime_ms)
    else:
        stage1_best_cand = None

    # Pick the best of (baseline, stage1_best) as Stage 2's starting point.
    baseline_runtime = baseline_ev.performance.latency_ms
    if stage1_best_cand is not None and stage1_best_cand.runtime_ms < baseline_runtime:
        s2_start = Stage2Best(
            source=stage1_best_cand.source,
            runtime_ms=stage1_best_cand.runtime_ms,
            evaluation=stage1_best_cand.evaluation,
        )
    else:
        s2_start = Stage2Best(
            source=baseline_src,
            runtime_ms=baseline_runtime,
            evaluation=baseline_ev,
        )

    # Stage 2: LLM exploration (zero iters if max_stage2_iters=0 — same code path).
    final = run_stage2(
        definition=definition,
        initial_best=s2_start,
        attempt_memory=memory,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
        max_iters=max_stage2_iters,
    )

    promote_to_traces(definition, final.source, TRACES_ROOT)

    result = TuneResult(
        baseline_runtime_ms=baseline_runtime,
        stage1_count=len(stage1_results),
        stage1_correct=len(correct),
        stage1_best_runtime_ms=(stage1_best_cand.runtime_ms if stage1_best_cand else 0.0),
        stage1_best_params=(stage1_best_cand.parameter_tuple if stage1_best_cand else None),
        final_runtime_ms=final.runtime_ms,
        final_source=final.source,
        overall_speedup=baseline_runtime / final.runtime_ms,
    )
    _print_report(definition_name, report, result, memory)
    return result


def _print_report(definition_name: str, report, result: TuneResult, memory: AttemptMemory) -> None:
    print(f"=== tune: {definition_name} ===")
    print()
    print(f"Baseline runtime:        {result.baseline_runtime_ms:.4f} ms")
    print(f"Baseline bottleneck:     {report.bottleneck_type}")
    print(f"  memory throughput:     {report.evidence.memory_throughput_pct:.1f}%")
    print(f"  SM throughput:         {report.evidence.sm_throughput_pct:.1f}%")
    print(f"  load efficiency:       {report.evidence.load_efficiency_pct:.1f}%")
    print()
    print(f"Stage 1 (template sweep): {result.stage1_correct}/{result.stage1_count} correct")
    print(f"  best runtime:          {result.stage1_best_runtime_ms:.4f} ms")
    print(f"  best params:           {result.stage1_best_params}")
    print()
    print(f"Stage 2 (LLM exploration): {len(memory.attempted_sources)} candidates evaluated")
    print(f"  final runtime:         {result.final_runtime_ms:.4f} ms")
    print()
    print(f"Overall speedup:         {result.overall_speedup:.2f}x")
    print(f"  ({result.baseline_runtime_ms:.4f} ms → {result.final_runtime_ms:.4f} ms)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unified two-stage Triton kernel tuner")
    ap.add_argument("--definition", required=True)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--llm-endpoint", required=True)
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--max-stage2-iters", type=int, default=30)
    args = ap.parse_args(argv)

    tune(
        definition_name=args.definition,
        baseline_path=args.baseline,
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        max_stage2_iters=args.max_stage2_iters,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
