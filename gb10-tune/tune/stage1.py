"""Stage 1: deterministic template-grid sweep.

For every (template, parameter tuple) in the registry for the current
allowed_action, check AttemptMemory; if unseen, compile + verify + bench.
Returns all CandidateResults. The orchestrator picks the best.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .attempt_memory import AttemptMemory
from .bench import bench_solution
from .data import Definition, Evaluation, EvaluationStatus
from .loop_helpers import _build_solution_for_runner
from .templates import templates_for
from .templates.base import Template


@dataclass
class CandidateResult:
    template_name: str
    parameter_tuple: Tuple
    source: str
    evaluation: Evaluation
    runtime_ms: float
    correct: bool


def _eval(definition: Definition, source: str) -> Evaluation:
    sol = _build_solution_for_runner(definition, source)
    # Stage 1 does NOT need NCU per-candidate (only timing + correctness).
    # NCU is run at Stage 0 (baseline) and Stage 2 (re-profile on accept).
    return bench_solution(definition, sol, n_iters=None, run_ncu=False)


def run_stage1(
    definition: Definition,
    allowed_action: str,
    attempt_memory: AttemptMemory,
) -> List[CandidateResult]:
    results: List[CandidateResult] = []
    for template in templates_for(allowed_action):
        for params in template.parameter_grid():
            key = (allowed_action, template.name, tuple(params))
            if attempt_memory.has(key):
                continue
            attempt_memory.add(key)
            src = template.render(params)
            ev = _eval(definition, src)
            results.append(
                CandidateResult(
                    template_name=template.name,
                    parameter_tuple=tuple(params),
                    source=src,
                    evaluation=ev,
                    runtime_ms=(ev.performance.latency_ms if ev.performance else 0.0),
                    correct=(ev.status == EvaluationStatus.PASSED),
                )
            )
    return results
