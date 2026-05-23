"""Stage 2: LLM-augmented exploration on top of Stage 1's best.

Each iteration:
  1. Build prompt with the current best source + runtime + fresh NCU findings
     + AttemptMemory's list of tried tuples.
  2. LLM emits a candidate source.
  3. sha256(source) is the AttemptMemory key — duplicates are rejected
     without running the candidate.
  4. Compile + verify + bench. If status is PASSED and runtime is at least
     1% faster than current best (i.e., runtime < current_best * 0.99), accept,
     re-profile NCU on the new best so the next iteration's findings reflect
     the change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

from .agent import build_prompt, extract_code_block, llm_complete
from .attempt_memory import AttemptMemory
from .bench import bench_solution
from .data import Definition, Evaluation, EvaluationStatus
from .loop_helpers import _build_solution_for_runner, ncu_profile_subprocess


@dataclass
class Stage2Best:
    source: str
    runtime_ms: float
    evaluation: Evaluation


def _eval(definition: Definition, source: str) -> Evaluation:
    return bench_solution(
        definition, _build_solution_for_runner(definition, source),
        n_iters=None, run_ncu=False,
    )


def run_stage2(
    definition: Definition,
    initial_best: Stage2Best,
    attempt_memory: AttemptMemory,
    llm_endpoint: str,
    llm_model: str,
    max_iters: int,
) -> Stage2Best:
    best = initial_best
    # Initial NCU diag on the best — feeds the first prompt's findings section.
    ncu_diag = ncu_profile_subprocess(best.source, definition)
    history: List[str] = []

    for i in range(max_iters):
        prompt = build_prompt(
            definition=definition,
            current_best_source=best.source,
            current_best_runtime_ms=best.runtime_ms,
            ncu_diag=ncu_diag,
            tried_tuples_summary=attempt_memory.attempted_tuples_summary(),
            history_tail=history[-5:],
        )
        resp = llm_complete(prompt, model=llm_model, endpoint=llm_endpoint)
        src = extract_code_block(resp)
        if src is None:
            history.append(f"iter {i}: no_code_block")
            continue
        h = hashlib.sha256(src.encode()).hexdigest()
        if attempt_memory.has_source(h):
            history.append(f"iter {i}: duplicate_source")
            continue
        attempt_memory.add_source(h)
        ev = _eval(definition, src)
        if ev.status != EvaluationStatus.PASSED:
            history.append(f"iter {i}: {ev.status.value}: {ev.extra_msg[:80]}")
            continue
        runtime_ms = ev.performance.latency_ms
        delta_pct = (best.runtime_ms - runtime_ms) / best.runtime_ms * 100.0
        if runtime_ms < best.runtime_ms * 0.99:
            best = Stage2Best(source=src, runtime_ms=runtime_ms, evaluation=ev)
            ncu_diag = ncu_profile_subprocess(src, definition)
            history.append(
                f"iter {i}: ACCEPT runtime={runtime_ms:.4f}ms ({delta_pct:+.2f}%)"
            )
        else:
            history.append(
                f"iter {i}: revert runtime={runtime_ms:.4f}ms ({delta_pct:+.2f}%)"
            )
    return best
