"""Per-dtype correctness validators (spec §5).

Three classes dispatched via Definition.validator_class:
- deterministic        : BF16 / FP16 / integer outputs. atol=1e-2, rtol=1e-2.
- matched_ratio        : FP8 e4m3 fast-accum. atol=0.5, rtol=5e-2, rho=0.95.
- matched_ratio_loose  : FP4 mxfp4/nvfp4 GEMM. atol=0.5, rtol=1e-1, rho=0.85.

All checks compare in float32 to avoid silent dtype-induced false-passes.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .data import Correctness, Definition, EvaluationStatus


@dataclass(frozen=True)
class ValidatorThresholds:
    atol: float
    rtol: float
    required_matched_ratio: float


_THRESHOLDS = {
    "deterministic": ValidatorThresholds(atol=1e-2, rtol=1e-2, required_matched_ratio=1.0),
    "matched_ratio": ValidatorThresholds(atol=0.5, rtol=5e-2, required_matched_ratio=0.95),
    "matched_ratio_loose": ValidatorThresholds(atol=0.5, rtol=1e-1, required_matched_ratio=0.85),
}


def _error_stats(
    output: torch.Tensor, reference: torch.Tensor, t: ValidatorThresholds
) -> Tuple[float, float, float]:
    """Returns (max_abs, max_rel, matched_ratio) — float32 comparison."""
    x = output.to(torch.float32)
    y = reference.to(torch.float32)
    eps = 1e-8
    abs_err = torch.abs(x - y)
    rel_err = abs_err / (torch.abs(y) + eps)
    n = abs_err.numel()
    exceeds = (abs_err > t.atol) & (rel_err > t.rtol)
    matched = 1.0 - float(exceeds.sum().item()) / float(n)
    return float(abs_err.max().item()), float(rel_err.max().item()), matched


def validate(
    definition: Definition,
    sol_outputs: List[torch.Tensor],
    ref_outputs: List[torch.Tensor],
) -> Tuple[EvaluationStatus, Correctness, str]:
    """Run the validator selected by definition.validator_class.

    Returns (status, correctness, extra_msg). status is one of:
      PASSED, INCORRECT_SHAPE, INCORRECT_DTYPE, INCORRECT_NUMERICAL.

    Caller is responsible for catching runtime/compile errors before calling this.
    """
    t = _THRESHOLDS[definition.validator_class]
    max_abs = 0.0
    max_rel = 0.0
    min_matched = 1.0

    for sol_t, ref_t in zip(sol_outputs, ref_outputs):
        if tuple(sol_t.shape) != tuple(ref_t.shape):
            return (
                EvaluationStatus.INCORRECT_SHAPE,
                Correctness(),
                f"shape mismatch: sol={tuple(sol_t.shape)} ref={tuple(ref_t.shape)}",
            )
        if sol_t.dtype != ref_t.dtype:
            return (
                EvaluationStatus.INCORRECT_DTYPE,
                Correctness(),
                f"dtype mismatch: sol={sol_t.dtype} ref={ref_t.dtype}",
            )
        a, r, m = _error_stats(sol_t, ref_t, t)
        max_abs = max(max_abs, a)
        max_rel = max(max_rel, r)
        min_matched = min(min_matched, m)

    correctness = Correctness(
        max_absolute_error=max_abs,
        max_relative_error=max_rel,
        extra={"matched_ratio": min_matched},
    )

    if min_matched < t.required_matched_ratio:
        return (
            EvaluationStatus.INCORRECT_NUMERICAL,
            correctness,
            f"matched_ratio={min_matched:.4f} < required={t.required_matched_ratio} "
            f"(max_abs={max_abs:.4g}, max_rel={max_rel:.4g})",
        )

    return EvaluationStatus.PASSED, correctness, ""
