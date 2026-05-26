"""FIB 3-class robust correctness (FlashInfer-Bench, arXiv:2601.00227 §5), selected by
Definition.validator_class: deterministic (elementwise, dtype-specific atol), matched_ratio
/ matched_ratio_loose (FP8/FP4 ratio rule), stochastic (TVD vs reference). Compared in fp32."""

from dataclasses import dataclass

import torch

from .data import Correctness, EvaluationStatus


@dataclass(frozen=True)
class ValidatorThresholds:
    atol: float
    rtol: float
    required_matched_ratio: float
    tvd_threshold: float = 0.0


_THRESHOLDS = {
    "matched_ratio": ValidatorThresholds(0.5, 5e-2, 0.95),
    "matched_ratio_loose": ValidatorThresholds(0.5, 1e-1, 0.85),
    "stochastic": ValidatorThresholds(0.0, 0.0, 0.0, tvd_threshold=0.05),
}

# Deterministic atol/rtol by output dtype (AutoKernel §5): bf16 looser than fp16.
_DETERMINISTIC_BY_DTYPE = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
    torch.float32: (1e-4, 1e-4),
    torch.float64: (1e-6, 1e-6),
}


def _deterministic_thresholds(ref_dtype):
    atol, rtol = _DETERMINISTIC_BY_DTYPE[ref_dtype]
    return ValidatorThresholds(atol, rtol, 1.0)


def _error_stats(output, reference, t):
    x = output.to(torch.float32)
    y = reference.to(torch.float32)
    abs_err = torch.abs(x - y)
    rel_err = abs_err / (torch.abs(y) + 1e-8)
    exceeds = (abs_err > t.atol) & (rel_err > t.rtol)
    matched = 1.0 - float(exceeds.sum().item()) / float(abs_err.numel())
    return float(abs_err.max().item()), float(rel_err.max().item()), matched


def _tvd_from_samples(sol_outputs, ref_outputs):
    """Total-variation distance between solution/reference empirical pmfs (on-device)."""
    sol = torch.cat([t.detach().reshape(-1).to(torch.int64) for t in sol_outputs])
    ref = torch.cat([t.detach().reshape(-1).to(torch.int64) for t in ref_outputs])
    n_bins = int(torch.maximum(sol.max(), ref.max()).item()) + 1
    p = torch.bincount(sol, minlength=n_bins).double()
    q = torch.bincount(ref, minlength=n_bins).double()
    p, q = p / p.sum().clamp_min(1.0), q / q.sum().clamp_min(1.0)
    return 0.5 * float(torch.abs(p - q).sum().item())


def validate(definition, sol_outputs, ref_outputs):
    """Returns (status, Correctness, msg). status ∈ {PASSED, INCORRECT_SHAPE/DTYPE/NUMERICAL}.
    Caller catches compile/runtime errors before this."""
    if definition.validator_class == "deterministic" and ref_outputs:
        t = _deterministic_thresholds(ref_outputs[0].dtype)
    else:
        t = _THRESHOLDS[definition.validator_class]

    for sol_t, ref_t in zip(sol_outputs, ref_outputs):
        if tuple(sol_t.shape) != tuple(ref_t.shape):
            return EvaluationStatus.INCORRECT_SHAPE, Correctness(), f"shape {tuple(sol_t.shape)} != {tuple(ref_t.shape)}"
        if sol_t.dtype != ref_t.dtype:
            return EvaluationStatus.INCORRECT_DTYPE, Correctness(), f"dtype {sol_t.dtype} != {ref_t.dtype}"

    if definition.validator_class == "stochastic":
        tvd = _tvd_from_samples(sol_outputs, ref_outputs)
        c = Correctness(extra={"tvd": tvd})
        if tvd > 0.05:
            return EvaluationStatus.INCORRECT_NUMERICAL, c, f"tvd={tvd:.4f} > 0.05"
        return EvaluationStatus.PASSED, c, ""

    max_abs = max_rel = 0.0
    min_matched = 1.0
    for sol_t, ref_t in zip(sol_outputs, ref_outputs):
        a, r, m = _error_stats(sol_t, ref_t, t)
        max_abs, max_rel, min_matched = max(max_abs, a), max(max_rel, r), min(min_matched, m)

    c = Correctness(max_absolute_error=max_abs, max_relative_error=max_rel, extra={"matched_ratio": min_matched})
    if min_matched < t.required_matched_ratio:
        return (EvaluationStatus.INCORRECT_NUMERICAL, c,
                f"matched_ratio={min_matched:.4f} < {t.required_matched_ratio} (max_abs={max_abs:.4g}, max_rel={max_rel:.4g})")
    return EvaluationStatus.PASSED, c, ""
