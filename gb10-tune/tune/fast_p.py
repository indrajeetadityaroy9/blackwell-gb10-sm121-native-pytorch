"""FIB fast_p AUC: trapezoidal integral of fraction(correct & speedup>p) over p in [0, p_max]."""

import statistics

from .data import EvaluationStatus


def fast_p_auc(evaluations, baseline_ms, p_max=2.0):
    if not evaluations or not baseline_ms or len(evaluations) != len(baseline_ms):
        return 0.0
    n = len(evaluations)
    speedups = []
    for ev, base in zip(evaluations, baseline_ms):
        if ev.status != EvaluationStatus.PASSED:  # PASSED guarantees performance is set
            continue
        cand = ev.performance.latency_ms
        if cand > 0.0 and base > 0.0:
            speedups.append(base / cand)

    n_points = 21
    step = p_max / (n_points - 1)
    frac = [sum(1 for s in speedups if s > i * step) / n for i in range(n_points)]
    integral = sum(0.5 * (frac[i] + frac[i + 1]) * step for i in range(n_points - 1))
    return integral / p_max


def median_speedup(evaluations, baseline_ms):
    """Median raw speedup (baseline_ms / candidate_ms) over PASSED workloads; 0.0 if none."""
    if not evaluations or not baseline_ms or len(evaluations) != len(baseline_ms):
        return 0.0
    speedups = []
    for ev, base in zip(evaluations, baseline_ms):
        if ev.status != EvaluationStatus.PASSED:  # PASSED guarantees performance is set
            continue
        cand = ev.performance.latency_ms
        if cand > 0.0 and base > 0.0:
            speedups.append(base / cand)
    return statistics.median(speedups) if speedups else 0.0
