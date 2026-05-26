"""AutoKernel roofline-tier classification (arXiv:2603.21331 §3) — on-device by arithmetic
intensity (FLOPs/byte) vs the GB10 ridge point."""

_RIDGE_FLOP_PER_BYTE = 300.0  # GB10 sm_121a bf16 peak / LPDDR5X bandwidth (approx)

_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float4_e2m1": 1,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "bool": 1,
}

TIER_MEMORY_BOUND = 1
TIER_COMPUTE_BOUND = 2
TIER_LATENCY_BOUND = 3


def flops_and_bytes(definition):
    """(FLOPs, bytes) from const axes + input dtype. None if op has no model."""
    ca = definition.const_axes
    b = _DTYPE_BYTES[next(iter(definition.inputs.values())).dtype.value]
    if definition.is_gemm and {"M", "N", "K"} <= ca.keys():
        m, n, k = ca["M"], ca["N"], ca["K"]
        return 2.0 * m * n * k, float((m * k + k * n + m * n) * b)
    if definition.is_reduction and "N" in ca:
        n = ca["N"]
        return float(n), float(n * b)  # ~1 add/elem read, scalar output
    if definition.is_norm and {"M", "H"} <= ca.keys():
        m, h = ca["M"], ca["H"]
        # read x+residual+weight, write y; ~5 flops/elem (memory-bound, AI ~ 0.8).
        return 5.0 * m * h, float((3 * m * h + h) * b)
    return None


def roofline_tier(definition):
    """2=compute-bound, 1=memory-bound, 3=latency/mixed (also the unknown-op fallback)."""
    fb = flops_and_bytes(definition)
    if fb is None or fb[1] == 0:
        return TIER_LATENCY_BOUND
    ai = fb[0] / fb[1]
    if ai >= _RIDGE_FLOP_PER_BYTE:
        return TIER_COMPUTE_BOUND
    if ai <= _RIDGE_FLOP_PER_BYTE / 8.0:
        return TIER_MEMORY_BOUND
    return TIER_LATENCY_BOUND
