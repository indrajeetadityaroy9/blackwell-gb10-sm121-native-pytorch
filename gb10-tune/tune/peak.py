"""GB10 roofline ceilings. Primary denominator is the THEORETICAL hardware peak (the
unlock metric — cuBLAS on immature sm_121a reaches only ~59% of it, so "% of cuBLAS"
would hide the opportunity). Secondary is the empirically-measured vendor ceiling.

Theoretical peaks are ESTIMATES (GB10 ~1 PFLOP FP4-sparse / 8 ≈ 125 TFLOPs bf16; DGX
Spark ~273 GB/s LPDDR5X) — refine when NVIDIA publishes sm_121a figures."""

import functools

import torch

from .bench import _cuda_event_time

_HW_PEAK_BF16_TFLOPS = 125.0
_HW_PEAK_BANDWIDTH_GBPS = 273.0


# lru_cache: an 8192³ bf16 GEMM is invoked on every keep + final report — cache makes the
# measurement once-per-process. Do not remove; the cost is otherwise paid each iteration.
@functools.lru_cache(maxsize=1)
def compute_peak_tflops():
    n = 8192
    a = torch.randn(n, n, device="cuda:0", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda:0", dtype=torch.bfloat16)
    st = _cuda_event_time(lambda: torch.matmul(a, b), warmup=5, iters=20)
    return (2.0 * n * n * n) / (st.median_ms / 1000.0) / 1e12


@functools.lru_cache(maxsize=1)
def bandwidth_peak_gbps():
    nbytes = 1 << 30
    x = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda:0")
    y = torch.empty_like(x)
    st = _cuda_event_time(lambda: y.copy_(x), warmup=5, iters=20)
    return (2.0 * nbytes) / (st.median_ms / 1000.0) / 1e9


def utilization(definition, flops, nbytes, latency_ms, tier):
    """pct_of_hw_peak (primary, vs theoretical) + pct_of_cublas (secondary, vs measured)."""
    secs = latency_ms / 1000.0
    achieved_tflops = flops / secs / 1e12
    achieved_gbps = nbytes / secs / 1e9
    out = {"achieved_tflops": achieved_tflops, "achieved_gbps": achieved_gbps}
    if tier == 2:
        out["pct_of_hw_peak"] = achieved_tflops / _HW_PEAK_BF16_TFLOPS
        out["pct_of_cublas"] = achieved_tflops / compute_peak_tflops()
        out["peak_basis"] = "bf16 TFLOPs (hw=125 est.)"
    elif tier == 1:
        out["pct_of_hw_peak"] = achieved_gbps / _HW_PEAK_BANDWIDTH_GBPS
        out["pct_of_cublas"] = achieved_gbps / bandwidth_peak_gbps()
        out["peak_basis"] = "GB/s (hw=273 est.)"
    return out
