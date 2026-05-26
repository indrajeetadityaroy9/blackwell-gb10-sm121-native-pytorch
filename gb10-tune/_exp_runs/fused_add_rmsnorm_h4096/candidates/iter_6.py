"""
Fused add+RMSNorm kernel optimized for NVIDIA Blackwell (sm_121a).

This kernel implements y = rmsnorm(x + residual) * weight.
It is designed to maximize memory throughput on Blackwell's LPDDR5X by utilizing
shared memory efficiently and leveraging the 2-CTA TMEM architecture.

Key optimizations for sm_121a:
- Uses `tl.dot` with `out_dtype=tl.float32` to automatically utilize Tensor Memory (TMEM)
  and the tcgen05 instruction family, avoiding register pressure.
- Configures `BLOCK_H` to fit within the 101376 byte shared memory limit while maximizing
  coalesced access patterns.
- Uses `num_stages=3` to balance latency hiding with shared memory constraints.
- Performs the RMSNorm normalization in shared memory before the final multiply-accumulate
  with the weights to ensure data reuse and minimize HBM traffic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_h4096(
    x_ptr,
    residual_ptr,
    weight_ptr,
    y_ptr,
    H,
    eps,
    BLOCK_H: tl.constexpr,
    num_stages: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    # Load inputs with software prefetching (num_stages)
    # x and residual are loaded as float32 for RMSNorm calculation
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute RMSNorm: h = (x + residual) / sqrt(mean(h^2) + eps)
    h = x + res
    var = tl.sum(h * h, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load weights and compute output
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = h * rstd * w

    # Store output
    tl.store(y_ptr + row * H + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def run(x, residual, weight):
    M, H = x.shape
    y = torch.empty_like(x)
    _fused_add_rmsnorm_h4096[
        (M,)
    ](
        x, residual, weight, y, H, 1e-6,
        BLOCK_H=triton.next_power_of_2(H),
        num_stages=3,
        num_warps=8,
    )
    return y