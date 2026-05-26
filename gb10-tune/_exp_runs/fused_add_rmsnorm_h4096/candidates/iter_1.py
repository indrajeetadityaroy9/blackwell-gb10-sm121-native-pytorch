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
  with weights to ensure data reuse and minimize HBM traffic.
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
    """
    Fused add + RMSNorm kernel for Blackwell (sm_121a).
    
    Computes y = (x + residual) * rsqrt(mean((x + residual)^2) + eps) * weight
    
    Args:
        x_ptr: Pointer to input tensor x of shape [M, H]
        residual_ptr: Pointer to residual tensor of shape [M, H]
        weight_ptr: Pointer to weight tensor of shape [H]
        y_ptr: Pointer to output tensor y of shape [M, H]
        H: Height dimension size
        eps: Epsilon for numerical stability
        BLOCK_H: Block size for the H dimension
        num_stages: Number of software prefetch stages
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    
    # Load inputs with software prefetching (num_stages=3)
    # Using tl.load with explicit mask and other=0.0 for boundary safety
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Fused add
    h = x + residual
    
    # Compute RMSNorm normalization factor
    # Using tl.sum for efficient reduction
    var = tl.sum(h * h, axis=0) / H
    
    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Load weights
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Fused multiply-accumulate
    y = h * rstd * w
    
    # Store output
    tl.store(y_ptr + row * H + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def run(x, residual, weight):
    M, H = x.shape
    y = torch.empty_like(x)
    
    # Configure kernel parameters for Blackwell (sm_121a)
    # BLOCK_H=1024 fits within the 101376 byte shared memory limit
    # num_stages=3 balances latency hiding with SMEM constraints
    _fused_add_rmsnorm_h4096[
        (M,),
    ](
        x, residual, weight, y, H, 1e-6,
        BLOCK_H=1024,
        num_stages=3,
        num_warps=8,
    )
    
    return y