"""Fused add+RMSNorm seed kernel — one program per row, single HBM pass.

y = rmsnorm(x + residual) * weight, with fp32 accumulation of the sum-of-squares.
Eager PyTorch runs this as ~5 unfused ATen kernels (add, pow, mean, rsqrt, mul, mul)
with repeated HBM round-trips; fusing into one Triton pass is the memory-bound win
(cf. AutoKernel RMSNorm 5.29x over eager). This seed is a straightforward fused
version; the tuner's job is to push it toward the LPDDR5X bandwidth roofline.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm(x_ptr, res_ptr, w_ptr, y_ptr, H, eps, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    off = row * H + cols
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
    h = x + res
    var = tl.sum(h * h, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = h * rstd * w
    tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty), mask=mask)


def run(x, residual, weight):
    M, H = x.shape
    y = torch.empty_like(x)
    _fused_add_rmsnorm[(M,)](
        x, residual, weight, y, H, 1e-6,
        BLOCK_H=triton.next_power_of_2(H), num_warps=8,
    )
    return y
