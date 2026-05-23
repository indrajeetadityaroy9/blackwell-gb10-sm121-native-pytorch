"""FP4 (mxfp4 / nvfp4) GEMM seed — torch._scaled_mm wrapper with FP4 inputs + scales.

Output bf16. Run B measured 157-227 TFLOPs across shapes at M=512.
Phase 3 prereq (spec §11): smoke this against PyTorch 2.10's actual FP4 API entry
point before Phase 3 launches — `torch._scaled_mm` signature for FP4 has shifted
across PyTorch versions.
"""

import torch


def run(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    # A, B: float4_e2m1fn_x2 packed (uint8 storage), scale_a/scale_b: e8m0 (mxfp4).
    return torch._scaled_mm(
        A, B,
        scale_a=scale_a, scale_b=scale_b,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
