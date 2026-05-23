"""FP8 (e4m3fn) GEMM seed — torch._scaled_mm wrapper.

Per-tensor unit scales; output bf16 with fast-accum.
Run B measured 102-137 TFLOPs across shapes at M=512.
"""

import torch


def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K] e4m3fn, B: [K, N] e4m3fn → bf16 [M, N].
    # _scaled_mm wants B in column-major (transposed), so pass B.T.contiguous().T.
    scale_a = torch.tensor(1.0, device=A.device, dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=A.device, dtype=torch.float32)
    return torch._scaled_mm(
        A, B,
        scale_a=scale_a, scale_b=scale_b,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
