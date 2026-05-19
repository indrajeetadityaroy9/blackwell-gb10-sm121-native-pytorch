"""
Direct kernel-level GEMM benchmark for the compute-bound regime.

Decode is memory-bound on GB10's 273 GB/s LPDDR5X; all wheels saturate
the same bandwidth ceiling. At M=512 (small prefill), the same Llama-3-8B
projections are compute-bound (arithmetic intensity > GB10's 330 FLOPs/byte
crossover) and cuBLAS / cuDNN heuristic dispatch matters — exactly where
native sm_121 cubins (Run B) should differentiate from the PTX-JIT path
(Runs A, C).

dtype is selected via argv:

  bf16   (default)  torch.matmul (a @ b)
  fp16              torch.matmul (a @ b)
  fp8               torch._scaled_mm  e4m3, scalar scales, use_fast_accum=True
  fp4               torch._scaled_mm  e2m1, 1x16 block scales (mxfp4)

Splitting one dtype per invocation isolates failures: cuBLAS Lt's FP8
heuristic table for sm_121 is incomplete in cuBLAS 13.x and raises
CUBLAS_STATUS_NOT_INITIALIZED on FP8 matmuls. By running each dtype as a
separate subprocess, the bf16+fp16 results survive even when fp8/fp4
fail upstream.

Shapes (Llama-3.1-8B, M = 512):

  qkv_proj   (M, 12288, 4096)
  attn_out   (M,  4096, 4096)
  mlp_up     (M, 14336, 4096)
  mlp_down   (M,  4096, 14336)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Result, cuda_event_time


SHAPES = [
    ("qkv_proj", 512, 12288, 4096),
    ("attn_out", 512,  4096, 4096),
    ("mlp_up",   512, 14336, 4096),
    ("mlp_down", 512,  4096, 14336),
]


def _bf16_gemm(M: int, N: int, K: int):
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    return lambda: a @ b


def _fp16_gemm(M: int, N: int, K: int):
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    return lambda: a @ b


def _fp8_gemm(M: int, N: int, K: int):
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).t()
    scale_a = torch.tensor(1.0, device="cuda")
    scale_b = torch.tensor(1.0, device="cuda")
    return lambda: torch._scaled_mm(
        a, b,
        scale_a=scale_a, scale_b=scale_b,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )


def _fp4_gemm(M: int, N: int, K: int):
    # torch.randn(...).to(float4_e2m1fn_x2) raises in torch 2.10+
    # ("copy_() does not support casting Float to Float4_e2m1fn_x2").
    # Construct via random uint8 storage (2 fp4 packed per byte) reinterpreted
    # as fp4_e2m1fn_x2; for benchmark timing, the exact values don't matter.
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8,
                      device="cuda").view(torch.float4_e2m1fn_x2)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8,
                      device="cuda").view(torch.float4_e2m1fn_x2).t()
    scale_a = torch.ones(M, K // 16, dtype=torch.float8_e4m3fn, device="cuda")
    scale_b = torch.ones(N, K // 16, dtype=torch.float8_e4m3fn, device="cuda")
    return lambda: torch._scaled_mm(
        a, b,
        scale_a=scale_a, scale_b=scale_b,
        out_dtype=torch.bfloat16,
    )


DTYPE_BUILDERS = {
    "bf16": _bf16_gemm,
    "fp16": _fp16_gemm,
    "fp8":  _fp8_gemm,
    "fp4":  _fp4_gemm,
}


def _measure(label: str, dtype: str, fn, M: int, N: int, K: int) -> Result:
    stats = cuda_event_time(fn, warmup=5, iters=50)
    flops = 2.0 * M * N * K
    tflops = flops / (stats.median_ms / 1000.0) / 1e12
    return Result(
        name=f"kernel/{dtype}/{label}/M={M}/N={N}/K={K}",
        unit="TFLOPs",
        measured=tflops,
        sol=None,
        stats=stats,
        extra={
            "tier": "kernel",
            "shape_label": label,
            "M": M, "N": N, "K": K,
            "dtype": dtype,
            "flops": flops,
        },
    )


def main() -> int:
    dtype = sys.argv[1] if len(sys.argv) > 1 else "bf16"
    builder = DTYPE_BUILDERS[dtype]
    results = [_measure(lbl, dtype, builder(M, N, K), M, N, K)
               for lbl, M, N, K in SHAPES]
    print(json.dumps({"results": [asdict(r) for r in results]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
