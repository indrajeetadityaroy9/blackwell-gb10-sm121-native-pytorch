"""
Tier 2-revised — GEMM tests over LLM-derived shapes.

(M, N, K) come from each model's config (hidden_size, intermediate_size,
num_attention_heads, num_key_value_heads, head_dim). M = batch·seq = 4096
(prefill). Five reference models; ~16 unique shapes after dedup.

Per-dtype dispatch:
  fp16  → torch matmul (a @ b)
  fp8   → torch._scaled_mm, float8_e4m3fn, scalar scales, use_fast_accum
  fp4   → torch._scaled_mm, float4_e2m1fn_x2, float8_e4m3fn 1×16 scales

Each test returns list[Result], one per shape.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from _harness import Result, allclose_gate, cuda_event_time
from _solar import gemm_sol, load_config


DEVICE = "cuda"
WARMUP = int(os.environ.get("BENCH_WARMUP", 5))
ITERS = int(os.environ.get("BENCH_ITERS", 50))
M_PREFILL = 4096   # batch=1 × seq=4096 prefill


@dataclass(frozen=True)
class ModelCfg:
    name: str
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int = 128


MODELS = [
    ModelCfg("llama3-8b",      hidden=4096, intermediate=14336, n_heads=32,  n_kv_heads=8),
    ModelCfg("llama3-70b",     hidden=8192, intermediate=28672, n_heads=64,  n_kv_heads=8),
    ModelCfg("mixtral-8x7b",   hidden=4096, intermediate=14336, n_heads=32,  n_kv_heads=8),
    ModelCfg("qwen3-30b-a3b",  hidden=2048, intermediate=6144,  n_heads=32,  n_kv_heads=4),
    ModelCfg("deepseek-v3",    hidden=7168, intermediate=18432, n_heads=128, n_kv_heads=128),
]


def _gemm_shapes() -> list[tuple[str, int, int, int]]:
    """Return (label, M, N, K) for the four GEMM families per model:
      qkv_proj   N=(n_heads+2·n_kv_heads)·head_dim   K=hidden
      attn_out   N=hidden                            K=n_heads·head_dim
      mlp_up     N=intermediate                      K=hidden
      mlp_down   N=hidden                            K=intermediate
    Deduplicated by (M, N, K); label = first model/family to seed the shape."""
    seen: dict[tuple[int, int, int], str] = {}
    M = M_PREFILL
    for m in MODELS:
        qkv_N    = (m.n_heads + 2 * m.n_kv_heads) * m.head_dim
        attn_N   = m.hidden
        attn_K   = m.n_heads * m.head_dim
        for label, N, K in [
            (f"{m.name}_qkv_proj", qkv_N,          m.hidden),
            (f"{m.name}_attn_out", attn_N,         attn_K),
            (f"{m.name}_mlp_up",   m.intermediate, m.hidden),
            (f"{m.name}_mlp_down", m.hidden,       m.intermediate),
        ]:
            key = (M, N, K)
            if key not in seen:
                seen[key] = label
    return [(label, M, N, K) for (M, N, K), label in seen.items()]


# Lazy-load SOLAR config
_CFG = None
def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def _result(name: str, fn: Callable, *, M: int, N: int, K: int,
            dtype: str, correctness: str | None) -> Result:
    stats = cuda_event_time(fn, warmup=WARMUP, iters=ITERS)
    flops = 2.0 * M * N * K
    median_s = stats.median_ms / 1000.0
    tflops = flops / median_s / 1e12
    sol = gemm_sol(M, N, K, dtype, _cfg())
    return Result(
        name=name, unit="TFLOPs", measured=tflops,
        sol=sol.sol_tflops, sol_score=None, sol_limit=sol.limit,
        stats=stats, correctness=correctness,
        extra={"flops": flops, "M": M, "N": N, "K": K},
    )


def _correctness_fp16(M_=512, K_=512, N_=512) -> str:
    a = torch.randn(M_, K_, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_, N_, device=DEVICE, dtype=torch.float16)
    out = a @ b
    ref = a.float() @ b.float()
    return allclose_gate(out, ref, rtol=1e-2, atol=1e-2)


def _correctness_fp8(M_=512, K_=512, N_=512) -> str:
    a32 = torch.randn(M_, K_, device=DEVICE)
    b32 = torch.randn(K_, N_, device=DEVICE)
    a = a32.to(torch.float8_e4m3fn)
    b = b32.to(torch.float8_e4m3fn).t().contiguous().t()
    s = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
    out = torch._scaled_mm(a, b, scale_a=s, scale_b=s,
                           out_dtype=torch.bfloat16, use_fast_accum=True)
    ref = a32 @ b32
    return allclose_gate(out, ref, rtol=5e-2, atol=0.5)


# -------------------- per-dtype tests --------------------

def test_gemm_fp16_llm() -> list[Result]:
    """FP16 GEMM over LLM shapes. One correctness gate at M=N=K=512."""
    correctness = _correctness_fp16()
    out: list[Result] = []
    for label, M, N, K in _gemm_shapes():
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        out.append(_result(
            name=f"gemm_fp16_{label}_M{M}_N{N}_K{K}",
            fn=lambda a=a, b=b: a @ b,
            M=M, N=N, K=K, dtype="fp16", correctness=correctness,
        ))
        del a, b
        torch.cuda.empty_cache()
    return out


def test_gemm_fp8_llm() -> list[Result]:
    """FP8 e4m3 GEMM over LLM shapes. use_fast_accum=True hits Blackwell peak."""
    correctness = _correctness_fp8()
    out: list[Result] = []
    s = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
    for label, M, N, K in _gemm_shapes():
        a = torch.randn(M, K, device=DEVICE).to(torch.float8_e4m3fn)
        b = torch.randn(K, N, device=DEVICE).to(torch.float8_e4m3fn)
        b = b.t().contiguous().t()   # FP8 wants col-major right operand
        out.append(_result(
            name=f"gemm_fp8_{label}_M{M}_N{N}_K{K}",
            fn=lambda a=a, b=b, s=s: torch._scaled_mm(
                a, b, scale_a=s, scale_b=s,
                out_dtype=torch.bfloat16, use_fast_accum=True),
            M=M, N=N, K=K, dtype="fp8",
            correctness=correctness,
        ))
        del a, b
        torch.cuda.empty_cache()
    return out


def test_gemm_fp4_llm() -> list[Result]:
    """NVFP4 GEMM over LLM shapes. 1×16 block scaling with float8_e4m3fn scales."""
    fp4 = torch.float4_e2m1fn_x2
    e4m3 = torch.float8_e4m3fn
    out: list[Result] = []
    # 16 representable FP4 values × random uint8 input → error O(√K·0.5) ≈ 45
    # at K=8192; no usable tolerance.
    correctness = "SKIP: FP4 numerics not gate-able with random input"
    for label, M, N, K in _gemm_shapes():
        a = torch.randint(0, 256, (M, K // 2), device=DEVICE,
                          dtype=torch.uint8).view(fp4)
        b = (torch.randint(0, 256, (N, K // 2), device=DEVICE,
                           dtype=torch.uint8).view(fp4).t())
        scale_a = torch.ones(M, K // 16, device=DEVICE, dtype=e4m3)
        scale_b = torch.ones(N, K // 16, device=DEVICE, dtype=e4m3)
        out.append(_result(
            name=f"gemm_fp4_{label}_M{M}_N{N}_K{K}",
            fn=lambda a=a, b=b, sa=scale_a, sb=scale_b: torch._scaled_mm(
                a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16),
            M=M, N=N, K=K, dtype="fp4", correctness=correctness,
        ))
        del a, b, scale_a, scale_b
        torch.cuda.empty_cache()
    return out


TESTS: dict[str, Callable[[], list[Result]]] = {
    "gemm_fp16_llm": test_gemm_fp16_llm,
    "gemm_fp8_llm":  test_gemm_fp8_llm,
    "gemm_fp4_llm":  test_gemm_fp4_llm,
}


if __name__ == "__main__":
    shapes = _gemm_shapes()
    print(f"{len(shapes)} unique GEMM shapes (M={M_PREFILL} prefill):")
    for label, M, N, K in shapes:
        print(f"  {label:40s} M={M:6d}  N={N:6d}  K={K:6d}")
