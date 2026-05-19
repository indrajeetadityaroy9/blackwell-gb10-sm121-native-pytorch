"""
Tier 2 — Triton autotuned matmul (TritonForge, arXiv 2512.09196) over
LLM-derived GEMM shapes from tests_gemm_llm._gemm_shapes.

triton.autotune over the 162-config TritonForge grid; key=[M,N,K] caches
the best config per shape.

Runs alongside the fixed-tile `triton` test:
  triton            — codegen quality at one fixed config
  triton_autotuned  — best Triton across all LLM shapes

First-shape autotune cost: ~5-10 min (162 compiles). Cache mounted per container
at bench/cache/triton/sm121/run{A,B,C}/ via run_bakeoff.sh.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from _harness import (
    Result, allclose_gate, cuda_event_time, load_config,
)
from _solar import gemm_sol
from tests_gemm_llm import _gemm_shapes


DEVICE = "cuda"
WARMUP = int(os.environ.get("BENCH_WARMUP", 5))
ITERS = int(os.environ.get("BENCH_ITERS", 50))


def test_triton_autotuned() -> list[Result]:
    """TritonForge sweep over 162 configs (BLOCK_M/N/K × num_stages × num_warps),
    GROUP_M=8 fixed. Iterates the same LLM shapes as tests_gemm_llm; the autotune
    cache keyed by [M,N,K] picks the best config per shape."""
    import triton
    import triton.language as tl

    configs = [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
            num_stages=ns, num_warps=nw,
        )
        for bm in (64, 128, 256)
        for bn in (64, 128, 256)
        for bk in (32, 64, 128)
        for ns in (2, 3, 4)
        for nw in (4, 8)
    ]

    @triton.autotune(configs=configs, key=["M", "N", "K"])
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c = acc.to(tl.float16)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    cfg = load_config()
    out: list[Result] = []
    for label, M, N, K in _gemm_shapes():
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        c = torch.empty(M, N, device=DEVICE, dtype=torch.float16)

        def run() -> None:
            # Under autotune, the grid lambda receives the chosen config via META.
            grid = lambda META: (
                triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            )
            matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
            )

        # First call triggers autotune (slow); pre-run outside the timing window.
        run()
        torch.cuda.synchronize()

        # Correctness gate vs torch fp16 matmul reference
        ref = a @ b
        correctness = allclose_gate(c, ref, rtol=1e-2, atol=1e-2)

        # Time it — cache hits the autotuned best config
        stats = cuda_event_time(run, warmup=WARMUP, iters=ITERS)

        flops_per_call = 2.0 * M * N * K
        median_s = stats.median_ms / 1000.0
        tflops = flops_per_call / median_s / 1e12
        sol = gemm_sol(M, N, K, "fp16", cfg)

        # Record chosen config in extra payload (shown in SUMMARY)
        cached = matmul_kernel.cache[(M, N, K)]
        chosen = {"BLOCK_M": cached.kwargs["BLOCK_M"],
                  "BLOCK_N": cached.kwargs["BLOCK_N"],
                  "BLOCK_K": cached.kwargs["BLOCK_K"],
                  "num_stages": cached.num_stages,
                  "num_warps": cached.num_warps}

        out.append(Result(
            name=f"triton_autotuned_{label}_M{M}_N{N}_K{K}",
            unit="TFLOPs", measured=tflops,
            sol=sol.sol_tflops, sol_score=None, sol_limit=sol.limit,
            stats=stats, correctness=correctness,
            extra={"flops": flops_per_call, "configs_searched": len(configs),
                   "best_config": chosen, "M": M, "N": N, "K": K},
        ))
        del a, b, c
        torch.cuda.empty_cache()
    return out


TESTS: dict[str, Callable[[], list[Result]]] = {
    "triton_autotuned": test_triton_autotuned,
}
