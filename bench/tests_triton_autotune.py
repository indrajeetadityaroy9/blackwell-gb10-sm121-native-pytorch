"""
Tier 2 — Triton autotuned matmul (TritonForge methodology, arXiv 2512.09196).

Plain triton.autotune over the same 162-config search grid as TritonForge
(no profiling feedback). Cache key=[M,N,K] so iters reuse the best config.

Runs as a SECOND test alongside the fixed-tile `triton` test:
  - `triton`           : compiler codegen quality (one fixed config)
  - `triton_autotuned` : best Triton performance on this hardware

First-run autotune cost: ~5-10 min (162 compiles). Cache mounted per-container
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
    Result, Stats, allclose_gate, cuda_event_time, load_config,
)
from _solar import gemm_sol


DEVICE = "cuda"
WARMUP = int(os.environ.get("BENCH_WARMUP", 5))
ITERS = int(os.environ.get("BENCH_ITERS", 50))
M_DEFAULT = int(os.environ.get("BENCH_M", 8192))


def _skip(name: str, reason: str) -> Result:
    stats = Stats.from_samples([0.0])
    return Result(name=name, unit="TFLOPs", measured=0.0, sol=None,
                  sol_score=None, sol_limit=None, stats=stats,
                  correctness=None, note=reason)


def test_triton_autotuned() -> Result:
    """TritonForge-style sweep over 162 configs:
      BLOCK_M ∈ {64,128,256}, BLOCK_N ∈ {64,128,256}, BLOCK_K ∈ {32,64,128},
      num_stages ∈ {2,3,4}, num_warps ∈ {4,8}.
    GROUP_M=8 fixed (TritonForge recipe)."""
    name = "triton_autotuned_matmul_8192_fp16"
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        return _skip(name, f"triton: {e}")

    M = N = K = M_DEFAULT

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

    try:
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        c = torch.empty(M, N, device=DEVICE, dtype=torch.float16)
    except Exception as e:
        return _skip(name, f"setup: {e}")

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
    try:
        run()
        torch.cuda.synchronize()
    except Exception as e:
        return _skip(name, f"autotune: {type(e).__name__}: {str(e)[:240]}")

    # Correctness gate vs torch fp16 matmul reference
    try:
        ref = a @ b
        correctness = allclose_gate(c, ref, rtol=1e-2, atol=1e-2)
    except Exception as e:
        correctness = f"FAIL: {type(e).__name__}: {e}"

    # Time it — cache hits the autotuned best config
    try:
        stats = cuda_event_time(run, warmup=WARMUP, iters=ITERS)
    except Exception as e:
        return _skip(name, f"{type(e).__name__}: {str(e)[:240]}")

    cfg = load_config()
    flops_per_call = 2.0 * M * N * K
    median_s = stats.median_ms / 1000.0
    tflops = flops_per_call / median_s / 1e12
    sol = gemm_sol(M, N, K, "fp16", cfg)

    # Record chosen config in extra payload (shown in SUMMARY)
    chosen = None
    try:
        cached = matmul_kernel.cache.get((M, N, K))
        if cached is not None:
            chosen = {"BLOCK_M": cached.kwargs["BLOCK_M"],
                      "BLOCK_N": cached.kwargs["BLOCK_N"],
                      "BLOCK_K": cached.kwargs["BLOCK_K"],
                      "num_stages": cached.num_stages,
                      "num_warps": cached.num_warps}
    except Exception:
        pass

    return Result(
        name=name, unit="TFLOPs", measured=tflops,
        sol=sol.sol_tflops, sol_score=None, sol_limit=sol.limit,
        stats=stats, correctness=correctness, note=None,
        extra={"flops": flops_per_call, "configs_searched": len(configs),
               "best_config": chosen},
    )


TESTS: dict[str, Callable[[], Result]] = {
    "triton_autotuned": test_triton_autotuned,
}
