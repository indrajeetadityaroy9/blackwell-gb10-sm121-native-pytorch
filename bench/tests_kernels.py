"""
Tier 2 — additional kernel tests beyond the original 6 GEMM/attention benchmarks.
All use the SOL-ExecBench harness from _harness.py.

  bandwidth      Grace LPDDR5X memory bandwidth (triad)
  attn_bwd       FlashAttention backward (5·B·H·S²·D causal)
  rmsnorm        F.rms_norm (bandwidth-bound)
  softmax        F.softmax over [B, H, S, S]
  cross_entropy  F.cross_entropy over [N, V=128k]

Each test returns a Result; merged into bench_full.py ALL_TESTS.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from _harness import (
    Result, Stats, allclose_gate, cuda_event_time, bandwidth_sol_gbs,
    attn_sol, load_config,
)


DEVICE = "cuda"
WARMUP = int(os.environ.get("BENCH_WARMUP", 5))
ITERS = int(os.environ.get("BENCH_ITERS", 50))

# Lazy-loaded SOL config; first call to any test populates it.
_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def _skip(name: str, reason: str, unit: str = "TFLOPs") -> Result:
    stats = Stats.from_samples([0.0])
    return Result(name=name, unit=unit, measured=0.0, sol=None,
                  sol_score=None, sol_limit=None, stats=stats,
                  correctness=None, note=reason)


def _bandwidth_result(name: str, fn: Callable, *, bytes_moved: float,
                      correctness: str | None = None,
                      extra: dict | None = None) -> Result:
    """Time a bandwidth-bound kernel; report GB/s vs LPDDR5X peak."""
    try:
        stats = cuda_event_time(fn, warmup=WARMUP, iters=ITERS)
    except Exception as e:
        return _skip(name, f"{type(e).__name__}: {str(e)[:240]}", unit="GB/s")
    median_s = stats.median_ms / 1000.0
    gbs = bytes_moved / median_s / 1e9
    sol = bandwidth_sol_gbs(_cfg())
    return Result(
        name=name, unit="GB/s", measured=gbs,
        sol=sol, sol_score=None, sol_limit="bandwidth",
        stats=stats, correctness=correctness, note=None,
        extra={"bytes_moved": bytes_moved, **(extra or {})},
    )


# -------------------- bandwidth (Grace LPDDR5X) --------------------

def test_bandwidth() -> Result:
    """STREAM triad `c = a + α·b` over 1 GiB fp32 (3 GiB moved/iter).
    Reported GB/s should approach GB10's 273 GB/s LPDDR5X spec."""
    name = "lpddr5x_triad_1gib"
    N_BYTES = 1 << 30   # 1 GiB
    n = N_BYTES // 4    # fp32 elements
    try:
        a = torch.randn(n, device=DEVICE, dtype=torch.float32)
        b = torch.randn(n, device=DEVICE, dtype=torch.float32)
        c = torch.empty_like(a)
        alpha = 3.0
    except Exception as e:
        return _skip(name, f"setup: {e}", unit="GB/s")
    # triad: read a + read b + write c = 3 × N_BYTES
    fn = lambda: torch.add(a, b, alpha=alpha, out=c)
    return _bandwidth_result(name, fn, bytes_moved=3 * N_BYTES,
                              extra={"shape": [n], "dtype": "fp32"})


# -------------------- FlashAttention backward --------------------

def test_attn_bwd() -> Result:
    """FlashAttention causal backward.
    FLOPs = 5·B·H·S²·D (forward 2 + backward 3 GEMM-equivalents, both causally halved)."""
    B, H, S, D = 4, 32, 4096, 128
    name = f"flash_attn_bwd_B{B}H{H}S{S}D{D}_causal"

    flash_fn = None
    try:
        from flash_attn import flash_attn_func  # type: ignore
        flash_fn = flash_attn_func
        layout = "BSHD"  # (B, S, H, D)
    except ImportError:
        # Fall back to torch SDPA-flash (backward supported)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        flash_fn = None
        layout = "BHSD"

    try:
        if layout == "BSHD":
            q = torch.randn(B, S, H, D, device=DEVICE, dtype=torch.float16,
                            requires_grad=True)
            k = torch.randn(B, S, H, D, device=DEVICE, dtype=torch.float16,
                            requires_grad=True)
            v = torch.randn(B, S, H, D, device=DEVICE, dtype=torch.float16,
                            requires_grad=True)
        else:
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16,
                            requires_grad=True)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16,
                            requires_grad=True)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16,
                            requires_grad=True)
    except Exception as e:
        return _skip(name, f"setup: {e}")

    def run() -> None:
        for t in (q, k, v):
            if t.grad is not None:
                t.grad = None
        if flash_fn is not None:
            out = flash_fn(q, k, v, causal=True)
        else:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.sum().backward()

    try:
        stats = cuda_event_time(run, warmup=WARMUP, iters=ITERS)
    except Exception as e:
        return _skip(name, f"{type(e).__name__}: {str(e)[:240]}")
    flops_per_call = 5.0 * B * H * S * S * D  # 2 fwd + 3 bwd, causally halved
    median_s = stats.median_ms / 1000.0
    tflops = flops_per_call / median_s / 1e12
    sol = attn_sol(B, H, S, D, "fp16", _cfg(), causal=True, backward=True)
    return Result(
        name=name, unit="TFLOPs", measured=tflops,
        sol=sol.sol_tflops, sol_score=None, sol_limit=sol.limit,
        stats=stats, correctness=None, note=None,
        extra={"flops": flops_per_call, "B": B, "H": H, "S": S, "D": D,
               "causal": True, "backward": True, "kernel": "flash_attn" if flash_fn else "sdpa"},
    )


# -------------------- RMSNorm (bandwidth-bound) --------------------

def test_rmsnorm() -> Result:
    """F.rms_norm on [N=8192, H=8192] fp16.
    Bytes: 2 × N·H·2 (read + write) + H·2 (weight)."""
    name = "rmsnorm_N8192_H8192_fp16"
    if not hasattr(F, "rms_norm"):
        return _skip(name, "F.rms_norm not available (need torch 2.6+)", unit="GB/s")
    N_, H_ = 8192, 8192
    try:
        x = torch.randn(N_, H_, device=DEVICE, dtype=torch.float16)
        w = torch.randn(H_, device=DEVICE, dtype=torch.float16)
    except Exception as e:
        return _skip(name, f"setup: {e}", unit="GB/s")
    # Correctness vs fp32 reference
    try:
        ref = F.rms_norm(x.float(), [H_], weight=w.float()).half()
        out = F.rms_norm(x, [H_], weight=w)
        correctness = allclose_gate(out, ref, rtol=1e-2, atol=1e-2)
    except Exception as e:
        correctness = f"FAIL: {type(e).__name__}: {e}"
    fn = lambda: F.rms_norm(x, [H_], weight=w)
    bytes_moved = 2 * N_ * H_ * 2 + H_ * 2   # x + out + weight
    return _bandwidth_result(name, fn, bytes_moved=bytes_moved,
                              correctness=correctness,
                              extra={"shape": [N_, H_], "dtype": "fp16"})


# -------------------- Softmax (bandwidth-bound) --------------------

def test_softmax() -> Result:
    """F.softmax on attention-shape [B=2, H=16, S=4096, S=4096] fp16.
    Bytes: 2 × B·H·S·S·2 (read + write)."""
    name = "softmax_B2H16S4096_fp16"
    B_, H_, S_ = 2, 16, 4096
    try:
        x = torch.randn(B_, H_, S_, S_, device=DEVICE, dtype=torch.float16)
    except Exception as e:
        return _skip(name, f"setup: {e}", unit="GB/s")
    try:
        ref = F.softmax(x.float(), dim=-1).half()
        out = F.softmax(x, dim=-1)
        correctness = allclose_gate(out, ref, rtol=1e-2, atol=1e-2)
    except Exception as e:
        correctness = f"FAIL: {type(e).__name__}: {e}"
    fn = lambda: F.softmax(x, dim=-1)
    bytes_moved = 2 * B_ * H_ * S_ * S_ * 2
    return _bandwidth_result(name, fn, bytes_moved=bytes_moved,
                              correctness=correctness,
                              extra={"shape": [B_, H_, S_, S_], "dtype": "fp16"})


# -------------------- Cross-entropy (LLM bottleneck) --------------------

def test_cross_entropy() -> Result:
    """F.cross_entropy on [N=8192, V=128k] fp32 logits + int64 targets.
    Bytes: N·V·4 (logits) + N·8 (targets) — logit read dominates."""
    name = "cross_entropy_N8192_V128k"
    N_, V_ = 8192, 128 * 1024
    try:
        logits = torch.randn(N_, V_, device=DEVICE, dtype=torch.float32)
        targets = torch.randint(0, V_, (N_,), device=DEVICE, dtype=torch.int64)
    except Exception as e:
        return _skip(name, f"setup: {e}", unit="GB/s")
    try:
        out = F.cross_entropy(logits, targets)
        ref = F.cross_entropy(logits.float(), targets)
        correctness = allclose_gate(
            torch.tensor([out.item()]), torch.tensor([ref.item()]),
            rtol=1e-3, atol=1e-3,
        )
    except Exception as e:
        correctness = f"FAIL: {type(e).__name__}: {e}"
    fn = lambda: F.cross_entropy(logits, targets)
    bytes_moved = N_ * V_ * 4 + N_ * 8   # logits + targets
    return _bandwidth_result(name, fn, bytes_moved=bytes_moved,
                              correctness=correctness,
                              extra={"N": N_, "V": V_, "dtype": "fp32"})


# test_key → callable, merged into bench_full.py ALL_TESTS
TESTS: dict[str, Callable[[], Result]] = {
    "bandwidth": test_bandwidth,
    "attn_bwd": test_attn_bwd,
    "rmsnorm": test_rmsnorm,
    "softmax": test_softmax,
    "cross_entropy": test_cross_entropy,
}
