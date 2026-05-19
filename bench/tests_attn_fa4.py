"""
Tier 2-revised — FA-4 attention shape grid (forward + backward).

Adapted from FlashAttention-4 (arXiv 2603.05451) §5.1:
  seqlens ∈ {2048, 4096, 8192, 16384}    (paper: 1k-32k; middle 4)
  configs:
    MHA d=64,  H=32              (paper baseline)
    MHA d=128, H=16              (paper baseline)
    GQA d=128, H_q=32, H_kv=8    (Llama-3.1-8B style)
  batch  = max(1, 32768 // seqlen)
  dtype  = bfloat16
  causal = True

DeepSeek MLA (Q=192, V=128) dropped — stock SDPA-flash requires q.head_dim ==
k.head_dim.

FLOPs:
  fwd  = 2·S²·d·H_q     (causal halves 4·S²·d·H_q)
  bwd  = fwd · 2.5      (5 mm in bwd vs 2 in fwd)

Each call wraps `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` — silent MATH
fallback aborts.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from _harness import Result, cuda_event_time
from _solar import attn_sol, load_config


DEVICE = "cuda"
WARMUP = int(os.environ.get("BENCH_WARMUP", 5))
ITERS = int(os.environ.get("BENCH_ITERS", 50))


@dataclass(frozen=True)
class AttnShape:
    label: str
    S: int          # sequence length
    d: int          # head_dim
    H_q: int        # query heads
    H_kv: int       # KV heads (== H_q for MHA, < H_q for GQA)

    @property
    def B(self) -> int:
        return max(1, 32768 // self.S)

    @property
    def is_gqa(self) -> bool:
        return self.H_q != self.H_kv


SEQLENS = [2048, 4096, 8192, 16384]
HEAD_CONFIGS = [
    ("mha_d64",  dict(d=64,  H_q=32, H_kv=32)),    # FA-4 paper: 32 heads, head_dim 64
    ("mha_d128", dict(d=128, H_q=16, H_kv=16)),    # FA-4 paper: 16 heads, head_dim 128
    ("gqa_llama3", dict(d=128, H_q=32, H_kv=8)),   # Llama-3.1-8B GQA
]

SHAPES: list[AttnShape] = [
    AttnShape(label=f"{cfg_name}_S{S}", S=S, **cfg)
    for S in SEQLENS
    for cfg_name, cfg in HEAD_CONFIGS
]


_CFG = None
def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def _assert_flash_backend(q, k, v) -> None:
    """Confirm FLASH_ATTENTION handles this shape; abort if it can't.
    `sdpa_kernel(...)` restricts the backend, so unsupported shapes raise."""
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                        enable_gqa=(q.shape[1] != k.shape[1]))


def _attn_fn(shape: AttnShape, *, backward: bool):
    """Build the fwd or bwd closure for one shape."""
    B, S, d, H_q, H_kv = shape.B, shape.S, shape.d, shape.H_q, shape.H_kv
    enable_gqa = shape.is_gqa
    # SDPA layout: (B, H, S, D)
    q = torch.randn(B, H_q,  S, d, device=DEVICE, dtype=torch.bfloat16,
                    requires_grad=backward)
    k = torch.randn(B, H_kv, S, d, device=DEVICE, dtype=torch.bfloat16,
                    requires_grad=backward)
    v = torch.randn(B, H_kv, S, d, device=DEVICE, dtype=torch.bfloat16,
                    requires_grad=backward)
    # Backend assertion before timing (raises if FLASH_ATTENTION rejects shape)
    _assert_flash_backend(q, k, v)

    if backward:
        def run() -> None:
            for t in (q, k, v):
                t.grad = None
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, enable_gqa=enable_gqa)
            out.sum().backward()
    else:
        def run() -> torch.Tensor:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, enable_gqa=enable_gqa)

    return run, (q, k, v)


def _attn_result(shape: AttnShape, *, backward: bool) -> Result:
    dir_label = "bwd" if backward else "fwd"
    name = f"sdpa_flash_{dir_label}_{shape.label}_Hq{shape.H_q}_Hkv{shape.H_kv}_B{shape.B}"
    run, tensors = _attn_fn(shape, backward=backward)
    stats = cuda_event_time(run, warmup=WARMUP, iters=ITERS)
    # FA-4 §5.1 with causal halving: fwd = 2·S²·d·H_q, bwd = fwd · 2.5
    flops_fwd = 2.0 * shape.S * shape.S * shape.d * shape.H_q
    flops = flops_fwd * (2.5 if backward else 1.0)
    median_s = stats.median_ms / 1000.0
    tflops = flops / median_s / 1e12
    sol = attn_sol(shape.B, shape.H_q, shape.S, shape.d, "bf16", _cfg(),
                   causal=True, backward=backward)
    # Free GPU memory before the next shape
    del tensors
    torch.cuda.empty_cache()
    return Result(
        name=name, unit="TFLOPs", measured=tflops,
        sol=sol.sol_tflops, sol_score=None, sol_limit=sol.limit,
        stats=stats, correctness=None,
        extra={"flops": flops, "B": shape.B, "H_q": shape.H_q, "H_kv": shape.H_kv,
               "S": shape.S, "d": shape.d, "causal": True, "backward": backward,
               "is_gqa": shape.is_gqa, "kernel": "sdpa_flash"},
    )


def test_attn_fa4_fwd() -> list[Result]:
    """FA-4 grid forward pass (12 shapes)."""
    return [_attn_result(s, backward=False) for s in SHAPES]


def test_attn_fa4_bwd() -> list[Result]:
    """FA-4 grid backward pass (12 shapes)."""
    return [_attn_result(s, backward=True) for s in SHAPES]


TESTS: dict[str, Callable[[], list[Result]]] = {
    "attn_fa4_fwd": test_attn_fa4_fwd,
    "attn_fa4_bwd": test_attn_fa4_bwd,
}


if __name__ == "__main__":
    print(f"{len(SHAPES)} attention shapes per direction:")
    for s in SHAPES:
        kind = "GQA" if s.is_gqa else "MHA"
        print(f"  {s.label:20s} B={s.B:3d}  H_q={s.H_q:3d}  H_kv={s.H_kv:3d}  "
              f"S={s.S:5d}  d={s.d:3d}  ({kind})")
