"""
FlashAttention-4 attention benchmark — paper §5 grid (arXiv 2603.05451).

  - Seqlens: 1024, 2048, 4096, 8192, 16384, 32768
  - Head configs: MHA d=64 H=32; MHA d=128 H=16
    (MLA d_q != d_v removed — FA-4 v4.0.0b13's qv path asserts
     `arch // 10 in [10, 11]` (sm_100/sm_110 only) which fails on sm_121)
  - Causal: True and False
  - Direction: forward and backward
  - dtype: bfloat16
  - batch = max(1, 32768 // seqlen)   (constant total tokens = 32k)

Outputs JSON to stdout matching the harness Result schema.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Result, cuda_event_time

from flash_attn.cute import flash_attn_func


HEAD_CONFIGS = [
    ("MHA_d64_H32", 64, 64, 32),
    ("MHA_d128_H16", 128, 128, 16),
]


def _flops(seqlen: int, head_dim_q: int, n_heads: int, causal: bool,
           backward: bool) -> float:
    """Paper §5: FLOPs(fwd) = 4·S²·d·H_q, halved for causal,
    ×2.5 for backward (5 matmuls bwd vs 2 fwd)."""
    fwd = 4.0 * seqlen * seqlen * head_dim_q * n_heads
    if causal:
        fwd /= 2.0
    return fwd * 2.5 if backward else fwd


def _result(name: str, fn, *, flops: float, **extra) -> Result:
    stats = cuda_event_time(fn, warmup=5, iters=50)
    tflops = flops / (stats.median_ms / 1000.0) / 1e12
    return Result(
        name=name,
        unit="TFLOPs",
        measured=tflops,
        sol=None,
        stats=stats,
        extra={"flops": flops, **extra},
    )


def run_forward(seqlen: int, config_name: str, d_q: int, d_v: int,
                n_heads: int, causal: bool) -> Result:
    batch = max(1, 32768 // seqlen)
    q = torch.randn(batch, seqlen, n_heads, d_q, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, seqlen, n_heads, d_q, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, seqlen, n_heads, d_v, dtype=torch.bfloat16, device="cuda")
    return _result(
        f"fa4_fwd/S={seqlen}/cfg={config_name}/causal={'T' if causal else 'F'}",
        lambda: flash_attn_func(q, k, v, causal=causal),
        flops=_flops(seqlen, d_q, n_heads, causal, backward=False),
        seqlen=seqlen, config=config_name,
        head_dim_q=d_q, head_dim_v=d_v, n_heads=n_heads,
        causal=causal, direction="fwd", batch=batch,
    )


def run_backward(seqlen: int, config_name: str, d_q: int, d_v: int,
                 n_heads: int, causal: bool) -> Result:
    batch = max(1, 32768 // seqlen)
    q = torch.randn(batch, seqlen, n_heads, d_q, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(batch, seqlen, n_heads, d_q, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(batch, seqlen, n_heads, d_v, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    grad_out = torch.randn_like(flash_attn_func(q, k, v, causal=causal))

    def step():
        q.grad = None
        k.grad = None
        v.grad = None
        flash_attn_func(q, k, v, causal=causal).backward(grad_out)

    return _result(
        f"fa4_bwd/S={seqlen}/cfg={config_name}/causal={'T' if causal else 'F'}",
        step,
        flops=_flops(seqlen, d_q, n_heads, causal, backward=True),
        seqlen=seqlen, config=config_name,
        head_dim_q=d_q, head_dim_v=d_v, n_heads=n_heads,
        causal=causal, direction="bwd", batch=batch,
    )


def main() -> int:
    results: list[Result] = []
    for seqlen in (1024, 2048, 4096, 8192, 16384, 32768):
        for config_name, d_q, d_v, n_heads in HEAD_CONFIGS:
            for causal in (True, False):
                results.append(run_forward(seqlen, config_name, d_q, d_v, n_heads, causal))
                results.append(run_backward(seqlen, config_name, d_q, d_v, n_heads, causal))

    print(json.dumps({"results": [asdict(r) for r in results]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
