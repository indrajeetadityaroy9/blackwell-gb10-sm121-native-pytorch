"""
FlashAttention-4 attention benchmark — paper §5 grid (arXiv 2603.05451).

Exercises flash_attn.cute.flash_attn_func across the published shape grid:
  - Seqlens: 1024, 2048, 4096, 8192, 16384, 32768
  - Head configs: MHA d=64 H=32; MHA d=128 H=16; (d_q=192, d_v=128) H=16
  - Causal: True and False
  - Direction: forward and backward
  - dtype: bfloat16
  - batch_size: max(1, 32768 // seqlen)  (constant total tokens = 32k)

Outputs JSON to stdout matching the schema bench/normalize.from_fa4 expects.
No try/except; no SDPA fallback. If FA-4 import or any kernel call fails,
the process exits non-zero with the real traceback.

Arch-guard patching happens at install time in bench/build/build_fa4.py;
the import here is unmodified.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Set arch flags before any torch CUDA initialization triggers a JIT compile.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1;12.1a")
os.environ.setdefault(
    "NVCC_APPEND_FLAGS",
    "-gencode arch=compute_121,code=sm_121a -ptxas-options=-O3 "
    "-D__CUDA_ARCH_FEAT_SM90_ALL",
)

import torch

# Make _harness importable from same dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import L2Flusher, Result, Stats, cuda_event_time

from flash_attn.cute import flash_attn_func


DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP = int(os.environ.get("BENCH_WARMUP", 5))
ITERS = int(os.environ.get("BENCH_ITERS", 50))

# FA-4 paper §5: seqlens 1k, 2k, 4k, 8k, 16k, 32k.
SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768]

# Head configs from the paper. (d_q, d_v, n_heads).
HEAD_CONFIGS = [
    ("MHA_d64_H32", 64, 64, 32),
    ("MHA_d128_H16", 128, 128, 16),
    ("MLA_dq192_dv128_H16", 192, 128, 16),
]


def _flops(seqlen: int, head_dim_q: int, n_heads: int, causal: bool,
           backward: bool) -> float:
    """Per the FA-4 paper §5: FLOPs(fwd) = 4·S²·d·H_q, halved for causal,
    ×2.5 for backward (5 matmuls bwd vs 2 fwd)."""
    fwd = 4.0 * seqlen * seqlen * head_dim_q * n_heads
    if causal:
        fwd /= 2.0
    return fwd * 2.5 if backward else fwd


def _build_qkv(batch: int, seqlen: int, d_q: int, d_v: int,
               n_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch, seqlen, n_heads, d_q, dtype=DTYPE, device=DEVICE)
    k = torch.randn(batch, seqlen, n_heads, d_q, dtype=DTYPE, device=DEVICE)
    v = torch.randn(batch, seqlen, n_heads, d_v, dtype=DTYPE, device=DEVICE)
    return q, k, v


def _measure_one(name: str, fn, *, flops: float) -> Result:
    stats = cuda_event_time(fn, warmup=WARMUP, iters=ITERS)
    median_s = stats.median_ms / 1000.0
    tflops = flops / median_s / 1e12
    return Result(
        name=name,
        unit="TFLOPs",
        measured=tflops,
        sol=None,
        sol_score=None,
        sol_limit=None,
        stats=stats,
        correctness=None,
        extra={"flops": flops},
    )


def run_forward(seqlen: int, config_name: str, d_q: int, d_v: int,
                n_heads: int, causal: bool) -> Result:
    batch = max(1, 32768 // seqlen)
    q, k, v = _build_qkv(batch, seqlen, d_q, d_v, n_heads)
    flops = _flops(seqlen, d_q, n_heads, causal, backward=False)
    name = (f"fa4_fwd/S={seqlen}/cfg={config_name}/"
            f"causal={'T' if causal else 'F'}")

    def step():
        return flash_attn_func(q, k, v, causal=causal)

    r = _measure_one(name, step, flops=flops)
    r.extra.update({
        "seqlen": seqlen, "config": config_name,
        "head_dim_q": d_q, "head_dim_v": d_v, "n_heads": n_heads,
        "causal": causal, "direction": "fwd", "batch": batch,
    })
    return r


def run_backward(seqlen: int, config_name: str, d_q: int, d_v: int,
                 n_heads: int, causal: bool) -> Result:
    batch = max(1, 32768 // seqlen)
    q, k, v = _build_qkv(batch, seqlen, d_q, d_v, n_heads)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    flops = _flops(seqlen, d_q, n_heads, causal, backward=True)
    name = (f"fa4_bwd/S={seqlen}/cfg={config_name}/"
            f"causal={'T' if causal else 'F'}")

    out = flash_attn_func(q, k, v, causal=causal)
    grad_out = torch.randn_like(out)

    def step():
        # zero grads each iter so backward is a fair measurement
        for t in (q, k, v):
            if t.grad is not None:
                t.grad = None
        out_ = flash_attn_func(q, k, v, causal=causal)
        out_.backward(grad_out)

    r = _measure_one(name, step, flops=flops)
    r.extra.update({
        "seqlen": seqlen, "config": config_name,
        "head_dim_q": d_q, "head_dim_v": d_v, "n_heads": n_heads,
        "causal": causal, "direction": "bwd", "batch": batch,
    })
    return r


def main() -> int:
    results: list[Result] = []
    for seqlen in SEQLENS:
        for config_name, d_q, d_v, n_heads in HEAD_CONFIGS:
            for causal in (True, False):
                results.append(
                    run_forward(seqlen, config_name, d_q, d_v, n_heads, causal)
                )
                results.append(
                    run_backward(seqlen, config_name, d_q, d_v, n_heads, causal)
                )

    doc = {"results": [asdict(r) for r in results]}
    print(json.dumps(doc, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
