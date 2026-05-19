"""
Harness primitives for the GB10 bake-off:
  Stats             latency aggregation from N samples
  cuda_event_time   kernel-only CUDA-event timing with cold L2
  Result, emit_json JSON-serializable schema consumed by _summarize.py
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch


@dataclass
class Stats:
    """Latency stats over N timed iterations (all in ms)."""

    mean_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    stdev_ms: float
    stdev_pct: float
    min_ms: float
    max_ms: float
    n: int

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "Stats":
        n = len(samples_ms)
        mean = statistics.fmean(samples_ms)
        median = statistics.median(samples_ms)
        if n >= 2:
            quantiles = statistics.quantiles(samples_ms, n=10, method="inclusive")
            p10, p90 = quantiles[0], quantiles[-1]
            stdev = statistics.stdev(samples_ms)
        else:
            p10 = p90 = median
            stdev = 0.0
        return cls(
            mean_ms=mean,
            median_ms=median,
            p10_ms=p10,
            p90_ms=p90,
            stdev_ms=stdev,
            stdev_pct=stdev / mean * 100.0,
            min_ms=min(samples_ms),
            max_ms=max(samples_ms),
            n=n,
        )


def cuda_event_time(fn: Callable[[], Any], warmup: int, iters: int) -> Stats:
    """Kernel-only timing via CUDA events with cold-cache L2 flush.
    GB10 L2 = 24 MB; 2× L2 = 48 MB evicts any prior footprint."""
    flush_buf = torch.zeros(48 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        flush_buf.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))

    return Stats.from_samples(samples)


@dataclass
class Result:
    """JSON-serializable measurement.

    Fields:
      name      stable identifier (e.g. "kernel/bf16/qkv_proj/M=512/N=12288/K=4096")
      unit      "TFLOPs" / "ms" — interpreted by _summarize.py per row
      measured  absolute measured value in `unit`
      sol       SOL bound in `unit`, or None for tiers that don't model SOL
      stats     Stats over the timed iterations
      extra     per-tier metadata (must include "tier": str)
    """

    name: str
    unit: str
    measured: float
    sol: float | None
    stats: Stats
    extra: dict[str, Any]


def emit_json(results: list[Result]) -> None:
    """Write the bake-off JSON document to stdout."""
    print(json.dumps({
        "schema_version": 2,
        "ts": time.time(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_properties(0).name,
        "arch_list": torch.cuda.get_arch_list(),
        "results": [asdict(r) for r in results],
    }, indent=2, default=str))
