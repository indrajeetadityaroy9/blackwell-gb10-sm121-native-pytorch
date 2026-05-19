"""
Harness primitives for the GB10 bake-off:
  - Stats: latency aggregation from N samples (mean/median/p10/p90/stdev)
  - L2Flusher: cold-cache reset between iterations (GB10 L2 = 24 MB)
  - cuda_event_time: kernel-only CUDA-event timing with cold L2
  - Result + emit_json: JSON-serializable schema consumed by _summarize.py

Stdlib only — numpy is not assumed in every container.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


# ---------- statistics ----------


@dataclass
class Stats:
    """Latency stats over N timed iterations (all in ms)."""

    mean_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    stdev_ms: float
    stdev_pct: float  # stdev_ms / mean_ms × 100
    min_ms: float
    max_ms: float
    n: int

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "Stats":
        n = len(samples_ms)
        if n == 0:
            raise ValueError("Stats.from_samples requires at least 1 sample")
        mean = statistics.fmean(samples_ms)
        median = statistics.median(samples_ms)
        if n >= 2:
            quantiles = statistics.quantiles(samples_ms, n=10, method="inclusive")
            p10, p90 = quantiles[0], quantiles[-1]
            stdev = statistics.stdev(samples_ms)
        else:
            p10 = p90 = median
            stdev = 0.0
        stdev_pct = (stdev / mean * 100.0) if mean > 0 else 0.0
        return cls(
            mean_ms=mean,
            median_ms=median,
            p10_ms=p10,
            p90_ms=p90,
            stdev_ms=stdev,
            stdev_pct=stdev_pct,
            min_ms=min(samples_ms),
            max_ms=max(samples_ms),
            n=n,
        )


# ---------- L2 cache flusher ----------


class L2Flusher:
    """Cold-cache reset between iterations. GB10 L2 = 24 MB; 2× L2 = 48 MB
    is enough to evict any prior kernel footprint."""

    DEFAULT_MB = 48

    def __init__(self, size_mb: int = DEFAULT_MB):
        n = size_mb * 1024 * 1024 // 4  # int32 elements
        self.buf = torch.zeros(n, dtype=torch.int32, device="cuda")

    def flush(self) -> None:
        self.buf.zero_()


# ---------- CUDA event timing ----------


def cuda_event_time(
    fn: Callable[[], Any],
    warmup: int = 5,
    iters: int = 50,
) -> Stats:
    """Kernel-only timing via CUDA events with cold-cache L2 flush between
    every timed iteration. Single deterministic GPU path; flush is always
    on (the bake-off has no warm-cache regime)."""
    flusher = L2Flusher()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        flusher.flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))  # ms

    return Stats.from_samples(samples)


# ---------- Result schema ----------


@dataclass
class Result:
    """JSON-serializable measurement.

    Fields:
      name      stable identifier (e.g. "fa4_fwd/S=4096/cfg=MHA_d128_H16/causal=T")
      unit      "TFLOPs" / "ms" / "tokens/s" — interpreted by _summarize.py per row
      measured  absolute measured value in `unit`
      sol       SOL bound in `unit`, or None for tiers that don't model SOL
                (back-derived from NCU achieved-% for the roofline tier).
      stats     Stats over the timed iterations
      extra     free-form per-tier metadata (must include "tier": str)
    """

    name: str
    unit: str
    measured: float
    sol: float | None
    stats: Stats
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def emit_json(results: list[Result], path: Path | None = None) -> None:
    """Write JSON document. If path is None, write to stdout."""
    doc = {
        "schema_version": 2,
        "ts": time.time(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_properties(0).name,
        "arch_list": torch.cuda.get_arch_list(),
        "results": [r.to_dict() for r in results],
    }
    s = json.dumps(doc, indent=2, default=str)
    if path is None:
        print(s)
    else:
        path.write_text(s)
