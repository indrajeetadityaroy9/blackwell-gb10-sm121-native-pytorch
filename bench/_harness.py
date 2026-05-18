"""
SOL-ExecBench harness primitives for DGX Spark GB10.

Provides:
  - L2 cache flushing (cold-cache iterations)
  - CUDA event timing (kernel-only)
  - Stats aggregation (mean/median/p10/p90/stdev) via stdlib statistics
  - SOL Score (re-exported from _solar)
  - JSON-serializable Result schema
  - Subprocess isolation for per-test memory cleanup

Stdlib only — numpy is not installed in Run A or Run B containers.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

# Re-export SOL primitives so callers only import from _harness.
from _solar import (  # noqa: F401
    GB10Config, SOL, attn_sol, bandwidth_sol_gbs, bytes_per_elem,
    gemm_sol, load_config, sol_score,
)


_BENCH_DIR = Path(__file__).parent
_ENTRYPOINT = _BENCH_DIR / "bench_full.py"


# ---------- statistics ----------

@dataclass
class Stats:
    """Latency stats over N timed iterations (all in ms)."""
    mean_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    stdev_ms: float
    stdev_pct: float   # stdev_ms / mean_ms × 100 — easier to eyeball than raw stdev
    min_ms: float
    max_ms: float
    n: int

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "Stats":
        n = len(samples_ms)
        if n == 0:
            raise ValueError("Stats.from_samples requires at least 1 sample")
        mean = statistics.fmean(samples_ms)
        # statistics.median is robust to single-sample input; quantiles needs n>=2
        median = statistics.median(samples_ms)
        if n >= 2:
            # p10/p90 via inclusive method (matches numpy default)
            quantiles = statistics.quantiles(samples_ms, n=10, method="inclusive")
            p10, p90 = quantiles[0], quantiles[-1]
            stdev = statistics.stdev(samples_ms)
        else:
            p10 = p90 = median
            stdev = 0.0
        stdev_pct = (stdev / mean * 100.0) if mean > 0 else 0.0
        return cls(
            mean_ms=mean, median_ms=median, p10_ms=p10, p90_ms=p90,
            stdev_ms=stdev, stdev_pct=stdev_pct,
            min_ms=min(samples_ms), max_ms=max(samples_ms), n=n,
        )


# ---------- L2 cache flusher ----------

class L2Flusher:
    """Cold-cache reset between iterations (SOL-ExecBench).
    GB10 L2 = 24 MB; 2× L2 = 48 MB is enough to evict prior footprint."""
    DEFAULT_MB = 48

    def __init__(self, size_mb: int = DEFAULT_MB, device: str = "cuda"):
        n = size_mb * 1024 * 1024 // 4   # int32 elements
        # Halve on OOM (rare; only at very large M)
        try:
            self.buf = torch.zeros(n, dtype=torch.int32, device=device)
        except torch.cuda.OutOfMemoryError:
            n //= 2
            self.buf = torch.zeros(n, dtype=torch.int32, device=device)
        self.size_mb = (n * 4) // (1024 * 1024)

    def flush(self) -> None:
        self.buf.zero_()


# ---------- CUDA event timing ----------

def cuda_event_time(
    fn: Callable[[], Any],
    warmup: int = 5,
    iters: int = 50,
    flush: bool = True,
) -> Stats:
    """Kernel-only timing via CUDA events with cold-cache L2 flush.
    Flush is queued on the same stream before start.record(), so the
    timed window measures only fn() with a cold L2."""
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_event_time requires CUDA")
    flusher = L2Flusher() if flush else None
    # Discard initialization, autotune, lazy compile costs
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        if flusher is not None:
            flusher.flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))   # ms

    return Stats.from_samples(samples)


# ---------- Result schema ----------

@dataclass
class Result:
    """Single test result. JSON-serializable; emitted under --json."""
    name: str                      # test key, e.g. "fp16_gemm"
    unit: str                      # "TFLOPs" or "GB/s"
    measured: float                # throughput in `unit`
    sol: float | None              # SOL bound, same unit; None if unmodeled
    sol_score: float | None        # in [0, 1]; None if baseline/SOL unknown
    sol_limit: str | None          # "compute" or "bandwidth"
    stats: Stats
    correctness: str | None        # "PASS" / "FAIL" / None
    note: str | None = None        # human comment (e.g. SKIPPED reason)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def emit_json(results: list[Result], path: Path | None = None) -> None:
    """Write JSON document. If path is None, write to stdout."""
    doc = {
        "schema_version": 1,
        "ts": time.time(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else None,
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else None,
        "results": [r.to_dict() for r in results],
    }
    s = json.dumps(doc, indent=2, default=str)
    if path is None:
        print(s)
    else:
        path.write_text(s)


# ---------- Subprocess isolation ----------

def run_isolated(test_name: str, env_overrides: dict[str, str] | None = None) -> dict:
    """Spawn one test in its own subprocess (SOL-ExecBench isolation).
    Returns the parsed JSON document the child wrote to stdout.
    Note: __file__ here is _harness.py; we invoke bench_full.py explicitly."""
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    out = subprocess.check_output(
        [sys.executable, str(_ENTRYPOINT), "--only", test_name, "--json"],
        env=env, stderr=subprocess.STDOUT,
    )
    return json.loads(out)


# ---------- Correctness gate helper ----------

def allclose_gate(
    actual: torch.Tensor, reference: torch.Tensor,
    rtol: float = 1e-2, atol: float = 1e-2, name: str = "test",
) -> str:
    """Return 'PASS' or 'FAIL: <detail>'. Never raises."""
    try:
        if torch.allclose(actual.float(), reference.float(), rtol=rtol, atol=atol):
            return "PASS"
        max_diff = (actual.float() - reference.float()).abs().max().item()
        return f"FAIL: max|diff|={max_diff:.4g} (rtol={rtol}, atol={atol})"
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"


# ---------- Self-test ----------

if __name__ == "__main__":
    print("smoke test on CPU stats:")
    s = Stats.from_samples([10.0, 11.0, 9.5, 10.5, 10.2])
    print(f"  mean={s.mean_ms:.2f} median={s.median_ms:.2f} stdev_pct={s.stdev_pct:.2f}")
    print("  expected mean ~10.24, stdev_pct ~5%")

    if torch.cuda.is_available():
        print("\nsmoke test cuda_event_time(no-op):")
        st = cuda_event_time(lambda: torch.cuda.synchronize(), warmup=2, iters=5)
        print(f"  no-op median={st.median_ms:.4f} ms over {st.n} iters")
