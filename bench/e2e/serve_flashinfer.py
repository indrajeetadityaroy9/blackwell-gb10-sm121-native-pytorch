"""
Tier 4: FlashInfer-Bench attention serving benchmark.

Uses the real flashinfer-bench==0.1.2 API (Benchmark + TraceSet + BenchmarkConfig).
The prior agent's `fib.run_trace(...)` symbol was hallucinated — verified against
github.com/flashinfer-ai/flashinfer-bench. See plan Risk Register entry #9.

Requires:
  - FlashInfer source-built with TORCH_CUDA_ARCH_LIST=12.1 (PyPI wheels are
    sm_120-only; trtllm-gen FMHA cubins lack sm_121 → cudaErrorIllegalInstruction).
    Build via: bash bench/e2e/build_flashinfer.sh
  - flashinfer-bench==0.1.2 installed: pip install flashinfer-bench==0.1.2
  - A trace file at bench/e2e/traces/<model>.jsonl (placeholder included; real
    traces should be captured from an actual workload via FlashInfer's tracer).

Standalone invocation:
  python bench/e2e/serve_flashinfer.py [--trace llama31_8b] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make _harness importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import Result, Stats   # noqa: E402

TRACE_DIR = Path(__file__).resolve().parent / "traces"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="llama31_8b",
                    help="trace file basename under bench/e2e/traces/ (no .jsonl)")
    ap.add_argument("--backend", default="flashinfer",
                    choices=["flashinfer", "torch_sdpa", "cudnn"],
                    help="attention backend to benchmark")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON to stdout instead of human-readable")
    args = ap.parse_args()

    try:
        from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet  # type: ignore
    except ImportError as e:
        print(f"[serve_flashinfer] FATAL: flashinfer-bench not installed: {e}",
              file=sys.stderr)
        return 2

    trace_path = TRACE_DIR / f"{args.trace}.jsonl"
    if not trace_path.exists():
        print(f"[serve_flashinfer] FATAL: trace not found: {trace_path}",
              file=sys.stderr)
        return 2

    cfg = BenchmarkConfig(
        backend=args.backend,
        warmup=int(os.environ.get("BENCH_WARMUP", 5)),
        iters=int(os.environ.get("BENCH_ITERS", 50)),
    )
    trace = TraceSet.from_path(str(trace_path))
    print(f"[serve_flashinfer] running {len(trace)} traces × backend={args.backend}",
          file=sys.stderr)

    bench = Benchmark(trace, cfg)
    results = bench.run_all()

    # Adapt flashinfer-bench's per-op Result objects to our schema
    out: list[Result] = []
    for r in results:
        # API: r.op, r.median_ms, r.tflops (best-effort field names per v0.1.2)
        stats = Stats(
            mean_ms=getattr(r, "mean_ms", r.median_ms),
            median_ms=r.median_ms,
            p10_ms=getattr(r, "p10_ms", r.median_ms),
            p90_ms=getattr(r, "p90_ms", r.median_ms),
            stdev_ms=getattr(r, "stdev_ms", 0.0),
            stdev_pct=getattr(r, "stdev_pct", 0.0),
            min_ms=getattr(r, "min_ms", r.median_ms),
            max_ms=getattr(r, "max_ms", r.median_ms),
            n=cfg.iters,
        )
        out.append(Result(
            name=f"flashinfer_{r.op}_{args.backend}",
            unit="TFLOPs", measured=getattr(r, "tflops", 0.0),
            sol=None, sol_score=None, sol_limit=None,
            stats=stats, correctness=None, note=None,
            extra={"trace": str(trace_path), "op": r.op,
                   "backend": args.backend},
        ))

    if args.json:
        print(json.dumps({"results": [r.to_dict() for r in out]}, indent=2))
    else:
        for r in out:
            print(f"  {r.name:60s} : {r.measured:7.2f} {r.unit} "
                  f"(med={r.stats.median_ms:.2f}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
