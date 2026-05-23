"""Timing harness + standalone bench CLI.

Vendors bench/_harness.cuda_event_time semantics (48 MB L2 flush, CUDA event
timing). Device is always "cuda:0" — no configurable surface; if the GB10's
single GPU is unavailable the import or the first kernel launch raises.

CLI:
  python -m tune.bench --definition <name> --baseline <path> [--no-ncu] [--n-iters N]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import multiprocessing as mp

from .compile import BuilderRegistry, Runnable
from .data import (
    BuildSpec,
    Correctness,
    Definition,
    Evaluation,
    EvaluationStatus,
    Performance,
    Solution,
    SourceFile,
    SupportedLanguages,
    env_snapshot,
    load_json_file,
    save_json_file,
)
from .trace import DEFINITIONS_ROOT
from .validators import validate

_FLUSH_BYTES = 48 * 1024 * 1024  # GB10 L2 is 24 MB; 2× evicts any prior footprint.
_DEFAULT_WARMUP = 5
_DEFAULT_ITERS = 50
_INPUT_SEED = 0xC0DE

_TIMING_LOCK = mp.get_context("spawn").Lock()


@dataclass
class _Stats:
    mean_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    stdev_ms: float
    stdev_pct: float
    n: int

    @classmethod
    def from_samples(cls, samples: List[float]) -> "_Stats":
        n = len(samples)
        mean = statistics.fmean(samples)
        median = statistics.median(samples)
        q = statistics.quantiles(samples, n=10, method="inclusive")
        stdev = statistics.stdev(samples)
        return cls(
            mean_ms=mean, median_ms=median, p10_ms=q[0], p90_ms=q[-1],
            stdev_ms=stdev, stdev_pct=stdev / mean * 100.0, n=n,
        )


def cuda_event_time(fn: Callable[[], Any], warmup: int, iters: int) -> _Stats:
    """Kernel-only CUDA-event timing with cold-cache L2 flush (48 MB on cuda:0)."""
    flush_buf = torch.zeros(_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda:0")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device="cuda:0")

    samples: List[float] = []
    for _ in range(iters):
        flush_buf.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(device="cuda:0")
        samples.append(s.elapsed_time(e))
    return _Stats.from_samples(samples)


def time_runnable(runnable: Runnable, inputs: List[Any], n_iters: int) -> _Stats:
    """Time a Runnable; serializes against other workers via the module lock."""
    with _TIMING_LOCK:
        return cuda_event_time(
            lambda: runnable(*inputs), warmup=_DEFAULT_WARMUP, iters=n_iters
        )


def generate_inputs(definition: Definition) -> List[torch.Tensor]:
    """Random inputs sized + dtyped per Definition.inputs. Seed 0xC0DE for repro."""
    g = torch.Generator(device="cuda:0").manual_seed(_INPUT_SEED)
    shapes = definition.get_input_shapes()
    dtypes = definition.torch_input_dtypes

    out: List[torch.Tensor] = []
    for shape, dtype in zip(shapes, dtypes):
        if dtype in (torch.bfloat16, torch.float16, torch.float32):
            t = torch.randn(*shape, generator=g, device="cuda:0", dtype=torch.float32).to(dtype)
        elif dtype == torch.float8_e4m3fn:
            t = torch.randn(*shape, generator=g, device="cuda:0", dtype=torch.float32)
            t = t.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        elif dtype in (torch.int8, torch.uint8):
            t = torch.full(tuple(shape), 127, dtype=torch.uint8, device="cuda:0")
        else:
            t = torch.randn(*shape, generator=g, device="cuda:0", dtype=torch.float32).to(dtype)
        out.append(t)
    return out


def _materialize_solution(definition: Definition, source_code: str) -> Solution:
    return Solution(
        name=f"_bench_{hashlib.sha256(source_code.encode()).hexdigest()[:12]}",
        definition=definition.name,
        author="bench-cli",
        spec=BuildSpec(
            language=SupportedLanguages("triton"),
            target_hardware=["sm_121a"],
            entry_point="candidate.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="candidate.py", content=source_code)],
    )


def _run_ncu_profile(definition: Definition, solution: Solution) -> Dict[str, Any]:
    """NCU subprocess for the standalone bench CLI. (Stage 1 uses
    loop_helpers.ncu_profile_subprocess instead.)"""
    import subprocess
    from .data import RandomInput, Workload
    from .ncu import parse_report

    with tempfile.TemporaryDirectory(prefix="gb10_tune_bench_ncu_") as tmp:
        tmp = Path(tmp)
        workload = Workload(
            axes={},
            inputs={k: RandomInput() for k in definition.inputs.keys()},
            uuid=f"bench_{hashlib.sha256(definition.name.encode()).hexdigest()[:8]}",
        )
        save_json_file(definition, tmp / "definition.json")
        save_json_file(solution, tmp / "solution.json")
        save_json_file(workload, tmp / "workload.json")
        rep = tmp / "report.ncu-rep"
        cmd = [
            "ncu", "--set", "full",
            "--nvtx", "--nvtx-include", "gb10_tune_profile]",
            "--launch-count", "30", "--target-processes", "all",
            "--force-overwrite", "--export", str(rep),
            "--", sys.executable, "-u", "-m", "tune.runner._solution_runner",
            "--data-dir", str(tmp),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return parse_report(rep)


def _flops_for_gemm(definition: Definition) -> Optional[float]:
    if definition.op_type != "gemm":
        return None
    ca = definition.const_axes
    return float(2 * ca["M"] * ca["N"] * ca["K"])


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bench_solution(
    definition: Definition,
    solution: Solution,
    *,
    n_iters: Optional[int] = None,
    run_ncu: bool = True,
) -> Evaluation:
    """Validate + time + (optionally) NCU-profile a Solution. Returns Evaluation.

    Maps execution failures to EvaluationStatus:
      - BuildError              → COMPILE_ERROR
      - any other exception during runnable invocation → RUNTIME_ERROR
    Status mapping is the feedback signal Stage 2's LLM consumes — not
    defensive fallback.
    """
    from .compile.registry import BuildError

    if n_iters is None:
        n_iters = _DEFAULT_ITERS

    registry = BuilderRegistry.get_instance()
    try:
        sol_runnable = registry.build(definition, solution)
    except BuildError as e:
        return Evaluation(
            status=EvaluationStatus.COMPILE_ERROR,
            environment=env_snapshot("cuda:0"),
            timestamp=_now_iso(),
            extra_msg=f"{type(e).__name__}: {e}"[:1000],
        )
    except Exception as e:
        return Evaluation(
            status=EvaluationStatus.COMPILE_ERROR,
            environment=env_snapshot("cuda:0"),
            timestamp=_now_iso(),
            extra_msg=f"{type(e).__name__}: {e}"[:1000],
        )

    ref_runnable = registry.build_reference(definition)
    inputs = generate_inputs(definition)

    try:
        with torch.no_grad():
            sol_outputs_raw = sol_runnable(*inputs)
        torch.cuda.synchronize(device="cuda:0")
    except Exception as e:
        return Evaluation(
            status=EvaluationStatus.RUNTIME_ERROR,
            environment=env_snapshot("cuda:0"),
            timestamp=_now_iso(),
            extra_msg=f"{type(e).__name__}: {e}"[:1000],
        )

    with torch.no_grad():
        ref_outputs_raw = ref_runnable(*inputs)
    torch.cuda.synchronize(device="cuda:0")

    sol_outputs = [sol_outputs_raw] if isinstance(sol_outputs_raw, torch.Tensor) else list(sol_outputs_raw)
    ref_outputs = [ref_outputs_raw] if isinstance(ref_outputs_raw, torch.Tensor) else list(ref_outputs_raw)

    status, correctness, extra_msg = validate(definition, sol_outputs, ref_outputs)

    if status != EvaluationStatus.PASSED:
        return Evaluation(
            status=status,
            environment=env_snapshot("cuda:0"),
            timestamp=_now_iso(),
            extra_msg=extra_msg,
            correctness=correctness if status == EvaluationStatus.INCORRECT_NUMERICAL else None,
        )

    stats = time_runnable(sol_runnable, inputs, n_iters=n_iters)

    flops = _flops_for_gemm(definition)
    tflops = flops / (stats.median_ms / 1000.0) / 1e12 if flops else 0.0

    ncu_diag: Dict[str, Any] = {}
    if run_ncu:
        ncu_diag = _run_ncu_profile(definition, solution)

    if correctness.extra is None:
        correctness = correctness.model_copy(update={"extra": {}})
    correctness.extra["ncu_diag"] = ncu_diag

    return Evaluation(
        status=EvaluationStatus.PASSED,
        environment=env_snapshot("cuda:0"),
        timestamp=_now_iso(),
        correctness=correctness,
        performance=Performance(
            latency_ms=stats.median_ms,
            tflops=tflops,
            p10_ms=stats.p10_ms,
            p90_ms=stats.p90_ms,
            stdev_pct=stats.stdev_pct,
            n_iters=stats.n,
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Standalone bench: load a Definition + a baseline kernel source, "
        "compile/verify/time, emit Evaluation JSON."
    )
    ap.add_argument("--definition", required=True, help="Definition name (loaded from DEFINITIONS_ROOT)")
    ap.add_argument("--baseline", required=True, type=Path, help="Path to a kernel source file (def run(...))")
    ap.add_argument("--no-ncu", action="store_true", help="Skip NCU profiling (timing only)")
    ap.add_argument("--n-iters", type=int, default=None)
    args = ap.parse_args(argv)

    defn = load_json_file(Definition, DEFINITIONS_ROOT / f"{args.definition}.json")
    sol = _materialize_solution(defn, args.baseline.read_text())
    ev = bench_solution(defn, sol, n_iters=args.n_iters, run_ncu=not args.no_ncu)
    print(ev.model_dump_json(indent=2))
    return 0 if ev.status == EvaluationStatus.PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
