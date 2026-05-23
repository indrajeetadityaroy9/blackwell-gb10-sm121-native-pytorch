"""Timing harness + bench CLI.

Vendors bench/_harness.cuda_event_time semantics directly (48 MB L2 flush,
CUDA event timing, warmup 5, iters 50/100). Per-device multiprocessing.Lock
serializes timing across the agent's persistent subprocess workers.

CLI:
  python -m explore.bench --definition <name> [--seed | --solution <path>]
                          [--no-ncu] [--device cuda:0] [--n-iters N]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import multiprocessing as mp

from .compile import BuilderRegistry, Runnable
from .data import (
    BuildSpec,
    Correctness,
    Definition,
    Environment,
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

_FLUSH_BYTES = 48 * 1024 * 1024  # GB10 L2 is 24 MB; 2x evicts any prior footprint.
_DEFAULT_WARMUP = 5
_DEFAULT_ITERS = 50
_FP4_ITERS = 100  # spec §6: FP4 variance is high; use n=100.
_INPUT_SEED = 0xC0DE

_DEVICE_LOCKS: Dict[str, Any] = {}


def _device_lock(device: str):
    """Returns a per-device multiprocessing.Lock — serializes timing across workers."""
    if device not in _DEVICE_LOCKS:
        _DEVICE_LOCKS[device] = mp.get_context("spawn").Lock()
    return _DEVICE_LOCKS[device]


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
        quantiles = statistics.quantiles(samples, n=10, method="inclusive")
        p10, p90 = quantiles[0], quantiles[-1]
        stdev = statistics.stdev(samples)
        return cls(
            mean_ms=mean,
            median_ms=median,
            p10_ms=p10,
            p90_ms=p90,
            stdev_ms=stdev,
            stdev_pct=stdev / mean * 100.0,
            n=n,
        )


def cuda_event_time(
    fn: Callable[[], Any], warmup: int, iters: int, device: str = "cuda:0"
) -> _Stats:
    """Kernel-only timing via CUDA events with cold-cache L2 flush.

    Vendored from bench/_harness.cuda_event_time:58 with explicit device argument.
    """
    flush_buf = torch.zeros(_FLUSH_BYTES // 4, dtype=torch.int32, device=device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device=device)

    samples: List[float] = []
    for _ in range(iters):
        flush_buf.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(device=device)
        samples.append(s.elapsed_time(e))
    return _Stats.from_samples(samples)


def time_runnable(
    runnable: Runnable, inputs: List[Any], n_iters: int, device: str = "cuda:0"
) -> _Stats:
    """Time a Runnable with the given inputs. Holds the per-device lock."""
    with _device_lock(device):
        return cuda_event_time(
            lambda: runnable(*inputs), warmup=_DEFAULT_WARMUP, iters=n_iters, device=device
        )


def generate_inputs(definition: Definition, device: str = "cuda:0") -> List[torch.Tensor]:
    """Random inputs sized + dtyped per Definition.inputs. Uses seed=0xC0DE.

    BF16/FP16: torch.randn(shape, dtype=...).
    FP8 (e4m3fn): randn → clamp+quantize to e4m3.
    FP4 (mxfp4 packed): randn → pack into uint8 + e8m0 scale. Returned as
    (A, B, scale_a, scale_b) for the 4-input Definitions.
    """
    g = torch.Generator(device=device).manual_seed(_INPUT_SEED)
    shapes = definition.get_input_shapes()
    dtypes = definition.torch_input_dtypes

    out: List[torch.Tensor] = []
    for name, shape, dtype in zip(definition.inputs.keys(), shapes, dtypes):
        if dtype in (torch.bfloat16, torch.float16, torch.float32):
            t = torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(dtype)
        elif dtype == torch.float8_e4m3fn:
            t = torch.randn(*shape, generator=g, device=device, dtype=torch.float32)
            t = t.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        elif dtype in (torch.int8, torch.uint8):
            # mxfp4 scale tensors are e8m0 stored as uint8 — fill with a neutral scale.
            t = torch.full(tuple(shape), 127, dtype=torch.uint8, device=device)
        else:
            # FP4 packed via float4_e2m1fn_x2: pack random fp32 → x2 uint8.
            t = torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(dtype)
        out.append(t)
    return out


def _materialize_solution(
    definition: Definition, source_code: str, language: str = "triton"
) -> Solution:
    """Build a Solution wrapping the given source code under candidate.py::run."""
    return Solution(
        name=f"_bench_{hashlib.sha256(source_code.encode()).hexdigest()[:12]}",
        definition=definition.name,
        author="bench-cli",
        spec=BuildSpec(
            language=SupportedLanguages(language),
            target_hardware=["sm_121a"],
            entry_point="candidate.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="candidate.py", content=source_code)],
    )


def _load_seed_source(definition: Definition) -> str:
    """Pick the right seed kernel by Definition.inputs['A'].dtype."""
    seed_by_dtype = {
        "bfloat16": "seed_gemm_bf16_fp16.py",
        "float16": "seed_gemm_bf16_fp16.py",
        "float8_e4m3fn": "seed_gemm_fp8.py",
        "float4_e2m1": "seed_gemm_fp4.py",
    }
    a_dtype = definition.inputs["A"].dtype
    seed_file = seed_by_dtype[a_dtype]
    seed_root = Path(__file__).parent.parent / "kernels" / "seed"
    return (seed_root / seed_file).read_text()


def _run_ncu_profile(
    definition: Definition, solution: Solution, device: str = "cuda:0"
) -> Dict[str, Any]:
    """Spawn NCU on a fresh _solution_runner subprocess. Mirrors loop_helpers but
    inline here for the standalone bench CLI."""
    import subprocess
    from .data import RandomInput, Workload
    from .ncu import parse_report

    with tempfile.TemporaryDirectory(prefix="gb10_bench_ncu_") as tmp:
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
            "ncu", "--set", "roofline",
            "--nvtx", "--nvtx-include", "gb10_explore_profile]",
            "--launch-count", "30", "--target-processes", "all",
            "--force-overwrite", "--export", str(rep),
            "--", sys.executable, "-u", "-m", "explore.runner._solution_runner",
            "--data-dir", str(tmp), "--device", device,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return parse_report(rep)


def _flops_for_gemm(definition: Definition) -> Optional[float]:
    """For op_type='gemm', return 2*M*N*K from the const axes. Else None."""
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
    device: str = "cuda:0",
    n_iters: Optional[int] = None,
    run_ncu: bool = True,
) -> Evaluation:
    """Validate + time + (optionally) NCU-profile a Solution. Returns Evaluation.

    Single execution path — any failure propagates as the exception.
    """
    a_dtype = definition.inputs["A"].dtype
    if n_iters is None:
        n_iters = _FP4_ITERS if a_dtype == "float4_e2m1" else _DEFAULT_ITERS

    registry = BuilderRegistry.get_instance()
    sol_runnable = registry.build(definition, solution)
    ref_runnable = registry.build_reference(definition)

    inputs = generate_inputs(definition, device=device)
    with torch.no_grad():
        sol_outputs_raw = sol_runnable(*inputs)
        ref_outputs_raw = ref_runnable(*inputs)
    torch.cuda.synchronize(device=device)

    sol_outputs = [sol_outputs_raw] if isinstance(sol_outputs_raw, torch.Tensor) else list(sol_outputs_raw)
    ref_outputs = [ref_outputs_raw] if isinstance(ref_outputs_raw, torch.Tensor) else list(ref_outputs_raw)

    status, correctness, extra_msg = validate(definition, sol_outputs, ref_outputs)

    if status != EvaluationStatus.PASSED:
        return Evaluation(
            status=status,
            environment=env_snapshot(device),
            timestamp=_now_iso(),
            extra_msg=extra_msg,
            correctness=correctness if status == EvaluationStatus.INCORRECT_NUMERICAL else None,
        )

    stats = time_runnable(sol_runnable, inputs, n_iters=n_iters, device=device)

    flops = _flops_for_gemm(definition)
    tflops = flops / (stats.median_ms / 1000.0) / 1e12 if flops else 0.0

    ncu_diag: Dict[str, Any] = {}
    if run_ncu:
        ncu_diag = _run_ncu_profile(definition, solution, device=device)

    # Stash ncu_diag in correctness.extra for the agent prompt.
    if correctness.extra is None:
        correctness = correctness.model_copy(update={"extra": {}})
    correctness.extra["ncu_diag"] = ncu_diag

    return Evaluation(
        status=EvaluationStatus.PASSED,
        environment=env_snapshot(device),
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
    ap = argparse.ArgumentParser(description="GB10 bench CLI")
    ap.add_argument("--definition", required=True, help="Definition name (loaded from DEFINITIONS_ROOT)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seed", action="store_true", help="Use the seed kernel for this Definition")
    src.add_argument("--solution", type=Path, help="Path to a Solution JSON")
    ap.add_argument("--no-ncu", action="store_true", help="Skip NCU profiling (timing only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-iters", type=int, default=None)
    args = ap.parse_args(argv)

    defn = load_json_file(Definition, DEFINITIONS_ROOT / f"{args.definition}.json")

    if args.seed:
        sol = _materialize_solution(defn, _load_seed_source(defn))
    else:
        sol = load_json_file(Solution, args.solution)

    ev = bench_solution(
        defn, sol, device=args.device, n_iters=args.n_iters, run_ncu=not args.no_ncu
    )
    print(ev.model_dump_json(indent=2))
    return 0 if ev.status == EvaluationStatus.PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
