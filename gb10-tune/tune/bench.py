"""Timing + benchmark harness.
  bench(definition, source, workloads) -> [Evaluation]
  bench_reference(definition, workloads) -> [float]   (fast_p baseline latencies)
CLI: python -m tune.bench --definition <name> --source <path>
"""

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import multiprocessing as mp
from safetensors.torch import load_file

from .data import (Definition, Evaluation, EvaluationStatus, Performance, Workload,
                   env_snapshot, load_json_file)
from .data.workload import SafetensorsInput, ScalarInput
from .validators import validate

_FLUSH_BYTES = 48 * 1024 * 1024  # 2× GB10's 24 MB L2 to evict prior footprint
_TIMING_LOCK = mp.get_context("spawn").Lock()
_FLUSH_BUF = torch.zeros(_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda:0")


@dataclass
class _Stats:
    median_ms: float
    p10_ms: float
    p90_ms: float
    stdev_pct: float
    n: int


def _stats_from(samples):
    mean = statistics.fmean(samples)
    q = statistics.quantiles(samples, n=10, method="inclusive")
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return _Stats(statistics.median(samples), q[0], q[-1],
                  (stdev / mean * 100.0) if mean > 0 else 0.0, len(samples))


def _cuda_event_time(fn, warmup=5, iters=50):
    # Per-iteration sync: each kernel timed in isolation with an L2 flush before it.
    # Batching all records with one trailing sync lets kernels queue back-to-back and the
    # events capture scheduling gaps — a heavy tail that inflates stdev ~32% → ~10%.
    flush_buf = _FLUSH_BUF
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device="cuda:0")
    samples = []
    for _ in range(iters):
        flush_buf.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize(device="cuda:0")
        samples.append(s.elapsed_time(e))
    return _stats_from(samples)


def _load_run_callable(source, module_name):
    # Triton @jit needs a real source file (inspect.getsourcefile); exec() of a synthetic
    # module fails, so write a temp .py and import it.
    name = f"_tune_{module_name}_{hashlib.sha256(source.encode()).hexdigest()[:12]}"
    path = Path(tempfile.gettempdir()) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "run", None)
    if not callable(fn):
        raise RuntimeError(f"{module_name}: no callable `run`")
    return fn


def _materialize_inputs(definition, workload):
    shapes = definition.get_input_shapes(dict(workload.axes))
    dtypes = definition.torch_input_dtypes
    seed = int.from_bytes(workload.uuid.encode()[:8].ljust(8, b"\0"), "little") & 0x7FFFFFFF
    g = torch.Generator(device="cuda:0").manual_seed(seed)
    out = []
    for (name, _), shape, dtype in zip(definition.inputs.items(), shapes, dtypes):
        inp = workload.inputs[name]
        if isinstance(inp, ScalarInput):
            out.append(inp.value)
        elif isinstance(inp, SafetensorsInput):
            out.append(load_file(inp.path)[inp.tensor_key].to("cuda:0").to(dtype))
        elif dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            out.append(torch.randn(*shape, generator=g, device="cuda:0", dtype=torch.float32).to(dtype))
        elif dtype == torch.float8_e4m3fn:
            out.append(torch.randn(*shape, generator=g, device="cuda:0", dtype=torch.float32).clamp(-448.0, 448.0).to(dtype))
        else:
            out.append(torch.randint(0, 127, tuple(shape), generator=g, device="cuda:0", dtype=torch.int64).to(dtype))
    return out


def _now_iso():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _err_eval(status, msg, correctness=None):
    return Evaluation(status=status, environment=env_snapshot("cuda:0"), timestamp=_now_iso(),
                      extra_msg=msg[:1000], correctness=correctness)


def _eval_one(definition, candidate_run, reference_run, workload):
    try:
        inputs = _materialize_inputs(definition, workload)
        with torch.no_grad():
            sol_raw = candidate_run(*inputs)
        torch.cuda.synchronize(device="cuda:0")
    except Exception as e:
        return _err_eval(EvaluationStatus.RUNTIME_ERROR, f"{type(e).__name__}: {e}")

    with torch.no_grad():
        ref_raw = reference_run(*inputs)
    torch.cuda.synchronize(device="cuda:0")

    sol = [sol_raw] if isinstance(sol_raw, torch.Tensor) else list(sol_raw)
    ref = [ref_raw] if isinstance(ref_raw, torch.Tensor) else list(ref_raw)
    status, correctness, msg = validate(definition, sol, ref)
    if status != EvaluationStatus.PASSED:
        return _err_eval(status, msg, correctness if status == EvaluationStatus.INCORRECT_NUMERICAL else None)

    with _TIMING_LOCK:
        st = _cuda_event_time(lambda: candidate_run(*inputs))
    return Evaluation(
        status=EvaluationStatus.PASSED, environment=env_snapshot("cuda:0"), timestamp=_now_iso(),
        correctness=correctness,
        performance=Performance(latency_ms=st.median_ms, p10_ms=st.p10_ms, p90_ms=st.p90_ms,
                                stdev_pct=st.stdev_pct, n_iters=st.n),
    )


def bench(definition, source, workloads):
    """Compile `source` once; one Evaluation per Workload. Compile failure → COMPILE_ERROR for all."""
    try:
        candidate_run = _load_run_callable(source, "candidate")
    except Exception as e:
        return [_err_eval(EvaluationStatus.COMPILE_ERROR, f"{type(e).__name__}: {e}") for _ in workloads]
    reference_run = _load_run_callable(definition.reference, "reference")
    return [_eval_one(definition, candidate_run, reference_run, w) for w in workloads]


def bench_reference(definition, workloads):
    """Time the fast_p baseline (perf_baseline / native eager) per Workload."""
    src = definition.perf_baseline if definition.perf_baseline is not None else definition.reference
    baseline_run = _load_run_callable(src, "perf_baseline")
    out = []
    for w in workloads:
        inputs = _materialize_inputs(definition, w)
        with _TIMING_LOCK:
            out.append(_cuda_event_time(lambda: baseline_run(*inputs)).median_ms)
    return out


def _definitions_root():
    return Path(os.environ.get("GB10_DEFINITIONS", "/opt/tune/definitions"))


def _load_workloads(definition_name):
    root = _definitions_root() / "workloads" / definition_name
    return [load_json_file(Workload, p) for p in sorted(root.glob("*.json"))]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bench a kernel source against a Definition's workloads.")
    ap.add_argument("--definition", required=True)
    ap.add_argument("--source", required=True, type=Path)
    args = ap.parse_args(argv)
    defn = load_json_file(Definition, _definitions_root() / f"{args.definition}.json")
    evals = bench(defn, args.source.read_text(), _load_workloads(args.definition))
    print(json.dumps([json.loads(e.model_dump_json()) for e in evals], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
