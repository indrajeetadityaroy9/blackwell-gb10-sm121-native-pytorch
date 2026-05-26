"""AutoKernel 5-stage correctness gate (arXiv:2603.21331 §5). Each stage returns
(passed, reason); reason is the ASI diagnostic (exception or validator status) fed back to
the proposer on failure. Execution primitives (`_load_run_callable`, `_rand`) and op
taxonomy (`Definition.is_*`) are single-homed in bench.py / the schema."""

import torch

from .bench import _load_run_callable, _rand
from .data import EvaluationStatus
from .validators import validate


def _norm_inputs(definition, m, value=None):
    # (x[m,H], residual[m,H], weight[H]) at the Definition's declared dtypes. H from axes.
    h = definition.const_axes["H"]
    dts = definition.torch_input_dtypes
    def mk(shape, dt):
        if value is not None:
            return torch.full(shape, value, device="cuda:0", dtype=torch.float32).to(dt)
        return _rand(shape, dt)
    return [mk((m, h), dts[0]), mk((m, h), dts[1]), mk((h,), dts[2])]


def _check(definition, cand, ref, inputs, tag):
    try:
        with torch.no_grad():
            s, r = cand(*inputs), ref(*inputs)
        torch.cuda.synchronize(device="cuda:0")
    except Exception as e:
        return False, f"{tag}: {type(e).__name__}: {e}"[:300]
    s_list = [s] if isinstance(s, torch.Tensor) else list(s)
    r_list = [r] if isinstance(r, torch.Tensor) else list(r)
    status, _, msg = validate(definition, s_list, r_list)
    return (True, "") if status == EvaluationStatus.PASSED else (False, f"{tag}: {status.name} {msg}"[:300])


# All stages generate inputs at the Definition's DECLARED dtypes — never a fixed
# fp16/bf16/fp32 sweep. Testing dtypes outside the kernel's contract (e.g. fp32 on a
# bf16 Definition) forces SMEM-cap failures unrelated to the kernel's actual job.
def _gemm_ab(definition, m, n, k):
    dt_a, dt_b = definition.torch_input_dtypes[0], definition.torch_input_dtypes[1]
    return [_rand((m, k), dt_a), _rand((k, n), dt_b)]


def _reduce_x(definition, length, value=None):
    dt = definition.torch_input_dtypes[0]
    if value is not None:
        return [torch.full((length,), value, device="cuda:0", dtype=torch.float32).to(dt)]
    return [_rand((length,), dt)]


_GEMM_SHAPES = [
    (128, 128, 128), (512, 512, 512), (2048, 2048, 2048), (4096, 4096, 4096),
    (8192, 1024, 1024), (1024, 1024, 8192), (4096, 4096, 512), (4096, 11008, 4096),
    (512, 12288, 4096),  # llama-3.1-8b qkv_proj target
]
_REDUCTION_LENGTHS = [1024, 16384, 1_048_576, 16_777_216]


def smoke(definition, cand, ref):
    # 5 random inputs (KernelBench/AutoKernel) to catch input-dependent bugs.
    for seed in range(5):
        torch.manual_seed(seed)
        if definition.is_gemm:
            ok, why = _check(definition, cand, ref, _gemm_ab(definition, 128, 128, 128), f"smoke gemm 128^3 seed{seed}")
        elif definition.is_reduction:
            ok, why = _check(definition, cand, ref, _reduce_x(definition, 1024), f"smoke reduce 1024 seed{seed}")
        elif definition.is_norm:
            ok, why = _check(definition, cand, ref, _norm_inputs(definition, 64), f"smoke norm M=64 seed{seed}")
        else:
            return True, ""
        if not ok:
            return False, why
    return True, ""


def shape_sweep(definition, cand, ref):
    if definition.is_gemm:
        for m, n, k in _GEMM_SHAPES:
            ok, why = _check(definition, cand, ref, _gemm_ab(definition, m, n, k), f"shape_sweep gemm {m}x{n}x{k}")
            if not ok:
                return False, why
    elif definition.is_reduction:
        for length in _REDUCTION_LENGTHS:
            ok, why = _check(definition, cand, ref, _reduce_x(definition, length), f"shape_sweep reduce {length}")
            if not ok:
                return False, why
    elif definition.is_norm:
        for m in (1, 64, 512, 2048, 4096):
            ok, why = _check(definition, cand, ref, _norm_inputs(definition, m), f"shape_sweep norm M={m}")
            if not ok:
                return False, why
    return True, ""


def stability(definition, cand, ref):
    if definition.is_gemm:  # dynamic range [1e-4, 1e4]
        dt_a, dt_b = definition.torch_input_dtypes[0], definition.torch_input_dtypes[1]
        a = (torch.rand(512, 512, device="cuda:0") * 1e4 + 1e-4).to(dt_a)
        b = (torch.rand(512, 512, device="cuda:0") * 1e4 + 1e-4).to(dt_b)
        return _check(definition, cand, ref, [a, b], "stability gemm dynamic-range")
    if definition.is_reduction:  # near-zero variance
        return _check(definition, cand, ref, _reduce_x(definition, 1_048_576, value=1e3), "stability reduce near-zero-variance")
    if definition.is_norm:  # all-equal rows → tiny variance, rsqrt stress
        return _check(definition, cand, ref, _norm_inputs(definition, 512, value=1e-3), "stability norm near-zero-variance")
    return True, ""


def determinism(definition, cand, ref):
    if definition.is_gemm:
        inputs = _gemm_ab(definition, 256, 256, 256)
    elif definition.is_reduction:
        inputs = _reduce_x(definition, 65536)
    elif definition.is_norm:
        inputs = _norm_inputs(definition, 512)
    else:
        return True, ""
    try:
        with torch.no_grad():
            outs = [cand(*inputs) for _ in range(3)]
        torch.cuda.synchronize(device="cuda:0")
    except Exception as e:
        return False, f"determinism: {type(e).__name__}: {e}"[:300]
    o0 = outs[0] if isinstance(outs[0], torch.Tensor) else outs[0][0]
    for o in outs[1:]:
        if not torch.equal(o0, o if isinstance(o, torch.Tensor) else o[0]):
            return False, "determinism: outputs differ across 3 identical-input runs"
    return True, ""


def edge_cases(definition, cand, ref):
    if definition.is_gemm:
        for m, n, k in ((1023, 1023, 1023), (4097, 4097, 4097), (1537, 1537, 1537)):
            ok, why = _check(definition, cand, ref, _gemm_ab(definition, m, n, k), f"edge_cases gemm {m}x{n}x{k}")
            if not ok:
                return False, why
    elif definition.is_reduction:
        for length in (1023, 4097, 8191):
            ok, why = _check(definition, cand, ref, _reduce_x(definition, length), f"edge_cases reduce {length}")
            if not ok:
                return False, why
    elif definition.is_norm:
        for m in (1, 1023, 4097):  # non-power-of-two row counts
            ok, why = _check(definition, cand, ref, _norm_inputs(definition, m), f"edge_cases norm M={m}")
            if not ok:
                return False, why
    return True, ""


def run_5_stage(source, definition):
    """Returns (ok, reason). reason="" on full pass, else the first failing stage's ASI."""
    try:
        cand = _load_run_callable(source, "candidate")
        ref = _load_run_callable(definition.reference, "reference")
    except Exception as e:
        return False, f"compile: {type(e).__name__}: {e}"[:300]
    for stage in (smoke, shape_sweep, stability, determinism, edge_cases):
        ok, why = stage(definition, cand, ref)
        if not ok:
            return False, why
    return True, ""
