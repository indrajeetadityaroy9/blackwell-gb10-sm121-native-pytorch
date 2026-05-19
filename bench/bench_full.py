"""
DGX Spark bake-off benchmark — SOL-ExecBench methodology (arXiv 2603.19173).

Test catalogue (each TESTS entry returns list[Result]):
  Tier 1 single-shape (codegen / reference): sparse, triton
  Tier 2 single-shape bandwidth-bound:        rmsnorm, softmax, cross_entropy
  Tier 2 Triton autotune across LLM shapes:   triton_autotuned
  Tier 2-revised LLM-config GEMM sweep:       gemm_fp16_llm, gemm_fp8_llm, gemm_fp4_llm
  Tier 2-revised FA-4 attention grid:         attn_fa4_fwd, attn_fa4_bwd

Methodology:
  - CUDA event timing via bench._harness (kernel-only, cold L2)
  - 5 warmup + 50 timed iters per shape (BENCH_ITERS / BENCH_WARMUP)
  - SOL Score per shape from bench/sol_config.toml
  - Subprocess isolation per ALL_TESTS entry (SOL-ExecBench requirement)

All listed features are required; missing ones fail loudly at import / call time.

CLI:
  bench_full.py                            # all tests, human-readable
  bench_full.py --only gemm_fp16_llm       # one test (which may yield many shapes)
  bench_full.py --only attn_fa4_fwd --json # JSON to stdout
  bench_full.py --json                     # all tests, JSON to stdout

Env (CLI takes precedence):
  BENCH_M=8192       GEMM dim for the single-shape sparse + triton tests
  BENCH_ITERS=50     timed iters per shape
  BENCH_WARMUP=5     warmup iters per shape
  BENCH_ONLY=...     same as --only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

# Make bench/ importable from both `python bench/bench_full.py` and subprocess.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from _harness import (
    Result, Stats, allclose_gate, cuda_event_time, emit_json,
    gemm_sol, load_config,
)
import json
import subprocess


DEVICE = "cuda"
M_DEFAULT = int(os.environ.get("BENCH_M", 8192))
ITERS = int(os.environ.get("BENCH_ITERS", 50))
WARMUP = int(os.environ.get("BENCH_WARMUP", 5))
M = N = K = M_DEFAULT
CFG = None  # lazy-loaded in main()


# ---------- helpers ----------

def _human_row(r: Result, width: int = 44) -> str:
    sol_str = f"SOL={r.sol:.1f}" if r.sol is not None else "SOL=?"
    score = f"score={r.sol_score:.2f}" if r.sol_score is not None else "score=?"
    return (f"  {r.name:{width}s} : {r.measured:7.2f} {r.unit:6s} "
            f"({sol_str}, {score}, "
            f"med={r.stats.median_ms:.2f}ms, σ={r.stats.stdev_pct:.1f}%, n={r.stats.n})")


def _gemm_result(name: str, fn: Callable, *, M_: int, N_: int, K_: int,
                 dtype: str, correctness: str | None = None,
                 out_bpe: float = 2.0) -> Result:
    """Time a GEMM and package as Result. flops = 2·M·N·K."""
    stats = cuda_event_time(fn, warmup=WARMUP, iters=ITERS)
    flops_per_call = 2.0 * M_ * N_ * K_
    median_s = stats.median_ms / 1000.0
    tflops = flops_per_call / median_s / 1e12
    sol = gemm_sol(M_, N_, K_, dtype, CFG, out_bytes_per_elem=out_bpe)
    return Result(
        name=name, unit="TFLOPs", measured=tflops,
        sol=sol.sol_tflops, sol_score=None,   # filled by orchestrator
        sol_limit=sol.limit, stats=stats,
        correctness=correctness,
        extra={"flops": flops_per_call, "compute_bound_s": sol.compute_bound_s,
               "bandwidth_bound_s": sol.bandwidth_bound_s},
    )


# ---------- tests ----------
# Single-shape GEMMs and attention moved to tests_gemm_llm.py and
# tests_attn_fa4.py. test_sparse_24 and test_triton_matmul stay single-shape:
# cuSPARSELt + fixed-tile Triton are codegen references.


def test_sparse_24() -> Result:
    name = "cusparselt_2of4_sparse_8192"
    from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
    SparseSemiStructuredTensor._FORCE_CUTLASS = False
    w = torch.randn(N, K, device=DEVICE, dtype=torch.float16)
    mask = torch.zeros_like(w, dtype=torch.bool)
    mask[:, 0::4] = True
    mask[:, 1::4] = True
    w = w * mask
    w_sparse = to_sparse_semi_structured(w)
    x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    # Correctness gate: dense linear with the same masked weight must match
    out = torch.nn.functional.linear(x, w_sparse)
    ref = torch.nn.functional.linear(x, w)
    correctness = allclose_gate(out, ref, rtol=1e-2, atol=1e-2, name="sparse")
    fn = lambda: torch.nn.functional.linear(x, w_sparse)
    # Sparse 2:4 effective ceiling = fp8 peak (2× dense per cuSPARSELt headline)
    return _gemm_result(name, fn, M_=M, N_=N, K_=K, dtype="sparse_fp8",
                        correctness=correctness)


def test_triton_matmul() -> Result:
    name = "triton_matmul_8192_fp16_fixed_tile"
    import triton
    import triton.language as tl

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c = acc.to(tl.float16)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c = torch.empty(M, N, device=DEVICE, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 256, 64, 8

    def run() -> None:
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        )

    # Correctness: triton output vs torch fp16 matmul reference
    run()
    torch.cuda.synchronize()
    ref = a @ b
    correctness = allclose_gate(c, ref, rtol=1e-2, atol=1e-2, name="triton")

    return _gemm_result(name, run, M_=M, N_=N, K_=K,
                        dtype="fp16", correctness=correctness)


# ---------- env_info (human output) ----------

def env_info() -> dict:
    import triton
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton.__version__,
    }
    p = torch.cuda.get_device_properties(0)
    info["gpu"] = f"{p.name}, SMs={p.multi_processor_count}, cc={p.major}.{p.minor}, mem={p.total_memory / 1e9:.1f} GB"
    info["arch_list"] = torch.cuda.get_arch_list()
    info["M"] = M
    info["iters"] = ITERS
    info["warmup"] = WARMUP
    return info


def print_env_info(info: dict) -> None:
    print("\n Environment ")
    for k, v in info.items():
        print(f"  {k:18s}: {v}")


# ---------- main ----------

ALL_TESTS: dict[str, Callable[[], list[Result]]] = {
    "sparse":             lambda: [test_sparse_24()],
    "triton":             lambda: [test_triton_matmul()],
}

# Tier 2: bandwidth-bound singletons — wrapped to list[Result]
from tests_kernels import TESTS as _TIER2_TESTS
ALL_TESTS.update({k: (lambda fn=fn: [fn()]) for k, fn in _TIER2_TESTS.items()})

# Tier 2: Triton autotune sweep over LLM GEMM shapes
from tests_triton_autotune import TESTS as _TIER2_TRITON_TESTS
ALL_TESTS.update(_TIER2_TRITON_TESTS)

# Tier 2-revised: LLM-config GEMMs (fp16/fp8/fp4)
from tests_gemm_llm import TESTS as _GEMM_LLM_TESTS
ALL_TESTS.update(_GEMM_LLM_TESTS)

# Tier 2-revised: FA-4 attention grid (fwd + bwd)
from tests_attn_fa4 import TESTS as _ATTN_FA4_TESTS
ALL_TESTS.update(_ATTN_FA4_TESTS)


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    ap.add_argument("--only", default=os.environ.get("BENCH_ONLY", "").strip(),
                    help="comma-separated subset of test keys (CLI overrides BENCH_ONLY env)")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON document to stdout instead of human-readable")
    ap.add_argument("--no-isolate", action="store_true",
                    help="run all selected tests in-process (default: subprocess "
                         "per test for SOL-ExecBench memory isolation)")
    return ap.parse_args(argv)


def _run_tests_isolated(test_keys: list[str], env: dict[str, str]) -> list[Result]:
    """Spawn each test in its own Python subprocess (SOL-ExecBench isolation).
    Returns Results parsed from child --json output."""
    out: list[Result] = []
    for key in test_keys:
        # Keep stderr separate: torch warnings would corrupt JSON if merged.
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--only", key, "--json", "--no-isolate"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True,
        )
        doc = json.loads(proc.stdout)
        for rd in doc["results"]:
            stats = Stats(**rd["stats"])
            out.append(Result(
                name=rd["name"], unit=rd["unit"],
                measured=rd["measured"], sol=rd["sol"],
                sol_score=rd["sol_score"], sol_limit=rd["sol_limit"],
                stats=stats, correctness=rd["correctness"],
                extra=rd["extra"],
            ))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    global CFG
    CFG = load_config()

    # Resolve which tests to run
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        unknown = [k for k in keys if k not in ALL_TESTS]
        if unknown:
            print(f"[ERROR] --only unknown keys: {unknown} (valid: {list(ALL_TESTS)})",
                  file=sys.stderr)
            return 1
        selected = [(k, ALL_TESTS[k]) for k in keys]
    else:
        selected = list(ALL_TESTS.items())

    # Run tests
    info = env_info()
    if not args.json:
        print_env_info(info)
        print(f"\n Benchmarks (iters={ITERS}, warmup={WARMUP}, cold-cache L2 flush) ")

    # Multi-test runs spawn a subprocess per test (SOL-ExecBench isolation),
    # unless --no-isolate. Single-test runs always go in-process.
    use_isolation = (len(selected) > 1) and (not args.no_isolate)

    results: list[Result] = []
    if use_isolation:
        results = _run_tests_isolated([k for k, _ in selected], dict(os.environ))
        if not args.json:
            for r in results:
                print(_human_row(r))
    else:
        for key, fn in selected:
            for r in fn():
                results.append(r)
                if not args.json:
                    print(_human_row(r))
            torch.cuda.empty_cache()

    if args.json:
        emit_json(results)
    else:
        print("\n Summary ")
        for r in results:
            print(_human_row(r))

    return 0


if __name__ == "__main__":
    sys.exit(main())
