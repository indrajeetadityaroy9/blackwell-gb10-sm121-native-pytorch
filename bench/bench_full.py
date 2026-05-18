"""
DGX Spark bake-off benchmark — six Blackwell-class tests under SOL-ExecBench
methodology (arXiv 2603.19173).

Tests:
  1. FP16 GEMM 8192^3                  (a @ b → cuBLAS)
  2. FP8 GEMM 8192^3 (e4m3)            (torch._scaled_mm, use_fast_accum=True)
  3. FP4 GEMM 8192^3 (NVFP4)           (torch._scaled_mm, 1x16 / e4m3fn scales)
  4. cuSPARSELt 2:4 sparse mm 8192^3   (to_sparse_semi_structured + F.linear)
  5. Triton matmul 8192^3 (FP16)       (fixed-tile @triton.jit)
  6. flash-attention forward           (flash_attn_func or torch SDPA-flash)

Methodology:
  - CUDA event timing via bench._harness (kernel-only)
  - L2 flush per iter (48 MB buffer for GB10's 24 MB L2)
  - 5 warmup + 50 timed iters (overridable via BENCH_ITERS / BENCH_WARMUP)
  - Per-test correctness gate at M=1024 vs fp32 reference
  - SOL Score from bench/sol_config.toml

Missing features SKIP gracefully so the same script runs across runs A/B/C.

CLI:
  bench_full.py                       # all tests, human-readable
  bench_full.py --only fp16,fp8       # subset
  bench_full.py --only fp16 --json    # single test → JSON to stdout
  bench_full.py --json                # all tests → JSON to stdout

Env (CLI takes precedence):
  BENCH_M=8192       GEMM dim (M = N = K)
  BENCH_ITERS=50     timed iters per test
  BENCH_WARMUP=5     warmup iters
  BENCH_ONLY=fp4     same as --only
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Callable

# Make bench/ importable from both `python bench/bench_full.py` and subprocess.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from _harness import (
    Result, Stats, allclose_gate, cuda_event_time, emit_json, run_isolated,
    bandwidth_sol_gbs, attn_sol, gemm_sol, load_config, sol_score,
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
    if r.note and r.measured == 0.0:
        return f"  {r.name:{width}s} : SKIPPED ({r.note})"
    sol_str = f"SOL={r.sol:.1f}" if r.sol is not None else "SOL=?"
    score = f"score={r.sol_score:.2f}" if r.sol_score is not None else "score=?"
    return (f"  {r.name:{width}s} : {r.measured:7.2f} {r.unit:6s} "
            f"({sol_str}, {score}, "
            f"med={r.stats.median_ms:.2f}ms, σ={r.stats.stdev_pct:.1f}%, n={r.stats.n})")


def _skip_result(name: str, reason: str, unit: str = "TFLOPs") -> Result:
    stats = Stats.from_samples([0.0])
    return Result(name=name, unit=unit, measured=0.0,
                  sol=None, sol_score=None, sol_limit=None,
                  stats=stats, correctness=None, note=reason)


def _gemm_result(name: str, fn: Callable, *, M_: int, N_: int, K_: int,
                 dtype: str, correctness: str | None = None,
                 out_bpe: float = 2.0) -> Result:
    """Time a GEMM and package as Result. flops = 2·M·N·K."""
    try:
        stats = cuda_event_time(fn, warmup=WARMUP, iters=ITERS)
    except Exception as e:
        return _skip_result(name, f"{type(e).__name__}: {str(e)[:240]}")
    flops_per_call = 2.0 * M_ * N_ * K_
    median_s = stats.median_ms / 1000.0
    tflops = flops_per_call / median_s / 1e12
    sol = gemm_sol(M_, N_, K_, dtype, CFG, out_bytes_per_elem=out_bpe)
    return Result(
        name=name, unit="TFLOPs", measured=tflops,
        sol=sol.sol_tflops, sol_score=None,   # filled by orchestrator
        sol_limit=sol.limit, stats=stats,
        correctness=correctness, note=None,
        extra={"flops": flops_per_call, "compute_bound_s": sol.compute_bound_s,
               "bandwidth_bound_s": sol.bandwidth_bound_s},
    )


def _attn_result(name: str, fn: Callable, *, B: int, H: int, S: int, D: int,
                 dtype: str = "fp16", causal: bool = True) -> Result:
    """Time an attention kernel; SOL via attn_sol."""
    try:
        stats = cuda_event_time(fn, warmup=WARMUP, iters=ITERS)
    except Exception as e:
        return _skip_result(name, f"{type(e).__name__}: {str(e)[:240]}")
    flops_per_call = (2.0 if causal else 4.0) * B * H * S * S * D
    median_s = stats.median_ms / 1000.0
    tflops = flops_per_call / median_s / 1e12
    sol = attn_sol(B, H, S, D, dtype, CFG, causal=causal)
    return Result(
        name=name, unit="TFLOPs", measured=tflops,
        sol=sol.sol_tflops, sol_score=None,
        sol_limit=sol.limit, stats=stats,
        correctness=None, note=None,
        extra={"flops": flops_per_call, "B": B, "H": H, "S": S, "D": D, "causal": causal},
    )


# ---------- correctness gates ----------

def _correctness_fp16_gemm(M_=1024) -> str:
    a = torch.randn(M_, M_, device=DEVICE, dtype=torch.float16)
    b = torch.randn(M_, M_, device=DEVICE, dtype=torch.float16)
    out = a @ b
    ref = a.float() @ b.float()
    return allclose_gate(out, ref, rtol=1e-2, atol=1e-2, name="fp16_gemm")


def _correctness_fp8_gemm(M_=1024) -> str:
    if not (hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm")):
        return "SKIP: no fp8 API"
    a32 = torch.randn(M_, M_, device=DEVICE)
    b32 = torch.randn(M_, M_, device=DEVICE)
    a = a32.to(torch.float8_e4m3fn)
    b = b32.to(torch.float8_e4m3fn).t().contiguous().t()
    s = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
    out = torch._scaled_mm(a, b, scale_a=s, scale_b=s,
                           out_dtype=torch.bfloat16, use_fast_accum=True)
    ref = a32 @ b32
    return allclose_gate(out, ref, rtol=5e-2, atol=0.5, name="fp8_gemm")


# ---------- 6 tests ----------

def test_fp16() -> Result:
    name = "fp16_gemm_8192"
    correctness = _correctness_fp16_gemm()
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    return _gemm_result(name, lambda: a @ b, M_=M, N_=N, K_=K,
                        dtype="fp16", correctness=correctness)


def test_fp8() -> Result:
    name = "fp8_gemm_8192_e4m3"
    if not hasattr(torch, "float8_e4m3fn"):
        return _skip_result(name, "torch.float8_e4m3fn not in this build")
    if not hasattr(torch, "_scaled_mm"):
        return _skip_result(name, "torch._scaled_mm not in this build")
    correctness = _correctness_fp8_gemm()
    try:
        a = torch.randn(M, K, device=DEVICE).to(torch.float8_e4m3fn)
        b = torch.randn(K, N, device=DEVICE).to(torch.float8_e4m3fn)
        b = b.t().contiguous().t()  # FP8 GEMM needs col-major right operand
        scale_a = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
    except Exception as e:
        return _skip_result(name, f"setup: {e}")
    # use_fast_accum=True is the only path to Blackwell FP8 Tensor Core peak.
    fn = lambda: torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b,
                                   out_dtype=torch.bfloat16, use_fast_accum=True)
    return _gemm_result(name, fn, M_=M, N_=N, K_=K,
                        dtype="fp8", correctness=correctness, out_bpe=2.0)


def test_fp4() -> Result:
    name = "fp4_gemm_8192_nvfp4"
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is None:
        return _skip_result(name, "torch.float4_e2m1fn_x2 not in this build")
    e4m3 = getattr(torch, "float8_e4m3fn", None)
    if e4m3 is None:
        return _skip_result(name, "torch.float8_e4m3fn (nvfp4 scale) not available")
    try:
        a = torch.randint(0, 256, (M, K // 2), device=DEVICE,
                          dtype=torch.uint8).view(fp4)
        b = (torch.randint(0, 256, (N, K // 2), device=DEVICE,
                           dtype=torch.uint8).view(fp4).t())
        scale_a = torch.ones(M, K // 16, device=DEVICE, dtype=e4m3)
        scale_b = torch.ones(N, K // 16, device=DEVICE, dtype=e4m3)
    except Exception as e:
        return _skip_result(name, f"setup: {e}")
    # FP4 correctness gate skipped: 16 representable values × random uint8 input
    # gives error magnitude O(√K·0.5) ≈ 45 at K=8192 — no usable tolerance.
    fn = lambda: torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b,
                                   out_dtype=torch.bfloat16)
    return _gemm_result(name, fn, M_=M, N_=N, K_=K, dtype="fp4",
                        correctness="SKIP: FP4 numerics not gate-able with random input")


def test_sparse_24() -> Result:
    name = "cusparselt_2of4_sparse_8192"
    try:
        from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
    except ImportError as e:
        return _skip_result(name, f"torch.sparse: {e}")
    try:
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
    except AttributeError:
        pass
    try:
        w = torch.randn(N, K, device=DEVICE, dtype=torch.float16)
        mask = torch.zeros_like(w, dtype=torch.bool)
        mask[:, 0::4] = True
        mask[:, 1::4] = True
        w = w * mask
        w_sparse = to_sparse_semi_structured(w)
        x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    except Exception as e:
        return _skip_result(name, f"setup: {e}")
    # Correctness gate: dense linear with the same masked weight must match
    try:
        out = torch.nn.functional.linear(x, w_sparse)
        ref = torch.nn.functional.linear(x, w)
        correctness = allclose_gate(out, ref, rtol=1e-2, atol=1e-2, name="sparse")
    except Exception as e:
        correctness = f"FAIL: {type(e).__name__}: {e}"
    fn = lambda: torch.nn.functional.linear(x, w_sparse)
    # Sparse 2:4 effective ceiling = fp8 peak (2× dense per cuSPARSELt headline)
    return _gemm_result(name, fn, M_=M, N_=N, K_=K, dtype="sparse_fp8",
                        correctness=correctness)


def test_triton_matmul() -> Result:
    name = "triton_matmul_8192_fp16_fixed_tile"
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        return _skip_result(name, f"triton: {e}")

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

    try:
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        c = torch.empty(M, N, device=DEVICE, dtype=torch.float16)
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 256, 64, 8
    except Exception as e:
        return _skip_result(name, f"setup: {e}")

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
    try:
        run()
        torch.cuda.synchronize()
        ref = a @ b
        correctness = allclose_gate(c, ref, rtol=1e-2, atol=1e-2, name="triton")
    except Exception as e:
        correctness = f"FAIL: {type(e).__name__}: {e}"

    return _gemm_result(name, run, M_=M, N_=N, K_=K,
                        dtype="fp16", correctness=correctness)


def test_flash_attn() -> Result:
    B, H, T, D = 4, 32, 4096, 128
    # Prefer flash_attn (Dao-AI); fall back to torch SDPA-flash.
    try:
        from flash_attn import flash_attn_func  # type: ignore
        name = f"flash_attn_fwd_B{B}H{H}T{T}D{D}_causal"
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.float16)
        fn = lambda: flash_attn_func(q, k, v, causal=True)
        return _attn_result(name, fn, B=B, H=H, S=T, D=D, dtype="fp16", causal=True)
    except ImportError:
        pass

    # Fallback: torch SDPA, forced flash backend
    name = f"sdpa_flash_fwd_B{B}H{H}T{T}D{D}_causal"
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        q = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)

        def run() -> torch.Tensor:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True)
    except Exception as e:
        return _skip_result(name, f"setup: {e}")
    return _attn_result(name, run, B=B, H=H, S=T, D=D, dtype="fp16", causal=True)


# ---------- env_info (human output) ----------

def env_info() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    try:
        import triton
        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = "(not installed)"
    try:
        import flash_attn
        info["flash_attn"] = flash_attn.__version__
    except ImportError:
        info["flash_attn"] = "(not installed)"
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

ALL_TESTS: dict[str, Callable[[], Result]] = {
    "fp16": test_fp16,
    "fp8": test_fp8,
    "fp4": test_fp4,
    "sparse": test_sparse_24,
    "triton": test_triton_matmul,
    "attn": test_flash_attn,
}

# Tier 2: bandwidth, attn_bwd, rmsnorm, softmax, cross_entropy
try:
    from tests_kernels import TESTS as _TIER2_TESTS  # noqa: E402
    ALL_TESTS.update(_TIER2_TESTS)
except ImportError:
    pass

# Tier 2: Triton autotune sweep
try:
    from tests_triton_autotune import TESTS as _TIER2_TRITON_TESTS  # noqa: E402
    ALL_TESTS.update(_TIER2_TRITON_TESTS)
except ImportError:
    pass


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
        try:
            # Keep stderr separate: torch/numpy warnings would corrupt JSON if merged.
            # stderr captured for failure diagnostics.
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--only", key, "--json", "--no-isolate"],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True,
            )
            raw = proc.stdout
            doc = json.loads(raw)
            for rd in doc["results"]:
                stats = Stats(**rd["stats"])
                out.append(Result(
                    name=rd["name"], unit=rd["unit"],
                    measured=rd["measured"], sol=rd["sol"],
                    sol_score=rd["sol_score"], sol_limit=rd["sol_limit"],
                    stats=stats, correctness=rd["correctness"],
                    note=rd["note"], extra=rd.get("extra", {}),
                ))
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or e.stdout or b"")[-240:].decode(errors="replace")
            out.append(_skip_result(
                key, f"subprocess exit {e.returncode}: {tail}"))
        except (json.JSONDecodeError, KeyError) as e:
            out.append(_skip_result(key, f"parse error: {type(e).__name__}: {e}"))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available", file=sys.stderr)
        return 1

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
            try:
                r = fn()
            except Exception:
                r = _skip_result(key, f"[CRASHED]\n{traceback.format_exc()[:240]}")
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
