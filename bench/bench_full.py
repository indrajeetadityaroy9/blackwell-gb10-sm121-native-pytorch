"""
DGX Spark bake-off benchmark — six tests exercising Blackwell-class capabilities:

  1. FP16 GEMM 8192^3                    (cuBLAS dispatch via a @ b)
  2. FP8 GEMM 8192^3 (e4m3)              (torch._scaled_mm, use_fast_accum=True)
  3. FP4 GEMM 8192^3 (e2m1, mxfp4)       (torch._scaled_mm, best-effort)
  4. cuSPARSELt 2:4 sparse mm 8192^3     (to_sparse_semi_structured + F.linear)
  5. Triton matmul 8192^3 (FP16)         (hand-written @triton.jit kernel)
  6. flash-attention forward             (flash_attn_func or torch SDPA-flash fallback)

Each test reports SKIPPED if a feature is missing from the current torch /
triton / flash_attn build, so the same script runs against the PyPI baseline
(Run A), the from-source wheel (Run B), and NGC (Run C).

Env overrides:
  BENCH_M=8192        size for the GEMM tests (M = N = K)
  BENCH_TARGET_S=15   per-test burn-in seconds (default 15)
"""

import os
import sys
import time
import traceback

import torch

DEVICE = "cuda"
M = N = K = int(os.environ.get("BENCH_M", 8192))
TARGET_S = float(os.environ.get("BENCH_TARGET_S", 15.0))
WARMUP = 10

RESULTS: dict[str, float | None] = {}


def header(title: str) -> None:
    print(f"\n {title} ", flush=True)


def env_info() -> None:
    print(f"torch.__version__       : {torch.__version__}")
    print(f"torch.version.cuda      : {torch.version.cuda}")
    print(f"torch.backends.cudnn    : {torch.backends.cudnn.version()}")
    try:
        import triton

        print(f"triton.__version__      : {triton.__version__}")
    except ImportError:
        print(f"triton.__version__      : (not installed)")
    try:
        import flash_attn

        print(f"flash_attn.__version__  : {flash_attn.__version__}")
    except ImportError:
        print(f"flash_attn.__version__  : (not installed)")
    p = torch.cuda.get_device_properties(0)
    print(
        f"GPU                     : {p.name}, SMs={p.multi_processor_count}, "
        f"cc={p.major}.{p.minor}, mem={p.total_memory / 1e9:.1f} GB"
    )
    print(f"M = N = K               : {M}")
    print(f"per-test target seconds : {TARGET_S}")


def skip(name: str, reason: str) -> None:
    print(f"  {name:42s} : SKIPPED ({reason})")
    RESULTS[name] = None


def bench(
    name: str,
    fn,
    flops_per_call: float,
    target_s: float = TARGET_S,
    warmup: int = WARMUP,
) -> None:
    """Run fn() until target_s elapsed; print TFLOPs; store in RESULTS."""
    try:
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        iters = 0
        while True:
            _ = fn()
            iters += 1
            if time.perf_counter() - start >= target_s:
                break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        avg_ms = elapsed / iters * 1e3
        tflops = (flops_per_call * iters / elapsed) / 1e12
        print(
            f"  {name:42s} : {tflops:7.2f} TFLOPs (iters={iters}, avg={avg_ms:.2f} ms)"
        )
        RESULTS[name] = tflops
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:140]}"
        print(f"  {name:42s} : SKIPPED ({msg})")
        RESULTS[name] = None


# ------------------------------------------------------------------------------------ #
# Test 1: FP16 GEMM (baseline, same workload as bench_gemm.py)
# ------------------------------------------------------------------------------------ #
def test_fp16() -> None:
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    bench("FP16 GEMM 8192^3", lambda: a @ b, flops_per_call=2 * M * N * K)


# ------------------------------------------------------------------------------------ #
# Test 2: FP8 GEMM via torch._scaled_mm (e4m3)
# ------------------------------------------------------------------------------------ #
def test_fp8() -> None:
    name = "FP8 GEMM 8192^3 (e4m3)"
    if not hasattr(torch, "float8_e4m3fn"):
        return skip(name, "torch.float8_e4m3fn not in this build")
    if not hasattr(torch, "_scaled_mm"):
        return skip(name, "torch._scaled_mm not in this build")
    try:
        a = torch.randn(M, K, device=DEVICE).to(torch.float8_e4m3fn)
        b = torch.randn(K, N, device=DEVICE).to(torch.float8_e4m3fn)
        b = b.t().contiguous().t()  # FP8 GEMM wants col-major right operand
        scale_a = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
    except Exception as e:
        return skip(name, f"setup: {e}")
    # use_fast_accum=True selects the FP32-accumulate-with-reduced-precision-rounding
    # kernel which is the only path that reaches Blackwell FP8 Tensor Core peak.
    # Without it cuBLASLt picks a slower precise-accum kernel (~50-70% of peak).
    bench(
        name,
        lambda: torch._scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        ),
        flops_per_call=2 * M * N * K,
    )


# ------------------------------------------------------------------------------------ #
# Test 3: FP4 GEMM (mxfp4, best-effort; skipped on every torch that doesn't expose it)
# ------------------------------------------------------------------------------------ #
def test_fp4() -> None:
    name = "FP4 GEMM 8192^3 (e2m1, mxfp4)"
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        return skip(name, "torch.float4_e2m1fn_x2 not in this build")
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is None:
        return skip(name, "torch.float8_e8m0fnu (mxfp scale dtype) not available")
    try:
        # 2 fp4 values per byte; scales are per 32-element block along the K axis.
        # For (M,K) @ (K,N) mxfp4 GEMM, cuBLASLt expects:
        #   scale_a: (M, K/32)   -- per (row of a, K-block)
        #   scale_b: (K/32, N)   -- per (K-block, col of b)
        # The previous version transposed scale_b which silently triggered SKIP.
        a = torch.randint(0, 256, (M, K // 2), device=DEVICE, dtype=torch.uint8).view(
            fp4_dtype
        )
        b = torch.randint(0, 256, (K // 2, N), device=DEVICE, dtype=torch.uint8).view(
            fp4_dtype
        )
        b = b.t().contiguous().t()  # col-major right operand (same idiom as FP8 path)
        scale_a = torch.ones(M, K // 32, device=DEVICE, dtype=e8m0)
        scale_b = torch.ones(K // 32, N, device=DEVICE, dtype=e8m0)
    except Exception as e:
        return skip(name, f"setup: {e}")
    bench(
        name,
        lambda: torch._scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        ),
        flops_per_call=2 * M * N * K,
    )


# ------------------------------------------------------------------------------------ #
# Test 4: cuSPARSELt 2:4 sparse matmul (semi-structured)
# ------------------------------------------------------------------------------------ #
def test_sparse_24() -> None:
    name = "cuSPARSELt 2:4 sparse mm 8192^3"
    try:
        from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
    except ImportError as e:
        return skip(name, f"torch.sparse: {e}")
    # Prefer cuSPARSELt backend (vs CUTLASS) so we test the lib the README names
    try:
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
    except AttributeError:
        pass
    try:
        w = torch.randn(N, K, device=DEVICE, dtype=torch.float16)
        # 2-of-4 mask: keep first two of every 4 along the K axis
        mask = torch.zeros_like(w, dtype=torch.bool)
        mask[:, 0::4] = True
        mask[:, 1::4] = True
        w = w * mask
        w_sparse = to_sparse_semi_structured(w)
        x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    except Exception as e:
        return skip(name, f"setup: {e}")
    bench(
        name,
        lambda: torch.nn.functional.linear(x, w_sparse),
        flops_per_call=2 * M * N * K,
    )


# ------------------------------------------------------------------------------------ #
# Test 5: Triton matmul (hand-written kernel, FP16 in, FP16 out)
# ------------------------------------------------------------------------------------ #
def test_triton_matmul() -> None:
    name = "Triton matmul 8192^3 (FP16)"
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        return skip(name, f"triton: {e}")

    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
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
        return skip(name, f"setup: {e}")

    def run() -> None:
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        matmul_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
        )

    bench(name, run, flops_per_call=2 * M * N * K)


# ------------------------------------------------------------------------------------ #
# Test 6: flash-attention forward (prefer flash_attn pkg; fall back to torch SDPA-flash)
# ------------------------------------------------------------------------------------ #
def test_flash_attn() -> None:
    B, H, T, D = 4, 32, 4096, 128
    causal_flops = 2 * B * H * T * T * D  # 2 mm's, causal halves each
    # Prefer flash_attn (Dao-AI / Tri Dao version)
    try:
        from flash_attn import flash_attn_func  # type: ignore

        q = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.float16)
        bench(
            f"flash_attn fwd (B={B},H={H},T={T},D={D},causal)",
            lambda: flash_attn_func(q, k, v, causal=True),
            flops_per_call=causal_flops,
        )
        return
    except ImportError:
        pass
    except Exception as e:
        skip(f"flash_attn fwd (B={B},H={H},T={T},D={D},causal)", f"setup: {e}")

    # Fallback: torch SDPA, forced flash backend
    name = f"torch SDPA-flash fwd (B={B},H={H},T={T},D={D},causal)"
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)

        def run() -> torch.Tensor:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )
    except Exception as e:
        return skip(name, f"setup: {e}")
    bench(name, run, flops_per_call=causal_flops)


# ------------------------------------------------------------------------------------ #
def main() -> int:
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available", file=sys.stderr)
        return 1

    header("Environment")
    env_info()

    header("Benchmarks (each ~%.0fs burn-in)" % TARGET_S)
    for fn in (
        test_fp16,
        test_fp8,
        test_fp4,
        test_sparse_24,
        test_triton_matmul,
        test_flash_attn,
    ):
        try:
            fn()
        except Exception:
            # Defensive: never let a single test crash the whole suite
            print(f"  [test crashed]\n{traceback.format_exc()}")
        torch.cuda.empty_cache()

    header("Summary")
    if not RESULTS:
        print("  (no test results recorded — every test crashed before bench()/skip())")
        return 1
    width = max((len(k) for k in RESULTS), default=40)
    for name, val in RESULTS.items():
        if val is None:
            print(f"  {name:{width}s} : SKIPPED")
        else:
            print(f"  {name:{width}s} : {val:7.2f} TFLOPs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
