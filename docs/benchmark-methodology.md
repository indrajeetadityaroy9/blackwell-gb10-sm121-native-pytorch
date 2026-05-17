# Benchmark Methodology — `bench/bench_full.py`

A 6-test PyTorch benchmark suite for DGX Spark (Blackwell GB10, sm_121). Each test exercises one capability claimed by Blackwell's spec sheet and reports numbers comparable to public benchmarks at the same shape.

## Tests

| # | Test | Workload | What it measures |
|---|---|---|---|
| 1 | FP16 GEMM 8192³ | `a @ b`, fp16 inputs | cuBLAS dispatch quality for Blackwell's primary matmul path |
| 2 | FP8 GEMM 8192³ (e4m3) | `torch._scaled_mm` with `use_fast_accum=True` | Blackwell 5th-gen Tensor Cores' FP8 throughput (theoretical 2× FP16) |
| 3 | FP4 GEMM 8192³ (mxfp4) | `torch._scaled_mm` with `float4_e2m1fn_x2` + per-block `float8_e8m0fnu` scales | Blackwell FP4 Tensor Cores (best-effort; API not stable across torch versions) |
| 4 | cuSPARSELt 2:4 sparse mm | `to_sparse_semi_structured` + `F.linear` | Hardware-accelerated 2:4 structured sparsity (Blackwell theoretical 2× dense) |
| 5 | Triton matmul 8192³ FP16 | Hand-written tiled `@triton.jit` kernel | Triton compiler's sm_121 PTXAS codegen quality |
| 6 | flash-attention forward | `flash_attn_func` (preferred) or `torch SDPA-flash` (fallback) | Attention kernel quality; B=4 H=32 T=4096 D=128 causal |

## Common harness

Each test uses the same `bench()` function:

```python
for _ in range(WARMUP):       # 10 iters, discarded
    fn()
torch.cuda.synchronize()
start = time.perf_counter()
iters = 0
while time.perf_counter() - start < TARGET_S:   # 15 seconds default
    fn()
    iters += 1
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
tflops = (flops_per_call * iters / elapsed) / 1e12
```

Key properties:

- **Wall-clock timing.** Includes Python overhead per iteration but at 8192³ matmul (~30 ms/iter for FP16), kernel time dominates by ~99%.
- **No `synchronize()` between iterations.** CUDA queue fills up; PyTorch matmuls are async; the synchronize at the end yields the true sustained GPU throughput.
- **Each test sets up fresh tensors**; no state leakage across tests.
- **Graceful skip on any failure** (missing dtype, missing package, runtime error). The remaining tests continue.

## FLOPs counting conventions

| Test | Formula | Note |
|---|---|---|
| FP16/FP8/FP4 GEMM | `2 * M * N * K` | Standard GEMM convention; 2 ops per multiply-accumulate |
| cuSPARSELt 2:4 | `2 * M * N * K` | Reports *effective* TFLOPs vs dense baseline. GPU actually performs ~M*N*K MACs (half) but reporting against dense convention is standard. |
| flash-attention causal | `2 * B * H * T * T * D` | Full attention is `4 * B * H * T * T * D` (Q@K^T + attn@V), causal halves both → 2 |

## Methodological caveats and known limitations

1. **Wall-clock vs CUDA event timing.** Our `bench()` uses `time.perf_counter()` around a synchronize-bracketed burn-in. This includes Python overhead per iteration, but at 8192³ matmul (~30 ms/iter for FP16) kernel time dominates by ~99%. CUDA-event timing (measuring only kernel time, excluding Python and launch overhead) typically reports lower TFLOPs than wall-clock, especially for short-iter benches; the *relative* ordering between A/B/C is robust within this methodology.

2. **No GPU clock locking.** SOL-ExecBench (arXiv 2603.19173) recommends `nvidia-smi -lgc <freq>,<freq>` before benchmarking to eliminate ±10% throughput swing from thermal/power management. Our bench does not do this. The 60 s burn-in averages over most thermal transients.

3. **No L2 cache clearing between iterations.** SOL-ExecBench recommends flushing L2 between iters to measure cold-cache performance. Our sustained burn-in produces warm-cache numbers, which are typical for production workloads but overestimate vs. SOL-style measurement.

4. **No median + variance reporting.** Each test produces a single average TFLOPs over ~500 iterations. With ~500-sample averaging, variance is low (<2%) but not explicitly reported.

5. **Triton kernel uses fixed tile sizes** (128×256×64 with GROUP_M=8), not autotuned. TritonForge (arXiv 2512.09196) shows 1.76× average speedup from profiling-guided autotune over fixed configs. Our Triton number is "untuned Triton baseline", which is what we want for comparing compiler codegen, but not "best Triton can do."

6. **No verification of numerical correctness.** Tests measure throughput, not accuracy. We trust torch's own correctness.

## What this bench is not

- It is not a comprehensive PyTorch benchmark (no convolutions, no LayerNorm, no end-to-end model timing).
- It is not a Blackwell-peak benchmark (we don't hit Speed-of-Light; that requires TMEM + 2-CTA MMA + CuTe-DSL per FlashAttention-4).
- It does not isolate hardware effects — driver state, container overhead, and kernel selection are not controlled.

The bench is **right-sized for comparing PyTorch deployment paths on identical hardware** — which is the specific question this research asks.
