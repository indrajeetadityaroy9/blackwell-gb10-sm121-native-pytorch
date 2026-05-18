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

---

# 2026 SOTA upgrade — methodology refresh

The original 6-test bench above is preserved as Tier 1's input. Layered onto
that input is SOL-ExecBench methodology (arXiv 2603.19173) plus three new tiers
of capability. All execution is containerized; the host runs only `docker`
commands. See `docs/containerization.md` for topology and `docs/sol-score.md`
for the SOL Score formula.

## Tier 1 — Methodology rigor (replaces the old wall-clock harness)

Every test in `bench/bench_full.py` now runs through `bench/_harness.py`:

- **Clock lock via privileged controller container** (`bench/_clocklock.sh`,
  Phase 0 gate `bench/verify_clocklock.sh`). `nvidia-smi --lock-gpu-clocks` at
  2418 MHz from inside a `--privileged` container; bench containers stay
  unprivileged. Trap-based unlock on EXIT/INT/TERM/ERR.
  - **Resolves limitation 2** (no GPU clock locking). Verified ±0.3% on GB10.
- **L2 cache flush before every iteration** (`L2Flusher`, 48 MB buffer = 2×
  GB10's 24 MB L2). Cold-cache per iter, matching SOL-ExecBench convention.
  - **Resolves limitation 3** (no L2 cache clearing).
- **CUDA event timing** (`cuda_event_time`), not wall-clock. Kernel-only
  measurement; eliminates Python overhead bias. Default 5 warmup + 50 timed
  iters (configurable via `BENCH_ITERS`, `BENCH_WARMUP`).
  - **Resolves limitation 1** (wall-clock vs CUDA event timing).
- **Statistical reporting** — mean / median / p10 / p90 / stdev / stdev_pct
  per test via stdlib `statistics`. Reported in `Result.stats`.
  - **Resolves limitation 4** (no median + variance reporting).
- **Subprocess isolation** — each test runs in its own Python process when
  invoked through the multi-test orchestrator (SOL-ExecBench requirement;
  ~5s per-test Python startup overhead accepted in exchange for reproducibility).
- **Per-test correctness gates** at small M=1024 against fp32 reference:
  FP16/BF16/Triton/Sparse (`rtol=1e-2 atol=1e-2`); FP8 (`rtol=5e-2 atol=0.5`);
  FP4 gate skipped (16 representable values × random uint8 → error magnitude
  O(√K·0.5) ≈ 45 at K=8192; meaningful tolerance unusable).
  - **Resolves limitation 6** (no verification of numerical correctness).
- **SOL Score per test** — analytical Speed-of-Light from `bench/_solar.py`
  using GB10 specs in `bench/sol_config.toml`. See `docs/sol-score.md`.
- **JSON output mode** (`bench_full.py --json`) for machine-readable
  aggregation by `bench/_summarize.py`. Human-readable output preserved as
  default for direct CLI use.

Tier 1 output: `bench/logs/run{A,B,C}.json` + `bench/logs/SUMMARY.txt` with a
3-way SOL Score table using Run A (PyPI) as baseline.

## Tier 2 — New kernel coverage

Four tests added in `bench/tests_kernels.py` and one in
`bench/tests_triton_autotune.py`, integrated into `bench_full.py`'s `ALL_TESTS`:

| Test key | Workload | Unit | SOL basis |
|---|---|---|---|
| `attn_bwd` | SDPA-flash causal backward (B=4 H=32 S=4096 D=128) | TFLOPs | 5·B·H·S²·D causal |
| `rmsnorm` | `F.rms_norm` on `[8192, 8192]` fp16 | GB/s | bandwidth-bound |
| `softmax` | `F.softmax` on `[2, 16, 4096, 4096]` fp16 | GB/s | bandwidth-bound |
| `cross_entropy` | `F.cross_entropy` on `[8192, 128k]` fp32 | GB/s | bandwidth-bound |
| `triton_autotuned` | TritonForge sweep (162 configs: BLOCK_M/N/K × num_stages × num_warps) | TFLOPs | FP16 peak |

The existing fixed-tile `triton` test is retained alongside `triton_autotuned`
— they measure different things (compiler codegen quality vs best-Triton).
Both SKIP on Run A (PyPI triton 3.5.0 PTXAS bug on sm_121a) and Run B
(no triton installed); only Run C produces numbers.

## Tier 3 — External tool integration

- **nvbench cross-validator** (`bench/nvbench_shim/`): C++ binary linking
  NVIDIA's nvbench library, runs the same FP16 GEMM via cuBLASLt with
  nvbench's built-in CUDA-event timing + L2 flush + statistical sampling.
  If our Python harness diverges from nvbench's number by >3%, it flags a
  bug in `_harness.py`. **Verified working on sm_121** (Risk Register concern
  closed): nvbench builds cleanly with `CMAKE_CUDA_ARCHITECTURES=121`; FP16
  GEMM 8192³ at 84.66 TFLOPs vs our Python harness's 86.19 TFLOPs (1.8% gap,
  within ±3% threshold).
- **Roofline / Nsight Compute** (`bench/roofline.py`): opt-in via
  `BENCH_PROFILE=1`. Uses `ncu --set roofline` + `ncu_report==2025.3.1`
  Python API to extract `sol_sm` and `sol_mem` percentages per kernel.
  Verified on GB10 with Nsight Compute 2026.1.0: FP16 GEMM at sol_sm=75%,
  sol_mem=52% — useful for calibrating `sol_config.toml` placeholders.

## Tier 4 (experimental) — Full-stack end-to-end LLM

**Out of scope for hardware benchmarking** — moved to `bench/experimental/`
and not exercised by `run_bakeoff.sh`. These workloads measure software-stack
throughput (vLLM scheduler, KV cache layout, tokenizer batching, attention
engine choice) rather than GB10 hardware capability. Tiers 1-3 already cover
the hardware-level questions this repo cares about.

Kept as reference scaffolding for two adjacent research questions a future
follow-up might pursue:

- `bench/experimental/mlperf_llama31_8b.py` — MLPerf inference v5.1
  `llama3_1-8b` wrapper using MLCommons' `mlcr` CLI inside
  `ghcr.io/mlcommons/inference:5.1-dev`. Reports tokens/sec, TTFT, ITL p50/p99.
  Gated by `BENCH_DOWNLOAD_MODELS=1` (16 GB HuggingFace download under Meta
  Llama 3.1 Community License; `HF_TOKEN` required). Useful if you want
  "first publicly-reported GB10 MLPerf v5.1 numbers" as a separate artifact.
  **Note**: llama3.1-8b is v5.1-only (NOT in v5.0).
- `bench/experimental/serve_flashinfer.py` — FlashInfer-Bench attention serving
  via the authentic v0.1.2 API (`Benchmark + TraceSet + BenchmarkConfig`).
  Requires FlashInfer source-built with `TORCH_CUDA_ARCH_LIST=12.1` (PyPI
  wheels are sm_120-only); build via
  `bench/experimental/build_flashinfer.sh` (~30-60 min cold).
- `bench/experimental/run_e2e.sh` — driver script if you ever want to run both;
  inherits the same clock-lock controller pattern as `run_bakeoff.sh`.

## What this upgrade does NOT do

The original "What this bench is not" caveats from the pre-upgrade methodology
still apply for what we deliberately left out:

- **Multi-GPU / NVLink-C2C** — DGX Spark is single-GPU only.
- **Training throughput** — we benchmark inference + microkernels, not training.
- **Power / perf-per-watt** — `nvidia-smi --query-gpu=power.draw` accuracy is
  not calibrated on GB10.
- **Public leaderboard submission** — local numbers only; no official MLPerf
  submission or FlashInfer-Bench upload.
- **cuBLASLt heuristic auto-tuning** — we use cuBLAS defaults.

But the original limitations 1, 2, 3, 4, and 6 (clock lock, L2 flush, CUDA
events, variance, correctness) are **all resolved** by Tier 1. Limitation 5
(Triton untuned) is **resolved** by the Tier 2 `triton_autotuned` test
(retained alongside `triton` for codegen comparison).
