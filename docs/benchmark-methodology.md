# Benchmark Methodology

The bake-off pipeline driven by `bench/run_bakeoff.sh` exercises three measurement tiers inside each wheel's container. The host runs only `docker` commands. See [containerization.md](containerization.md) for the container topology.

## Entry points

| File | Role |
|---|---|
| `bench/run_bakeoff.sh` | Host-side driver. Spawns the clock-lock sidecar, runs A/B/C in sequence, calls `_summarize.py`. |
| `bench/run_tiers.py` | In-container orchestrator. Runs the three tiers in deterministic order; emits one JSON document to stdout. |
| `bench/_harness.py` | Shared measurement primitives: `Stats`, `cuda_event_time`, `Result`, `emit_json`. |
| `bench/kernel_bench.py` | Tier 2 kernel-level GEMM bench (per-dtype subprocess). |
| `bench/roofline.py` | Tier 3 NCU profiler wrapper. |
| `bench/normalize.py` | Schema adapters: `from_optimum`, `from_kernel`, `from_ncu` → `list[Result]`. |
| `bench/_summarize.py` | Aggregates `runA/B/C.json` → `SUMMARY.txt`. |

## Methodology rigor (applied to every tier)

- **Clock lock via privileged sidecar** (`bench/_clocklock.sh`, gated by `bench/verify_clocklock.sh`). `nvidia-smi --lock-gpu-clocks=2418,2418` is issued from inside a single `--privileged` controller container (`dgx-bench-clocklock`); bench containers themselves stay unprivileged (only `--cap-add=SYS_ADMIN` for NCU's HW counters). Trap-based unlock on EXIT/INT/TERM.
- **L2 cache flush before every iteration** — `_harness.cuda_event_time` zeroes a 48 MB int32 buffer (2× GB10's 24 MB L2) between samples for cold-cache measurement.
- **CUDA event timing** — kernel-only via `torch.cuda.Event`, eliminating Python-overhead bias. Default 5 warmup + 50 timed iters.
- **Statistical reporting** — `Stats.from_samples` produces mean / median / p10 / p90 / stdev / stdev_pct per measurement via stdlib `statistics`.
- **Per-wheel JSON output** — `_harness.emit_json` stamps each report with `torch_version`, `cuda_version`, `device_name`, `arch_list`; consumed by `_summarize.py`.

## Tier 1 — optimum-benchmark wall-clock (bandwidth-bound regime)

`run_tiers.run_optimum_tier` iterates each YAML in `bench/configs/optimum/${BENCH_OPTIMUM_SCENARIO}/` (default `recommended/`) and invokes `optimum-benchmark --config-dir <dir> --config-name <stem>`. Underscore-prefixed names (Hydra inheritance bases like `_base_.yaml`) are skipped; symlink targets shared between `recommended/` and `wide/` are deduped against their realpath so the same config is not run twice.

The configs pin the GB10-specific runtime overrides:

- `scenario.warmup_runs=5`, `scenario.iterations=10`, `duration=0` (count-based, not wall-clock).
- `backend.device=cuda`, `device_ids=0` (DGX Spark is single-GPU).
- `launcher.device_isolation=true`, `device_isolation_action=error` — missing CUDA aborts the run.
- `hydra.run.dir=/logs/optimum/${name}` for stable globbing by `from_optimum`.
- `OVERRIDE_BENCHMARKS=1` — forces optimum-benchmark to re-run when the run dir already exists from a previous wheel in the same `/logs` volume.

The default `recommended/llama3_8b_bf16.yaml` runs Llama-3.1-8B BF16 at batch=1, seq_len=512, 128 new tokens (prefill + decode + per-token latency).

After each run, `from_optimum(run_dir)` reads `benchmark_report.json` and produces one `Result(unit="ms")` per operation; throughput (for ops that report it — generate/forward/decode) is folded into `extra`.

## Tier 2 — kernel-level GEMM (compute-bound regime)

`run_tiers.run_kernel_tier` invokes `bench/kernel_bench.py` once per dtype (`bf16`, `fp16`, `fp8`, `fp4`) as an isolated subprocess. Per-dtype isolation is necessary because cuBLAS Lt's heuristic table on sm_121 is incomplete for FP8/FP4 and raises `CUBLAS_STATUS_NOT_INITIALIZED` mid-process — without subprocess isolation, an FP8 failure would lose the BF16/FP16 results in the same Python process. Per-dtype failures are recorded as zero-length result lists; the wheel still produces JSON for the dtypes that succeeded.

Shapes are the four Llama-3.1-8B projections at M=512 (small prefill):

| Label | (M, N, K) |
|---|---|
| `qkv_proj` | (512, 12288, 4096) |
| `attn_out` | (512, 4096, 4096) |
| `mlp_up` | (512, 14336, 4096) |
| `mlp_down` | (512, 4096, 14336) |

At M=512 these are above GB10's ~330 FLOPs/byte arithmetic-intensity crossover and therefore compute-bound — the regime where native sm_121 cubins are expected to differentiate from the PTX-JIT path. Decode-shape (M=1) matmuls are bandwidth-bound and all three wheels saturate the same 273 GB/s LPDDR5X ceiling, so they are not separately measured here.

Per-dtype kernels:

- `bf16` / `fp16` — `torch.matmul (a @ b)`.
- `fp8` — `torch._scaled_mm` with `float8_e4m3fn`, scalar scales, `use_fast_accum=True`, BF16 output.
- `fp4` — `torch._scaled_mm` with `float4_e2m1fn_x2` and `float8_e4m3fn` block scales at 1×16 (mxfp4), BF16 output. The fp4 storage is built via random `uint8 → view(float4_e2m1fn_x2)` because `torch.randn(...).to(float4_e2m1fn_x2)` raises in torch 2.10+; for timing the exact values don't matter.

FLOPs are counted with the standard `2·M·N·K` convention; throughput is `flops / median_ms / 1e9` TFLOPs.

## Tier 3 — NCU roofline

`run_tiers.run_roofline_tier` profiles the BF16 `kernel_bench.py` under Nsight Compute. BF16 is the most stable codepath across all three wheels (FP8/FP4 may skip on some).

`bench/roofline.py` runs:

```
ncu --set roofline \
    --kernel-name regex:(?i)(gemm|mma|flash|sdpa|attention|cutlass) \
    --launch-count 200 \
    --target-processes all \
    --force-overwrite \
    --export <rep_path> \
    -- <command>
```

The 200-launch budget captures ~3–5 launches per unique GEMM kernel (4 shapes × multiple iters) without exploding NCU's 10–30× replay overhead.

`from_ncu(rep_path)` parses the `.ncu-rep` via the `ncu_report` Python API and emits one `Result` per profiled kernel:

- `measured` = `gpu__time_duration.sum` in ms
- `sol` = back-derived ideal duration (`duration × achieved_pct / 100`) where `achieved_pct = max(sol_sm_pct, sol_mem_pct)`
- `extra` records `achieved_pct`, `sol_sm_pct`, `sol_mem_pct`, and a `limit ∈ {compute, bandwidth}` classification

This is the only tier with a non-`None` SOL bound, so it is the only tier whose rows produce a numeric SOL Score column in `SUMMARY.txt`.

## Aggregation: SOL Score

`bench/_summarize.py` reads `runA/B/C.json`, groups results by stable `name` across wheels, and produces a SOL Score per (wheel, test) using Run A as baseline:

```
Score(wheel, test) = clamp01( (measured − baseline) / (sol − baseline) )
```

The formula is sign-correct in both directions: TFLOPs (higher-is-better) yields positive / positive; latency-ms (lower-is-better) yields negative / negative.

- **Score = 0.0** → wheel matches PyPI baseline (no improvement)
- **Score = 1.0** → wheel reaches the NCU-derived per-kernel ceiling
- Tiers without a `sol` value (optimum, kernel) render '—' in the score column

Rows where wheel coverage is incomplete (typically NCU kernel names that differ across wheels) are counted and reported as a skipped-row footnote rather than dropped silently.

## What this bench does not do

- **Multi-GPU / NVLink-C2C** — DGX Spark is single-GPU.
- **Training throughput** — inference + microkernels only.
- **Power / perf-per-watt** — `nvidia-smi --query-gpu=power.draw` accuracy is not calibrated on GB10; `energy: false` in the optimum-benchmark configs.
- **Public leaderboard submission** — local numbers only.
- **cuBLASLt heuristic auto-tuning** — cuBLAS defaults.
- **Per-shape Triton autotuning** — Triton is in the bench-base image but is not exercised as a separate measurement tier; the kernel bench uses `torch.matmul` / `torch._scaled_mm`, whose backend selection is left to PyTorch's dispatcher.
