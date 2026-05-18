# SOL Score — Speed-of-Light scoring for DGX Spark GB10

The bench's primary throughput metric for kernel work is **SOL Score** — the
fraction of the gap between the PyPI baseline (Run A) and the hardware's
analytical Speed-of-Light (SOL) bound that a given wheel closes.

This document explains the formula, how the SOL bound is computed (`SOLAR`),
the GB10-specific peak values it uses, and the calibration step.

## Formula (SOL-ExecBench, arXiv 2603.19173)

```
SOL Score = clamp01( (measured − baseline) / (SOL − baseline) )
```

Where:
- `measured` is the wheel's median throughput from the timed run
- `baseline` is Run A's (PyPI `torch==2.9.0+cu130`) median for the same test
- `SOL` is the analytical Speed-of-Light bound from `bench/_solar.py`

Interpretation:
- **Score = 0.0** → wheel matches baseline (no improvement over PyPI)
- **Score = 1.0** → wheel reaches the hardware SOL ceiling
- Scores >1.0 are clamped to 1.0 (means SOL placeholder is too low; recalibrate)
- Reported as N/A when the test SKIPPED on Run A (no baseline)

## SOLAR — how SOL is computed

`bench/_solar.py` derives SOL from operation FLOPs, memory bytes touched, and
hardware peaks read from `bench/sol_config.toml`. The bound is the lower of:

- **Compute-bound rate**: `FLOPs / peak_tflops_for_dtype`
- **Bandwidth-bound rate**: `bytes / mem_bandwidth_tb_s`

Take the **longer** of the two times → that's the SOL latency. SOL throughput
is `FLOPs / SOL_latency`.

For 8192³ GEMMs at FP16/FP8/FP4 the workload is compute-bound on GB10
(arithmetic intensity ≫ FLOPs/byte), so SOL = `peak_tflops[dtype]`.

For RMSNorm/softmax/cross-entropy, the workload is bandwidth-bound, so SOL is
reported as `mem_bandwidth_tb_s × 1000` GB/s.

## GB10 peaks in `sol_config.toml`

| Field | Value | Source |
|---|---|---|
| `sm` | sm_121 | cudaDeviceProp.major.minor |
| `sm_count` | 48 | cudaDeviceProp.multi_processor_count |
| `l2_cache_mb` | 24 | cudaDeviceProp.L2_cache_size |
| `unified_mem_gb` | 124.6 | cudaDeviceProp.total_memory |
| `mem_bandwidth_tb_s` | 0.273 | Grace LPDDR5X-8533 × 4 channels (verified by Tier 2 `bandwidth` test) |
| `peak_tflops.fp16` | 90.0 | **placeholder** (1.1× highest measured) |
| `peak_tflops.fp8` | 185.0 | **placeholder** |
| `peak_tflops.fp4` | 350.0 | **placeholder** |
| `peak_tflops.sparse_fp8` | 370.0 | **placeholder** |

## Why the peak TFLOPs are placeholders

NVIDIA has not published official GB10 GEMM peak TFLOPs at the time of writing.
The placeholders are derived as **1.1× the highest measured TFLOPs across runs
A/B/C** on this hardware (FP16=82, FP8=170, FP4=314), then rounded.

This means:
- **SOL Scores using placeholder peaks are only meaningful for relative
  comparison across wheels.**
- A wheel scoring 0.85 on FP16 closes 85% of the (PyPI→placeholder-SOL) gap.
  If the real SOL is higher (e.g., 110 TFLOPs from cross-validation with
  Nsight Compute's SOL%), the absolute score would be lower.
- Cross-validation: Nsight Compute roofline (`bench/roofline.py`) reports a
  hardware-grounded SOL% per kernel. Compare against our placeholder-derived
  SOL Score to detect when the placeholder needs updating.

## Calibration

To replace the placeholder peaks with empirical values from this hardware:

```
sg docker -c 'bash bench/calibrate_peaks.sh'   # (one-shot, optional)
```

This (when implemented) runs each kernel at 60s burn-in on Run C (best-case
wheel), takes max-seen TFLOPs × 1.05 as the new SOL placeholder, and rewrites
`bench/sol_config.toml`. **NOT included in default Tier 1 yet** — file a
follow-up if needed.

A better calibration is `Nsight Compute` SOL% from the roofline rule:

```
sg docker -c 'docker run --rm --gpus all --cap-add=SYS_ADMIN \
  -v dgx-spark-build-strict:/work:ro -v $PWD:/repo \
  nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04 \
  bash -c "<setup>; python /repo/bench/roofline.py fp16"'
```

This reports `sol_sm` (e.g., 75%) and `sol_mem` (e.g., 52%). If our analytical
SOL Score is 96% but Nsight's SOL% is 75%, the real hardware peak is ~28% higher
than our placeholder. Multiply `peak_tflops[fp16]` by that ratio.

## Citations

- SOL-ExecBench, arXiv 2603.19173 — methodology + score formula
- Quartet II, arXiv 2601.22813 — NVFP4 training reference numbers
- TritonForge, arXiv 2512.09196 — Triton autotune speedup reference
- Nsight Compute 2026.1.0 — roofline rule implementation
