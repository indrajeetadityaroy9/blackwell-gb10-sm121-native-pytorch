# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A from-source PyTorch 2.10.0 build for NVIDIA DGX Spark (GB10 / Grace + Blackwell, sm_121) with native `sm_121` and `sm_121a` cubins — plus a SOL-ExecBench benchmark harness that compares it against PyPI `torch==2.9.0+cu130` and NGC `pytorch:26.04-py3`. Two parts only:

- `build/source_build.sh` — produces the wheel inside a docker named volume.
- `bench/` — the SOL-ExecBench harness and the 3-way bake-off driver.

`bench/` is run as scripts, not installed as a package (`pyproject.toml` `packages = []`).

## Common commands

```bash
# Build the wheel (~1.5–2 h fresh; ~30 min warm via ccache). Artifacts live in docker volume dgx-spark-build-strict.
bash build/source_build.sh

# 3-way bake-off A/B/C — requires Docker + NVIDIA Container Toolkit + `docker login nvcr.io` for Run C.
bash bench/run_bakeoff.sh

# Phase-0 gate: confirm NVML --lock-gpu-clocks works on this GB10 before relying on the controller.
bash bench/verify_clocklock.sh

# Run one test directly (inside a torch-equipped env, e.g. NGC container or installed wheel)
python bench/bench_full.py                          # all tests, human-readable
python bench/bench_full.py --only gemm_fp16_llm     # subset
python bench/bench_full.py --only fp4 --json        # JSON to stdout
python bench/bench_full.py --no-isolate             # skip subprocess-per-test SOL-ExecBench isolation

# Tier 3 roofline (opt-in; needs Nsight Compute 2025.3+ and SYS_ADMIN cap)
BENCH_PROFILE=1 python bench/roofline.py fp16

# Cleanup the build volume (frees ~25 GB)
docker volume rm dgx-spark-build-strict
```

Bench env vars (`run_bakeoff.sh` forwards these into each container):

- `BENCH_ONLY` (same as `--only`), `BENCH_ITERS=50`, `BENCH_WARMUP=5`, `BENCH_M=8192`
- `BENCH_GPU_MHZ=2418` (clock-lock target — GB10 Default Applications clock; Max boost is 3003)
- `BENCH_PROFILE=1` enables Tier 3 ncu profiling

## Architecture

### Build pipeline (`build/source_build.sh`)

Everything happens inside `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04` with the host repo bind-mounted read-only and the docker named volume `dgx-spark-build-strict` for source/ccache/wheel. No host filesystem state outside that volume. Key knob: `TORCH_CUDA_ARCH_LIST="12.1;12.1a"` — produces native SASS for both sm_121 and the arch-specific `sm_121a` feature paths (TMEM, 2-CTA MMA, async tensor-core writes) instead of relying on PTX-JIT from `compute_120`.

The script strips many optional components (`USE_DISTRIBUTED=0`, `USE_NCCL=0`, `USE_FBGEMM=0`, `USE_MAGMA=0`, `USE_KINETO=0`, `USE_MKLDNN=0`, etc.) and `rm -rf /work/pytorch/build` before each invocation because CMake caches `LDFLAGS_INIT` and other env-derived vars from the first configure — env changes are silently ignored on re-runs otherwise.

### Bake-off driver (`bench/run_bakeoff.sh`)

Spawns four containers:

1. **Clock-lock controller** — `--privileged`, holds the GPU clocks via NVML for the duration of the run. Bench containers stay unprivileged. A trap on `EXIT INT TERM ERR` resets clocks and removes the controller.
2. **Run A** — `cuda:13.0.0-devel` + pip install `torch==2.9.0+cu130` (PyPI baseline).
3. **Run B** — `cuda:13.2.0-devel` + pip install the wheel from `dgx-spark-build-strict` (skips if absent).
4. **Run C** — `nvcr.io/nvidia/pytorch:26.04-py3` (NGC vendor reference).

Each container has its own Triton cache directory under `bench/cache/triton/sm121/run{A,B,C}/` because Triton's `backend.hash()` differs between sm_120 and sm_121 — sharing a cache would silently feed in wrong-arch artifacts. Each container writes `bench/logs/run{A,B,C}.json`; `bench/_summarize.py` aggregates them into `SUMMARY.txt` with SOL Score = `(measured − baselineA) / (sol − baselineA)`.

### Benchmark harness (`bench/`)

- **`bench_full.py`** — entrypoint and orchestrator. Builds `ALL_TESTS: dict[str, () -> list[Result]]` from `tests_kernels.TESTS`, `tests_triton_autotune.TESTS`, `tests_gemm_llm.TESTS`, `tests_attn_fa4.TESTS`, plus inline `test_sparse_24` and `test_triton_matmul`. Multi-test runs **spawn a subprocess per test** (SOL-ExecBench memory isolation requirement) by re-invoking the same script with `--only <key> --json --no-isolate`; single-test runs always go in-process.
- **`_harness.py`** — Stdlib-only (no numpy — not installed in Run A/B containers). Provides `cuda_event_time` (kernel-only via `torch.cuda.Event`, cold-cache by L2-flushing 48 MB / 2× the 24 MB L2 between iters), `Stats.from_samples` (mean/median/p10/p90/stdev via `statistics`), `Result` dataclass, `emit_json`, `allclose_gate`, subprocess isolation.
- **`_solar.py`** — Analytical SOL bounds (compute-bound vs bandwidth-bound, picks the limiting resource). `gemm_sol` and `attn_sol` use the FLOPs/bytes formulas documented in `docs/benchmark-methodology.md`. Config in `sol_config.toml`.
- **`sol_config.toml`** — GB10 spec: 48 SMs, 24 MB L2, 273 GB/s LPDDR5X (verify via the bandwidth test before trusting), `peak_tflops` per dtype. The `peak_tflops` values are calibration **placeholders** (~1.1× highest measured across A/B/C) until replaced with datasheet numbers — SOL Scores are only meaningful for relative A/B/C comparison until then.
- **`_clocklock.sh`** — NVML helper. On GB10 only the GPU clock is lockable (mem clock = N/A on LPDDR5X). `nvidia-smi --lock-gpu-clocks="${mhz},${mhz}"` requires a `<min,max>` pair; the script also verifies the lock took effect because NVML can silently no-op on some SKUs.
- **`tests_gemm_llm.py`** — GEMM shapes derived from public model configs (Llama-3-8B/70B, Mixtral-8x7B, Qwen3-30B-A3B, DeepSeek-V3) at `M = 4096` prefill. fp16 via `a @ b`, fp8 via `torch._scaled_mm` (`float8_e4m3fn`, scalar scales, `use_fast_accum=True`), fp4 via `torch._scaled_mm` with `float4_e2m1fn_x2` + `float8_e4m3fn` 1×16 scales.
- **`tests_attn_fa4.py`** — FlashAttention-4 §5.1 grid: 4 seqlens × 3 configs (MHA d=64/H=32, MHA d=128/H=16, GQA d=128/Hq=32/Hkv=8), fwd + bwd, bf16, causal. Wraps `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` — silent MATH fallback aborts. DeepSeek MLA dropped because stock SDPA-flash requires `q.head_dim == k.head_dim`.
- **`tests_triton_autotune.py`** — TritonForge 162-config autotune over the same LLM GEMM shapes. First-shape autotune costs ~5–10 min (162 compiles); the per-container Triton cache amortizes this across runs.
- **`tests_kernels.py`** — Bandwidth-bound singletons: rmsnorm, softmax over `[B,H,S,S]`, cross_entropy over `[N, V=128k]`.
- **`roofline.py`** — Tier 3 opt-in. `ncu --set roofline` against `bench_full.py`, parses the `.ncu-rep` via the `ncu_report` Python API. Per-instruction SASS counters are not in the roofline set on Blackwell — use `--set full` separately if needed.
- **`nvbench_shim/`** — sm_121-targeted nvbench GEMM binary (CMake + `build_nvbench.sh` + `run_nvbench.py`); built inside a CUDA-devel container with `CMAKE_CUDA_ARCHITECTURES=121`. Ubuntu 24.04's cmake 3.28 is too old for nvbench main; the build script pip-installs `cmake>=3.31` into a venv.
- **`experimental/`** — Tier 4 end-to-end LLM workloads (FlashInfer attention serving, MLPerf v5.1 llama3.1-8b). Inherits the clock-lock pattern from `run_bakeoff.sh`. Gated by `BENCH_DOWNLOAD_MODELS=1`.

### Result schema

All tests produce `Result(name, unit, measured, sol, sol_score, sol_limit, stats, correctness, extra)`. `sol_score` is filled by the orchestrator (`_summarize.py`), not by the test itself. `correctness` is `"PASS"` / `"FAIL: max|diff|=..."` / `None`. Adding a new test means appending a `() -> list[Result]` to one of the `TESTS` dicts that `bench_full.py` imports.

## Things that bite

- The bench tree is run as scripts; `sys.path.insert(0, str(Path(__file__).resolve().parent))` at the top of each test module makes `bench/` importable from both direct invocation and subprocess isolation. Don't refactor that away.
- `bench/logs/`, `bench/wheels/`, `bench/build/`, and `docs/` are in `.gitignore`. `docs/` is gitignored because it's writer-only context — do not assume those files ship with the repo for downstream consumers.
- Re-running `source_build.sh` without `rm -rf /work/pytorch/build` will silently reuse a stale CMakeCache and ignore env-var changes.
- A failed bench inside a container does not stop subsequent runs — `run_bakeoff.sh` uses `set -uo pipefail` (no `-e`) so it records exit codes and continues to `_summarize.py`.
- `--no-isolate` exists for debugging but disables the SOL-ExecBench memory-isolation guarantee; do not use it for reported numbers.
