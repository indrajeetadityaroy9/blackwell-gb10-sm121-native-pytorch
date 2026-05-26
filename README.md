# NVIDIA GB10 / sm_121a: native PyTorch wheel + kernel-optimization research

This repository has two layers, built bottom-up for NVIDIA GB10 (Grace + Blackwell, **sm_121a**) on a DGX Spark:

1. **The wheel** — a from-source PyTorch build with native `sm_121`/`sm_121a` cubins, the foundation everything else runs on (`build/`, `bench/`).
2. **The research** — `gb10-tune/`, a prototype that adapts recent AI-kernel-optimization mechanisms *specifically* for sm_121a hardware utilization, running on top of that wheel.

---

# Layer 1 — Native sm_121/sm_121a PyTorch wheel

A from-source build that produces the first PyTorch wheel on this hardware with **native sm_121 and sm_121a cubins** — eliminating the runtime PTX-JIT path that every other distribution falls back to on GB10.

## Why it exists

Empirically verified via `torch.cuda.get_arch_list()`:

| Wheel | Compiled arch list |
|---|---|
| PyPI `torch==2.10.0+cu130` | `sm_80, sm_90, sm_100, sm_110, sm_120, compute_120` |
| NGC `pytorch:26.04-py3` (`torch 2.12.0a0+nv26.04`) | `sm_80, sm_86, sm_90, sm_100, sm_110, sm_120, compute_120` |
| **This build (PyTorch 2.10.0)** | **`sm_121, sm_121a`** |

On every wheel that doesn't ship `sm_121`, the CUDA driver JIT-compiles `compute_120` PTX to `sm_121` SASS at first kernel launch. This build emits sm_121 SASS ahead of time and additionally compiles arch-specific feature paths (`sm_121a` — TMEM, 2-CTA MMA, async tensor-core writes) that several CUTLASS-derived kernels in PyTorch rely on.

## Build the wheel (~1.5–2 h fresh, ~30 min warm via ccache)

```bash
bash build/source_build.sh
```

PyTorch 2.10.0 builds inside `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04` with `TORCH_CUDA_ARCH_LIST="12.1;12.1a"`. All source, build artifacts, and the resulting wheel live in the docker named volume `dgx-spark-build-strict` — no host filesystem state outside the captured launcher log.

## Three-way bake-off (validating the wheel)

```bash
bash bench/build/build_bench_base.sh        # one-time, ~5–10 min, image ~10 GB
bash bench/build/prefetch_hf_models.sh      # one-time, pulls Llama-3.1-8B into dgx-spark-hf-cache
bash bench/run_bakeoff.sh
```

`run_bakeoff.sh` spawns a privileged clock-lock sidecar (`dgx-bench-clocklock`, NVML lock at 2418 MHz), then runs three ephemeral containers in sequence:

- **Run A** — PyPI `torch==2.10.0+cu130` on the `dgx-spark-bench-base:cuda13.2` image
- **Run B** — source-built wheel mounted read-only from `dgx-spark-build-strict`
- **Run C** — NGC `nvcr.io/nvidia/pytorch:26.04-py3`

Each run executes `bench/run_tiers.py` inside its container. Three tiers run in deterministic order:

1. **optimum-benchmark** — Llama-3.1-8B BF16, batch=1, seq_len=512, 128 new tokens. Bandwidth-bound regime; reports per-op latency.
2. **kernel-level GEMM** — four Llama-3.1-8B projection shapes at M=512 across bf16/fp16/fp8/fp4 (each dtype as an isolated subprocess so cuBLAS Lt's incomplete FP8/FP4 sm_121 heuristic table can't take out BF16/FP16 results).
3. **NCU roofline** — `ncu --set roofline` over the BF16 kernel_bench with kernel-name filtering and a 200-launch budget; per-kernel achieved % of peak parsed via the `ncu_report` Python API.

Each run emits a JSON document to `bench/logs/run{A,B,C}.json`. `bench/_summarize.py` aggregates them into `bench/logs/SUMMARY.txt` with a per-test SOL Score column for the roofline tier (gap-closure: `(measured − baseline) / (sol − baseline)`, clamped to `[0, 1]`). The optimum and kernel tiers render '—' in the score column since no SOL bound is modeled there.

Requires Docker with the NVIDIA Container Toolkit, an NGC API key (`docker login nvcr.io`) for Run C, and an `hf` CLI token for the model prefetch. Tear down the bake-off volumes with `bash bench/cleanup_volumes.sh`.

## When NGC is still the right call

Despite the build matching NGC on the workloads this bench exercises, NGC remains the path of least resistance without rebuilding PyTorch or pinning a specific build configuration:

```bash
docker run --rm -it --gpus all --ipc=host -v $HOME:/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:26.04-py3
```

NGC ships triton 3.6.0 (the only stack where the Triton matmul test succeeds on sm_121a) and a private NVIDIA `flash_attn==2.7.4+nv26.04` fork — neither reproducible from source.

---

# Layer 2 — Adapting AI-kernel-optimization mechanisms for sm_121a (research, WIP)

`gb10-tune/` runs on top of the native wheel. The novel angle is **domain-specific adaptation**: taking the mechanisms from four recent papers and specializing them for GB10 / sm_121a hardware utilization, rather than applying any of them off-the-shelf.

| Paper | Mechanism drawn on |
|---|---|
| **FlashInfer-Bench** ([2601.00227](https://arxiv.org/abs/2601.00227)) | Trace schema (Definition / Workload / Solution / Evaluation), the `fast_p` correctness-and-speed metric, robust multi-class benchmark |
| **AutoKernel** ([2603.21331](https://arxiv.org/abs/2603.21331)) | Single-agent keep/revert loop, 5-stage correctness harness, tiered playbook (extended with an sm_121a tier: TMEM, 2-CTA MMA, `tcgen05`, the 101376-byte SMEM cap) |
| **RecursiveMAS** ([2604.25917](https://arxiv.org/abs/2604.25917)) | Inner/Outer latent links and multi-agent collaboration patterns |
| **GEPA** ([2507.19457](https://arxiv.org/abs/2507.19457)) | Reflective search via Actionable Side Information (full execution traces fed back to the proposer) |

clock-locked benchmarking (`gb10-tune/run_tune.sh`, NVML 2418 MHz), per-iteration CUDA-event timing (~10% stdev), `fast_p` AUC scoring, a visible/held-out workload split for contamination resistance, on-device roofline classification, and `% of theoretical hardware peak` reporting with clock provenance in every result.

fused `add + RMSNorm` Triton kernel runs **3.07× over unfused eager PyTorch at 76.5% of LPDDR5X bandwidth** on GB10 (clock-locked, correctness-validated). This is the op class where the reference papers also find their gains; AutoKernel reports 5.29× on RMSNorm but *loses* on GEMM, where eager is already cuBLAS at the ceiling.
