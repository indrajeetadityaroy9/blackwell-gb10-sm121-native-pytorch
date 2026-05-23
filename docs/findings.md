# Findings — DGX Spark PyTorch Performance Analysis

## TL;DR

A PyTorch 2.10.0 wheel built from source with `TORCH_CUDA_ARCH_LIST="12.1;12.1a"` is the first PyTorch on the NVIDIA DGX Spark (Grace + Blackwell GB10, sm_121) with native sm_121 and sm_121a cubins. Every other public PyTorch wheel falls back to runtime PTX-JIT on this hardware. This build ships SASS for sm_121 ahead of time and additionally compiles the arch-specific feature variant sm_121a.

## The configurational gap

Empirically verified by reading `torch.cuda.get_arch_list()` from each wheel:

| Wheel | Compiled arch list | Native sm_121 cubin? |
|---|---|---|
| PyPI `torch==2.10.0+cu130` | `sm_80, sm_90, sm_100, sm_110, sm_120, compute_120` | ❌ |
| NGC `pytorch:26.04-py3` (`torch 2.12.0a0+nv26.04`) | `sm_80, sm_86, sm_90, sm_100, sm_110, sm_120, compute_120` | ❌ |
| **This build (PyTorch 2.10.0)** | **`sm_121, sm_121a`** | **✓** |

On the GB10 (cc=12.1), every CUDA kernel call from either public wheel falls through to the embedded `compute_120` PTX, which the driver JIT-compiles to sm_121 SASS at runtime. The PTX JIT is well-tuned but has overhead — both per-launch (first call pays the JIT) and structural (PTX → SASS retains hints the JIT compiler may or may not honor).

Setting `TORCH_CUDA_ARCH_LIST="12.1;12.1a"` produces SASS for sm_121 (and the arch-specific feature variant sm_121a) ahead of time. The wheel ships native cubins and skips the JIT step entirely on this hardware.

## What the bake-off measures

Hardware: NVIDIA DGX Spark, GB10 (compute capability 12.1, 48 SMs, ~130 GB unified memory), Ubuntu 24.04.

`bench/run_bakeoff.sh` runs the same three-tier workload inside each wheel's container under a privileged NVML clock-lock controller at 2418 MHz. See [benchmark-methodology.md](benchmark-methodology.md) for the full pipeline. Each tier targets a different regime:

| Tier | Workload | Unit | Regime |
|---|---|---|---|
| optimum | Llama-3.1-8B BF16 prefill + decode (batch=1, seq_len=512, 128 new tokens) | ms | bandwidth-bound (273 GB/s LPDDR5X ceiling) |
| kernel | Four Llama-3.1-8B projection shapes at M=512, dtypes bf16/fp16/fp8/fp4 | TFLOPs | compute-bound (arith intensity > GB10's ~330 FLOPs/byte crossover) |
| roofline | NCU `--set roofline` over the BF16 kernel_bench | ms + achieved % of peak | per-kernel SOL diagnosis |

The compute-bound tier is where native sm_121 cubins (Run B) should differentiate from the PTX-JIT path (Runs A, C); the bandwidth-bound tier should saturate near identical numbers across all three wheels since they all hit the same DRAM ceiling.

The bake-off emits `bench/logs/SUMMARY.txt` with per-test rows for each wheel and a gap-closure SOL Score column for the roofline tier (other tiers render '—' since no SOL bound is modeled there).

## Interpretation

### What the configurational comparison tells you

The arch-list table above is the only assertion that holds independent of any single measurement run. The structural advantages previously assumed unrecoverable without NVIDIA-internal access — cuBLAS 13.2 sm_121 heuristic tables, curated NVIDIA-branch torch, the private `flash_attn==2.7.4+nv26.04` fork — are partially or wholly addressed by emitting native cubins from a stock PyTorch source build. The bake-off's role is to quantify how much of the PyPI→NGC gap is closed by replacing the PTX-JIT path with native SASS.

### What this bench is not

- Not a comprehensive PyTorch benchmark (no convolutions, no training, no end-to-end multi-model serving).
- Not a Blackwell-peak benchmark — reaching Speed-of-Light requires TMEM + 2-CTA MMA + CuTe-DSL hand-tuned kernels per FlashAttention-4, which neither cuBLAS Lt nor the SDPA-flash backend currently emit on sm_121.
- Does not isolate hardware effects independent of driver state, container overhead, or kernel selection.

It is right-sized for the specific question this research asks: **how much of the configurational gap between PyPI PyTorch and NGC PyTorch on GB10 is closed by a from-source build with native sm_121 cubins.**

## Deployment

```bash
docker run --rm -it --gpus all --ipc=host \
  -v $HOME:/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:26.04-py3
```

To use the from-source wheel built by this repo, install it into a CUDA 13.2 environment (the runtime needs `cudnn9-cuda-13-2`, `cusparselt-cuda-13`, `libcusparselt0-cuda-13`):

```bash
pip install dist/torch-2.10.0-cp312-cp312-linux_aarch64.whl
```

`bench/run_bakeoff.sh`'s Run B does this end-to-end and provides a working template.
