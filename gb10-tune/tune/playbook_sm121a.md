# Triton kernel optimization playbook — NVIDIA GB10 (sm_121a)

The proposer prompt is filled with one tier from this file, selected by the
roofline classifier:

- tier returned `1` (memory-bound, `mem% > 80`) → Tier 2: Memory access
- tier returned `2` (compute-bound, `sm% > 80`) → Tier 3: Compute
- tier returned `3` (latency-bound)             → Tier 4: Advanced + Tier 5c: sm_121a

Always include Tier 1 (Block-size sweep) and Tier 6 (Kernel-specific) as
secondary context when relevant.

---

## Tier 1 — Block-size sweep

Sweep tile dimensions through powers of 2. Try rectangular tiles. Adjust
`num_warps` (4, 8) and `num_stages` (2, 3, 4). Typical wins: 10–50%.

```python
@triton.autotune(configs=[
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8,  num_stages=4),
], key=["M", "N", "K"])
```

## Tier 2 — Memory access

- Coalesced loads: contiguous tensor strides; align tile starts to 128 B.
- Software prefetching: `num_stages=3` or `4`; explicit `tl.load(..., other=0.0)` lookahead.
- L2 swizzling: `group_size_m = 8` to keep adjacent program IDs in the same L2 set.
- Shared-memory padding: pad inner stride by 1 dtype-element to break 32-bank conflicts.
- Vector loads via `tl.load` with `boundary_check=(0, 1)` on aligned chunks.

Typical wins: 10–30%.

## Tier 3 — Compute

- TF32 accumulation: `acc = tl.dot(a, b, acc, allow_tf32=True)`.
- Epilogue fusion: scale + bias + activation inside the same kernel; avoid extra kernel launches.
- Loop invariant hoisting: pull constants out of the K-loop.
- Mixed-precision accumulation: load fp16/bf16, accumulate fp32, store dtype.

Typical wins: 5–15%.

## Tier 4 — Advanced

- Split-K: split the K dimension across multiple programs, atomic-add the partial outputs;
  ideal for tall-skinny matmul (M small, K large).
- Persistent kernels: one program per SM iterates over a worklist; reduces launch overhead
  for many-small-tile workloads.
- Warp specialization: separate producer / consumer warps with `tl.async_load` + barriers.
- Triton `@triton.autotune` over `Config` grid.

Typical wins: 5–20%.

## Tier 5a — Hopper (sm_90)

Tensor Memory Accelerator (TMA) descriptors for 2D loads, `cp.async.bulk.tensor`.
Not directly applicable on sm_121a but several patterns transfer.

## Tier 5b — Ampere (sm_80)

`cp.async` global → shared with `async_wait_group`. Triton emits these via
`tl.load(..., cache_modifier=".ca")` and `num_stages >= 2`.

## Tier 5c — Blackwell sm_121a (this hardware)

**Key constraints:**

- Per-block shared-memory cap: **101376 bytes** (smaller than sm_100). Configurations
  that exceed this fail at launch — Triton's `num_stages` × tile-size product must fit.
- Tensor Memory (TMEM): a separate 256 KB-per-SM accumulator memory distinct from SMEM.
  Triton 3.7's `tl.dot()` automatically allocates TMEM when the accumulator dtype permits.
- 2-CTA MMA: a single MMA instruction spans two CTAs sharing TMEM, reducing per-CTA
  register pressure. Emitted automatically by `tl.dot` when block dims align.
- `tcgen05` tensor instruction family: replaces `wmma`/`hmma` on Blackwell. Triton 3.7
  selects `tcgen05` automatically; avoid explicit inline PTX `wmma.*` (it falls back to
  older SM-class codegen).
- LPDDR5X unified memory: bandwidth-asymmetric vs. HBM. Memory-bound kernels benefit
  disproportionately from L2-aware swizzling and tile-size choices that fit in the 24 MB L2.

**Concrete actions for sm_121a:**

- Prefer `tl.dot(..., out_dtype=tl.float32)` and let Triton allocate TMEM; do not
  manually accumulate in shared memory.
- Cap `BLOCK_M × BLOCK_K × sizeof(input_dtype) × num_stages` to under 101376 bytes.
- Use `num_stages=3` as the default starting point; sm_121a's TMA path tolerates 3 well
  but 4 frequently exceeds the SMEM cap.
- For Llama-3.1-8B QKV (M=512, N=12288, K=4096), try `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64`
  with `num_warps=8`; this fits the SMEM cap and tiles cleanly.

Typical wins: 5–15% over a portable Blackwell baseline.

## Tier 6 — Kernel-specific

- **Attention**: online softmax (FlashAttention-style); avoid materializing the QK^T matrix.
- **Normalization**: Welford single-pass; warp-shuffle reductions across rows.
- **Reduction (sum, max)**: hierarchical warp shuffle → block shuffle → atomic_add tail.
- **Tall-skinny matmul (M small)**: split-K with atomic-add output accumulation.
- **MoE GEMM**: grouped GEMM with expert-indexed tile assignment.
