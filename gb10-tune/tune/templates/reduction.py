"""VectorizedCoalescedReductionTemplate — Triton 1D sum-reduction emitter.

Parameter grid: block ∈ {128, 256}, vector_width ∈ {1, 2, 4},
elements_per_thread ∈ {1, 2} → 12 candidates, all within sm_121a budgets.

Template behavior:
  1. Contiguous global loads (offsets = pid * BLOCK*VEC*EPT + arange()).
  2. Optional vectorization (multiple inner loads — Triton folds VEC>1 into wider ops).
  3. Per-thread accumulation in fp32.
  4. Block-level reduction via tl.sum.
  5. One partial result per block — final reduction host-side via partials.sum().
"""

from itertools import product
from typing import Iterator, Tuple

from .base import Template


class VectorizedCoalescedReductionTemplate(Template):
    name = "vectorized_coalesced_reduction"
    allowed_action = "vectorized_coalesced_reduction"

    def parameter_grid(self) -> Iterator[Tuple[int, int, int]]:
        yield from product((128, 256), (1, 2, 4), (1, 2))

    def render(self, params: Tuple[int, int, int]) -> str:
        block_size, vector_width, elems_per_thread = params
        return f'''\
"""Generated reduction kernel — block={block_size}, vec={vector_width}, ept={elems_per_thread}."""

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_kernel(
    in_ptr, out_ptr, N,
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
    EPT: tl.constexpr,
):
    pid = tl.program_id(0)
    block_elements = BLOCK_SIZE * VEC * EPT
    block_start = pid * block_elements
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for e in tl.static_range(EPT):
        for v in tl.static_range(VEC):
            offs = block_start + e * BLOCK_SIZE * VEC + v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x = tl.load(in_ptr + offs, mask=offs < N, other=0.0)
            acc = acc + x.to(tl.float32)
    block_sum = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, block_sum)


def run(x: torch.Tensor) -> torch.Tensor:
    N = x.numel()
    BLOCK_SIZE = {block_size}
    VEC = {vector_width}
    EPT = {elems_per_thread}
    block_elements = BLOCK_SIZE * VEC * EPT
    num_blocks = (N + block_elements - 1) // block_elements
    partials = torch.empty(num_blocks, dtype=torch.float32, device=x.device)
    _reduce_kernel[(num_blocks,)](
        x, partials, N,
        BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, EPT=EPT,
    )
    return partials.sum()
'''
