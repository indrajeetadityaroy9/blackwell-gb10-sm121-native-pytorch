"""1D sum-reduction baseline — deliberately weak config (BLOCK=32, VEC=1, EPT=1).

32 is outside the template grid (which uses block ∈ {128, 256}), so Stage 1 has
headroom over the baseline by construction.
"""

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
    BLOCK_SIZE = 32
    VEC = 1
    EPT = 1
    block_elements = BLOCK_SIZE * VEC * EPT
    num_blocks = (N + block_elements - 1) // block_elements
    partials = torch.empty(num_blocks, dtype=torch.float32, device=x.device)
    _reduce_kernel[(num_blocks,)](
        x, partials, N,
        BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, EPT=EPT,
    )
    return partials.sum()
