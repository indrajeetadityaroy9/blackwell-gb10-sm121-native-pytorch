"""TiledGroupMGEMMTemplate — Triton BF16/FP16 GEMM emitter with Group-M tiling.

Parameter grid (filtered by sm_121a shared-memory budget):

  BLOCK_M  ∈ {64, 128}
  BLOCK_N  ∈ {64, 128, 256}
  BLOCK_K  ∈ {32, 64}
  GROUP_M  ∈ {4, 8}
  num_warps ∈ {4, 8}
  num_stages ∈ {2, 3}

  Cartesian product = 96 tuples; budget filter
      2 * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * num_stages <= 101376
  (2 bytes/element for bf16; 101376 bytes is the sm_121a per-block shared-memory
  limit) keeps the configs that won't OutOfResources at launch.
"""

from itertools import product
from typing import Iterator, Tuple

from .base import Template


class TiledGroupMGEMMTemplate(Template):
    name = "tiled_groupm_gemm"
    allowed_action = "tiled_groupm_gemm"

    def parameter_grid(
        self,
    ) -> Iterator[Tuple[int, int, int, int, int, int]]:
        for bm, bn, bk, gm, nw, ns in product(
            (64, 128), (64, 128, 256), (32, 64), (4, 8), (4, 8), (2, 3)
        ):
            if 2 * (bm * bk + bk * bn) * ns <= 101376:
                yield (bm, bn, bk, gm, nw, ns)

    def render(self, params: Tuple[int, int, int, int, int, int]) -> str:
        bm, bn, bk, gm, nw, ns = params
        return f'''\
"""Generated GEMM kernel — BLOCK={bm}x{bn}x{bk}, GROUP_M={gm}, num_warps={nw}, num_stages={ns}."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"K mismatch: A.shape={{A.shape}}, B.shape={{B.shape}}"
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = {bm}, {bn}, {bk}, {gm}
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps={nw}, num_stages={ns},
    )
    return C
'''
