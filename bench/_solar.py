"""
SOLAR — analytical Speed-of-Light bounds for DGX Spark GB10.
Methodology: SOL-ExecBench (arXiv 2603.19173).

SOL = throughput at the limiting resource (min of compute-bound and
bandwidth-bound rates). SOL Score = fraction of (baseline → SOL) gap
closed, in [0, 1].

Reads sol_config.toml from the same directory. Stdlib-only. Python 3.11+.
"""

from __future__ import annotations
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


_CFG_PATH = Path(__file__).parent / "sol_config.toml"


@dataclass(frozen=True)
class GB10Config:
    sm: str
    sm_count: int
    l2_cache_mb: int
    unified_mem_gb: float
    mem_bandwidth_tb_s: float
    peak_tflops: dict[str, float]


def load_config(path: Path = _CFG_PATH) -> GB10Config:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    g = raw["gb10"]
    return GB10Config(
        sm=g["sm"],
        sm_count=int(g["sm_count"]),
        l2_cache_mb=int(g["l2_cache_mb"]),
        unified_mem_gb=float(g["unified_mem_gb"]),
        mem_bandwidth_tb_s=float(g["mem_bandwidth_tb_s"]),
        peak_tflops={k: float(v) for k, v in g["peak_tflops"].items()},
    )


# Bytes per element. float4_e2m1fn_x2 packs 2 fp4 values per byte = 0.5 B/elem.
_BYTES_PER_ELEM = {
    "fp16": 2.0, "bf16": 2.0,
    "fp32": 4.0, "fp64": 8.0,
    "fp8": 1.0, "fp8_e4m3fn": 1.0, "fp8_e5m2": 1.0,
    "fp4": 0.5, "float4_e2m1fn_x2": 0.5,
    "int8": 1.0, "uint8": 1.0,
}


def bytes_per_elem(dtype: str) -> float:
    if dtype not in _BYTES_PER_ELEM:
        raise KeyError(
            f"unknown dtype '{dtype}' for SOLAR (valid: {list(_BYTES_PER_ELEM)})"
        )
    return _BYTES_PER_ELEM[dtype]


@dataclass(frozen=True)
class SOL:
    """Analytical SOL with the bounding resource identified."""
    sol_tflops: float            # SOL throughput (TFLOPs)
    sol_gbs: float | None        # SOL bandwidth (GB/s) if bandwidth-bound, else None
    compute_bound_s: float       # compute-limited time (s)
    bandwidth_bound_s: float     # bandwidth-limited time (s)
    limit: str                   # "compute" or "bandwidth"


def gemm_sol(M: int, N: int, K: int, dtype: str, cfg: GB10Config,
             out_bytes_per_elem: float = 2.0) -> SOL:
    """SOL for dense GEMM (M,K) @ (K,N) → (M,N).
    Bytes = M·K·bpe + K·N·bpe + M·N·out_bpe."""
    flops = 2.0 * M * N * K
    bpe = bytes_per_elem(dtype)
    bytes_ = bpe * (M * K + K * N) + out_bytes_per_elem * (M * N)
    peak_tflops = cfg.peak_tflops.get(dtype)
    if peak_tflops is None:
        raise KeyError(f"sol_config.toml has no peak_tflops['{dtype}'] entry")
    compute_s = flops / (peak_tflops * 1e12)
    bandwidth_s = bytes_ / (cfg.mem_bandwidth_tb_s * 1e12)
    if compute_s >= bandwidth_s:
        sol_tflops = peak_tflops
        limit = "compute"
    else:
        sol_tflops = flops / bandwidth_s / 1e12
        limit = "bandwidth"
    return SOL(sol_tflops=sol_tflops, sol_gbs=None,
               compute_bound_s=compute_s, bandwidth_bound_s=bandwidth_s,
               limit=limit)


def attn_sol(B: int, H: int, S: int, D: int, dtype: str, cfg: GB10Config,
             causal: bool = True, backward: bool = False) -> SOL:
    """SOL for FlashAttention-style fused attention.
    Forward FLOPs:  4·B·H·S²·D  (causal halves to 2·B·H·S²·D)
    Backward FLOPs: ~2.5× forward
    Bytes: 4·B·H·S·D (Q,K,V,O)."""
    fwd_flops = (2.0 if causal else 4.0) * B * H * S * S * D
    flops = fwd_flops * (2.5 if backward else 1.0)  # 5/2 fwd-equivalents for bwd
    bpe = bytes_per_elem(dtype)
    bytes_ = bpe * 4 * B * H * S * D
    peak_tflops = cfg.peak_tflops.get(dtype, cfg.peak_tflops.get("fp16"))
    compute_s = flops / (peak_tflops * 1e12)
    bandwidth_s = bytes_ / (cfg.mem_bandwidth_tb_s * 1e12)
    if compute_s >= bandwidth_s:
        sol_tflops = peak_tflops
        limit = "compute"
    else:
        sol_tflops = flops / bandwidth_s / 1e12
        limit = "bandwidth"
    return SOL(sol_tflops=sol_tflops, sol_gbs=None,
               compute_bound_s=compute_s, bandwidth_bound_s=bandwidth_s,
               limit=limit)


def bandwidth_sol_gbs(cfg: GB10Config) -> float:
    """SOL ceiling in GB/s for bandwidth-bound tests."""
    return cfg.mem_bandwidth_tb_s * 1000.0


def sol_score(measured: float, baseline: float, sol: float) -> float:
    """SOL-ExecBench score: (measured − baseline) / (SOL − baseline), clamped [0,1].
    0 = matches baseline, 1 = reaches SOL. Returns 0 if SOL ≤ baseline."""
    if sol <= baseline:
        return 0.0
    return max(0.0, min(1.0, (measured - baseline) / (sol - baseline)))


# Self-test: print SOL for representative kernels.
if __name__ == "__main__":
    cfg = load_config()
    print(f"loaded sol_config.toml for {cfg.sm}, {cfg.sm_count} SMs")
    print(f"  mem_bandwidth = {cfg.mem_bandwidth_tb_s * 1000:.1f} GB/s")
    print(f"  peak_tflops   = {cfg.peak_tflops}")
    s = gemm_sol(8192, 8192, 8192, "fp16", cfg)
    print(f"  FP16 8192^3 GEMM SOL: {s.sol_tflops:.1f} TFLOPs (limit={s.limit})")
    s = gemm_sol(8192, 8192, 8192, "fp8", cfg)
    print(f"  FP8  8192^3 GEMM SOL: {s.sol_tflops:.1f} TFLOPs (limit={s.limit})")
    s = gemm_sol(8192, 8192, 8192, "fp4", cfg)
    print(f"  FP4  8192^3 GEMM SOL: {s.sol_tflops:.1f} TFLOPs (limit={s.limit})")
    print(f"  bandwidth SOL: {bandwidth_sol_gbs(cfg):.1f} GB/s")
    print(f"  sol_score(measured=82, baseline=56.8, sol=90) = {sol_score(82, 56.8, 90):.3f}")
