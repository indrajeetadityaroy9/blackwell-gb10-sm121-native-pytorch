"""BottleneckReport — single classification rule.

Stage 0 produces an `ncu_diag` dict via `tune.ncu.parse_report`. This module
turns it into a structured decision: which template family Stage 1 will sweep.

v1 supports memory_bound only — for reduction and GEMM kernels Triton's
pipeline keeps mem and SM throughputs balanced, so "memory_bound" means
mem >= sm. Strict `<` raises; downstream code does not handle other
bottleneck classes yet.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal


@dataclass
class BottleneckEvidence:
    memory_throughput_pct: float
    dram_throughput_pct: float
    sm_throughput_pct: float
    load_efficiency_pct: float


@dataclass
class BottleneckReport:
    kernel_name: str
    bottleneck_type: Literal["memory_bound"]
    evidence: BottleneckEvidence
    allowed_action: str


# op_type → template family. Hardcoded dispatch; not LLM-driven.
_OP_TYPE_TO_ACTION = {
    "reduction": "vectorized_coalesced_reduction",
    "gemm": "tiled_groupm_gemm",
}


def classify(ncu_diag: Dict[str, Any], op_type: str) -> BottleneckReport:
    mem = ncu_diag["mem_throughput_pct"]
    sm = ncu_diag["sm_throughput_pct"]
    if mem < sm:
        raise RuntimeError(
            f"Unsupported bottleneck: mem={mem:.1f}% < sm={sm:.1f}%. "
            "v1 supports memory_bound only."
        )
    return BottleneckReport(
        kernel_name=ncu_diag["kernel_name"],
        bottleneck_type="memory_bound",
        evidence=BottleneckEvidence(
            memory_throughput_pct=mem,
            dram_throughput_pct=ncu_diag.get("dram_throughput_pct", 0.0),
            sm_throughput_pct=sm,
            load_efficiency_pct=ncu_diag.get("load_efficiency_pct", 0.0),
        ),
        allowed_action=_OP_TYPE_TO_ACTION[op_type],
    )
