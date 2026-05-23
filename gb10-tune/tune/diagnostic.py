"""NCU-driven diagnostic engine — produces Findings(severity, title, detail, action)
from a KernelData. Ported from cuda_auto_tune/ncu_analyse.py, scoped to Triton.

The agent prompt is built from these Findings; the model is told to cite the
metric values in `detail` as evidence for each proposed change. No more
generic "try larger tiles" — every recommendation maps to an NCU symptom.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List

from .ncu import KernelData, kernel_data_from_dict


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2


@dataclass
class Finding:
    severity: Severity
    title: str
    detail: str   # cite NCU metric values here
    action: str   # 1–4 concrete code patterns


# ---------------------------------------------------------------------------
# Analyzers (Triton-scoped subset of cuda_auto_tune/ncu_analyse.py)
# ---------------------------------------------------------------------------


def _analyze_warp_stalls(kd: KernelData) -> List[Finding]:
    out: List[Finding] = []
    breakdown = kd.stall_breakdown()
    if not breakdown:
        return out

    if 0 < kd.warps_eligible_per_cycle < 1.0:
        out.append(Finding(
            Severity.WARNING,
            f"Low Warp Scheduling Efficiency ({kd.warps_eligible_per_cycle:.2f}/cycle)",
            f"Only {kd.warps_eligible_per_cycle:.2f} eligible warps per cycle "
            "(ideal >= 2). The scheduler frequently has no work to issue.",
            "Increase occupancy (reduce smem/register pressure) "
            "or reduce per-warp latency (better pipelining).",
        ))

    actions = {
        "Long Scoreboard":
            "Warps wait for global/L2 memory. Increase num_stages (2→3→4) for "
            "deeper software pipelining; ensure tl.load offsets are coalesced; "
            "verify innermost dim stride == 1.",
        "Short Scoreboard":
            "Warps wait for shared memory / L1 results. Reduce bank conflicts "
            "(pad shared-memory tile by +1 along the leading dim), reorder "
            "shared-memory accesses, reduce dependency chains.",
        "Wait":
            "Warps stalled on cp.async.wait or barrier. Reduce num_stages "
            "(over-buffered) OR increase BLOCK_M/BLOCK_N to add more compute "
            "per stage.",
        "Sleeping": "Warps explicitly sleeping. Check for nanosleep/yield calls.",
        "Barrier":
            "Warps stalled at __syncthreads(). Reduce barrier frequency, "
            "use tl.broadcast / warp-level primitives.",
        "MIO Throttle":
            "Memory I/O pipeline saturated. Reduce shared-memory or SFU rate, "
            "interleave with other instructions.",
        "LG Throttle":
            "Local/global memory pipeline throttled. Reduce outstanding "
            "memory requests, improve access patterns.",
        "Math Pipe Throttle":
            "Math pipeline fully utilized (positive signal for compute-bound). "
            "Consider Tensor Cores or reduce FLOPs.",
        "Drain": "Warp draining at kernel/block end. Balance work across warps.",
        "Not Selected": "Scheduler contention; usually not actionable.",
        "Selected": "Productive work (not a concern).",
    }
    for reason, count, pct in breakdown[:5]:
        if pct < 10:
            continue
        sev = Severity.CRITICAL if pct >= 25 else Severity.WARNING
        out.append(Finding(
            sev,
            f"Warp Stall: {reason} ({pct:.1f}%)",
            f"{reason} accounts for {pct:.1f}% of stall samples ({count:,.0f}).",
            actions.get(reason, "Consult NCU documentation."),
        ))
    return out


def _analyze_occupancy(kd: KernelData) -> List[Finding]:
    out: List[Finding] = []

    if 0 < kd.warps_active_pct < 50:
        limiter, val = kd.occupancy_limiter()
        action_map = {
            "Registers":
                "Registers are the occupancy bottleneck. Reduce num_warps "
                "(try 4 or 2), reduce BLOCK_M*BLOCK_N (lower per-thread state), "
                "or accumulate in smaller chunks.",
            "Shared Memory":
                "Shared memory is the occupancy bottleneck. Reduce num_stages "
                "(each stage doubles the smem buffer), or reduce "
                "BLOCK_M/BLOCK_N/BLOCK_K tile sizes.",
            "Warps":
                "Warp count per block limits occupancy. Increase num_warps "
                "(8 or 16 — but watch register pressure).",
            "Blocks":
                "Block count limit reached. Reduce num_warps so more blocks "
                "fit per SM.",
        }
        out.append(Finding(
            Severity.WARNING,
            f"Low Occupancy ({kd.warps_active_pct:.1f}%), Limited by {limiter}",
            f"Achieved occupancy {kd.warps_active_pct:.1f}% "
            f"(theoretical {kd.theoretical_occupancy_pct:.1f}%). "
            f"Primary limiter: {limiter} ({val:.0f} blocks/SM).",
            action_map.get(limiter, "Adjust launch configuration."),
        ))

    rpt = kd.registers_per_thread
    if rpt >= 128:
        out.append(Finding(
            Severity.CRITICAL,
            f"Very High Register Usage ({rpt:.0f} regs/thread)",
            "Extremely high register count severely limits occupancy "
            "(each warp needs its own register file).",
            "Reduce num_warps (try 4), reduce BLOCK_M*BLOCK_N to lower "
            "per-thread state, or split accumulator into chunks.",
        ))
    elif rpt >= 64:
        out.append(Finding(
            Severity.WARNING,
            f"High Register Usage ({rpt:.0f} regs/thread)",
            "High register count may limit occupancy.",
            "Consider reducing num_warps to 4 or reducing tile dimensions.",
        ))

    if kd.local_mem_store_sectors > 0:
        out.append(Finding(
            Severity.CRITICAL,
            f"Register Spills Detected ({kd.local_mem_store_sectors:.0f} sectors)",
            f"Local-memory store sectors > 0 indicates the kernel is spilling "
            f"registers to local memory (DRAM). This is catastrophic for perf.",
            "Reduce per-thread state: smaller BLOCK_M*BLOCK_N, reduce num_warps, "
            "or break the accumulator into smaller chunks.",
        ))

    return out


def _analyze_divergence(kd: KernelData) -> List[Finding]:
    out: List[Finding] = []
    div = kd.divergence_pct()
    if div > 20:
        out.append(Finding(
            Severity.WARNING,
            f"Significant Thread Divergence ({div:.1f}%)",
            f"Avg threads executed: {kd.avg_thread_executed:.1f}, "
            f"avg active (true): {kd.avg_thread_executed_true:.1f}. "
            f"Divergence: {div:.1f}%.",
            "Restructure branch logic for warp-uniform conditions; ensure "
            "tile-shape evenly covers the problem (predicated tail handling).",
        ))
    elif div > 10:
        out.append(Finding(
            Severity.INFO,
            f"Moderate Thread Divergence ({div:.1f}%)",
            f"Divergence: {div:.1f}%. Monitor if correlated with perf issues.",
            "",
        ))
    return out


def _analyze_instruction_mix(kd: KernelData) -> List[Finding]:
    out: List[Finding] = []

    if kd.pipe_lsu_pct > 0 and kd.pipe_fma_pct > 0:
        if kd.pipe_lsu_pct > 2 * kd.pipe_fma_pct:
            out.append(Finding(
                Severity.WARNING,
                f"LSU-Dominated Instruction Mix "
                f"(LSU={kd.pipe_lsu_pct:.1f}%, FMA={kd.pipe_fma_pct:.1f}%)",
                "Load/Store instructions dominate; kernel spends most cycles "
                "on memory ops rather than compute.",
                "Increase BLOCK_K (more compute reuse per loaded element); "
                "verify tl.dot is hitting the Tensor Core path (see Tensor Core finding).",
            ))

    if kd.pipe_tensor_pct > 50:
        out.append(Finding(
            Severity.INFO,
            f"Good Tensor Core Utilization ({kd.pipe_tensor_pct:.1f}%)",
            "Tensor Core pipeline is well-utilized.",
            "",
        ))
    return out


def _analyze_triton_specific(kd: KernelData) -> List[Finding]:
    """Triton-specific findings keyed off num_warps inferred from block_size."""
    out: List[Finding] = []

    # block_size from NCU is "(threads,1,1)" — first dim is total threads per block.
    num_warps = 0
    try:
        bs = str(kd.block_size).strip().split(",")[0].strip("()[] ")
        total_threads = int(bs)
        num_warps = total_threads // 32
    except (ValueError, IndexError):
        pass

    if num_warps > 0:
        if kd.registers_per_thread >= 128 and num_warps >= 8:
            out.append(Finding(
                Severity.CRITICAL,
                f"Triton: Excessive num_warps ({num_warps}) with High Register Pressure "
                f"({kd.registers_per_thread:.0f} regs)",
                f"num_warps={num_warps} combined with "
                f"{kd.registers_per_thread:.0f} registers/thread severely limits "
                "occupancy (each warp needs its own register file).",
                "Reduce num_warps to 4 or 2. Alternatively reduce BLOCK_* sizes "
                "to lower per-thread register demand.",
            ))
        elif kd.registers_per_thread >= 64 and num_warps >= 8:
            out.append(Finding(
                Severity.WARNING,
                f"Triton: High num_warps ({num_warps}) with Elevated Register Usage "
                f"({kd.registers_per_thread:.0f} regs)",
                f"num_warps={num_warps} with {kd.registers_per_thread:.0f} regs "
                "may limit occupancy.",
                "Consider reducing num_warps to 4, or reducing tile dimensions.",
            ))

    smem_kb = kd.shared_mem_per_block_kb
    if smem_kb > 0:
        limiter, _ = kd.occupancy_limiter()
        # sm_121a max dynamic smem ≈ 99 KB; above ~95 KB severely caps occupancy.
        if smem_kb > 90:
            out.append(Finding(
                Severity.WARNING,
                f"Triton: Very High Shared Memory ({smem_kb:.1f} KB/block)",
                f"Shared memory usage ({smem_kb:.1f} KB) is at or near the "
                "sm_121a per-block limit (~99 KB). This caps blocks/SM at 1.",
                "Reduce BLOCK_* tile dimensions or decrease num_stages. "
                "Each num_stages level doubles the smem buffer.",
            ))
        if limiter == "Shared Memory" and kd.warps_active_pct < 40:
            out.append(Finding(
                Severity.CRITICAL,
                f"Triton: Shared Memory Limits Occupancy ({kd.warps_active_pct:.1f}%)",
                f"Shared memory ({smem_kb:.1f} KB/block) is the occupancy "
                f"bottleneck. Achieved occupancy: {kd.warps_active_pct:.1f}%.",
                "Reduce num_stages (each stage doubles smem buffer), or "
                "reduce BLOCK_M/BLOCK_N/BLOCK_K tile sizes.",
            ))

    # Tensor Core check — for GEMM kernels, low pipe_tensor + high pipe_fma
    # implies tl.dot is falling back to the scalar FMA path.
    if kd.pipe_tensor_pct < 5 and kd.pipe_fma_pct > 20:
        out.append(Finding(
            Severity.WARNING,
            f"Triton: Tensor Cores Not Utilized "
            f"(tensor={kd.pipe_tensor_pct:.1f}%, FMA={kd.pipe_fma_pct:.1f}%)",
            f"tl.dot likely falling back to scalar FMA path "
            f"(Tensor={kd.pipe_tensor_pct:.1f}%, FMA={kd.pipe_fma_pct:.1f}%).",
            "1) Ensure BLOCK_M, BLOCK_N, BLOCK_K are multiples of 16. "
            "2) Verify input dtypes are bf16/fp16/fp8 (not fp32 without allow_tf32). "
            "3) Check tl.dot operand layout is [M,K] x [K,N]. "
            "4) On sm_121a, GROUP_M tiling helps L2 reuse but doesn't affect dot path.",
        ))

    total_stalls = kd.total_stall_samples()
    if total_stalls > 0:
        long_sb_pct = kd.stall_long_scoreboard / total_stalls * 100
        wait_pct = kd.stall_wait / total_stalls * 100

        if long_sb_pct > 30:
            out.append(Finding(
                Severity.WARNING,
                f"Triton: Long Scoreboard Stall ({long_sb_pct:.1f}%) — Increase num_stages",
                f"Global memory latency dominates ({long_sb_pct:.1f}% of stalls). "
                "Triton's software pipelining (num_stages) can hide this.",
                "Increase num_stages (2→3→4 — subject to shared-memory budget). "
                "Verify shared-memory usage stays under 99 KB.",
            ))
        elif wait_pct > 30 and long_sb_pct < 15:
            out.append(Finding(
                Severity.WARNING,
                f"Triton: Wait Stall ({wait_pct:.1f}%) — Reduce num_stages or Grow Tile",
                f"Async-copy wait dominates ({wait_pct:.1f}%), long scoreboard low "
                f"({long_sb_pct:.1f}%). Pipeline is over-buffered or tile is too small.",
                "Reduce num_stages (saves smem, may improve occupancy), OR "
                "increase BLOCK_M/BLOCK_N for more compute per stage.",
            ))

    return out


def _classify_overall(kd: KernelData) -> str:
    sm = kd.sm_throughput_pct
    mem = kd.mem_throughput_pct
    if sm > mem + 20:
        return "COMPUTE_BOUND"
    if mem > sm + 20:
        return "MEMORY_BOUND"
    if sm < 40 and mem < 40:
        return "LATENCY_BOUND"
    if sm > 60 and mem > 60:
        return "BALANCED (near peak)"
    return "MIXED"


def analyze(ncu_diag: Dict[str, Any]) -> Dict[str, Any]:
    """Run all Triton analyzers against an ncu_diag dict (from ncu.parse_report).

    Returns a structured analysis:
      {
        overall, sm_pct, mem_pct, dram_pct, occupancy_pct, theoretical_occupancy_pct,
        limiter, regs_per_thread, smem_kb_per_block, stall_top_pct, divergence_pct,
        findings: [Finding, ...]
      }
    """
    if not ncu_diag or "error" in ncu_diag:
        return {
            "overall": "UNKNOWN", "findings": [],
            "error": ncu_diag.get("error") if isinstance(ncu_diag, dict) else "no diag",
        }

    kd = kernel_data_from_dict(ncu_diag)
    findings: List[Finding] = []
    findings.extend(_analyze_warp_stalls(kd))
    findings.extend(_analyze_occupancy(kd))
    findings.extend(_analyze_divergence(kd))
    findings.extend(_analyze_instruction_mix(kd))
    findings.extend(_analyze_triton_specific(kd))
    findings.sort(key=lambda f: -int(f.severity))

    return {
        "overall": _classify_overall(kd),
        "kernel_name": kd.kernel_name,
        "duration_us": kd.duration_us,
        "sm_pct": kd.sm_throughput_pct,
        "mem_pct": kd.mem_throughput_pct,
        "dram_pct": kd.dram_throughput_pct,
        "occupancy_pct": kd.warps_active_pct,
        "theoretical_occupancy_pct": kd.theoretical_occupancy_pct,
        "limiter": kd.occupancy_limiter()[0],
        "regs_per_thread": kd.registers_per_thread,
        "smem_kb_per_block": kd.shared_mem_per_block_kb,
        "pipe_tensor_pct": kd.pipe_tensor_pct,
        "pipe_fma_pct": kd.pipe_fma_pct,
        "stall_breakdown": kd.stall_breakdown(),
        "divergence_pct": kd.divergence_pct(),
        "findings": findings,
    }


def render_analysis(analysis: Dict[str, Any]) -> str:
    """Render the analysis as a multi-line prompt section."""
    if "error" in analysis:
        return f"  NCU diagnostic unavailable: {analysis['error']}"

    lines = [
        "  === Conclusion ===",
        f"  Kernel:    {analysis.get('kernel_name', '?')}",
        f"  Duration:  {analysis.get('duration_us', 0):.1f} us",
        f"  Overall:   {analysis['overall']}",
        f"  Roofline:  SM {analysis['sm_pct']:.1f}%, MEM {analysis['mem_pct']:.1f}%, "
        f"DRAM {analysis['dram_pct']:.1f}%",
        f"  Occupancy: {analysis['occupancy_pct']:.1f}% "
        f"(theoretical {analysis['theoretical_occupancy_pct']:.1f}%), "
        f"limited by {analysis['limiter']}",
        f"  Regs/Thread: {analysis['regs_per_thread']:.0f}, "
        f"Smem/Block: {analysis['smem_kb_per_block']:.1f} KB",
        f"  Pipe: Tensor {analysis['pipe_tensor_pct']:.1f}%, "
        f"FMA {analysis['pipe_fma_pct']:.1f}%",
    ]

    breakdown = analysis.get("stall_breakdown", [])
    if breakdown:
        top = ", ".join(f"{n}={pct:.1f}%" for n, _, pct in breakdown[:3])
        lines.append(f"  Top stalls: {top}")
    div = analysis.get("divergence_pct", 0.0)
    if div > 5:
        lines.append(f"  Divergence: {div:.1f}%")

    findings = analysis.get("findings", [])
    if findings:
        lines.append("")
        lines.append("  === Findings (severity-sorted) ===")
        for i, f in enumerate(findings, 1):
            sev_name = Severity(f.severity).name
            lines.append(f"  [{sev_name}] {f.title}")
            lines.append(f"    detail: {f.detail}")
            lines.append(f"    action: {f.action}")
    else:
        lines.append("")
        lines.append("  (No findings — kernel may already be near optimal "
                     "for this shape, or NCU metrics insufficient.)")

    return "\n".join(lines)
