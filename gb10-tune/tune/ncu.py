"""NCU report parsing — extracts the metric set used by the diagnostic engine
to produce evidence-cited findings (modeled on cuda_auto_tune/ncu_analyse.py).

Reads the .ncu-rep binary via the `ncu_report` Python API. Requires
`ncu --set full` so the heavier-cost metrics (occupancy, stalls, pipes,
divergence) are present; `--set roofline` only populates sm/mem throughput.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


_M_DURATION_US = "gpu__time_duration.sum"
_M_SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_M_MEM = "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed"
_M_DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
_M_DRAM_R = "dram__bytes_read.sum"
_M_DRAM_W = "dram__bytes_write.sum"
_M_L1_HIT = "l1tex__t_sector_hit_rate.pct"
_M_L2_HIT = "lts__t_sector_hit_rate.pct"
_M_SHMEM_BANK_CONFLICTS = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
_M_LOCAL_STORE_SECTORS = "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
_M_WARPS_ACTIVE = "sm__warps_active.avg.pct_of_peak_sustained_active"
_M_REGS_PER_THREAD = "launch__registers_per_thread"
_M_SMEM_PER_BLOCK = "launch__shared_mem_per_block"
_M_OCC_LIMIT_REGS = "launch__occupancy_limit_registers"
_M_OCC_LIMIT_SHMEM = "launch__occupancy_limit_shared_mem"
_M_OCC_LIMIT_WARPS = "launch__occupancy_limit_warps"
_M_OCC_LIMIT_BLOCKS = "launch__occupancy_limit_blocks"
_M_THEORETICAL_OCC = "sm__maximum_warps_per_active_cycle_pct"
_M_WARPS_ELIGIBLE = "smsp__warps_eligible.avg.per_cycle_active"

_M_STALL_LONG_SB = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
_M_STALL_SHORT_SB = "smsp__pcsamp_warps_issue_stalled_short_scoreboard"
_M_STALL_WAIT = "smsp__pcsamp_warps_issue_stalled_wait"
_M_STALL_SLEEPING = "smsp__pcsamp_warps_issue_stalled_sleeping"
_M_STALL_BARRIER = "smsp__pcsamp_warps_issue_stalled_barrier"
_M_STALL_MIO_THROTTLE = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
_M_STALL_LG_THROTTLE = "smsp__pcsamp_warps_issue_stalled_lg_throttle"
_M_STALL_MATH_THROTTLE = "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"
_M_STALL_DRAIN = "smsp__pcsamp_warps_issue_stalled_drain"
_M_STALL_NOT_SELECTED = "smsp__pcsamp_warps_issue_stalled_not_selected"
_M_STALL_SELECTED = "smsp__pcsamp_warps_issue_stalled_selected"

_M_PIPE_FMA = "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active"
_M_PIPE_ALU = "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active"
_M_PIPE_LSU = "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"
_M_PIPE_TENSOR = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
_M_PIPE_TENSOR_HMMA = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
_M_PIPE_FMA_FP16 = "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active"

_M_AVG_THREAD_EXECUTED = "smsp__thread_inst_executed_per_inst_executed.ratio"
_M_AVG_THREAD_EXECUTED_TRUE = "smsp__thread_inst_executed_per_inst_executed.pct"

# Load efficiency (PoC §4B): avg bytes-per-sector for global loads / peak.
# Computed as ratio / max_rate * 100 (no `.pct` field exists on sm_121a).
# 100% = perfectly coalesced 32-byte sector loads.
_M_LOAD_EFFICIENCY_RATIO = "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio"
_M_LOAD_EFFICIENCY_MAX = "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.max_rate"


@dataclass
class KernelData:
    """All NCU metrics for one kernel launch. Field names mirror cuda_auto_tune."""

    kernel_name: str = ""
    grid_size: str = ""
    block_size: str = ""

    duration_us: float = 0.0
    sm_throughput_pct: float = 0.0
    mem_throughput_pct: float = 0.0
    dram_throughput_pct: float = 0.0

    l1_hit_rate_pct: float = 0.0
    l2_hit_rate_pct: float = 0.0
    shared_mem_bank_conflicts: float = 0.0
    local_mem_store_sectors: float = 0.0
    dram_read_gbytes: float = 0.0
    dram_write_gbytes: float = 0.0

    warps_active_pct: float = 0.0
    registers_per_thread: float = 0.0
    shared_mem_per_block_kb: float = 0.0
    occupancy_limit_registers: float = 0.0
    occupancy_limit_shared_mem: float = 0.0
    occupancy_limit_warps: float = 0.0
    occupancy_limit_blocks: float = 0.0
    theoretical_occupancy_pct: float = 0.0
    warps_eligible_per_cycle: float = 0.0

    stall_long_scoreboard: float = 0.0
    stall_short_scoreboard: float = 0.0
    stall_wait: float = 0.0
    stall_sleeping: float = 0.0
    stall_barrier: float = 0.0
    stall_mio_throttle: float = 0.0
    stall_lg_throttle: float = 0.0
    stall_math_pipe_throttle: float = 0.0
    stall_drain: float = 0.0
    stall_not_selected: float = 0.0
    stall_selected: float = 0.0

    pipe_fma_pct: float = 0.0
    pipe_alu_pct: float = 0.0
    pipe_lsu_pct: float = 0.0
    pipe_tensor_pct: float = 0.0
    pipe_tensor_hmma_pct: float = 0.0
    pipe_fma_fp16_pct: float = 0.0

    avg_thread_executed: float = 0.0
    avg_thread_executed_true: float = 0.0

    raw: Dict[str, float] = field(default_factory=dict)

    def total_stall_samples(self) -> float:
        return (
            self.stall_long_scoreboard + self.stall_short_scoreboard
            + self.stall_wait + self.stall_sleeping + self.stall_barrier
            + self.stall_mio_throttle + self.stall_lg_throttle
            + self.stall_math_pipe_throttle + self.stall_drain
            + self.stall_not_selected + self.stall_selected
        )

    def stall_breakdown(self) -> List[Tuple[str, float, float]]:
        """Returns [(name, count, pct)] sorted by count desc, filtered to nonzero."""
        total = self.total_stall_samples()
        if total == 0:
            return []
        reasons = [
            ("Long Scoreboard", self.stall_long_scoreboard),
            ("Short Scoreboard", self.stall_short_scoreboard),
            ("Wait", self.stall_wait),
            ("Sleeping", self.stall_sleeping),
            ("Barrier", self.stall_barrier),
            ("MIO Throttle", self.stall_mio_throttle),
            ("LG Throttle", self.stall_lg_throttle),
            ("Math Pipe Throttle", self.stall_math_pipe_throttle),
            ("Drain", self.stall_drain),
            ("Not Selected", self.stall_not_selected),
            ("Selected", self.stall_selected),
        ]
        out = [(n, c, c / total * 100) for n, c in reasons if c > 0]
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def occupancy_limiter(self) -> Tuple[str, float]:
        """Returns (limiter_name, blocks_per_sm) for the SMALLEST of the four limits."""
        limits = [
            ("Registers", self.occupancy_limit_registers),
            ("Shared Memory", self.occupancy_limit_shared_mem),
            ("Warps", self.occupancy_limit_warps),
            ("Blocks", self.occupancy_limit_blocks),
        ]
        nonzero = [(n, v) for n, v in limits if v > 0]
        if not nonzero:
            return ("Unknown", 0.0)
        return min(nonzero, key=lambda t: t[1])

    def divergence_pct(self) -> float:
        """Fraction of issued threads that were inactive (predicated off)."""
        if self.avg_thread_executed <= 0:
            return 0.0
        diff = self.avg_thread_executed - self.avg_thread_executed_true
        if diff <= 0:
            return 0.0
        return diff / self.avg_thread_executed * 100

    def load_coalescing_ratio(self) -> float:
        """Sectors-per-request — proxy for non-coalesced loads. Not available
        without explicit metrics; returns 0 if unset."""
        return float(self.raw.get("load_coalescing_ratio", 0.0))


def _metric(act, name: str) -> float:
    """Look up a metric by name; return 0.0 if not present or non-numeric."""
    m = act.metric_by_name(name)
    if m is None:
        return 0.0
    return m.as_double()


def parse_report(rep_path: Path) -> Dict[str, Any]:
    """Parse a .ncu-rep file via ncu_report API. Returns the longest-duration
    kernel's metrics as a dict. The dict mirrors KernelData fields so the
    diagnostic engine can consume it directly.

    Backward-compat: the old fields (kernel_name, duration_ms, sol_sm_pct,
    sol_mem_pct, limit) are still present so the loop's earlier consumers
    (build_prompt's NCU section) keep working.
    """
    import ncu_report

    ctx = ncu_report.load_report(str(rep_path))
    best: Dict[str, Any] = {}
    best_duration_us = -1.0

    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            act = rng.action_by_idx(ai)
            duration_us = _metric(act, _M_DURATION_US) / 1e3   # ns → us
            if duration_us <= best_duration_us:
                continue

            kd = KernelData(
                kernel_name=act.name(),
                duration_us=duration_us,
                sm_throughput_pct=_metric(act, _M_SM),
                mem_throughput_pct=_metric(act, _M_MEM),
                dram_throughput_pct=_metric(act, _M_DRAM),
                dram_read_gbytes=_metric(act, _M_DRAM_R) / 1e9,
                dram_write_gbytes=_metric(act, _M_DRAM_W) / 1e9,
                l1_hit_rate_pct=_metric(act, _M_L1_HIT),
                l2_hit_rate_pct=_metric(act, _M_L2_HIT),
                shared_mem_bank_conflicts=_metric(act, _M_SHMEM_BANK_CONFLICTS),
                local_mem_store_sectors=_metric(act, _M_LOCAL_STORE_SECTORS),
                warps_active_pct=_metric(act, _M_WARPS_ACTIVE),
                registers_per_thread=_metric(act, _M_REGS_PER_THREAD),
                shared_mem_per_block_kb=_metric(act, _M_SMEM_PER_BLOCK) / 1024.0,
                occupancy_limit_registers=_metric(act, _M_OCC_LIMIT_REGS),
                occupancy_limit_shared_mem=_metric(act, _M_OCC_LIMIT_SHMEM),
                occupancy_limit_warps=_metric(act, _M_OCC_LIMIT_WARPS),
                occupancy_limit_blocks=_metric(act, _M_OCC_LIMIT_BLOCKS),
                theoretical_occupancy_pct=_metric(act, _M_THEORETICAL_OCC),
                warps_eligible_per_cycle=_metric(act, _M_WARPS_ELIGIBLE),
                stall_long_scoreboard=_metric(act, _M_STALL_LONG_SB),
                stall_short_scoreboard=_metric(act, _M_STALL_SHORT_SB),
                stall_wait=_metric(act, _M_STALL_WAIT),
                stall_sleeping=_metric(act, _M_STALL_SLEEPING),
                stall_barrier=_metric(act, _M_STALL_BARRIER),
                stall_mio_throttle=_metric(act, _M_STALL_MIO_THROTTLE),
                stall_lg_throttle=_metric(act, _M_STALL_LG_THROTTLE),
                stall_math_pipe_throttle=_metric(act, _M_STALL_MATH_THROTTLE),
                stall_drain=_metric(act, _M_STALL_DRAIN),
                stall_not_selected=_metric(act, _M_STALL_NOT_SELECTED),
                stall_selected=_metric(act, _M_STALL_SELECTED),
                pipe_fma_pct=_metric(act, _M_PIPE_FMA),
                pipe_alu_pct=_metric(act, _M_PIPE_ALU),
                pipe_lsu_pct=_metric(act, _M_PIPE_LSU),
                pipe_tensor_pct=_metric(act, _M_PIPE_TENSOR),
                pipe_tensor_hmma_pct=_metric(act, _M_PIPE_TENSOR_HMMA),
                pipe_fma_fp16_pct=_metric(act, _M_PIPE_FMA_FP16),
                avg_thread_executed=_metric(act, _M_AVG_THREAD_EXECUTED),
                avg_thread_executed_true=_metric(act, _M_AVG_THREAD_EXECUTED_TRUE),
            )
            # Stash load efficiency (PoC §4B): ratio / max_rate * 100.
            _ld_ratio = _metric(act, _M_LOAD_EFFICIENCY_RATIO)
            _ld_max = _metric(act, _M_LOAD_EFFICIENCY_MAX)
            kd.raw["load_efficiency_pct"] = (
                _ld_ratio / _ld_max * 100.0 if _ld_max > 0 else 0.0
            )

            # Flatten into a dict the diagnostic + prompt code can consume.
            # Backward-compat fields first:
            out: Dict[str, Any] = {
                "kernel_name": kd.kernel_name,
                "duration_ms": kd.duration_us / 1000.0,
                "sol_sm_pct": kd.sm_throughput_pct,
                "sol_mem_pct": kd.mem_throughput_pct,
                "limit": "compute" if kd.sm_throughput_pct >= kd.mem_throughput_pct else "bandwidth",
            }
            # All other fields by attribute name:
            for f in KernelData.__dataclass_fields__:
                if f in ("kernel_name", "raw"):
                    continue
                out[f] = getattr(kd, f)

            out["__stall_breakdown"] = kd.stall_breakdown()
            out["__occupancy_limiter"] = list(kd.occupancy_limiter())
            out["__divergence_pct"] = kd.divergence_pct()
            out["load_efficiency_pct"] = kd.raw.get("load_efficiency_pct", 0.0)

            best = out
            best_duration_us = duration_us

    return best


def kernel_data_from_dict(d: Dict[str, Any]) -> KernelData:
    """Reconstruct a KernelData from the dict produced by parse_report.

    Used by the diagnostic engine when consuming an Evaluation's ncu_diag.
    """
    kd = KernelData()
    for f in KernelData.__dataclass_fields__:
        if f == "raw":
            continue
        if f in d:
            setattr(kd, f, d[f])
    return kd
