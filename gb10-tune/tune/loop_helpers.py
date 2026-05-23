"""Helpers shared by stage1.py, stage2.py, tune.py.

- new_run_id: timestamp-based unique workspace dir name.
- _build_solution_for_runner / _build_workload_for_runner: pack a Solution +
  Workload for the NCU subprocess.
- ncu_profile_subprocess: spawn ncu --set full + _solution_runner; return
  parsed metric dict. Raises on any failure — single-path execution.
- promote_to_traces: write the accepted candidate source to
  traces/<def_name>/best.py (atomic write via os.replace).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .data import (
    BuildSpec,
    Definition,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    Workload,
    save_json_file,
)
from .ncu import parse_report


def new_run_id(definition: Definition) -> str:
    return f"{definition.name}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _build_solution_for_runner(definition: Definition, candidate_src: str) -> Solution:
    sha = hashlib.sha256(candidate_src.encode()).hexdigest()
    return Solution(
        name=f"_iter_{sha[:12]}",
        definition=definition.name,
        author="tune",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=["sm_121a"],
            entry_point="candidate.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="candidate.py", content=candidate_src)],
    )


def _build_workload_for_runner(definition: Definition) -> Workload:
    return Workload(
        axes={},
        inputs={k: RandomInput() for k in definition.inputs.keys()},
        uuid=f"tune_{hashlib.sha256(definition.name.encode()).hexdigest()[:8]}",
    )


def ncu_profile_subprocess(
    candidate_src: str, definition: Definition
) -> Dict[str, Any]:
    """Run ncu --set full against a fresh _solution_runner subprocess on
    candidate_src; parse the .ncu-rep via ncu_report.

    NCU --set full collects roofline + occupancy + 11 stall categories + 5 pipe
    utilizations + divergence + smem/regs/limiters — the full metric set the
    diagnostic engine consumes. Roughly 30-60 s/invocation overhead.
    """
    with tempfile.TemporaryDirectory(prefix="gb10_tune_ncu_") as tmp:
        tmp = Path(tmp)
        solution = _build_solution_for_runner(definition, candidate_src)
        workload = _build_workload_for_runner(definition)
        save_json_file(definition, tmp / "definition.json")
        save_json_file(solution, tmp / "solution.json")
        save_json_file(workload, tmp / "workload.json")
        rep = tmp / "report.ncu-rep"
        cmd = [
            "ncu", "--set", "full",
            "--nvtx", "--nvtx-include", "gb10_tune_profile]",
            "--launch-count", "30", "--target-processes", "all",
            "--force-overwrite", "--export", str(rep),
            "--", sys.executable, "-u", "-m", "tune.runner._solution_runner",
            "--data-dir", str(tmp),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return parse_report(rep)


def promote_to_traces(
    definition: Definition,
    candidate_src: str,
    traces_root: Path,
) -> None:
    """Write the accepted candidate source to traces/<def_name>/best.py
    (atomic via tempfile + os.replace).
    """
    def_dir = traces_root / definition.name
    def_dir.mkdir(parents=True, exist_ok=True)
    target = def_dir / "best.py"
    tmp = def_dir / "best.py.tmp"
    tmp.write_text(candidate_src)
    os.replace(tmp, target)
