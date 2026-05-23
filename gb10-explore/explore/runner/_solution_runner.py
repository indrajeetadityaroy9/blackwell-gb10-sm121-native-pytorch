"""NCU-targeted subprocess entry point.

Reads definition.json/solution.json/workload.json from --data-dir, builds the
Solution, runs it once inside an NVTX range named 'gb10_explore_profile' so
`ncu --nvtx --nvtx-include 'gb10_explore_profile]'` can filter for it.

Single-shot, single-path: any failure exits non-zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

from ..bench import generate_inputs
from ..compile import BuilderRegistry
from ..data import Definition, Solution, Workload, load_json_file


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    data_dir: Path = args.data_dir
    definition = load_json_file(Definition, data_dir / "definition.json")
    solution = load_json_file(Solution, data_dir / "solution.json")
    workload = load_json_file(Workload, data_dir / "workload.json")

    runnable = BuilderRegistry.get_instance().build(definition, solution)
    inputs = generate_inputs(definition, device=args.device)

    # Warmup outside the NVTX range so NCU filtering only catches the timed call.
    with torch.no_grad():
        runnable(*inputs)
    torch.cuda.synchronize(device=args.device)

    with torch.cuda.nvtx.range("gb10_explore_profile"):
        with torch.no_grad():
            runnable(*inputs)
        torch.cuda.synchronize(device=args.device)

    runnable.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
