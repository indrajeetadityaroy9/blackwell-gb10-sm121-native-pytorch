"""Public re-export façade for the data layer.

`from explore.trace import Definition, Solution, Workload, Evaluation, ...`
"""

from .data import (
    AxisConst,
    AxisSpec,
    AxisVar,
    BuildSpec,
    Correctness,
    Definition,
    DType,
    Environment,
    Evaluation,
    EvaluationStatus,
    InputSpec,
    Performance,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedLanguages,
    TensorSpec,
    Trace,
    Workload,
    dtype_str_to_torch_dtype,
    env_snapshot,
    load_json_file,
    save_json_file,
)

# Module-level paths consumed by load_definition / promote_to_traces / agent_loop.
# Imported here (not just in loop.py) so any module can read them.
import os
from pathlib import Path

RUNS_ROOT = Path(os.environ.get("GB10_RUNS", "/workspace/runs"))
DEFINITIONS_ROOT = Path(os.environ.get("GB10_DEFINITIONS", "/workspace/definitions"))
TRACES_ROOT = Path(os.environ.get("GB10_TRACES", "/workspace/traces"))


def load_definition(name: str) -> Definition:
    """Read DEFINITIONS_ROOT / f'{name}.json' and parse as Definition."""
    return load_json_file(Definition, DEFINITIONS_ROOT / f"{name}.json")


__all__ = [
    "AxisConst",
    "AxisSpec",
    "AxisVar",
    "BuildSpec",
    "Correctness",
    "DEFINITIONS_ROOT",
    "Definition",
    "DType",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    "InputSpec",
    "Performance",
    "RUNS_ROOT",
    "RandomInput",
    "SafetensorsInput",
    "ScalarInput",
    "Solution",
    "SourceFile",
    "SupportedBindings",
    "SupportedLanguages",
    "TRACES_ROOT",
    "TensorSpec",
    "Trace",
    "Workload",
    "dtype_str_to_torch_dtype",
    "env_snapshot",
    "load_definition",
    "load_json_file",
    "save_json_file",
]
