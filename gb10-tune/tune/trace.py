"""Workspace path roots + load_definition + data-layer re-exports.

`RUNS_ROOT`, `DEFINITIONS_ROOT`, `TRACES_ROOT` are sourced from env vars with
container-mount defaults. `load_definition(name)` is the canonical way to
materialize a Definition by name.

The data-layer classes (Definition / Solution / Workload / Evaluation / …) are
re-exported here so `from tune.trace import Definition, ...` matches the
original explore.trace surface.
"""

import os
from pathlib import Path

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

RUNS_ROOT = Path(os.environ.get("GB10_RUNS", "/workspace/runs"))
DEFINITIONS_ROOT = Path(os.environ.get("GB10_DEFINITIONS", "/workspace/definitions"))
TRACES_ROOT = Path(os.environ.get("GB10_TRACES", "/workspace/traces"))


def load_definition(name: str) -> Definition:
    return load_json_file(Definition, DEFINITIONS_ROOT / f"{name}.json")
