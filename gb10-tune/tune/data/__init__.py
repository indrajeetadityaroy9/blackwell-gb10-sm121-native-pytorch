"""Strongly-typed data models for GB10-tune. Ported from flashinfer_bench/data/.

Schema additions over FIB:
- Definition.validator_class: Literal["deterministic","matched_ratio","matched_ratio_loose"]
  selects the per-dtype evaluator (spec §5).
"""

from .definition import AxisConst, AxisSpec, AxisVar, Definition, DType, TensorSpec
from .json_utils import load_json_file, save_json_file
from .solution import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedLanguages,
)
from .trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from .utils import (
    BaseModelWithDocstrings,
    NonEmptyString,
    NonNegativeInt,
    dtype_str_to_torch_dtype,
    env_snapshot,
)
from .workload import InputSpec, RandomInput, SafetensorsInput, ScalarInput, Workload

__all__ = [
    "AxisConst",
    "AxisSpec",
    "AxisVar",
    "BaseModelWithDocstrings",
    "BuildSpec",
    "Correctness",
    "Definition",
    "DType",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    "InputSpec",
    "NonEmptyString",
    "NonNegativeInt",
    "Performance",
    "RandomInput",
    "SafetensorsInput",
    "ScalarInput",
    "Solution",
    "SourceFile",
    "SupportedBindings",
    "SupportedLanguages",
    "TensorSpec",
    "Trace",
    "Workload",
    "dtype_str_to_torch_dtype",
    "env_snapshot",
    "load_json_file",
    "save_json_file",
]
