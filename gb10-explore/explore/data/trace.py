"""Trace + Evaluation + ancillary metric structs."""

import math
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString
from .workload import Workload


class Correctness(BaseModelWithDocstrings):
    model_config = ConfigDict(ser_json_inf_nan="strings")

    max_relative_error: float = Field(default=0.0)
    max_absolute_error: float = Field(default=0.0)
    extra: Optional[Dict[str, Any]] = Field(default=None)
    """Extra metrics — GB10 stuffs ncu_diag here under key 'ncu_diag'."""

    @field_validator("max_relative_error", "max_absolute_error")
    @classmethod
    def _non_negative_or_nan(cls, v: float) -> float:
        if math.isnan(v):
            return v
        if v < 0:
            raise ValueError("must be non-negative or NaN")
        return v


class Performance(BaseModelWithDocstrings):
    latency_ms: float = Field(default=0.0, ge=0.0)
    reference_latency_ms: float = Field(default=0.0, ge=0.0)
    speedup_factor: float = Field(default=0.0, ge=0.0)
    tflops: float = Field(default=0.0, ge=0.0)
    """GB10 addition: derived from 2*M*N*K / latency_ms / 1e9 (TFLOPs/s)."""
    p10_ms: float = Field(default=0.0, ge=0.0)
    p90_ms: float = Field(default=0.0, ge=0.0)
    stdev_pct: float = Field(default=0.0, ge=0.0)
    n_iters: int = Field(default=0, ge=0)


class Environment(BaseModelWithDocstrings):
    hardware: NonEmptyString
    libs: Dict[str, str] = Field(default_factory=dict)


class EvaluationStatus(str, Enum):
    PASSED = "PASSED"
    INCORRECT_SHAPE = "INCORRECT_SHAPE"
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    INCORRECT_DTYPE = "INCORRECT_DTYPE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMPILE_ERROR = "COMPILE_ERROR"
    TIMEOUT = "TIMEOUT"


class Evaluation(BaseModelWithDocstrings):
    status: EvaluationStatus
    environment: Environment
    timestamp: NonEmptyString
    log: str = ""
    extra_msg: str = ""
    """GB10 addition: short failure-mode hint for the agent prompt (first 200 chars
    of traceback / numerical-mismatch summary). Empty for PASSED."""
    correctness: Optional[Correctness] = None
    performance: Optional[Performance] = None

    @model_validator(mode="after")
    def _validate_status_metrics(self) -> "Evaluation":
        if self.status == EvaluationStatus.PASSED:
            if self.correctness is None or self.performance is None:
                raise ValueError(
                    f"PASSED requires both correctness and performance"
                )
        elif self.status == EvaluationStatus.INCORRECT_NUMERICAL:
            if self.correctness is None:
                raise ValueError("INCORRECT_NUMERICAL requires correctness")
            if self.performance is not None:
                raise ValueError("INCORRECT_NUMERICAL must not include performance")
        else:
            if self.correctness is not None or self.performance is not None:
                raise ValueError(
                    f"{self.status} must not include correctness/performance"
                )
        return self


class Trace(BaseModelWithDocstrings):
    """Definition + Workload + (Solution, Evaluation) link."""

    definition: NonEmptyString
    workload: Workload
    solution: Optional[str] = None
    evaluation: Optional[Evaluation] = None

    def is_workload_trace(self) -> bool:
        return self.solution is None and self.evaluation is None

    def is_successful(self) -> bool:
        return (
            not self.is_workload_trace()
            and self.evaluation.status == EvaluationStatus.PASSED
        )
