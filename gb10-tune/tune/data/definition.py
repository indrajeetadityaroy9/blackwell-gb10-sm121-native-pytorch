"""Kernel Definition + axis/tensor specs."""

from __future__ import annotations

import ast
from enum import Enum
from functools import cached_property
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import torch
from pydantic import BaseModel, Field, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt, dtype_str_to_torch_dtype

# GB10 validator dispatch — selects the per-dtype evaluator (spec §5).
ValidatorClass = Literal["deterministic", "matched_ratio", "matched_ratio_loose", "stochastic"]


class AxisConst(BaseModelWithDocstrings):
    """A compile-time-known dimension."""

    type: Literal["const"] = "const"
    value: NonNegativeInt
    description: Optional[str] = None


class AxisVar(BaseModel):
    """A runtime-bound dimension."""

    type: Literal["var"] = "var"
    description: Optional[str] = Field(default=None)


class DType(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT4_E2M1 = "float4_e2m1"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    BOOL = "bool"


class TensorSpec(BaseModelWithDocstrings):
    """Symbolic tensor shape + dtype. shape=None for scalars."""

    shape: Optional[List[NonEmptyString]]
    dtype: DType
    description: Optional[str] = None


AxisSpec = Union[AxisConst, AxisVar]


class Definition(BaseModelWithDocstrings):
    """Formal contract for a computational workload.

    GB10 additions:
    - validator_class: which per-dtype evaluator selects this kernel's correctness gate.
    """

    name: NonEmptyString
    op_type: NonEmptyString
    axes: Dict[NonEmptyString, Union[AxisConst, AxisVar]]
    inputs: Dict[NonEmptyString, TensorSpec]
    outputs: Dict[NonEmptyString, TensorSpec]
    reference: NonEmptyString
    perf_baseline: Optional[str] = None
    """Optional native-dtype eager baseline for fast_p speedup timing (defines `run`).

    `reference` is the correctness oracle (e.g., fp32-accumulated matmul) and must stay
    high-precision. fast_p speedup, however, must be measured against the realistic
    competitive baseline — PyTorch eager / cuBLAS in native dtype — per KernelBench and
    FlashInfer-Bench. When None, fast_p falls back to `reference` (correct only when the
    reference is already the native-dtype eager op, e.g., a reduction `x.sum()`)."""
    validator_class: ValidatorClass = "deterministic"
    tags: List[NonEmptyString] = Field(default_factory=list)
    description: Optional[str] = Field(default=None)
    constraints: List[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_reference_code(self) -> "Definition":
        mod = ast.parse(self.reference, mode="exec")
        has_run = any(
            isinstance(n, ast.FunctionDef) and n.name == "run" for n in mod.body
        )
        if not has_run:
            raise ValueError("Reference must define a top-level function named 'run'")
        return self

    @model_validator(mode="after")
    def _validate_perf_baseline_code(self) -> "Definition":
        if self.perf_baseline is None:
            return self
        mod = ast.parse(self.perf_baseline, mode="exec")
        if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in mod.body):
            raise ValueError("perf_baseline must define a top-level function named 'run'")
        return self

    @model_validator(mode="after")
    def _validate_input_output_names(self) -> "Definition":
        if set(self.inputs.keys()) & set(self.outputs.keys()):
            raise ValueError("Input and output names must not overlap")
        return self

    @model_validator(mode="after")
    def _validate_constraints_syntax(self) -> "Definition":
        for c in self.constraints:
            ast.parse(c, mode="eval")
        return self

    @model_validator(mode="after")
    def _validate_tensor_axis_references(self) -> "Definition":
        all_tensors = {**self.inputs, **self.outputs}
        for tensor_name, spec in all_tensors.items():
            if spec.shape is None:
                continue
            for axis_name in spec.shape:
                if axis_name not in self.axes:
                    role = "Input" if tensor_name in self.inputs else "Output"
                    raise ValueError(
                        f'{role} "{tensor_name}" references undefined axis "{axis_name}"'
                    )
        return self

    # Op taxonomy — single owner of op-type dispatch, read by correctness.py and ncu.py.
    @property
    def is_gemm(self) -> bool:
        return self.op_type.lower() == "gemm"

    @property
    def is_reduction(self) -> bool:
        return self.op_type.lower() in ("reduction", "reduce", "sum")

    @property
    def is_norm(self) -> bool:
        return self.op_type.lower() in ("rmsnorm", "layernorm", "norm")

    @cached_property
    def const_axes(self) -> Dict[str, int]:
        return {n: ax.value for n, ax in self.axes.items() if isinstance(ax, AxisConst)}

    @cached_property
    def var_axes(self) -> List[str]:
        return [n for n, ax in self.axes.items() if isinstance(ax, AxisVar)]

    @cached_property
    def var_axes_bindings(self) -> Dict[str, Tuple[str, int]]:
        bindings: Dict[str, Tuple[str, int]] = {}
        for inp_name, spec in self.inputs.items():
            if spec.shape is None:
                continue
            for dim_idx, axis in enumerate(spec.shape):
                ax_def = self.axes.get(axis)
                if isinstance(ax_def, AxisVar) and axis not in bindings:
                    bindings[axis] = (inp_name, dim_idx)
        return bindings

    def get_axes_values(
        self, input_shapes: Iterable[Optional[Tuple[int, ...]]]
    ) -> Dict[str, int]:
        var_vals: Dict[str, int] = {}
        for (inp_name, inp_spec), inp_shape in zip(self.inputs.items(), input_shapes):
            if inp_spec.shape is None:
                continue
            if len(inp_spec.shape) != len(inp_shape):
                raise ValueError(
                    f"Input '{inp_name}': defined ndim {len(inp_spec.shape)} != actual {len(inp_shape)}"
                )
            for axis_name, axis_value in zip(inp_spec.shape, inp_shape):
                if axis_name in self.axes and self.axes[axis_name].type == "var":
                    if axis_name in var_vals and var_vals[axis_name] != axis_value:
                        raise ValueError(
                            f"Axis '{axis_name}' bound to {var_vals[axis_name]} but later got {axis_value}"
                        )
                    var_vals[axis_name] = axis_value
        missing = set(self.var_axes) - set(var_vals.keys())
        if missing:
            raise ValueError(f"Missing values for variable axes: {missing}")
        return var_vals

    def get_axes_values_from_inputs(self, inputs: Iterable[Any]) -> Dict[str, int]:
        shapes = [tuple(v.shape) if hasattr(v, "shape") else None for v in inputs]
        return self.get_axes_values(shapes)

    def _get_shapes(
        self,
        tensors: Iterable[TensorSpec],
        var_axes_values: Optional[Dict[str, int]] = None,
    ) -> List[Optional[Tuple[int, ...]]]:
        var_axes_values = var_axes_values or {}
        shapes: List[Optional[Tuple[int, ...]]] = []
        for spec in tensors:
            if spec.shape is None:
                shapes.append(None)
                continue
            shape: List[int] = []
            for axis_name in spec.shape:
                axis = self.axes[axis_name]
                if isinstance(axis, AxisConst):
                    shape.append(axis.value)
                else:
                    shape.append(var_axes_values[axis_name])
            shapes.append(tuple(shape))
        return shapes

    def get_input_shapes(
        self, var_axes_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
        return self._get_shapes(self.inputs.values(), var_axes_values)

    def get_output_shapes(
        self, var_axes_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
        return self._get_shapes(self.outputs.values(), var_axes_values)

    @cached_property
    def torch_input_dtypes(self) -> List[torch.dtype]:
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.inputs.values()]

    @cached_property
    def torch_output_dtypes(self) -> List[torch.dtype]:
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.outputs.values()]
