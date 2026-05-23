"""Workload — concrete input specs for benchmarking."""

from typing import Dict, Literal, Union

from .utils import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt


class RandomInput(BaseModelWithDocstrings):
    """Random tensor input generated at workload time."""

    type: Literal["random"] = "random"


class ScalarInput(BaseModelWithDocstrings):
    """Scalar literal input (int / float / bool)."""

    type: Literal["scalar"] = "scalar"
    value: Union[int, float, bool]


class SafetensorsInput(BaseModelWithDocstrings):
    """Tensor loaded from a safetensors file at the given path/key."""

    type: Literal["safetensors"] = "safetensors"
    path: NonEmptyString
    tensor_key: NonEmptyString


InputSpec = Union[RandomInput, SafetensorsInput, ScalarInput]


class Workload(BaseModelWithDocstrings):
    """Concrete workload configuration for benchmarking.

    Binds variable axes to integers and inputs to specific data sources.
    """

    axes: Dict[str, NonNegativeInt]
    inputs: Dict[str, InputSpec]
    uuid: NonEmptyString
