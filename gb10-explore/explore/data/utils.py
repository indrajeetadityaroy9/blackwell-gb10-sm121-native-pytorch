"""Common utilities and base classes for data models."""

from functools import cache
from typing import Annotated, Dict

import torch
from pydantic import BaseModel, ConfigDict, Field

NonEmptyString = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class BaseModelWithDocstrings(BaseModel):
    """Base model exposing attribute docstrings to the model JSON schema."""

    model_config = ConfigDict(use_attribute_docstrings=True)


@cache
def _dtype_str_to_torch_dtype_map() -> Dict[str, torch.dtype]:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float4_e2m1": torch.float4_e2m1fn_x2,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "bool": torch.bool,
    }


def dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    return _dtype_str_to_torch_dtype_map()[dtype_str]


def env_snapshot(device: str):
    """Record torch/triton/cuda versions + device name. Returns Environment."""
    from .trace import Environment

    import triton

    libs: Dict[str, str] = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
    }
    hardware = torch.cuda.get_device_name(torch.device(device).index)
    return Environment(hardware=hardware, libs=libs)
