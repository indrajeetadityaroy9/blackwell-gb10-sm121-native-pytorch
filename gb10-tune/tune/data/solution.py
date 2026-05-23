"""Solution — a concrete implementation of a Definition."""

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString


class SupportedLanguages(str, Enum):
    PYTHON = "python"
    TRITON = "triton"
    CPP = "cpp"
    CUDA = "cuda"
    TILELANG = "tilelang"


class SupportedBindings(str, Enum):
    TVM_FFI = "tvm-ffi"
    TORCH = "torch"


class SourceFile(BaseModelWithDocstrings):
    """A single source file in a Solution — relative path + inline content."""

    path: NonEmptyString
    content: NonEmptyString

    @model_validator(mode="after")
    def _validate_source_path(self) -> "SourceFile":
        p = Path(self.path)
        if p.is_absolute():
            raise ValueError(f"Invalid source path (absolute not allowed): {self.path}")
        if ".." in p.parts:
            raise ValueError(
                f"Invalid source path (parent traversal not allowed): {self.path}"
            )
        return self


class BuildSpec(BaseModelWithDocstrings):
    """How to build/run a Solution: language, hardware, entry point.

    entry_point format: '{file_path}::{function_name}' (e.g., 'candidate.py::run').
    """

    language: SupportedLanguages
    target_hardware: List[str] = Field(min_length=1)
    entry_point: NonEmptyString
    dependencies: List[NonEmptyString] = Field(default_factory=list)
    destination_passing_style: bool = True
    binding: Optional[SupportedBindings] = None

    @model_validator(mode="after")
    def _validate_entry_point(self) -> "BuildSpec":
        if self.entry_point.count("::") != 1:
            raise ValueError(
                f"Invalid entry_point '{self.entry_point}': expected 'file::function'"
            )
        return self


class Solution(BaseModelWithDocstrings):
    """A concrete implementation for a Definition. Frozen/immutable; content-hashed."""

    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)

    _hash_cache: str = PrivateAttr()

    name: NonEmptyString
    definition: NonEmptyString
    author: NonEmptyString
    spec: BuildSpec
    sources: List[SourceFile] = Field(min_length=1)
    description: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _validate_source_path_entry_point(self) -> "Solution":
        seen = set()
        for s in self.sources:
            if s.path in seen:
                raise ValueError(f"Duplicate source path '{s.path}'")
            seen.add(s.path)
        entry_file = self.spec.entry_point.split("::")[0]
        if entry_file not in seen:
            raise ValueError(f"Entry source '{entry_file}' not in sources")
        return self

    def get_entry_path(self) -> Path:
        return Path(self.spec.entry_point.split("::")[0])

    def get_entry_symbol(self) -> str:
        return self.spec.entry_point.split("::")[-1]

    def get_entry_source(self) -> SourceFile:
        entry_path = self.spec.entry_point.split("::")[0]
        for s in self.sources:
            if s.path == entry_path:
                return s
        raise ValueError(f"Entry source '{entry_path}' not found")

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_hash_cache", self._compute_hash())

    def _compute_hash(self) -> str:
        h = hashlib.sha1()
        for s in (
            self.definition,
            self.spec.language.value,
            self.spec.entry_point,
            self.spec.binding.value if self.spec.binding else "",
            *self.spec.dependencies,
            *(part for src in self.sources for part in (src.path, src.content)),
        ):
            h.update(s.encode())
        return h.hexdigest()

    def hash(self) -> str:
        return self._hash_cache

    def __hash__(self) -> int:
        return hash(self._hash_cache)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Solution) and self._hash_cache == other._hash_cache
