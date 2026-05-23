"""Runnable — callable wrapper around a compiled Solution."""

from typing import Any, Callable, Optional

from pydantic import BaseModel

from ..data import Definition


class RunnableMetadata(BaseModel):
    build_type: str
    definition_name: str
    solution_name: str
    destination_passing_style: bool = False
    definition: Optional[Definition] = None

    model_config = {"arbitrary_types_allowed": True}


class Runnable:
    """Callable wrapper around a compiled Solution. Triton solutions are
    value-returning (`def run(*inputs) -> output`); destination_passing_style=False
    is the only mode we use."""

    def __init__(
        self,
        callable: Callable[..., Any],
        metadata: RunnableMetadata,
        cleaner: Optional[Callable[[], None]] = None,
    ) -> None:
        self._callable = callable
        self.metadata = metadata
        self._cleaner = cleaner

    def __call__(self, *args, **kwargs):
        return self._callable(*args, **kwargs)

    def cleanup(self) -> None:
        if self._cleaner is not None:
            self._cleaner()
