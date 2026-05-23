"""Template ABC — minimal contract for a parameterized Triton kernel emitter.

A Template emits Triton source for one parameter tuple at a time. The grid is
the template's job (so different templates can enforce their own
hardware/shape feasibility filters); render() turns one tuple into a complete
`def run(...)` source string.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple


class Template(ABC):
    name: str
    allowed_action: str

    @abstractmethod
    def parameter_grid(self) -> Iterator[Tuple]:
        """Yields only configurations that pass the template's feasibility filter
        (shared-memory budget, register caps, shape divisibility, etc.)."""

    @abstractmethod
    def render(self, params: Tuple) -> str:
        """Emit Triton source defining `def run(...)` for the given parameters."""
