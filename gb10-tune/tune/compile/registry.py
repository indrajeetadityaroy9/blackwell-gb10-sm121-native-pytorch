"""BuilderRegistry — dispatches Definition+Solution to a Runnable.

Single canonical path: every Solution has spec.language ∈ {python, triton} and
is built via the same importlib loader. The 'triton' vs 'python' distinction is
metadata-only — both call into Python files that may import triton.
"""

import importlib.util
import sys
import tempfile
import uuid
from pathlib import Path
from typing import ClassVar, Dict, Optional

from ..data import Definition, Solution
from .runnable import Runnable, RunnableMetadata


class BuildError(RuntimeError):
    pass


class BuilderRegistry:
    _instance: ClassVar[Optional["BuilderRegistry"]] = None

    @classmethod
    def get_instance(cls) -> "BuilderRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._cache: Dict[str, Runnable] = {}

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        key = solution.hash()
        if key in self._cache:
            return self._cache[key]

        # Materialize sources into a fresh temp dir.
        build_dir = Path(tempfile.mkdtemp(prefix="gb10_build_"))
        for src in solution.sources:
            (build_dir / src.path).parent.mkdir(parents=True, exist_ok=True)
            (build_dir / src.path).write_text(src.content)

        # Load the entry module via importlib (unique name per build to avoid cache collisions).
        entry_path = build_dir / solution.get_entry_path()
        entry_symbol = solution.get_entry_symbol()
        mod_name = f"_gb10_tune_{solution.name}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(mod_name, str(entry_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, entry_symbol):
            raise BuildError(
                f"Entry symbol '{entry_symbol}' not found in '{entry_path}'"
            )
        callable_ = getattr(module, entry_symbol)

        def cleaner():
            sys.modules.pop(mod_name, None)
            # Leave temp dir for inspection; tmpdir cleanup is the OS's problem.

        runnable = Runnable(
            callable=callable_,
            metadata=RunnableMetadata(
                build_type=solution.spec.language.value,
                definition_name=definition.name,
                solution_name=solution.name,
                destination_passing_style=solution.spec.destination_passing_style,
                definition=definition,
            ),
            cleaner=cleaner,
        )
        self._cache[key] = runnable
        return runnable

    def build_reference(self, definition: Definition) -> Runnable:
        """Build the Definition's `reference` Python source as a Runnable."""
        mod_name = f"_gb10_tune_ref_{definition.name}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_loader(mod_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        exec(definition.reference, module.__dict__)

        if not hasattr(module, "run"):
            raise BuildError(f"Definition '{definition.name}' reference has no 'run' function")

        return Runnable(
            callable=module.run,
            metadata=RunnableMetadata(
                build_type="python",
                definition_name=definition.name,
                solution_name=f"__reference__{definition.name}",
                destination_passing_style=False,
                definition=definition,
            ),
        )
