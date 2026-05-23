"""Anti-repetition memory shared across Stage 1 and Stage 2.

Stage 1 keys: (allowed_action, template_name, parameter_tuple).
Stage 2 keys: sha256 hex of raw candidate source.

Both sets are checked before evaluation; both are populated after evaluation.
This single store is what prevents the LLM lock-in observed when Stage 2 ran
standalone — every candidate the deterministic sweep evaluated, and every
candidate the LLM ever emitted in this run, is permanently off the table.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple


# (allowed_action, template_name, parameter_tuple)
AttemptKey = Tuple[str, str, Tuple[int, ...]]


@dataclass
class AttemptMemory:
    attempted_keys: Set[AttemptKey] = field(default_factory=set)
    attempted_sources: Set[str] = field(default_factory=set)

    # Stage 1
    def has(self, key: AttemptKey) -> bool:
        return key in self.attempted_keys

    def add(self, key: AttemptKey) -> None:
        self.attempted_keys.add(key)

    # Stage 2
    def has_source(self, sha256_hex: str) -> bool:
        return sha256_hex in self.attempted_sources

    def add_source(self, sha256_hex: str) -> None:
        self.attempted_sources.add(sha256_hex)

    # For LLM prompt context: compact list of "action/template/params" strings.
    def attempted_tuples_summary(self) -> List[str]:
        return [f"{a}/{t}/{p}" for (a, t, p) in sorted(self.attempted_keys)]
