"""Template registry: allowed_action → list of templates that target it.

Stage 1 looks up templates via `templates_for(BottleneckReport.allowed_action)`.
"""

from typing import List

from .base import Template
from .gemm import TiledGroupMGEMMTemplate
from .reduction import VectorizedCoalescedReductionTemplate


_TEMPLATES_BY_ACTION = {
    "vectorized_coalesced_reduction": [VectorizedCoalescedReductionTemplate()],
    "tiled_groupm_gemm": [TiledGroupMGEMMTemplate()],
}


def templates_for(allowed_action: str) -> List[Template]:
    return _TEMPLATES_BY_ACTION[allowed_action]
