"""LLM client + evidence-driven prompt builder for Stage 2.

The prompt has four sections: Definition summary, current best source + runtime,
NCU diagnostic findings (severity-sorted, each with metric values + action),
and the AttemptMemory-derived list of (action, template, params) tuples
already evaluated by Stage 1. The LLM is asked to cite one finding and emit
ONE ```python ... ``` block.

Failure-mode hints and repeated-status detection are not needed —
`AttemptMemory.has_source()` rejects any duplicate candidate at the source-hash
level before it reaches the evaluator.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import httpx

from .data import Definition
from .diagnostic import analyze, render_analysis


_SYSTEM = (
    "You are an expert GPU kernel engineer specializing in NVIDIA Blackwell "
    "(sm_121a) and Triton 3.7. You optimize a single-file Triton kernel "
    "iteratively. Cite one NCU finding by title and quote its metric value, "
    "name the single knob you will change, then emit the revised kernel "
    "inside ONE ```python ... ``` fenced block. Preserve the `def run(...)` "
    "function signature. sm_121a per-block shared-memory limit is 101376 "
    "bytes — configurations exceeding this fail at launch."
)


def llm_complete(
    messages: List[Dict[str, str]],
    model: str,
    endpoint: str,
    *,
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    timeout_s: float = 300.0,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    resp = httpx.post(
        f"{endpoint.rstrip('/')}/chat/completions", json=payload, timeout=timeout_s
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _summarize_definition(definition: Definition) -> str:
    ca = definition.const_axes
    dim_str = " ".join(f"{k}={v}" for k, v in ca.items())
    first_input_dtype = next(iter(definition.inputs.values())).dtype
    return f"{first_input_dtype} {definition.op_type.upper()}, {dim_str} on sm_121a"


def build_prompt(
    definition: Definition,
    current_best_source: str,
    current_best_runtime_ms: float,
    ncu_diag: Dict[str, Any],
    tried_tuples_summary: List[str],
    history_tail: List[str],
) -> List[Dict[str, str]]:
    diagnostic_block = render_analysis(analyze(ncu_diag))
    tried_block = (
        "\n".join(f"  - {t}" for t in tried_tuples_summary)
        if tried_tuples_summary
        else "  (none yet)"
    )
    history_block = (
        "\n".join(history_tail)
        if history_tail
        else "(no Stage-2 history yet — this is the first LLM iteration)"
    )
    user = f"""\
## 1. Definition
{_summarize_definition(definition)}

```json
{definition.model_dump_json(indent=2)}
```

## 2. Current best (runtime = {current_best_runtime_ms:.4f} ms)

```python
{current_best_source}
```

## 3. NCU diagnostic + Findings

{diagnostic_block}

## 4. Already-evaluated configurations (Stage 1 template grid)
{tried_block}

## 5. Stage 2 history (last {len(history_tail)} iterations)
```
{history_block}
```

## Your turn
Cite one finding from §3 (by title + metric value). Name one knob you will
change and why. Emit the revised kernel inside ONE ```python ... ``` block.
"""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def extract_code_block(resp: str) -> Optional[str]:
    m = re.search(r"```python\n(.*?)(?:```|\Z)", resp, re.DOTALL)
    return m.group(1).strip() if m else None
