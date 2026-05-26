"""Swarm of RecursiveMAS Qwen specialists, loaded in-process via transformers + the
vendored latent-link adapters (modeling.py) and adapter-file resolver (hf_resolver.py).

use_mixture=False (A/B/C): Distillation-Learner-4B proposes; Distillation-Expert-9B on escalation.
use_mixture=True  (D):     Mixture-Code-3B (Inner Link r=3) → Outer Link → Mixture-Summarizer-2B decodes.
All checkpoints pre-cached in gb10-swarm-cache; Mixture models load only for D."""

import json
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from ._recursivemas import hf_resolver, modeling

_LEARNER = "RecursiveMAS/Distillation-Learner-Qwen3.5-4B"
_EXPERT = "RecursiveMAS/Distillation-Expert-Qwen3.5-9B"
_CODE = "RecursiveMAS/Mixture-Code-Qwen2.5-Coder-3B"
_SUMMARIZER = "RecursiveMAS/Mixture-Summarizer-Qwen3.5-2B"
_OUTERLINKS = "RecursiveMAS/Mixture-Outerlinks"

_PLAYBOOK_PATH = Path(__file__).parent / "playbook_sm121a.md"
_CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _snapshot(repo_id):
    return Path(snapshot_download(repo_id, local_files_only=True)).resolve()


def _load_base(repo_id):
    """(model, tokenizer, hidden_size) at fp16 on cuda. The checkpoints declare
    tokenizer_class 'TokenizersBackend', so the fast tokenizer is loaded directly."""
    resolved = modeling.resolve_local_pretrained_path(str(_snapshot(repo_id)))
    tok = PreTrainedTokenizerFast.from_pretrained(resolved)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        resolved, torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()
    return model, tok, int(model.get_input_embeddings().weight.size(-1))


def _load_inner_adapter(repo_id, hidden_size):
    state = torch.load(hf_resolver.resolve_inner_adapter(_snapshot(repo_id), task=None), map_location="cpu")
    a = modeling.Adapter(hidden_size=hidden_size,
                         adapter_type=modeling.infer_inner_adapter_type_from_state_dict(state))
    a.load_state_dict(state, strict=True)
    return a.to("cuda", torch.float16).eval()


def _load_outer_adapter(legacy_key):
    # Dims come from outerlink_config.json (authoritative), not loaded model hidden sizes.
    outer_dir = _snapshot(_OUTERLINKS)
    cfg = json.loads((outer_dir / "outerlink_config.json").read_text())
    adapters = cfg["adapters"] if "adapters" in cfg else cfg["tasks"][None]["adapters"]
    entry = next(e for e in adapters if str(e["legacy_key"]) == legacy_key)
    state = torch.load(outer_dir / entry["filename"], map_location="cpu")
    a = modeling.CrossModelAdapter(in_dim=int(entry["in_dim"]), out_dim=int(entry["out_dim"]),
                                   adapter_type=modeling.infer_outer_adapter_type_from_state_dict(state))
    a.load_state_dict(state, strict=True)
    return a.to("cuda", torch.float16).eval()


def _tier_section(playbook, tier):
    # tier 1/2 map to their own section; the sm_121a Tier 5c is always appended.
    header = {1: "## Tier 2 — Memory access", 2: "## Tier 3 — Compute"}.get(tier, "## Tier 4 — Advanced")
    lines = playbook.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == header)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## Tier ")), len(lines))
    t5c = next(i for i, l in enumerate(lines) if l.strip().startswith("## Tier 5c"))
    t5c_end = next((i for i in range(t5c + 1, len(lines)) if lines[i].startswith("## Tier ")), len(lines))
    return "\n".join(lines[start:end] + [""] + lines[t5c:t5c_end])


def _render_asi(asi_records):
    # GEPA ASI: last few failed attempts + diagnostics, so the proposer learns what to avoid.
    if not asi_records:
        return ""
    out = ["\n## Prior attempts and why they failed (learn from these — do not repeat)"]
    for rec in asi_records[-3:]:
        out.append(f"\n### iter {rec['iter']} — FAILED: {rec['failure']}\n```python\n{rec['source']}\n```")
    return "\n".join(out)


def _build_prompt(definition, best_source, tier, playbook, asi_records=None):
    return (
        "You are an expert GPU kernel engineer specializing in NVIDIA Blackwell "
        "(sm_121a) and Triton 3.7. Emit the revised kernel inside ONE ```python ... ``` "
        "fenced block. Preserve the `def run(...)` signature. The sm_121a per-block "
        "shared-memory limit is 101376 bytes.\n\n"
        f"## Definition\n{definition.model_dump_json(indent=2)}\n\n"
        f"## Current best source\n```python\n{best_source}\n```\n\n"
        f"## Playbook tier {tier}\n{_tier_section(playbook, tier)}\n"
        f"{_render_asi(asi_records)}\n## Your turn\nEmit ONE ```python ... ``` block."
    )


def _extract_code(text):
    # The prompt demands exactly one ```python fenced block; a malformed generation that
    # omits it fails here loudly rather than silently passing raw text downstream.
    return _CODE_RE.search(text).group(1).strip()


class Swarm:
    def __init__(self, use_mixture):
        self.use_mixture = use_mixture
        self.playbook = _PLAYBOOK_PATH.read_text()
        self.learner = _load_base(_LEARNER)
        self.expert = _load_base(_EXPERT)
        self.code = self.summarizer = self.code_inner = self.outer_cs = None
        if use_mixture:
            self.code = _load_base(_CODE)
            self.summarizer = _load_base(_SUMMARIZER)
            self.code_inner = _load_inner_adapter(_CODE, self.code[2])
            self.outer_cs = _load_outer_adapter("outer_2s")

    def propose(self, definition, best_source, tier, asi_records=None):
        prompt = _build_prompt(definition, best_source, tier, self.playbook, asi_records)
        if self.use_mixture:
            return self._mixture_propose(prompt)
        return _extract_code(self._generate(self.learner, prompt))

    def expert_propose(self, definition, best_source, asi_records):
        # GEPA reflection_lm: diagnose the recurring failure from ASI, then fix — not blind retry.
        prompt = (
            "You are a senior GPU kernel engineer acting as a REVIEWER on NVIDIA Blackwell "
            "(sm_121a) / Triton 3.7. A junior agent has repeatedly failed to improve this "
            "kernel. Below are its failed attempts with diagnostics. First, in 2-3 sentences, "
            "diagnose the COMMON ROOT CAUSE. Then emit ONE corrected kernel inside ONE "
            "```python ... ``` block. Preserve `def run(...)`; respect the 101376-byte SMEM limit.\n\n"
            f"## Definition\n{definition.model_dump_json(indent=2)}\n\n"
            f"## Current best (correct) source\n```python\n{best_source}\n```\n\n"
            f"## Playbook\n{_tier_section(self.playbook, 4)}\n{_render_asi(asi_records)}\n"
            "## Your turn\nDiagnosis: <2-3 sentences>. Then ONE ```python ... ``` block."
        )
        return _extract_code(self._generate(self.expert, prompt))

    def _generate(self, base, prompt, max_new_tokens=4096):
        model, tok, _ = base
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
        return tok.decode(out_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def _mixture_propose(self, prompt):
        # Latent flow: Code reads prompt → Inner Link self-refines r=3 → Outer Link to
        # Summarizer's embedding space → Summarizer decodes (the only text decode).
        code_model, code_tok, _ = self.code
        summ_model, summ_tok, _ = self.summarizer
        ctx = code_model.get_input_embeddings()(code_tok(prompt, return_tensors="pt").to("cuda").input_ids)
        with torch.no_grad():
            h = code_model(inputs_embeds=ctx, output_hidden_states=True).hidden_states[-1][:, -1:, :]
        for _ in range(3):
            ctx = torch.cat([ctx, self.code_inner(h.to(torch.float16)).to(ctx.dtype)], dim=1)
            with torch.no_grad():
                h = code_model(inputs_embeds=ctx, output_hidden_states=True).hidden_states[-1][:, -1:, :]
        h_summ = self.outer_cs(h.to(torch.float16))
        summ_prompt = ("Emit a single Triton kernel for the Blackwell sm_121a definition above, "
                       "inside ONE ```python ... ``` block. Preserve the `def run(...)` signature.")
        summ_in = summ_tok(summ_prompt, return_tensors="pt").to("cuda")
        combined = torch.cat([summ_model.get_input_embeddings()(summ_in.input_ids),
                              h_summ.to(torch.float16)], dim=1)
        with torch.no_grad():
            gen = summ_model.generate(inputs_embeds=combined, max_new_tokens=4096, do_sample=False,
                                      pad_token_id=summ_tok.pad_token_id)
        return _extract_code(summ_tok.decode(gen[0], skip_special_tokens=True))
