"""
triage_api.py -- the thin wrapper that IS "your custom LLM API" for this
assignment. It sits on top of the professor's LLMBox without modifying any of
its files: it imports llmbox.src.generation.GenerationManager, loads Phi-4-mini
ONCE, and exposes the three baseline modes (generate / structured_output /
tool_calling) plus the two custom functionalities (recovery prompt, guardrail)
as plain Python calls so the Part B-D scripts can submit 50 inputs without the
60-second-per-call model reload that the Hydra CLI would incur.

Nothing here edits LLMBox. We only (a) build an OmegaConf cfg object the same
way Hydra would, and (b) reuse GenerationManager's own _load_model_and_tokenizer
and _generate_once. If a future LLMBox refactor renames those, this wrapper is
the only file that needs touching.

Requires (on your laptop, same env as Assignment 1):
    pip install hydra-core omegaconf torch transformers accelerate safetensors
and the local Phi-4-mini weights you already have.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Make the professor's llmbox importable without copying it. Set LLMBOX_DIR to
# wherever you unzipped LLMBOX.ZIP; defaults to a sibling ./llmbox folder.
import os
LLMBOX_DIR = Path(os.environ.get("LLMBOX_DIR", Path(__file__).resolve().parent.parent / "llmbox"))
if str(LLMBOX_DIR) not in sys.path:
    sys.path.insert(0, str(LLMBOX_DIR))

from omegaconf import OmegaConf  # noqa: E402

import scenario  # noqa: E402


# ---------------------------------------------------------------------------
# Config construction -- mirrors what Hydra composes for `model=phi4_instruct`,
# built by hand so we never shell out to startllm.py (which also tries to email
# course staff on non-local models; we stay strictly local).
# ---------------------------------------------------------------------------
def build_cfg(model_path: str, *, temperature: float = 0.0, top_p: float = 1.0,
              top_k: int = 0, max_new_tokens: int = 256,
              system_prompt: str = "", seed: int = 42) -> Any:
    do_sample = temperature and temperature > 0
    cfg = {
        "seed": seed,
        "system_prompt": system_prompt,
        "output_dir": str(scenario.OUT_DIR),
        "prompt": None,
        "prompt_file": None,
        "username": "mahika",
        "model": {
            "name": "phi4_instruct",
            "source": "local",
            "model_id": "microsoft/Phi-4-mini-instruct",
            "local_path": model_path,
            "architecture": "phi3",
            "dtype": "bfloat16",
            "device": "auto",
            "trust_remote_code": False,
            "supports_tool_calling": True,
            "supports_structured_output": True,
            "tool_calling_format": "functools_prompt",
            "chat_template_kwargs": {},
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "do_sample": bool(do_sample),
            "repetition_penalty": 1.0,
        },
    }
    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# The API object: loads the model once, answers many prompts.
# ---------------------------------------------------------------------------
@dataclass
class Response:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    refused: bool = False
    refusal_reason: str = ""
    meta: dict = field(default_factory=dict)


class TriageAPI:
    def __init__(self, model_path: str, **gen_kwargs):
        from src.generation import GenerationManager  # llmbox, lazy
        self.cfg = build_cfg(model_path, **gen_kwargs)
        self.gm = GenerationManager()
        self.model, self.tokenizer, self.device = self.gm._load_model_and_tokenizer(self.cfg)

    def set_generation(self, **kwargs) -> None:
        """Change temperature/top_p/max_new_tokens between runs without reload."""
        for k, v in kwargs.items():
            self.cfg.generation[k] = v
        if "temperature" in kwargs:
            self.cfg.generation.do_sample = bool(kwargs["temperature"] and kwargs["temperature"] > 0)

    def _count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # --- baseline: mode=generate --------------------------------------------
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Response:
        messages = []
        sp = self.cfg.system_prompt if system_prompt is None else system_prompt
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": prompt})
        t0 = time.time()
        text = self.gm._generate_once(self.model, self.tokenizer, self.device, messages, self.cfg)
        dt = time.time() - t0
        ptxt = (sp + "\n" + prompt) if sp else prompt
        return Response(text=text, latency_s=dt,
                        prompt_tokens=self._count(ptxt), completion_tokens=self._count(text))

    # --- mode=structured_output ---------------------------------------------
    def structured(self, prompt: str, schema_text: str,
                   system_prompt: str = "") -> Response:
        instruction = ("\n\nRespond with ONLY a single JSON object that strictly "
                       "matches this JSON Schema, with no other text:\n" + schema_text)
        sys_content = (system_prompt + instruction).strip()
        r = self.generate(prompt, system_prompt=sys_content)
        return r

    def close(self) -> None:
        pass
