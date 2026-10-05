"""Local Phi-4-mini runner (optionally with a LoRA adapter)."""
from __future__ import annotations

import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PHI_DEFAULT = REPO.parent / "cmu_application_of_nlx_llm" / "lab01" / "models" / "phi-4-mini-instruct"


def phi_path() -> Path:
    return Path(os.environ.get("PHI_MODEL_PATH", PHI_DEFAULT))


class PhiRunner:
    def __init__(self, model_path: Path | None = None, adapter_path: Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = model_path or phi_path()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True)
        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, system: str, user: str, max_new_tokens: int = 320) -> tuple[str, float, int, int]:
        """Greedy decode. Returns (text, latency_s, prompt_tokens, output_tokens)."""
        import torch

        t0 = time.perf_counter()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        # No repetition penalty: it also penalizes prompt tokens, and the ticket must
        # copy category and issue names verbatim from the prompt.
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return text, time.perf_counter() - t0, int(inputs["input_ids"].shape[1]), int(len(new))
