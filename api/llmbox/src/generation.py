'''
    LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
    Copyright (C) 2026  Sara Kingsley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import json
import logging
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Shared model / generation helpers
# --------------------------------------------------------------------------

class GenerationManager:

    def __init__(self) -> None:
         self.log = logging.getLogger(__name__)

    def _resolve_device_and_dtype(self, cfg):
        import torch
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[cfg.model.dtype]
        if cfg.model.device != "auto":
            device = cfg.model.device
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device, dtype


    def _resolve_model_path(self, cfg):
        """Turn model.source/model_id/local_path into a (path_or_repo_id,
        local_files_only) pair for from_pretrained().

        source=local uses local_files_only=True so a bad path fails fast and
        clearly with a filesystem-style error, instead of transformers quietly
        trying to interpret it as a hub repo id. source=huggingface leaves
        local_files_only off, so from_pretrained uses its normal cache-or-
        download behavior."""
        if cfg.model.source == "local":
            if not cfg.model.local_path:
                raise ValueError(
                    f"model.source=local requires model.local_path to be set for model '{cfg.model.name}'."
                )
            return cfg.model.local_path, True
        if cfg.model.source == "huggingface":
            return cfg.model.model_id, False
        raise ValueError(f"Unknown model.source '{cfg.model.source}'. Use 'huggingface' or 'local'.")

    def _ensure_remote_code_compat(self):
        """Some trust_remote_code model repos (e.g. Phi-4-mini-instruct's
        modeling_phi3.py) still import `LossKwargs` from `transformers.utils`,
        a name that newer transformers releases renamed to `TransformersKwargs`.
        That break isn't specific to this one model -- it hits a lot of
        trust_remote_code repos whose custom code hasn't been updated for the
        renamed symbol. Rather than pin an older transformers version (and
        lose whatever else changed since), alias the old name back in right
        before loading."""
        import transformers.utils as _tu
        if not hasattr(_tu, "LossKwargs") and hasattr(_tu, "TransformersKwargs"):
            _tu.LossKwargs = _tu.TransformersKwargs

    def _load_model_and_tokenizer(self, cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device, dtype = self._resolve_device_and_dtype(cfg)
        model_path, local_files_only = self._resolve_model_path(cfg)
        log.info(
            "Loading '%s' (source=%s, %s) onto %s as %s...",
            cfg.model.name, cfg.model.source, model_path, device, cfg.model.dtype,
        )
        if cfg.model.trust_remote_code:
            self._ensure_remote_code_compat()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=cfg.model.trust_remote_code, local_files_only=local_files_only
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=cfg.model.trust_remote_code,  #fixme
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return model, tokenizer, device

    def _generation_kwargs(self, cfg, tokenizer):
        gen = cfg.generation
        return dict(
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=max(gen.temperature, 1e-5),
            top_p=gen.top_p,
            top_k=gen.top_k,
            repetition_penalty=gen.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    def _generate_once(self, model, tokenizer, device, messages, cfg, **template_kwargs):
        return self._generate_measured(model, tokenizer, device, messages, cfg, seed=cfg.seed, **template_kwargs)["text"]

    def _generate_measured(self, model, tokenizer, device, messages, cfg, seed=None, **template_kwargs):
        """[A2 M0] Same generation as the original _generate_once, but it
        (1) seeds sampling when `seed` is given -- upstream never applied
        cfg.seed, so do_sample=True runs were not reproducible -- and
        (2) returns the measurements upstream discarded: prompt/completion
        token counts, wall-clock latency and finish_reason."""
        import time
        import torch

        if seed is not None:
            from transformers import set_seed
            set_seed(int(seed))

        template_kwargs = {**OmegaConf.to_container(cfg.model.chat_template_kwargs, resolve=True), **template_kwargs}
        started = time.perf_counter()
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **template_kwargs,
        ).to(device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(**inputs, **self._generation_kwargs(cfg, tokenizer))

        new_ids = output_ids[0][input_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        latency = time.perf_counter() - started
        completion_tokens = int(new_ids.shape[-1])
        return {
            "text": text,
            "prompt_tokens": int(input_len),
            "completion_tokens": completion_tokens,
            "latency_s": round(latency, 4),
            "finish_reason": "length" if completion_tokens >= cfg.generation.max_new_tokens else "stop",
            "seed": None if seed is None else int(seed),
        }

    def _resolve_prompt(self, cfg) -> str:
        if cfg.prompt_file:
            text = Path(cfg.prompt_file).read_text(encoding="utf-8").strip()
        elif cfg.prompt:
            text = cfg.prompt.strip()
        else:
            raise ValueError('This mode needs a prompt. Pass prompt="..." or prompt_file=path/to/file.txt')
        if not text:
            raise ValueError("Resolved prompt is empty.")
        return text

    # --------------------------------------------------------------------------
    # "functools_prompt" tool calling -- for models (e.g. Phi-4-mini-instruct)
    # whose chat template doesn't do structured tool calling via a `tools=`
    # kwarg, and instead expects tool definitions embedded in the system
    # prompt as <|tool|>[...]<|/tool|>, with the model replying with a
    # "functools[...]"-prefixed JSON list of calls when it wants to invoke
    # one. Mirrors the convention chat_phi4mini.py established.
    # --------------------------------------------------------------------------

    def build_functools_system_prompt(self, base_system_prompt: str, tools: list, tool_choice: str = "auto") -> str:
        parts = [base_system_prompt] if base_system_prompt else []
        tools_json = json.dumps(tools)
        parts.append(f"<|tool|>{tools_json}<|/tool|>")
        if tool_choice == "required":
            # [A2 M3] upstream ignored tool_calling.tool_choice on this path; its
            # "auto" wording tells the model to skip tools whenever it thinks
            # it can answer from internal knowledge.
            parts.append(
                "You must call at least one of the provided functions before giving "
                "your final answer.\n"
                "* prefix function calls with the functools marker (no closing "
                "marker required)\n"
                "* all function calls should be generated in a single JSON list "
                'formatted as functools[{"name": [function name], "arguments": '
                "[function arguments as JSON]}, ...]\n"
                "* follow the provided JSON schema. Do not hallucinate arguments "
                "or values.\n"
                "* respect the argument type formatting."
            )
            return "\n".join(parts)
        parts.append(
            "In addition to plain text responses, you can choose to call one "
            "or more of the provided functions.\n"
            "Use the following rule to decide when to call a function:\n"
            "* if the response can be generated from your internal knowledge, "
            "do so\n"
            "* if you need external information that can be obtained by "
            "calling one or more of the provided functions, generate "
            "function calls\n"
            "If you decide to call functions:\n"
            "* prefix function calls with the functools marker (no closing "
            "marker required)\n"
            "* all function calls should be generated in a single JSON list "
            'formatted as functools[{"name": [function name], "arguments": '
            "[function arguments as JSON]}, ...]\n"
            "* follow the provided JSON schema. Do not hallucinate arguments "
            "or values.\n"
            "* respect the argument type formatting."
        )
        return "\n".join(parts)

    def parse_tool_calls(self, text: str):
        """Look for a 'functools[...]' block and parse it into a list of
        {"name": ..., "arguments": {...}} calls. Returns None if no tool
        call is present."""
        marker = "functools"
        idx = text.find(marker)
        if idx == -1:
            return None
        json_str = text[idx + len(marker):].strip()
        try:
            calls, _ = json.JSONDecoder().raw_decode(json_str)
            return calls
        except json.JSONDecodeError:
            return None

    def run_tool_turn(self, model, tokenizer, device, messages, cfg):
        """Generate one assistant turn under the functools_prompt convention:
        if the model requests tool call(s), execute them locally against
        src.tools.TOOL_REGISTRY, feed the results back in, and generate the
        final answer. `messages` is mutated in place with the raw response
        and any tool result, same as run_turn did in chat_phi4mini.py.
        Returns (final_text, tool_call_records)."""
        from src.tools import TOOL_REGISTRY

        raw_response = self._generate_once(model, tokenizer, device, messages, cfg)
        tool_call_records = []

        tool_calls = self.parse_tool_calls(raw_response)
        if tool_calls:
            results = []
            for call in tool_calls:
                name = call.get("name")
                arguments = call.get("arguments", {})
                func = TOOL_REGISTRY.get(name)
                if func is None:
                    result = {"error": f"Unknown tool '{name}'"}
                else:
                    try:
                        result = func(**arguments)
                    except Exception as exc:  # noqa: BLE001 - surface bad args to the model
                        result = {"error": str(exc)}
                results.append({"name": name, "result": result})
                tool_call_records.append({"name": name, "arguments": arguments, "result": result})

            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "tool", "content": json.dumps(results)})

            raw_response = self._generate_once(model, tokenizer, device, messages, cfg)

        messages.append({"role": "assistant", "content": raw_response})
        return raw_response, tool_call_records

    # --------------------------------------------------------------------------
    # [A2 M3] Multi-round tool loop with allowlist + argument validation.
    # run_tool_turn above does one round and executes any registered tool.
    # This loop: up to `max_rounds` rounds of (generate -> parse functools[...]
    # -> check_call -> execute -> feed results back), then the final answer.
    # Rejected calls are not executed; the reason is returned to the model as
    # the tool result, and recorded. Tokens/latency are summed over rounds.
    # --------------------------------------------------------------------------
    def run_tool_loop(self, generate_fn, messages, offered, max_rounds=3, require_call=False):
        from src.tools import TOOL_REGISTRY, check_call

        messages = list(messages)
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
        records, rounds, nudged = [], 0, False
        while True:
            out = generate_fn(messages)
            for k in totals:
                totals[k] += out[k]
            calls = self.parse_tool_calls(out["text"])
            if not calls and require_call and rounds == 0 and not nudged:
                # tool_choice=required: one reminder, then accept whatever comes back
                nudged = True
                messages.append({"role": "assistant", "content": out["text"]})
                messages.append({"role": "user", "content": "You must call one of the provided functions "
                                 "(functools[...]) before giving your final answer."})
                continue
            if not calls or rounds >= max_rounds:
                break
            if isinstance(calls, dict):
                calls = [calls]
            rounds += 1
            results = []
            for call in calls if isinstance(calls, list) else [calls]:
                problem = check_call(call, offered)
                name = call.get("name") if isinstance(call, dict) else None
                args = call.get("arguments", {}) if isinstance(call, dict) else None
                if problem:
                    result = {"error": f"rejected: {problem}"}
                else:
                    try:
                        result = TOOL_REGISTRY[name](**args)
                    except Exception as exc:  # noqa: BLE001 - report tool errors to the model
                        result = {"error": str(exc)[:300]}
                results.append({"name": name, "result": result})
                records.append({"round": rounds, "name": name, "arguments": args,
                                "rejected": problem, "result": result})
            messages.append({"role": "assistant", "content": out["text"]})
            messages.append({"role": "tool", "content": json.dumps(results)})
        return {**out, **totals, "latency_s": round(totals["latency_s"], 4),
                "tool_calls": records, "tool_rounds": rounds, "nudged": nudged,
                "hit_round_limit": bool(calls) and rounds >= max_rounds}
