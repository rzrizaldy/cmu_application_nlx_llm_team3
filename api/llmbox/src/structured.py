"""
structured.py  [A2 M2]

Structured output with validation. Upstream mode=structured_output pasted a
JSON Schema file into the system prompt and, with strict=true, only tried
json.loads -- there was no schema validation and no second chance. Here:

  1. the schema comes from a Pydantic model (structured_output.pydantic_model,
     "module.path:ClassName"), so the prompt schema and the validator are
     the same object;
  2. the reply is cut down to its first JSON object (models often add ``` fences
     or chatter) and validated with model_validate_json;
  3. on failure, the validation errors are sent back as a user turn and the
     model tries again, up to structured_output.max_retries times.

Every attempt's tokens and latency are summed, so the cost of retries shows
up in the measurements rather than disappearing.
"""
import importlib
import json


def load_pydantic_model(spec):
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"pydantic_model must look like 'module.path:ClassName', got {spec!r}")
    return getattr(importlib.import_module(module_name), class_name)


def schema_instruction(model_cls):
    schema = json.dumps(model_cls.model_json_schema(), indent=1)
    return ("\n\nRespond with ONLY a single JSON object that strictly matches this JSON Schema, "
            f"with no other text:\n{schema}")


def first_json_object(text):
    start = text.find("{")
    while start != -1:
        try:
            obj, end = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return text[start:start + end]
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def structured_generate(generate_fn, messages, model_cls, max_retries=2):
    """generate_fn(messages) -> dict from GenerationManager._generate_measured."""
    from pydantic import ValidationError

    messages = list(messages)
    attempts, violation = [], None
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
    out = None
    for attempt in range(max_retries + 1):
        out = generate_fn(messages)
        for k in totals:
            totals[k] += out[k]
        attempts.append(out["text"])
        candidate = first_json_object(out["text"])
        try:
            if candidate is None:
                raise ValueError("no JSON object found in the reply")
            parsed = model_cls.model_validate_json(candidate).model_dump()
            return {**out, **totals, "latency_s": round(totals["latency_s"], 4), "parsed": parsed,
                    "violation": None, "retries": attempt, "attempts": attempts}
        except (ValidationError, ValueError) as exc:
            violation = str(exc)[:800]
        messages += [
            {"role": "assistant", "content": out["text"]},
            {"role": "user", "content": "Your reply did not match the required JSON Schema:\n"
                                        f"{violation}\nReply with only the corrected JSON object."},
        ]
    return {**out, **totals, "latency_s": round(totals["latency_s"], 4), "parsed": None,
            "violation": violation, "retries": max_retries, "attempts": attempts}
