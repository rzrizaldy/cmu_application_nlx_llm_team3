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

"""
tools.py

Example tool implementations for mode=tool_calling. When a model's chat
template doesn't have native structured tool-calling support and instead
uses a convention like Phi-4-mini-instruct's <|tool|>...<|/tool|> system
prompt + "functools[...]" response marker (see
GenerationManager.build_functools_system_prompt / parse_tool_calls /
run_tool_turn in generation.py), the model's requested tool name is looked
up here and actually executed locally.

Swap these out for whatever you actually want the model to be able to
call. Each tool needs:
  1. An entry under `tool_calling.tools` in your config (name, description,
     and a JSON-schema `parameters` block) -- that's what gets shown to the
     model.
  2. A matching Python callable registered here under the same name --
     that's what actually runs when the model asks for it.
"""

TOOL_REGISTRY = {}


def register_tool(name):
    def decorator(func):
        TOOL_REGISTRY[name] = func
        return func
    return decorator


@register_tool("get_current_weather")
def get_current_weather(location, unit="celsius"):
    # Stub data -- replace this body with a real weather API call.
    fake_data = {"Paris, France": 18, "New York, USA": 22, "Tokyo, Japan": 27}
    temp_c = fake_data.get(location, 20)
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32, 1)
    return {"location": location, "temperature": temp, "unit": unit}


@register_tool("calculate")
def calculate(expression):
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return {"error": "Expression contains disallowed characters."}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as exc:  # noqa: BLE001 - want to surface any eval error to the model
        return {"error": str(exc)}


# ==========================================================================
# [A2 M3] Pittsburgh 311 knowledge-base tagging tools.
#
# Two read-only lookups over local CSVs, and one *write* tool. The write tool
# never changes anything live: it appends a proposed tag update to a review
# queue (JSONL) that a human must approve. It is the only tool in this API
# that has an effect outside the conversation, which is why the governance
# policy treats it differently.
#
# configure_waste311(cfg) must be called before use; the batch runner sets
# CONTEXT["doc_id"] for every row, so the model cannot choose which record
# it writes to (it is never shown the doc_id).
# ==========================================================================
import csv as _csv
import json as _json
import re as _re
from datetime import datetime as _dt, timezone as _tz
from pathlib import Path as _Path

CONTEXT = {"doc_id": None, "codebook": [], "issue_summary": [], "review_queue": None}


def configure_waste311(cfg):
    tc = cfg.tool_calling
    with open(tc.codebook_path, encoding="utf-8") as f:
        CONTEXT["codebook"] = [r for r in _csv.DictReader(f)]
    with open(tc.issue_summary_path, encoding="utf-8") as f:
        CONTEXT["issue_summary"] = [r for r in _csv.DictReader(f)]
    CONTEXT["review_queue"] = _Path(tc.review_queue_path) if tc.review_queue_path else None


def _tokens(text):
    return set(_re.findall(r"[a-z0-9]+", str(text).lower())) - {"the", "of", "and", "a", "to", "for", "in", "or"}


@register_tool("lookup_codebook")
def lookup_codebook(query, limit=5):
    """Top matches from the 311 issue/category codebook by word overlap."""
    q = _tokens(query)
    if not q:
        return {"error": "query must contain words"}
    scored = []
    for r in CONTEXT["codebook"]:
        s = len(q & _tokens(f"{r['issue']} {r['category']} {r['department']}"))
        if s:
            scored.append((s, r))
    scored.sort(key=lambda x: (-x[0], x[1]["issue"]))
    limit = max(1, min(int(limit), 10))
    return {"matches": [{"issue": r["issue"], "category": r["category"] or None, "department": r["department"] or None}
                        for _, r in scored[:limit]]}


@register_tool("get_issue_stats")
def get_issue_stats(issue):
    """Historical request volume and closure-time percentiles for a waste issue."""
    rows = CONTEXT["issue_summary"]
    exact = [r for r in rows if r["request_type_name"].lower() == str(issue).lower()]
    if not exact:
        q = _tokens(issue)
        best = max(rows, key=lambda r: len(q & _tokens(r["request_type_name"])), default=None)
        exact = [best] if best is not None and q & _tokens(best["request_type_name"]) else []
    if not exact:
        return {"error": f"no statistics for issue '{issue}'"}
    r = exact[0]
    return {"issue": r["request_type_name"], "category": r["category"], "request_count": int(r["request_count"]),
            "median_hours": round(float(r["median_resolution_hours"]), 1),
            "p75_hours": round(float(r["p75_resolution_hours"]), 1),
            "p90_hours": round(float(r["p90_resolution_hours"]), 1)}


@register_tool("queue_tag_update")
def queue_tag_update(service_focus, information_type, responsible_department):
    """WRITE (sandboxed): propose tags for the current record; a human reviews the queue."""
    from src.pydantic_models.waste_tags import WasteTags
    tags = WasteTags(service_focus=service_focus, information_type=information_type,
                     responsible_department=responsible_department).model_dump()
    entry = {"doc_id": CONTEXT["doc_id"], "tags": tags, "status": "pending_human_review",
             "queued_at": _dt.now(_tz.utc).isoformat()}
    if CONTEXT["review_queue"] is not None:
        CONTEXT["review_queue"].parent.mkdir(parents=True, exist_ok=True)
        with CONTEXT["review_queue"].open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry) + "\n")
    return {"queued": True, "status": "pending_human_review"}


# --------------------------------------------------------------------------
# [A2 M3] Argument validation. Upstream executed any function in
# TOOL_REGISTRY with whatever arguments the model produced -- including
# tools that were never offered to it in the config. check_call() enforces
# an allowlist (only offered tools) and the offered JSON-schema (required
# keys, no unknown keys, basic types) before anything runs.
# --------------------------------------------------------------------------
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def check_call(call, offered):
    if not isinstance(call, dict):
        return "tool call is not an object"
    name, args = call.get("name"), call.get("arguments", {})
    if name not in offered:
        return f"tool '{name}' is not offered to this API"
    if not isinstance(args, dict):
        return "arguments must be an object"
    params = offered[name].get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    missing = [k for k in params.get("required", []) if k not in args]
    if missing:
        return f"missing required arguments: {missing}"
    unknown = [k for k in args if k not in props]
    if unknown:
        return f"unknown arguments: {unknown}"
    for k, v in args.items():
        want = _TYPES.get(props[k].get("type"))
        if want is not None and (not isinstance(v, want) or (want is int and isinstance(v, bool))):
            return f"argument '{k}' must be {props[k].get('type')}"
        enum = props[k].get("enum")
        if enum is not None and v not in enum:
            return f"argument '{k}' must be one of {enum}"
    return None
