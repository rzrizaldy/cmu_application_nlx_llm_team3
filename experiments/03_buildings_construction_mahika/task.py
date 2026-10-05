"""
task.py -- the triage task definition shared by every part: the prompt that
turns a 311 record into a classification/routing request, a tolerant parser for
the model's reply, and the scoring functions. Keeping this in one file means
Parts B, C and D score identically.
"""
from __future__ import annotations
import json
import re
from typing import Optional

import scenario

CATEGORIES = scenario.ISSUE_CATEGORIES
DEPARTMENTS = scenario.DEPARTMENTS

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# v1 = the plain baseline prompt (Part B). Deliberately minimal, like the
# Assignment 1 v1 prompt, so the same issue_category weakness reappears and the
# Part C recovery functionality has something to fix.
SYSTEM_V1 = (
    "You are a triage assistant for the City of Pittsburgh 311 service center. "
    "Read one 311 service request and classify it."
)

def user_prompt(text: str) -> str:
    cats = ", ".join(CATEGORIES)
    depts = ", ".join(DEPARTMENTS)
    return (
        f"311 request:\n{text}\n\n"
        f"Return a JSON object with exactly these keys:\n"
        f'  "issue_category": one of [{cats}]\n'
        f'  "responsible_department": one of [{depts}]\n'
        f"Return only the JSON object."
    )

# v2 = the recovery prompt (Part C custom functionality #1). Adds the category
# definitions that the Assignment 1 analysis showed were the missing piece:
# Phi collapsed permits->construction and accessibility->building_maintenance
# because it did not know what the buckets meant.
CATEGORY_DEFINITIONS = (
    "Category definitions (use these exactly):\n"
    "- permits: anything about work done WITHOUT a permit, unpermitted "
    "electrical/HVAC work, or building without a permit.\n"
    "- accessibility: sidewalks, curb ramps, ADA ramps, broken sidewalks, "
    "blocked sidewalks, or city steps.\n"
    "- construction: active construction-SITE problems (e.g. construction site "
    "maintenance), not permit violations.\n"
    "- building_maintenance: the condition of an existing building or city "
    "property (e.g. building maintenance, vacant building, fire safety system)."
)
SYSTEM_V2 = SYSTEM_V1 + "\n\n" + CATEGORY_DEFINITIONS


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse(reply: str) -> dict:
    """Pull the first JSON object out of the model's reply; tolerant of code
    fences and trailing prose. Returns {} if nothing parseable."""
    if not reply:
        return {}
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return {}
    blob = m.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        # try trimming to the first balanced object
        depth = 0
        end = None
        for i, ch in enumerate(blob):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return {}
        try:
            obj = json.loads(blob[:end])
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}


def normalize_cat(v: Optional[str]) -> str:
    if not v:
        return ""
    s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
    return s


def normalize_dept(v: Optional[str]) -> str:
    if not v:
        return ""
    return str(v).strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_one(pred: dict, gold: dict) -> dict:
    pc = normalize_cat(pred.get("issue_category"))
    gc = normalize_cat(gold.get("issue_category"))
    pd_ = normalize_dept(pred.get("responsible_department"))
    gd = normalize_dept(gold.get("responsible_department"))
    valid_schema = (pc in CATEGORIES) and (pd_ in [normalize_dept(d) for d in DEPARTMENTS])
    return {
        "category_correct": pc == gc and gc != "",
        "department_correct": pd_ == gd and gd != "",
        "both_correct": (pc == gc and gc != "") and (pd_ == gd and gd != ""),
        "schema_valid": valid_schema,
        "pred_category": pc,
        "gold_category": gc,
        "pred_department": pd_,
        "gold_department": gd,
    }


def aggregate(scored: list[dict]) -> dict:
    n = len(scored) or 1
    def rate(k):
        return round(sum(1 for s in scored if s[k]) / n, 4)
    # per-category breakdown of category accuracy
    per_cat = {}
    for c in CATEGORIES:
        rows = [s for s in scored if s["gold_category"] == c]
        if rows:
            per_cat[c] = round(sum(1 for s in rows if s["category_correct"]) / len(rows), 3)
    return {
        "n": len(scored),
        "category_accuracy": rate("category_correct"),
        "department_accuracy": rate("department_correct"),
        "both_correct": rate("both_correct"),
        "schema_valid_rate": rate("schema_valid"),
        "category_accuracy_by_gold": per_cat,
    }
