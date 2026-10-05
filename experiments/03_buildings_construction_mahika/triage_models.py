"""
triage_models.py -- Part C, Custom Evaluation 1: the Pydantic model for the
triage task, plus the JSON Schema that mode=structured_output sends to the model.

The Pydantic model is the authority: it constrains issue_category and
responsible_department to the allowed values via Enums, so a reply that invents
a category fails validation. We export its JSON Schema to data/triage_schema.json
(what LLMBox's structured_output mode appends to the system prompt) and validate
every model reply against the Pydantic model to get the structured-output
validity rate the assignment asks for.
"""
from __future__ import annotations
import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

import scenario


class IssueCategory(str, Enum):
    building_maintenance = "building_maintenance"
    construction = "construction"
    permits = "permits"
    accessibility = "accessibility"


# Build the department enum from the scenario list (": unknown" is the abstain).
Department = Enum(  # type: ignore
    "Department",
    {d.replace(", ", "_").replace(" ", "_").replace("-", "_"): d for d in scenario.DEPARTMENTS},
    type=str,
)


class TriageResult(BaseModel):
    """Structured triage decision for one 311 request."""
    issue_category: IssueCategory = Field(
        description="Which of the four issue buckets this request falls into.")
    responsible_department: str = Field(
        description="The department to route the request to, or 'unknown' if the "
                    "request does not clearly belong to one.")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="Model confidence in the routing decision, 0-1.")


def validate_reply(pred: dict):
    """Return (ok, TriageResult|None, error_str)."""
    try:
        obj = TriageResult(**pred)
        return True, obj, ""
    except Exception as e:  # pydantic ValidationError or TypeError
        return False, None, str(e).splitlines()[0]


def export_schema(path: Path | None = None) -> Path:
    path = path or (scenario.HERE / "data" / "triage_schema.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = TriageResult.model_json_schema()
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    p = export_schema()
    print(f"Wrote JSON schema -> {p}")
    print(json.dumps(TriageResult.model_json_schema(), indent=2))
