"""
triage.py -- Pydantic model for the Pittsburgh 311 triage task.

New file for Assignment 2 (Mahika Gunjkar, Group 3). Follows the LLMBox
src/pydantic_models/ convention (same as personrecords.py). Used by
mode=structured_output: export its JSON schema and pass it via
structured_output.schema_path.

    python -c "from src.pydantic_models.triage import export; export()"
    # writes triage_schema.json next to this file
"""
from __future__ import annotations
import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class IssueCategory(str, Enum):
    building_maintenance = "building_maintenance"
    construction = "construction"
    permits = "permits"
    accessibility = "accessibility"


class TriageResult(BaseModel):
    """Structured triage decision for one 311 service request."""
    issue_category: IssueCategory = Field(
        description="Which of the four issue buckets this request falls into.")
    responsible_department: str = Field(
        description="The city department to route to, or 'unknown' if unclear.")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="Model confidence in the decision, 0-1.")


def export(path: Optional[str] = None) -> str:
    p = Path(path) if path else Path(__file__).with_name("triage_schema.json")
    p.write_text(json.dumps(TriageResult.model_json_schema(), indent=2), encoding="utf-8")
    return str(p)


if __name__ == "__main__":
    print("wrote", export())
