"""
[A2 M2] Pydantic model for the Pittsburgh 311 knowledge-base tagging task.

Mirrors the Assignment 1 JSON Schema (assignment01/assignment1/extraction.py
SCHEMA) so C1 measures the same task as Part B. The label definitions live
in the task instruction; the Field descriptions here are short on purpose.

WasteTagsDeptRequired is the ablation for the brief's point that a required
field forces the model to produce a value whether or not the source supports
one: it demands at least one department.
"""
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ServiceFocus = Literal["collection", "dumping", "litter", "graffiti", "weeds/debris",
                       "other_neighborhood", "multiple", "not_stated"]
InformationType = Literal["service_guidance", "issue_taxonomy", "operational_summary"]


class WasteTags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_focus: ServiceFocus = Field(description="Central service focus of the record.")
    information_type: InformationType = Field(description="What kind of record this is.")
    responsible_department: List[str] = Field(
        description="Department names copied exactly from the record; [] if none is stated.")

    @field_validator("responsible_department")
    @classmethod
    def _clean(cls, names):
        names = [n.strip() for n in names]
        if any(not n for n in names):
            raise ValueError("department names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("department names must be unique")
        return names


class WasteTagsDeptRequired(WasteTags):
    responsible_department: List[str] = Field(
        min_length=1, description="Department names copied exactly from the record (at least one).")
