"""Structured Pittsburgh 311 intake ticket (team final API)."""
from typing import List, Optional

from pydantic import BaseModel, Field


class Ticket311(BaseModel):
    domain: str = Field(description="subtopic_key: streets_mobility, waste_neighborhood, buildings_construction, parks_public_spaces")
    category: str = Field(description="Exact WPRDC category string for the subtopic")
    issue: str = Field(description="Specific 311 issue name")
    department: str = Field(description="Responsible city department")
    missing_information: List[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    abstain: bool = False
    historical_resolution_range: Optional[str] = Field(
        default=None, description="Median or range of historical closure time when known"
    )
