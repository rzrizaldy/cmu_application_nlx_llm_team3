"""
complaint311.py — Pydantic model for Pittsburgh 311 Streets & Mobility complaint classification.
Used by LLMBox structured_output mode to constrain model outputs to the correct schema.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class ComplaintCategory(str, Enum):
    ROAD = "Road/Street Issues"
    TRAFFIC = "Traffic and Street Sign Issues"
    LIGHT = "Street Light"
    PEDESTRIAN = "Pedestrian/Bicycle Concerns"
    PARKING = "Parking"

class Complaint311Response(BaseModel):
    category: ComplaintCategory = Field(description="One of the five complaint categories")
    department: str = Field(description="The Pittsburgh city department responsible for this complaint")
    entities: List[str] = Field(description="Street names, intersections, neighborhoods, or landmarks mentioned")
    confidence: Optional[float] = Field(default=None, description="Model confidence 0-1 if available")
