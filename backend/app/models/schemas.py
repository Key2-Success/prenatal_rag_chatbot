from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DietType(str, Enum):
    non_vegetarian = "Non-Vegetarian"
    ovo_vegetarian = "Ovo-Vegetarian"
    vegetarian = "Vegetarian"


class MedicalCondition(str, Enum):
    low_iron = "Low iron"
    hypertension = "Hypertension"
    diabetes = "Diabetes"


# --- User Profile ---
# Mirrors the UserProfile class from the notebook, but as a Pydantic model.
# Pydantic automatically validates types and provides helpful error messages
# when the frontend sends bad data (e.g. age as a string).

class UserProfile(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=10, le=60)
    pregnancy_week: int = Field(..., ge=1, le=45)
    diet_type: DietType
    weight_kg: float = Field(..., gt=0, le=300)
    height_cm: float = Field(..., gt=0, le=250)
    medical_conditions: list[MedicalCondition] = Field(default_factory=list)

    def to_context_string(self) -> str:
        """Formats profile as a compact string for injection into LLM prompts."""
        conditions = (
            ", ".join(self.medical_conditions) if self.medical_conditions else "None"
        )
        return (
            f"Week of pregnancy: {self.pregnancy_week}, "
            f"Diet: {self.diet_type}, "
            f"Medical conditions: {conditions}, "
            f"Age: {self.age}, "
            f"Weight: {self.weight_kg}kg, "
            f"Height: {self.height_cm}cm"
        )


# --- Chat ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_profile: UserProfile


class Source(BaseModel):
    org_display_name: str
    doc_title: str
    page: int
    year_published: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    guardrail_triggered: bool = False
    fallback_triggered: bool = False
