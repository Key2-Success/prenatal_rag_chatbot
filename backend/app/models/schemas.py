"""
schemas.py — Pydantic models for the chat API.

Every value that crosses an external boundary (HTTP request, HTTP response,
or a YAML/JSON file) gets a Pydantic model defined here or in a peer module.
Internal data flow uses these same models — there's no shadow set of dicts.
"""

from enum import Enum

from pydantic import BaseModel, Field


# --- Enums -------------------------------------------------------------------

class DietType(str, Enum):
    non_vegetarian = "Non-Vegetarian"
    ovo_vegetarian = "Ovo-Vegetarian"
    vegetarian = "Vegetarian"


class MedicalCondition(str, Enum):
    low_iron = "Low iron"
    hypertension = "Hypertension"
    diabetes = "Diabetes"


class ResponseType(str, Enum):
    """
    What type of response the pipeline produced. A single discriminator
    makes downstream code (frontend, eval) trivial:

        if response.response_type is ResponseType.emergency: show red banner
        if response.response_type is ResponseType.answer:    show sources

    Rather than:

        if response.guardrail_triggered and response.answer == EMERGENCY_RESPONSE: ...

    which is what the previous dual-bool design required.
    """
    answer = "answer"             # LLM-generated answer with sources
    emergency = "emergency"       # safety guardrail tripped
    out_of_scope = "out_of_scope"  # off-topic guardrail tripped
    no_results = "no_results"     # retrieval found nothing above threshold


# --- User profile ------------------------------------------------------------

class UserProfile(BaseModel):
    """
    The clinical + lifestyle context attached to every chat request. Used
    both to personalise retrieval (diet appended to query) and to ground
    the LLM's answer (injected into the prompt).
    """
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=10, le=60)
    pregnancy_week: int = Field(..., ge=1, le=45)
    diet_type: DietType
    weight_kg: float = Field(..., gt=0, le=300)
    height_cm: float = Field(..., gt=0, le=250)
    medical_conditions: list[MedicalCondition] = Field(default_factory=list)

    def to_context_string(self) -> str:
        """Compact, human-readable summary for injection into the LLM prompt."""
        if self.medical_conditions:
            conditions = ", ".join(c.value for c in self.medical_conditions)
        else:
            conditions = "None"
        return (
            f"Week of pregnancy: {self.pregnancy_week}, "
            f"Diet: {self.diet_type.value}, "
            f"Medical conditions: {conditions}, "
            f"Age: {self.age}, "
            f"Weight: {self.weight_kg}kg, "
            f"Height: {self.height_cm}cm"
        )


# --- Chat request / response ------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_profile: UserProfile


class Source(BaseModel):
    """A citation surfaced alongside an answer."""
    org_display_name: str
    doc_title: str
    page: int
    year_published: int


class ChatResponse(BaseModel):
    """
    The single shape returned by /chat.

    `response_type` is the source of truth for what happened. `sources` is
    non-empty only when response_type == answer. The frontend should branch
    on `response_type`, never parse the answer text.
    """
    response_type: ResponseType
    answer: str
    sources: list[Source] = Field(default_factory=list)
