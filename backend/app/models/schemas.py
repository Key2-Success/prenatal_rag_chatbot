"""
schemas.py — Pydantic models for the chat API.

Every value that crosses an external boundary (HTTP request, HTTP response,
or a YAML/JSON file) gets a Pydantic model defined here or in a peer module.
Internal data flow uses these same models — there's no shadow set of dicts.

Personalisation architecture (see PreferenceEnum docstring for full detail):
  Profile-derived rules that shape the LLM's answer are NOT hardcoded in the
  system prompt. Instead, each preference (diet, conditions, trimester)
  carries its own `to_prompt_rule()` method. UserProfile assembles these into
  a per-request personalization block that gets injected into the user
  message. Adding a new profile dimension (e.g. allergies) is a focused
  edit here — the system prompt stays static and cacheable.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


# --- Personalisation architecture --------------------------------------------

@runtime_checkable
class PromptRuleProvider(Protocol):
    """
    Anything that contributes a single rule to the per-request personalization
    block. Implementations come in two shapes:
      - PreferenceEnum subclasses (DietType, MedicalCondition, future
        Allergy, ...) for discrete-value preferences.
      - Plain dataclasses (TrimesterRule, future BMIRule, ...) for
        continuous/numeric preferences that don't fit the enum pattern.

    Both shapes satisfy this Protocol via duck typing + runtime_checkable.
    Static type checkers also surface missing `to_prompt_rule` at edit time.
    """
    def to_prompt_rule(self) -> str: ...


class PreferenceEnum(str, Enum):
    """
    Base for all user-profile preference enums (DietType, MedicalCondition,
    future Allergy, ...). Subclasses MUST override `to_prompt_rule()`.

    Two-level enforcement runs at CLASS-DEFINITION time (not first request):

    1. Method-override check — TypeError if a subclass inherits the
       NotImplementedError stub unchanged. Catches "forgot to define
       to_prompt_rule entirely" at import time.

    2. Per-value coverage check — every enum member's `to_prompt_rule()`
       is called; TypeError if any raises (KeyError from a missing dict
       entry, NotImplementedError, AttributeError) or returns an empty/
       whitespace-only string. Catches "added a new enum value but forgot
       to update the rule dict" at import time.

    Net effect: a misconfigured preference enum crashes at import — the
    app refuses to start. Same loud-failure UX as a static type error,
    no silent KeyErrors at first request.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # 1. Method-override check.
        if cls.to_prompt_rule is PreferenceEnum.to_prompt_rule:
            raise TypeError(
                f"{cls.__name__} inherits from PreferenceEnum but does not "
                f"override to_prompt_rule(). Every preference enum must "
                f"provide its own rule — see DietType for an example."
            )

        # 2. Per-value coverage check. Iterate every enum member, call
        # its to_prompt_rule(), and verify the result is a non-empty string.
        missing: list[str] = []
        empty: list[str] = []
        for member in cls:
            try:
                rule = member.to_prompt_rule()
            except (KeyError, NotImplementedError, AttributeError) as e:
                missing.append(f"{member.name} ({type(e).__name__}: {e})")
                continue
            if not isinstance(rule, str) or not rule.strip():
                empty.append(member.name)

        if missing:
            raise TypeError(
                f"{cls.__name__}.to_prompt_rule() is missing or broken for "
                f"these values: {missing}. Add a rule for each value before "
                f"the app can start."
            )
        if empty:
            raise TypeError(
                f"{cls.__name__}.to_prompt_rule() returned empty strings for "
                f"these values: {empty}. Each rule must be non-empty."
            )

    def to_prompt_rule(self) -> str:
        raise NotImplementedError(
            "Override to_prompt_rule on the subclass."
        )


# --- Enums -------------------------------------------------------------------

class DietType(PreferenceEnum):
    non_vegetarian = "Non-Vegetarian"
    ovo_vegetarian = "Ovo-Vegetarian"
    vegetarian = "Vegetarian"

    def to_prompt_rule(self) -> str:
        # Keyed by self.name (the Python identifier) rather than enum members
        # because __init_subclass__ calls this method DURING class definition,
        # at which point DietType isn't bound yet — a reference like
        # `DietType.vegetarian` inside the dict literal would fail with
        # NameError. self.name is a plain string and always resolvable.
        return {
            "vegetarian":
                "Vegetarian — omit meat, poultry, fish, and eggs from any food recommendations.",
            "ovo_vegetarian":
                "Ovo-Vegetarian — omit meat, poultry, and fish (eggs are fine).",
            "non_vegetarian":
                "Non-Vegetarian — no food restrictions apply.",
        }[self.name]

    def to_validation_rule(self) -> str | None:
        """
        Optional check rule for the post-answer validator.

        Returns None when no validation is needed (e.g. non-vegetarian has
        no exclusions). Returns a string instruction otherwise. The string
        is fed to the validator LLM verbatim.
        """
        return {
            "vegetarian": (
                "User is Vegetarian. Flag ANY food the answer RECOMMENDS that "
                "is meat, poultry, fish, seafood, or eggs (including dishes "
                "containing them like 'chicken biryani', 'fish curry', 'egg "
                "bhurji'). Do NOT flag mentions where the answer is telling "
                "the user to AVOID those foods."
            ),
            "ovo_vegetarian": (
                "User is Ovo-Vegetarian. Flag ANY food the answer RECOMMENDS "
                "that is meat, poultry, fish, or seafood (including dishes "
                "containing them). Eggs ARE allowed. Do NOT flag mentions "
                "where the answer is telling the user to AVOID those foods."
            ),
            "non_vegetarian": None,
        }[self.name]


class MedicalCondition(PreferenceEnum):
    low_iron = "Low iron"
    hypertension = "Hypertension"
    diabetes = "Diabetes"

    def to_prompt_rule(self) -> str:
        # Keyed by self.name — see DietType.to_prompt_rule for why.
        return {
            "low_iron":
                "Has low iron — emphasize iron-rich foods, iron supplementation thresholds, and anaemia management.",
            "hypertension":
                "Has hypertension — emphasize sodium restriction and salt limits.",
            "diabetes":
                "Has gestational diabetes — emphasize blood-sugar management, glycemic load, and avoiding refined carbohydrates.",
        }[self.name]

    def to_validation_rule(self) -> str | None:
        """
        Optional check rule for the post-answer validator.

        Returns None for conditions where the guidance is EMPHASIS (do more
        of X) rather than RESTRICTION (don't do X). Low iron is emphasis only.
        """
        return {
            "low_iron": None,  # Emphasis (eat MORE iron), not a restriction
            "hypertension": (
                "User has hypertension. Flag ANY food the answer RECOMMENDS "
                "that is high in sodium: pickles, papad, salty snacks, "
                "namkeen, processed foods, instant noodles, packaged soups, "
                "salted nuts, or foods explicitly described as high-salt. "
                "Do NOT flag mentions where the answer is telling the user "
                "to AVOID those foods. Do NOT flag normal salt usage in "
                "cooking unless it's an explicit recommendation."
            ),
            "diabetes": (
                "User has gestational diabetes. Flag ANY food the answer "
                "RECOMMENDS that is high in refined sugars or refined carbs: "
                "sweets, mithai, sugary drinks, fruit juices, large amounts "
                "of white rice, refined-flour (maida) products, candies, "
                "biscuits, cakes, or desserts. Do NOT flag mentions where "
                "the answer is telling the user to AVOID those foods."
            ),
        }[self.name]


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


# --- Numeric / continuous rule providers -------------------------------------

@dataclass(frozen=True)
class TrimesterRule:
    """
    Wraps pregnancy_week (int) as a PromptRuleProvider.

    Why a wrapper class instead of a method on UserProfile: keeps every rule
    provider behind the same Protocol interface, so UserProfile._collect_rule_providers
    can iterate them uniformly. Continuous-numeric fields don't fit the
    PreferenceEnum per-value enforcement (no discrete set to enumerate), so
    they use range-based if/elif with an `else` fallback that guarantees no
    input is unhandled.

    Boundaries: weeks 1-12 = T1, 13-26 = T2, 27+ = T3. Pydantic constrains
    pregnancy_week to [1, 45] at the UserProfile boundary, so we don't need
    to defend against out-of-range values here.
    """
    week: int

    def to_prompt_rule(self) -> str:
        if self.week <= 12:
            return (
                f"1st trimester (week {self.week}) — emphasize folic acid "
                f"and neural tube development."
            )
        elif self.week <= 26:
            return (
                f"2nd trimester (week {self.week}) — iron supplementation "
                f"begins, weight gain ramps."
            )
        else:
            return (
                f"3rd trimester (week {self.week}) — continued iron and "
                f"calcium, late-pregnancy considerations."
            )


# --- User profile ------------------------------------------------------------

class UserProfile(BaseModel):
    """
    The clinical + lifestyle context attached to every chat request. Used
    both to personalise retrieval (diet appended to query) and to ground
    the LLM's answer (injected into the prompt via to_personalization_block).
    """
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=10, le=60)
    pregnancy_week: int = Field(..., ge=1, le=45)
    diet_type: DietType
    weight_kg: float = Field(..., gt=0, le=300)
    height_cm: float = Field(..., gt=0, le=250)
    medical_conditions: list[MedicalCondition] = Field(default_factory=list)

    def _collect_rule_providers(self) -> list[PromptRuleProvider]:
        """
        Gather all profile-derived rule providers in display order.

        Adding a new profile dimension is a focused edit here — append to
        the list, and the rest of the system (system prompt, the user
        message builder, the eval) stays unchanged. The schema is the
        single source of truth for what the LLM knows about the user.
        """
        providers: list[PromptRuleProvider] = [self.diet_type]
        providers.extend(self.medical_conditions)
        providers.append(TrimesterRule(self.pregnancy_week))
        return providers

    def to_personalization_block(self) -> str:
        """
        Build the per-request personalization block injected into the LLM's
        user message. One bulleted line per rule provider.

        Format chosen for explicit-rule readability — the LLM treats each
        bullet as a separate instruction, which gives stronger enforcement
        than dense prose. Goes in the USER message (not system) so the
        static system prompt stays cacheable across requests.
        """
        return "\n".join(
            f"- {p.to_prompt_rule()}" for p in self._collect_rule_providers()
        )

    def to_context_string(self) -> str:
        """
        Compact, human-readable summary for tracing/logging — NOT the LLM
        personalization block. Used by observability spans where we want a
        one-line profile snapshot for grep-ability. The personalization
        block (above) is what reaches the LLM.
        """
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
    """
    A citation surfaced alongside an answer.

    `chunk_text` and `relevance_score` expose the actual retrieved passage
    and its cross-encoder (bge-reranker-v2-m3) relevance score — the same
    score shown in Langfuse traces. They power the frontend's citation
    hover-card ("show the RAG working"). Both are pass-throughs from
    RetrievedChunk; no extra computation. The score is a sigmoid-normalised
    0–1 relevance value, not raw cosine similarity (cosine is not retained
    past the rerank stage, and the cross-encoder score is more meaningful).
    """
    org_display_name: str
    doc_title: str
    page: int
    year_published: int
    chunk_text: str
    relevance_score: float


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
