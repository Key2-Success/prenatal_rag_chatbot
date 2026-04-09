"""
guardrails.py — Two-layer safety checks applied before every LLM call.

Layer 1 — Emergency guardrail:
  Detects urgent medical symptoms. Returns a hard-coded redirect to
  healthcare professionals. We do NOT call the LLM for these — it's
  faster, cheaper, and more reliable.

Layer 2 — Scope guardrail:
  Detects clearly off-topic questions (e.g. finance, politics).
  Returns a polite out-of-scope response.

Design note: keyword matching is intentionally broad (we'd rather
over-trigger a "go to a doctor" message than miss a real emergency).
"""

from dataclasses import dataclass

EMERGENCY_RESPONSE = (
    "It sounds like you may be experiencing something urgent. "
    "Please contact a qualified healthcare provider or go to your nearest hospital immediately. "
    "In India, you can call 108 for emergency medical services."
)

FALLBACK_RESPONSE = (
    "I'm Poshan Saathi, a nutrition companion for pregnant women. "
    "I can only help with questions about nutrition and antenatal care. "
    "I don't have information about that topic."
)

# Words that suggest a medical emergency or symptoms needing immediate care.
# Erring on the side of over-detection is intentional.
EMERGENCY_KEYWORDS = [
    "bleeding", "blood", "severe pain", "can't breathe", "unconscious",
    "fainted", "seizure", "fit", "convulsion", "fever", "chest pain",
    "labour", "labor", "water broke", "waters broke", "no movement",
    "baby not moving", "emergency", "hospital", "ambulance", "urgent",
    "dizziness", "blurred vision", "swelling face", "headache severe",
]

# Topics clearly outside prenatal/nutritional scope.
OUT_OF_SCOPE_KEYWORDS = [
    "stock", "crypto", "bitcoin", "invest", "finance", "money",
    "politics", "election", "religion", "astrology", "recipe",
    "weather", "travel", "movie", "song", "celebrity",
]


@dataclass
class GuardrailResult:
    triggered: bool
    response: str | None = None
    kind: str | None = None  # "emergency" | "out_of_scope"


def check_guardrails(message: str) -> GuardrailResult:
    """
    Checks message against emergency and out-of-scope keyword lists.
    Returns a GuardrailResult — if triggered=True, use the response directly
    and skip the RAG + LLM pipeline entirely.
    """
    lowered = message.lower()

    for keyword in EMERGENCY_KEYWORDS:
        if keyword in lowered:
            return GuardrailResult(
                triggered=True, response=EMERGENCY_RESPONSE, kind="emergency"
            )

    for keyword in OUT_OF_SCOPE_KEYWORDS:
        if keyword in lowered:
            return GuardrailResult(
                triggered=True, response=FALLBACK_RESPONSE, kind="out_of_scope"
            )

    return GuardrailResult(triggered=False)
