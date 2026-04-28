"""
guardrails.py — Pre-LLM safety + scope checks.

Two layers, each a keyword list compiled into a single word-boundary regex:

  Emergency: detects urgent medical symptoms. Returns a hard-coded redirect
    to healthcare professionals — no LLM call (faster, cheaper, more reliable
    than relying on the model to do the right thing under stress).

  Out-of-scope: detects clearly unrelated questions (finance, politics, etc).
    Returns a polite redirect to in-scope topics.

Why word-boundary regex (not substring match):
  Substring `"blood" in "bloody mary"` returns True. Word boundaries make
  the matcher fire on `blood` but not `bloody`, on `fit` (seizure) but not
  `fitness`. Phrase keywords like "severe pain" still work — \b sits at
  word edges, not at every letter.
"""

import re
from dataclasses import dataclass

from backend.app.models.schemas import ResponseType

EMERGENCY_RESPONSE = (
    "It sounds like you may be experiencing something urgent. "
    "Please contact a qualified healthcare provider or go to your nearest hospital immediately. "
    "In India, you can call 108 for emergency medical services."
)

OUT_OF_SCOPE_RESPONSE = (
    "I'm Poshan Saathi, a nutrition companion for pregnant women. "
    "I can only help with questions about nutrition and antenatal care."
)

NO_RESULTS_RESPONSE = (
    "I'm Poshan Saathi, a nutrition companion for pregnant women. "
    "I don't have information about that topic in my reference guidelines."
)

# Words/phrases suggesting a medical emergency. Erring on the side of
# over-detection is intentional — we'd rather over-trigger "go to a doctor"
# than miss a real emergency.
EMERGENCY_KEYWORDS: tuple[str, ...] = (
    "bleeding", "blood", "severe pain", "can't breathe", "unconscious",
    "fainted", "seizure", "fit", "convulsion", "fever", "chest pain",
    "labour", "labor", "water broke", "waters broke", "no movement",
    "baby not moving", "emergency", "hospital", "ambulance", "urgent",
    "dizziness", "blurred vision", "swelling face", "headache severe",
)

# Topics clearly outside prenatal nutrition / antenatal care.
OUT_OF_SCOPE_KEYWORDS: tuple[str, ...] = (
    "stock", "crypto", "bitcoin", "invest", "finance", "money",
    "politics", "election", "religion", "astrology", "recipe",
    "weather", "travel", "movie", "song", "celebrity",
)


def _compile(keywords: tuple[str, ...]) -> re.Pattern[str]:
    """One alternation regex with word boundaries, case-insensitive."""
    alternation = "|".join(re.escape(k) for k in keywords)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_EMERGENCY_RE = _compile(EMERGENCY_KEYWORDS)
_OUT_OF_SCOPE_RE = _compile(OUT_OF_SCOPE_KEYWORDS)


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of `check_guardrails`. `response_type` is None when nothing fired."""
    triggered: bool
    response_type: ResponseType | None = None
    response: str | None = None


def check_guardrails(message: str) -> GuardrailResult:
    """
    Run both keyword checks. Emergency wins on any tie — if a message
    looks both urgent and off-topic, the urgent path is the safer default.
    """
    if _EMERGENCY_RE.search(message):
        return GuardrailResult(
            triggered=True,
            response_type=ResponseType.emergency,
            response=EMERGENCY_RESPONSE,
        )
    if _OUT_OF_SCOPE_RE.search(message):
        return GuardrailResult(
            triggered=True,
            response_type=ResponseType.out_of_scope,
            response=OUT_OF_SCOPE_RESPONSE,
        )
    return GuardrailResult(triggered=False)
