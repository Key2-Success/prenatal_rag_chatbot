"""
classifier.py — LLM-based intent triage for incoming chat messages.

Replaces the old keyword-regex guardrails. Word-boundary keywords could not
disambiguate cases like:

  "What should I avoid eating to keep my blood sugar in check?"

The literal word `blood` is in the emergency list, so the regex fired and
the user got an emergency redirect instead of a nutrition answer. Adding
more exceptions would have spiraled — every rule needed a counter-rule.
A small LLM call gets the obvious cases right without that maintenance tax.

Three labels, mapped 1:1 to pipeline behaviour:
  - in_scope     → continue to retrieval + answer LLM
  - emergency    → return EMERGENCY_RESPONSE, no retrieval
  - out_of_scope → return OUT_OF_SCOPE_RESPONSE, no retrieval

Failure mode is fail-open: if the classifier call raises, we log loudly
and return `in_scope`. Rationale — the answer LLM has its own scope rule
in its system prompt, and the emergency redirect is a *safety net*, not
the only line of defence. Blocking a legitimate nutrition question because
OpenAI hiccuped would be worse than letting the answer LLM handle it.

Why a separate enum (MessageClassification) instead of reusing ResponseType:
  ResponseType describes what the pipeline returned to the user. The
  `answer` value implies retrieval ran and an LLM answered. The classifier
  doesn't know that yet — it only knows whether the message *should* be
  routed to the answer path. Keeping them distinct keeps the routing
  signal from leaking into the response shape.
"""

import logging
from enum import Enum

from pydantic import BaseModel

from backend.app.clients import get_openai_client
from backend.app.config import settings

logger = logging.getLogger(__name__)


class MessageClassification(str, Enum):
    """How the pipeline should route a chat message."""
    in_scope = "in_scope"
    emergency = "emergency"
    out_of_scope = "out_of_scope"


class ClassificationResult(BaseModel):
    """
    Structured output from the classifier LLM.

    `reasoning` is captured for debuggability — when the classifier
    mislabels, the log line tells us why without re-running anything.
    """
    label: MessageClassification
    reasoning: str


# Prompt notes:
#  - Explicitly tells the model that mentioning a medical word is NOT
#    sufficient for emergency. The "blood sugar" failure case is called
#    out by name so the model has a concrete anchor.
#  - "When in doubt, prefer in_scope" biases the classifier toward letting
#    the answer LLM handle ambiguous cases — its system prompt also enforces
#    scope, so we don't need to be over-aggressive here.
_SYSTEM_PROMPT = """You are a triage classifier for Poshan Saathi, a prenatal nutrition chatbot serving pregnant women in India.

Classify the user's message into exactly one of three labels:

emergency
  The user describes an URGENT medical situation requiring immediate professional care.
  Examples: heavy bleeding, severe abdominal pain happening now, baby has stopped moving,
  water broke, loss of consciousness, seizure, chest pain, sudden severe headache with
  vision changes, high fever during pregnancy.
  IMPORTANT: do NOT mark a message as emergency just because it mentions a medical word.
  "What should I avoid eating to keep my blood sugar in check?" is NOT an emergency —
  it is a routine nutrition question from someone managing diabetes.

out_of_scope
  The user is asking about something unrelated to prenatal nutrition or antenatal care.
  Examples: cryptocurrency, stock investing, politics, weather, recipes for specific dishes,
  movie or song recommendations, travel destinations, religion, astrology, celebrity news.

in_scope
  The user is asking a legitimate question about prenatal nutrition, foods to eat or avoid
  during pregnancy, supplements, hydration, weight gain, antenatal care, or how their diet
  or medical conditions interact with pregnancy. This includes routine questions even when
  they mention chronic conditions like diabetes, hypertension, or anaemia.

Tie-breaking rules:
  - Between in_scope and emergency: choose emergency only if the message describes
    acute symptoms occurring now. Chronic conditions or general questions are in_scope.
  - Between in_scope and out_of_scope: prefer in_scope when the topic could plausibly
    relate to nutrition or pregnancy.

Return your label with one short sentence of reasoning."""


def classify_message(message: str) -> MessageClassification:
    """
    Classify the message. On any failure, log and return `in_scope` —
    the answer pipeline has its own scope and safety guards.
    """
    try:
        completion = get_openai_client().beta.chat.completions.parse(
            model=settings.classifier_model,
            temperature=settings.classifier_temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            response_format=ClassificationResult,
        )
        result = completion.choices[0].message.parsed
        if result is None:
            # `parsed` is None if the model refused or output couldn't be
            # coerced. Treat the same as a hard failure: fail open.
            logger.warning(
                "classifier returned no parsed result for message=%r; defaulting to in_scope",
                message,
            )
            return MessageClassification.in_scope
        logger.debug(
            "classifier label=%s reasoning=%s message=%r",
            result.label.value, result.reasoning, message,
        )
        return result.label
    except Exception:
        # Network blip, rate limit, schema drift — anything. Don't block
        # the user; let the answer pipeline run.
        logger.exception("classifier call failed for message=%r; defaulting to in_scope", message)
        return MessageClassification.in_scope
