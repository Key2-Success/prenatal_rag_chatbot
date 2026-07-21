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
from backend.app.models.schemas import ChatTurn
from backend.app.observability import observe, update_current_span

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
PROMPT_VERSION = "v1.2"  # v1.1: conversation-continuity rules + history block; v1.2: food/ingredient/meal questions are implicitly in_scope (only cooking recipes are out)

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
  Examples: cryptocurrency, stock investing, politics, weather, step-by-step cooking
  recipes (how to prepare a dish), movie or song recommendations, travel destinations,
  religion, astrology, celebrity news.

in_scope
  The user is asking a legitimate question about prenatal nutrition, foods to eat or avoid
  during pregnancy, supplements, hydration, weight gain, antenatal care, or how their diet
  or medical conditions interact with pregnancy. This includes routine questions even when
  they mention chronic conditions like diabetes, hypertension, or anaemia.
  IMPORTANT: every user is a pregnant woman talking to a prenatal nutrition companion, so
  any question naming a food, ingredient, dish, beverage, or meal is implicitly asking
  about eating it during pregnancy — "what about <some food>?", "is <some food> ok?",
  "what should I eat for <some meal>?" are all in_scope. Whether/how a food fits a
  pregnancy diet is in_scope; only step-by-step cooking instructions are out_of_scope.

Tie-breaking rules:
  - Between in_scope and emergency: choose emergency only if the message describes
    acute symptoms occurring now. Chronic conditions or general questions are in_scope.
  - Between in_scope and out_of_scope: prefer in_scope when the topic could plausibly
    relate to nutrition or pregnancy.

Conversation continuity:
  - The message may include a "Recent conversation:" block above the current
    message. Short follow-ups inherit the topic of that conversation: after an
    in-scope discussion of foods to eat, a follow-up like "what about <some
    food>?" or "and in the third trimester?" continues the SAME in-scope topic
    and must be classified in_scope.
  - Classify the CURRENT message in that conversational context, never in isolation.
  - Continuity does not override the other labels: a follow-up that switches to
    an unrelated topic is still out_of_scope, and acute symptoms are still emergency.

Return your label with one short sentence of reasoning."""


# Classifier context caps: the last 2 turns, assistant turns clipped hard.
# Triage only needs the topic gist ("we were discussing iron-rich foods"), not
# the full answer text — keeping this tiny keeps the nano call fast and cheap.
_CLASSIFIER_HISTORY_TURNS = 2
_CLASSIFIER_TURN_CHARS = 300


def _format_history(history: list[ChatTurn]) -> str:
    """Compact 'Recent conversation:' block for the classifier user message."""
    lines = []
    for turn in history[-_CLASSIFIER_HISTORY_TURNS:]:
        content = turn.content.strip()
        if len(content) > _CLASSIFIER_TURN_CHARS:
            content = content[:_CLASSIFIER_TURN_CHARS] + "…"
        lines.append(f"{turn.role.upper()}: {content}")
    return "\n".join(lines)


@observe(name="classify_message")
def classify_message(
    message: str, history: list[ChatTurn] | None = None
) -> MessageClassification:
    """
    Classify the message, optionally in the context of the recent conversation
    (so follow-ups like "what about <food>?" inherit the in-scope topic).
    On any failure, log and return `in_scope` — the answer pipeline has its
    own scope and safety guards.
    """
    # Set explicit input on the span instead of letting @observe capture
    # the function args (per Langfuse skill best practice).
    update_current_span(input={
        "message": message,
        "n_history": len(history) if history else 0,
        "prompt_version": PROMPT_VERSION,
    })
    if history:
        user_content = (
            "Recent conversation:\n" + _format_history(history)
            + "\n\nCurrent message:\n" + message
        )
    else:
        user_content = message
    try:
        completion = get_openai_client().beta.chat.completions.parse(
            model=settings.classifier_model,
            temperature=settings.classifier_temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
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
        # Surface label + reasoning on the span output — visible at a glance
        # in the trace UI without expanding the underlying generation.
        update_current_span(
            output={"label": result.label.value, "reasoning": result.reasoning},
        )
        return result.label
    except Exception:
        # Network blip, rate limit, schema drift — anything. Don't block
        # the user; let the answer pipeline run.
        logger.exception("classifier call failed for message=%r; defaulting to in_scope", message)
        return MessageClassification.in_scope
