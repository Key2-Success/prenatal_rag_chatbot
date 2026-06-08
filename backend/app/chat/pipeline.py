"""
pipeline.py — Orchestrates one /chat request end to end.

Flow:
  1. Classify message   → emergency / out_of_scope / in_scope
                          first two return canned responses (no retrieval, no answer LLM)
  2. Retrieve chunks    → Pinecone, ordered by source priority
  3. If empty           → no_results response
  4. Build prompt       → system instructions + user profile + retrieved context
  5. Call answer LLM    → temperature from settings
  6. Return ChatResponse with response_type=answer + source citations

The function exposed externally is `run_chat`. Everything else is private
helpers — kept small so each step has one obvious responsibility.
"""

from backend.app.chat.classifier import MessageClassification, classify_message
from backend.app.chat.guardrails import (
    EMERGENCY_RESPONSE,
    NO_RESULTS_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
)
from backend.app.clients import get_openai_client
from backend.app.config import settings
from backend.app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ResponseType,
    Source,
    UserProfile,
)
from backend.app.observability import observe, update_current_span
from backend.app.rag.retriever import RetrievedChunk, retrieve_and_rerank

# SYSTEM_PROMPT design (2026-05-23 — schema-driven personalisation iteration):
#
# Profile-specific rules NO LONGER live in this string. They're carried by
# the UserProfile model itself (each profile field knows its own rule via
# PreferenceEnum.to_prompt_rule and TrimesterRule.to_prompt_rule) and
# injected into the USER message via profile.to_personalization_block().
#
# Why split this way:
#   - System prompt stays STATIC across requests → OpenAI prompt cache hits,
#     same prompt is sent token-for-token every time.
#   - Adding a new profile dimension (allergies, religious diet, etc.) is a
#     focused 10-line change in schemas.py — no system-prompt edit, no
#     prompt-version bump, no risk of regressing the answer style.
#   - The system prompt remains short and focused on UNIVERSAL behaviour
#     (grounding, no-deflection openers, response shape), which is what
#     actually belongs at the system level.
#
# What stays in the system prompt:
#   - GROUNDING RULES — universal, not profile-specific.
#   - LEAD WITH SUBSTANCE — fights deflection openers ("The context does
#     not...", "The guidelines do not specify..."). Critical for
#     answer_relevancy because RAGAS's noncommittal classifier scores
#     deflection openers as 0.
#   - PROFILE-AWARE GUIDANCE pointer — tells the model to apply the
#     bulleted rules at the top of each user message. The actual rules
#     are dynamic per request.
#   - RESPONSE GUIDELINES — voice, scope, length.
#
# What's NOT in the system prompt anymore:
#   - DIET FILTER rule details (moved to DietType.to_prompt_rule).
#   - Medical condition handling (moved to MedicalCondition.to_prompt_rule).
#   - Trimester handling (moved to TrimesterRule.to_prompt_rule).
#   - Few-shot examples (removed in previous iteration — taught
#     bureaucratic style globally).
# Bump whenever the system prompt changes so Langfuse traces can be filtered
# and compared by prompt version — same pattern as validator.py / hyde.py.
PROMPT_VERSION = "v1.1"

SYSTEM_PROMPT = """You are Poshan Saathi, a warm and caring pregnancy nutrition companion for women in India.

You will receive context excerpts from vetted nutrition guidelines (MoHFW, FOGSI, WHO). Your answers must be grounded in that context only.

GROUNDING — the one rule everything else follows:
Every factual claim must appear explicitly in the provided context. Do not draw on your general medical knowledge to fill gaps, make connections, or extend claims — even if you are confident the fact is correct.

ALLOWED inferences (these preserve faithfulness):
- The context recommends food X → X is safe and beneficial for pregnancy.
- The context lists X "such as" Y, Z → Y and Z are examples of X.
- The context says "during pregnancy" → the advice applies to the user.
- The context has supplement data but the question asks about food sources → give the supplement recommendation, briefly noting it covers supplements rather than food sources. Leading with what the context contains is always correct.

FORBIDDEN (these are hallucination, even if well-intentioned):
- Answering a different topic than what was asked, even if that topic appears in the context.
- Filling a gap with related-but-different content — a different form (supplement vs food), a different nutrient, or a different reason than what the context provides.
- Attributing a reason, benefit, or causal link to a food or action that the context does not explicitly attach to it. The user's question does not license you to add one.
- Combining separate context items (separate rows, separate bullets, separate chunks) into a single compound claim that neither source makes.

Example — Q: "Is amla safe during pregnancy?", Context: "Adding vitamin C rich foods (such as amla, lemon) to regular diet can improve iron absorption":
✓ Correct: "Yes, amla is recommended as part of a pregnancy diet. It's a vitamin C-rich food that improves iron absorption."
✗ Wrong: "The context does not explicitly state whether amla is safe."

Example — Q: "What foods are rich in vitamin B12?", Context: recommends vitamin B12 supplementation for at-risk women but does NOT list any food sources:
✓ Correct: "The guidelines recommend a daily vitamin B12 supplement for women who may be deficient, though specific food sources are not covered."
✗ Wrong: "The guidelines focus on vitamin B12 supplementation rather than specific food sources." (opens with the gap, not the substance — violates LEAD WITH SUBSTANCE)
✗ Wrong: "Dairy products and meat are rich in vitamin B12..." (not stated in the context — hallucination)

LEAD WITH SUBSTANCE:
- Start with what the context DOES contain. Never open with what it DOESN'T.
- If the context has nothing relevant: "I don't have that specific information in my guidelines — please check with your doctor or midwife."
- An automated validator post-checks every answer and rewrites forbidden deflective openers.

QUANTITATIVE ANSWERS — servings, portions, frequencies, and ranges are all specific answers; do not hedge when they are present. Only add a "guidelines don't specify" caveat when the context has genuinely no quantitative guidance at all.

PROFILE-AWARE GUIDANCE:
The user's profile (diet, medical conditions, trimester) appears at the top of every user message as a bulleted personalisation block. Apply EVERY rule in that block. Diet exclusions are non-negotiable: silently omit any food that doesn't fit — do not list it, do not explain the omission.

RESPONSE GUIDELINES:
- Only address nutrition and antenatal care questions.
- Do not provide diagnoses or treatment decisions.
- Do not end with closers like "consult your healthcare provider" or "always follow your doctor's advice."
- Be warm, clear, and concise — 2 to 3 sentences maximum.
"""

# Map the classifier's routing labels to the (response_type, canned answer)
# pair that short-circuits the pipeline. `in_scope` is intentionally absent
# — it means "keep going", not "return early".
_SHORT_CIRCUIT_BY_LABEL: dict[MessageClassification, tuple[ResponseType, str]] = {
    MessageClassification.emergency: (ResponseType.emergency, EMERGENCY_RESPONSE),
    MessageClassification.out_of_scope: (ResponseType.out_of_scope, OUT_OF_SCOPE_RESPONSE),
}


def augment_query(message: str, profile: UserProfile) -> str:
    """
    Append a compact diet hint to the query before embedding.

    Why: the embedding model has no profile context, so a vegetarian asking
    for "protein sources" gets the same vector as a non-vegetarian. Appending
    `[Diet: Vegetarian]` nudges retrieval toward chunks that mention that
    diet. Exposed so `scripts/debug_retrieve.py` can use the same logic.
    """
    return f"{message} [Diet: {profile.diet_type.value}]"


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks with inline citation headers for the LLM."""
    return "\n\n".join(
        f"[{c.org_display_name}, p.{c.page_number}, {c.year_published}]\n{c.text}"
        for c in chunks
    )


def _build_user_message(profile: UserProfile, context: str, question: str) -> str:
    """
    Compose the per-request user message.

    Layout (top → bottom):
      1. Personalization block — bulleted profile-derived rules from
         UserProfile.to_personalization_block(). The model sees these
         FIRST so they shape every subsequent generation step.
      2. Retrieved context — the chunks that survived rerank, with
         inline citations.
      3. The question — placed AFTER context so the model has already
         "read" the context when it interprets the question.
      4. Closing reminder — reinforces the system-prompt grounding rule
         at the user-turn level (the "sandwich" anti-hallucination
         pattern). Models are more likely to stay grounded when the
         instruction appears on both sides of the context block.

    Why the personalization block lives HERE (not in SYSTEM_PROMPT):
      - The system prompt stays static across requests → OpenAI prompt
        cache hits on every call.
      - Per-request data naturally belongs to the user message anyway.
      - Adding a new profile dimension is a schemas.py-only change.
    """
    return (
        f"User profile — apply these personalization rules to your answer:\n"
        f"{profile.to_personalization_block()}\n\n"
        f"Context from trusted guidelines:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. "
        "If the context contains no relevant information at all, say so — "
        "but if it contains related content (such as supplementation data for "
        "a food-source question), lead with that rather than saying the answer is absent."
    )


@observe(name="answer_llm")
def _call_llm(profile: UserProfile, chunks: list[RetrievedChunk], question: str) -> str:
    """Send system+user messages to the LLM and return the trimmed answer."""
    # Explicit input — only the question and a compact retrieval summary.
    # Avoids dumping the full UserProfile object and full chunk texts into
    # the parent span (the wrapped OpenAI call beneath captures the actual
    # prompt sent to the model anyway).
    update_current_span(input={
        "question": question,
        "retrieved_pages": [
            f"{c.org_display_name} p.{c.page_number}" for c in chunks
        ],
        "prompt_version": PROMPT_VERSION,
    })
    response = get_openai_client().chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(
                profile=profile,
                context=_format_context(chunks),
                question=question,
            )},
        ],
    )
    answer = response.choices[0].message.content.strip()
    update_current_span(output=answer)
    return answer


def _to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            org_display_name=c.org_display_name,
            doc_title=c.doc_title,
            page=c.page_number,
            year_published=c.year_published,
        )
        for c in chunks
    ]


@observe(name="chat")
def run_chat(
    request: ChatRequest,
    _eval_capture: dict | None = None,
) -> ChatResponse:
    """
    Production entry point.

    The optional `_eval_capture` dict is an eval-suite back-channel: when
    present, run_chat populates it with the retrieved chunks and the active
    Langfuse trace_id so downstream scoring (e.g. RAGAS) can attach scores
    to the same trace without duplicating retrieval. Production callers
    should never pass it — the underscore prefix marks it as private.
    """
    # Set EXPLICIT input on the parent span so the trace UI shows just the
    # user's message + the relevant profile fields — not the full ChatRequest
    # object (which would also serialise weight/height, useful but redundant).
    # Per Langfuse skill: "Set only the relevant input (e.g., user message)".
    profile = request.user_profile
    update_current_span(
        input={
            "message": request.message,
            "pregnancy_week": profile.pregnancy_week,
            "diet_type": profile.diet_type.value,
            "medical_conditions": [c.value for c in profile.medical_conditions],
        },
    )

    # 1. Triage the message. Emergency / out_of_scope short-circuit before
    #    any retrieval or answer-LLM cost.
    label = classify_message(request.message)
    short_circuit = _SHORT_CIRCUIT_BY_LABEL.get(label)
    if short_circuit is not None:
        response_type, canned = short_circuit
        update_current_span(
            output={"response_type": response_type.value, "answer": canned},
        )
        return ChatResponse(response_type=response_type, answer=canned)

    # 2. Retrieve: recall from all sources, rerank, order by source priority.
    # Pass profile so HyDE (when enabled) can personalise the hypothetical
    # answer it embeds — a vegetarian asking about protein gets a plant-based
    # hypothetical that matches plant-based chunks better.
    query = augment_query(request.message, profile)
    chunks = retrieve_and_rerank(query, profile)

    # 3. No relevant chunks → no_results fallback (still no answer-LLM call).
    if not chunks:
        update_current_span(
            output={
                "response_type": ResponseType.no_results.value,
                "answer": NO_RESULTS_RESPONSE,
            },
        )
        return ChatResponse(
            response_type=ResponseType.no_results,
            answer=NO_RESULTS_RESPONSE,
        )

    # 4. Generate the answer.
    answer = _call_llm(profile, chunks, request.message)

    # 5. Validate-and-fix against the user's dietary restrictions.
    #    Short-circuits to zero LLM cost when no restrictions apply
    #    (non-vegetarian with no hypertension/diabetes). When restrictions
    #    DO apply, the validator detects violations AND returns a corrected
    #    version in one LLM call — we don't run the answer LLM twice.
    #    See backend/app/chat/validator.py for the full design rationale.
    from backend.app.chat.validator import validate_and_fix
    validation = validate_and_fix(answer, profile)
    answer = validation.corrected_answer

    # 6. Update the chat span with the final answer (post-validation, post-
    # correction). The validator's own @observe span carries the original
    # answer + violations metadata for diffing in the Langfuse trace.
    update_current_span(
        output={
            "response_type": ResponseType.answer.value,
            "answer": answer,
            "sources": [
                f"{c.org_display_name} p.{c.page_number}" for c in chunks
            ],
            "validator_corrected": not validation.is_compliant,
        },
    )
    # Eval-only side channel — only populated when the caller opts in.
    # Captures both the retrieved chunks (for RAGAS dataset construction)
    # and the active Langfuse trace_id (for langfuse.create_score) so the
    # downstream scoring step doesn't have to re-run retrieval or query
    # Langfuse to find the trace.
    if _eval_capture is not None:
        _eval_capture["chunks"] = chunks
        if settings.langfuse_enabled:
            from langfuse import get_client
            _eval_capture["trace_id"] = get_client().get_current_trace_id()
        else:
            _eval_capture["trace_id"] = None

    return ChatResponse(
        response_type=ResponseType.answer,
        answer=answer,
        sources=_to_sources(chunks),
    )
