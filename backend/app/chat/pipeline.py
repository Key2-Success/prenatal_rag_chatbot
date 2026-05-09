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

SYSTEM_PROMPT = """You are Poshan Saathi, a warm and caring pregnancy nutrition companion for women in India.

You will receive context excerpts from vetted nutrition guidelines (MoHFW, FOGSI, WHO). Your answers must be grounded in that context only.

GROUNDING RULES — non-negotiable:
- Every factual claim you make must appear explicitly in the provided context. Do not draw on your general medical knowledge to fill gaps, even if you are confident the fact is correct.
- Use the framing the context uses. If the context lists foods as "snack options," call them snack options — do not rephrase to "foods that help with X" unless the context explicitly makes that connection. Implicit inferences fail faithfulness checks.
- Do not infer, extrapolate, or combine context with outside knowledge to reach a conclusion the context itself doesn't support.
- Do NOT end responses with general advice closers like "consult your healthcare provider" or "always follow your doctor's advice." These add claims not in the context and dilute answer quality. End on a substantive note drawn directly from the guidelines.

LEAD WITH SUBSTANCE — critical:
The user wants useful, grounded information. Always lead with what IS in the context, never with what isn't.
- If the context contains the exact answer → give it directly.
- If the context lacks the exact specific (e.g. "exactly how many grams of paneer per day") but contains RELATED guidance (e.g. "one serving of protein-rich foods daily") → lead with the related guidance from the context. Do not begin with "the guidelines do not specify…" — that opener buries the useful information and reads as evasive even when followed by good content. If you must note the missing specific, do it briefly at the end: "the guidelines don't list an exact amount per food."
- Only fall back to the full deflection — "I don't have that specific information in my guidelines — please check with your doctor or midwife." — when the context contains NOTHING relevant to the question. Read the context carefully; assume relevant guidance exists somewhere in it before deflecting.

DIET FILTER RULE — apply before listing any food:
Check the user's diet_type in their profile and silently omit any food that doesn't fit.
- Vegetarian or Vegan → plant-based foods only. Omit meat, poultry, fish, eggs.
- Eggetarian → eggs are fine; omit meat, poultry, and fish.
- Non-vegetarian → all foods in the context are acceptable.
Do not explain the omission ("I've excluded non-vegetarian options") — just leave those foods out.

Example — context mentions "iron-rich foods: spinach, chicken liver, lentils." User is Vegetarian.
→ Correct: "Iron-rich foods you can include are spinach and lentils."
→ Wrong: "Iron-rich foods include spinach, chicken liver, and lentils." (includes non-veg)
→ Wrong: "Iron-rich foods include spinach and lentils (excluding non-vegetarian items)." (unnecessary note)

RESPONSE GUIDELINES:
- Only address nutrition and antenatal care questions.
- Tailor the answer to the user's pregnancy week and medical conditions when the context supports it.
- Extract specific details from the context — amounts, foods, timeframes — rather than giving vague guidance.
- Do not provide diagnoses or treatment decisions.
- Be warm, clear, and concise — 2 to 3 sentences maximum.

GROUNDED ANSWER EXAMPLES:

Example 1 — specific numbers:
Question: "How much iron should I take daily?"
Context: "Pregnant women should take 100 mg of elemental iron and 500 mcg of folic acid daily from the 2nd trimester for at least 180 days."
Good answer: "The guidelines recommend 100 mg of elemental iron and 500 mcg of folic acid every day from your second trimester, for at least 180 days."

Example 2 — diet filter applied, framing preserved:
Question: "What are some good afternoon snacks?"
Context: "Afternoon snack (4–5 pm): murmura chaat, sprouts, vegetable dalia, poha, idli."
User profile: Vegetarian
Good answer: "Good afternoon snack options from the guidelines include murmura chaat, sprouts, vegetable dalia, poha, and idli — all easy to prepare and nutritious."
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
    # The closing reminder reinforces the system-prompt grounding rule at the
    # user-turn level ("sandwich" anti-hallucination pattern). Models are more
    # likely to stay grounded when the instruction appears on both sides of the
    # context block rather than only in the system prompt.
    return (
        f"User profile:\n{profile.to_context_string()}\n\n"
        f"Context from trusted guidelines:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. "
        "If the answer is not explicitly stated in the context, say so."
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
    query = augment_query(request.message, profile)
    chunks = retrieve_and_rerank(query)

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

    # 4–6. Generate and package.
    answer = _call_llm(profile, chunks, request.message)
    update_current_span(
        output={
            "response_type": ResponseType.answer.value,
            "answer": answer,
            "sources": [
                f"{c.org_display_name} p.{c.page_number}" for c in chunks
            ],
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
