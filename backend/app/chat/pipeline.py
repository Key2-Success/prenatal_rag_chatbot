"""
pipeline.py — Orchestrates one /chat request end to end.

Flow:
  1. Guardrail check    → if tripped, return early (no LLM call)
  2. Retrieve chunks    → Pinecone, ordered by source priority
  3. If empty           → no_results response
  4. Build prompt       → system instructions + user profile + retrieved context
  5. Call LLM           → temperature from settings
  6. Return ChatResponse with response_type=answer + source citations

The function exposed externally is `run_chat`. Everything else is private
helpers — kept small so each step has one obvious responsibility.
"""

from backend.app.chat.guardrails import NO_RESULTS_RESPONSE, check_guardrails
from backend.app.clients import get_openai_client
from backend.app.config import settings
from backend.app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ResponseType,
    Source,
    UserProfile,
)
from backend.app.rag.retriever import RetrievedChunk, retrieve_ordered

SYSTEM_PROMPT = """You are Poshan Saathi, a friendly pregnancy nutrition companion for women in India.

Rules:
- Only answer questions about nutrition and antenatal care.
- Tailor your answer to the user's diet, pregnancy week, and medical conditions.
- Be warm, clear, and concise — 2 to 3 sentences maximum.
- Do not provide medical diagnoses or treatment decisions.
- If the provided context does not contain enough information to answer, say so honestly.
"""


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
    return (
        f"User profile:\n{profile.to_context_string()}\n\n"
        f"Context from trusted guidelines:\n{context}\n\n"
        f"Question: {question}"
    )


def _call_llm(profile: UserProfile, chunks: list[RetrievedChunk], question: str) -> str:
    """Send system+user messages to the LLM and return the trimmed answer."""
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
    return response.choices[0].message.content.strip()


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


def run_chat(request: ChatRequest) -> ChatResponse:
    # 1. Guardrail check — short-circuit before spending money on retrieval/LLM.
    guardrail = check_guardrails(request.message)
    if guardrail.triggered:
        assert guardrail.response_type is not None and guardrail.response is not None
        return ChatResponse(
            response_type=guardrail.response_type,
            answer=guardrail.response,
        )

    # 2. Retrieve.
    query = augment_query(request.message, request.user_profile)
    chunks = retrieve_ordered(query)

    # 3. No relevant chunks → no_results fallback (still no LLM call).
    if not chunks:
        return ChatResponse(
            response_type=ResponseType.no_results,
            answer=NO_RESULTS_RESPONSE,
        )

    # 4–6. Generate and package.
    answer = _call_llm(request.user_profile, chunks, request.message)
    return ChatResponse(
        response_type=ResponseType.answer,
        answer=answer,
        sources=_to_sources(chunks),
    )
