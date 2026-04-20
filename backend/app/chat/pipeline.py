"""
pipeline.py — Orchestrates the full RAG → LLM → response flow.

Flow for every /chat request:
  1. Guardrail check  →  if triggered, return immediately (no LLM call)
  2. Retrieve chunks from Pinecone (ordered: MoHFW → FOGSI → WHO)
  3. If no chunks above threshold  →  return fallback response
  4. Build prompt: system instructions + user profile + retrieved context + question
  5. Call OpenAI LLM
  6. Return answer + source citations
"""

from openai import OpenAI

from backend.app.config import settings
from backend.app.chat.guardrails import check_guardrails, FALLBACK_RESPONSE
from backend.app.rag.retriever import retrieve_ordered
from backend.app.models.schemas import ChatRequest, ChatResponse, Source

LLM_MODEL = "gpt-4.1-nano"  # matches your original; swap to gpt-4o for higher quality

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def build_prompt(user_context: str, retrieved_context: str, question: str) -> str:
    return f"""You are Poshan Saathi, a friendly pregnancy nutrition companion for women in India.

User profile:
{user_context}

Rules:
- Only answer questions about nutrition and antenatal care.
- Tailor your answer to the user's diet, pregnancy week, and medical conditions above.
- Be warm, clear, and concise — 2 to 3 sentences maximum.
- Do not provide medical diagnoses or treatment decisions.
- If the context does not contain enough information to answer, say so honestly.

Context from trusted guidelines:
{retrieved_context}

Question: {question}

Answer:"""


def run_chat(request: ChatRequest) -> ChatResponse:
    # Step 1: Guardrail check
    guardrail = check_guardrails(request.message)
    if guardrail.triggered:
        return ChatResponse(
            answer=guardrail.response,
            sources=[],
            guardrail_triggered=True,
        )

    # Step 2: Retrieve — append diet to query for better semantic match.
    # .value unwraps the Enum to its string ("Vegetarian" vs "DietType.vegetarian").
    augmented_query = (
        f"{request.message} [Diet: {request.user_profile.diet_type.value}]"
    )
    chunks = retrieve_ordered(augmented_query)

    # Step 3: Fallback if no relevant chunks found
    if not chunks:
        return ChatResponse(
            answer=FALLBACK_RESPONSE,
            sources=[],
            fallback_triggered=True,
        )

    # Step 4: Build context string with inline citations
    retrieved_context = "\n\n".join([
        f"[{c['org_display_name']}, p.{c['page_number']}, {c['year_published']}]\n{c['text']}"
        for c in chunks
    ])

    # Step 5: Call LLM
    prompt = build_prompt(
        user_context=request.user_profile.to_context_string(),
        retrieved_context=retrieved_context,
        question=request.message,
    )

    response = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # lower temp = more consistent, factual answers
    )
    answer = response.choices[0].message.content.strip()

    # Step 6: Format sources for the frontend
    sources = [
        Source(
            org_display_name=c["org_display_name"],
            doc_title=c["doc_title"],
            page=c["page_number"],
            year_published=c["year_published"],
        )
        for c in chunks
    ]

    return ChatResponse(answer=answer, sources=sources)
