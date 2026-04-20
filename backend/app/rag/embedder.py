"""
embedder.py — Text embedding using OpenAI's text-embedding-3-small.

Why text-embedding-3-small over ada-002 (what LlamaIndex used by default):
  - 5x cheaper than ada-002
  - Better performance on benchmarks
  - 1536 dimensions (same as ada-002) — Pinecone index dimension stays the same
"""

from openai import OpenAI

from backend.app.config import settings

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Pinecone free tier allows up to 100 upserts per request.
# Batching also reduces API round-trips during ingestion.
BATCH_SIZE = 100

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of strings in batches.
    Returns a list of float vectors in the same order as the input.
    """
    client = get_client()
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embeds a single query string. Used at retrieval time."""
    return embed_texts([query])[0]
