"""
embedder.py — Text embedding via OpenAI's text-embedding-3-small.

Why text-embedding-3-small over the legacy ada-002:
  - 5x cheaper
  - Better benchmark scores
  - Same 1536 dimensions as ada-002, so the Pinecone index dim is unchanged

EMBEDDING_DIMENSIONS lives here (not in config) because it's a property of
the model, not a tunable knob. Changing the model means changing both —
and the Pinecone index has to be recreated. Keeping them adjacent makes
that coupling explicit.
"""

from backend.app.clients import get_openai_client

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# OpenAI's embeddings endpoint accepts up to 2048 inputs per request, but
# 100 is a comfortable sweet spot — small enough to keep retries cheap if
# one batch fails, large enough to keep ingestion fast.
_EMBED_BATCH_SIZE = 100


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, batched. Returns vectors in input order."""
    client = get_openai_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend(item.embedding for item in response.data)
    return out


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Used at retrieval time."""
    return embed_texts([query])[0]
