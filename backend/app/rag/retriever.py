"""
retriever.py — Pinecone vector store: upsert and ordered retrieval.

Design decisions:

  1. Single Pinecone index with metadata filtering.
     The notebook used 3 in-memory LlamaIndex indexes (one per PDF). We use
     ONE Pinecone index and filter by `org_display_name` at query time —
     same effect, persistent and scalable.

  2. Ordered retrieval (MoHFW → FOGSI → WHO).
     We query each source in priority order and return as soon as we find
     chunks above the similarity threshold. Preserves the "prefer Indian
     guidelines first" design from the original notebook. Priority order
     comes from sources.json, not a hardcode.

  3. Tunables in Settings.
     similarity_threshold and top_k are env-overridable for tuning runs:
         SIMILARITY_THRESHOLD=0.55 python -m eval.run_eval
"""

import uuid

from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.observability import observe, update_current_span
from backend.app.rag.chunker import Chunk
from backend.app.rag.embedder import EMBEDDING_DIMENSIONS, embed_query
from backend.app.sources import priority_order

# Pinecone recommends ≤ 100 vectors per upsert request.
_UPSERT_BATCH_SIZE = 100

_pinecone_index = None


class RetrievedChunk(BaseModel):
    """One chunk returned from Pinecone, with its similarity score."""
    text: str
    org_display_name: str
    doc_title: str
    page_number: int
    year_published: int
    score: float


def get_index():
    """Lazily initialise and return the Pinecone index handle."""
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    pc = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    _pinecone_index = pc.Index(settings.pinecone_index_name)
    return _pinecone_index


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Upsert chunk embeddings + metadata into Pinecone, batched."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
            f"must have equal length"
        )

    index = get_index()
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": chunk.model_dump(),
        }
        for chunk, emb in zip(chunks, embeddings)
    ]

    for i in range(0, len(vectors), _UPSERT_BATCH_SIZE):
        index.upsert(vectors=vectors[i : i + _UPSERT_BATCH_SIZE])

    print(f"Upserted {len(vectors)} vectors to Pinecone.")


def _query_one_source(source_name: str, embedding: list[float]) -> list[RetrievedChunk]:
    """Query Pinecone for one source, applying the similarity threshold."""
    results = get_index().query(
        vector=embedding,
        top_k=settings.top_k,
        filter={"org_display_name": {"$eq": source_name}},
        include_metadata=True,
    )
    out: list[RetrievedChunk] = []
    for match in results["matches"]:
        if match["score"] < settings.similarity_threshold:
            continue
        meta = match["metadata"]
        out.append(RetrievedChunk(
            text=meta["text"],
            org_display_name=meta["org_display_name"],
            doc_title=meta["doc_title"],
            page_number=meta["page_number"],
            year_published=meta["year_published"],
            score=match["score"],
        ))
    return out


@observe(name="retrieve_ordered")
def retrieve_ordered(query: str) -> list[RetrievedChunk]:
    """
    Query each source in priority order. Return the first source's chunks
    that pass the similarity threshold; empty list if none do (callers
    treat this as the "no_results" fallback).
    """
    update_current_span(input={"query": query})

    embedding = embed_query(query)
    for source_name in priority_order():
        chunks = _query_one_source(source_name, embedding)
        if chunks:
            # Surface which source won and how many chunks passed threshold —
            # the most useful retrieval signals to see in the trace UI without
            # having to expand every span.
            update_current_span(
                output={
                    "winning_source": source_name,
                    "chunks_returned": len(chunks),
                    "top_score": chunks[0].score,
                    "pages": [c.page_number for c in chunks],
                },
            )
            return chunks
    update_current_span(output={"winning_source": None, "chunks_returned": 0})
    return []
