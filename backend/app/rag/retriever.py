"""
retriever.py — Pinecone vector store: upsert and ordered retrieval.

Key design decisions:
  1. Single Pinecone index with metadata filtering.
     The notebook had 3 separate in-memory LlamaIndex indexes (one per PDF).
     Here we use ONE Pinecone index and filter by `doc_reference_order` at
     query time — same logic, but persistent and scalable.

  2. Ordered retrieval (MoHFW → FOGSI → WHO).
     We query each source in priority order and return as soon as we find
     chunks above the similarity threshold. This preserves your original
     "prefer Indian guidelines first" design decision.

  3. Similarity threshold.
     Your original threshold was 0.7. Pinecone cosine scores are in [-1, 1]
     but typically land in [0.0, 1.0] for semantically related text.
     0.6 is a safe starting point — tune up if you get too many irrelevant
     results, tune down if you get too many "no results found" fallbacks.
"""

import uuid
from pinecone import Pinecone, ServerlessSpec

from backend.app.config import settings
from backend.app.sources import priority_order
from backend.app.rag.embedder import embed_query, EMBEDDING_DIMENSIONS

SIMILARITY_THRESHOLD = 0.6
TOP_K = 5  # retrieve top 5 chunks per source before threshold filtering

_pinecone_index = None


def get_index():
    """Lazily initialises and returns the Pinecone index."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index_name = settings.pinecone_index_name

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        _pinecone_index = pc.Index(index_name)
    return _pinecone_index


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Upserts chunk embeddings + metadata into Pinecone.
    Called once by scripts/ingest.py — not at query time.
    """
    index = get_index()
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "org_display_name": chunk["org_display_name"],
                "doc_title": chunk["doc_title"],
                "doc_reference_order": chunk["doc_reference_order"],
                "year_published": chunk["year_published"],
                "page_number": chunk["page_number"],
            },
        })

    # Pinecone recommends batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])

    print(f"Upserted {len(vectors)} vectors to Pinecone.")


def retrieve_ordered(query: str) -> list[dict]:
    """
    Queries Pinecone in priority order (MoHFW → FOGSI → WHO).
    Returns the first source's chunks that pass the similarity threshold.
    If no source passes the threshold, returns an empty list (triggers fallback).

    Each returned dict:
    {
        "text": str,
        "org_display_name": str,
        "doc_title": str,
        "page_number": int,
        "year_published": int,
        "score": float,
    }
    """
    index = get_index()
    query_embedding = embed_query(query)

    for source_name in priority_order():
        results = index.query(
            vector=query_embedding,
            top_k=TOP_K,
            filter={"org_display_name": {"$eq": source_name}},
            include_metadata=True,
        )

        above_threshold = [
            {
                "text": match["metadata"]["text"],
                "org_display_name": match["metadata"]["org_display_name"],
                "doc_title": match["metadata"]["doc_title"],
                "page_number": match["metadata"]["page_number"],
                "year_published": match["metadata"]["year_published"],
                "score": match["score"],
            }
            for match in results["matches"]
            if match["score"] >= SIMILARITY_THRESHOLD
        ]

        if above_threshold:
            return above_threshold

    return []
