"""
retriever.py — Pinecone vector store: upsert and two-stage retrieval.

Design decisions:

  1. Single Pinecone index with metadata filtering.
     ONE Pinecone index, filtered by `org_display_name` at query time —
     persistent, scalable, no per-source indexes needed.

  2. Two-stage retrieval: recall → rerank.
     Stage 1 (recall): all three sources are queried in parallel; results
     are pooled and deduplicated by text content. No source is excluded
     before the reranker sees the candidates.
     Stage 2 (rerank): Pinecone Inference bge-reranker-v2-m3 (cross-encoder)
     scores every (query, candidate) pair jointly — much better precision
     than cosine similarity alone.

  3. Source priority via ordering, not score nudges.
     After reranking, selected chunks are sorted by (doc_reference_order ASC,
     reranker_score DESC). Selection is pure relevance; source preference is
     expressed as position in the LLM context window, exploiting the model's
     primacy effect without any magic coefficients to tune.

  4. Tunables in Settings.
     similarity_threshold, top_k, reranker_candidate_k, reranker_model are
     all env-overridable:
         SIMILARITY_THRESHOLD=0.2 RERANKER_CANDIDATE_K=15 python -m eval.run_eval
"""

import hashlib
import uuid

from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.observability import observe, update_current_span
from backend.app.rag.chunker import Chunk
from backend.app.rag.embedder import EMBEDDING_DIMENSIONS, embed_query
from backend.app.sources import priority_order, priority_rank_by_org

# Pinecone recommends ≤ 100 vectors per upsert request.
_UPSERT_BATCH_SIZE = 100

# Module-level singletons. Both index queries and inference rerank share the
# same authenticated client — creating a new Pinecone() on every call would
# re-authenticate on each request and bypass connection reuse.
_pinecone_client: Pinecone | None = None
_pinecone_index = None


class RetrievedChunk(BaseModel):
    """One chunk returned from Pinecone, with its similarity score."""
    text: str
    org_display_name: str
    doc_title: str
    page_number: int
    year_published: int
    score: float


def _get_client() -> Pinecone:
    """Return the cached Pinecone client, initialising it on first call."""
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pinecone_client


def get_index():
    """Lazily initialise and return the Pinecone index handle."""
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    pc = _get_client()
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


def _query_source(source_name: str, embedding: list[float]) -> list[RetrievedChunk]:
    """
    Stage 1 recall: query Pinecone for one source.

    Uses reranker_candidate_k (not top_k) because we're building a candidate
    pool for the reranker, not the final context window. similarity_threshold
    acts as a noise floor — anything below it is too far from the query to be
    worth sending to the cross-encoder.
    """
    results = get_index().query(
        vector=embedding,
        top_k=settings.reranker_candidate_k,
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


def _dedup_by_text(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Remove exact-duplicate chunks (same text content) from the candidate pool.

    Deduplication is by text hash only — not by page or source. Two chunks
    from the same page are kept if they contain different text, since both
    could be genuinely relevant.

    When duplicates exist, the first occurrence wins (highest cosine score
    because _query_source returns results in score order).
    """
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        key = hashlib.md5(chunk.text.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


@observe(name="retrieve_and_rerank")
def retrieve_and_rerank(query: str) -> list[RetrievedChunk]:
    """
    Two-stage retrieval: recall from all sources, then cross-encoder rerank.

    Stage 1 — Recall:
      Query all sources simultaneously. Pool and deduplicate by text content.
      Every source gets a fair shot at the reranker — no hard waterfall.

    Stage 2 — Rerank:
      Pinecone Inference cross-encoder scores each (query, candidate) pair
      jointly. Selection is pure semantic relevance; the cross-encoder is
      significantly more precise than cosine similarity.

    Stage 3 — Order:
      Sort selected chunks by (doc_reference_order ASC, reranker_score DESC).
      Source priority is expressed as position in the LLM context window
      (primacy effect), not as an additive score nudge — no coefficients to tune.

    Returns an empty list if no candidates pass the similarity noise floor,
    which the pipeline treats as the "no_results" fallback.
    """
    update_current_span(input={"query": query})

    embedding = embed_query(query)

    # Stage 1: recall from all sources, pool, deduplicate.
    all_candidates: list[RetrievedChunk] = []
    sources_hit: dict[str, int] = {}
    for source_name in priority_order():
        source_chunks = _query_source(source_name, embedding)
        sources_hit[source_name] = len(source_chunks)
        all_candidates.extend(source_chunks)

    all_candidates = _dedup_by_text(all_candidates)

    if not all_candidates:
        update_current_span(output={"chunks_returned": 0, "sources_hit": sources_hit})
        return []

    # Stage 2: rerank. Pinecone Inference cross-encoder scores every
    # (query, candidate) pair and returns the top_k highest-relevance chunks.
    # Response shape verified against installed SDK source (pinecone_plugins
    # 5.4.2): RerankResult.data is List[RankedDocument]; each RankedDocument
    # has .index (int), .score (float), .document (dict, when return_documents=True).
    rerank_result = _get_client().inference.rerank(
        model=settings.reranker_model,
        query=query,
        documents=[{"id": str(i), "text": c.text} for i, c in enumerate(all_candidates)],
        top_n=settings.top_k,
        return_documents=True,
    )

    # Reconstruct RetrievedChunks from the reranker output, preserving all
    # original metadata. The reranker returns items in relevance order, but
    # we re-sort in Stage 3, so that order is intentionally discarded here.
    ranked: list[RetrievedChunk] = []
    rank_by_org = priority_rank_by_org()
    for item in rerank_result.data:
        original = all_candidates[int(item.index)]
        ranked.append(RetrievedChunk(
            text=original.text,
            org_display_name=original.org_display_name,
            doc_title=original.doc_title,
            page_number=original.page_number,
            year_published=original.year_published,
            score=item.score,  # reranker score replaces cosine similarity
        ))

    # Stage 3: sort by (source priority ASC, reranker score DESC).
    # MoHFW content appears first in the LLM context window; within each
    # source, the most relevant chunk leads. No score nudges — ordering only.
    ranked.sort(key=lambda c: (
        rank_by_org.get(c.org_display_name, 999),
        -c.score,
    ))

    update_current_span(
        output={
            "chunks_returned": len(ranked),
            "sources_hit": sources_hit,
            "sources_in_output": list({c.org_display_name for c in ranked}),
            "top_reranker_score": ranked[0].score if ranked else None,
            "pages": [f"{c.org_display_name} p.{c.page_number}" for c in ranked],
        },
    )
    return ranked
