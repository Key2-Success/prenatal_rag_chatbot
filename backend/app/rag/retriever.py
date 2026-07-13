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
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

from backend.app.config import PROJECT_ROOT, settings
from backend.app.observability import observe, update_current_span
from backend.app.rag.chunker import Chunk
from backend.app.timing import record_stage
from backend.app.rag.embedder import EMBEDDING_DIMENSIONS, embed_query
from backend.app.sources import priority_order, priority_rank_by_org

# Pinecone recommends ≤ 100 vectors per upsert request.
_UPSERT_BATCH_SIZE = 100

# Module-level singletons. The Pinecone client handles vector queries only —
# reranking has moved to a self-hosted CrossEncoder (see _get_reranker).
_pinecone_client: Pinecone | None = None
_pinecone_index = None
# Guards first-call index init so the concurrent per-source queries in
# retrieve_and_rerank can't race on create/validate. Double-checked, same
# pattern as the reranker + BM25 singletons below.
_pinecone_index_lock = threading.Lock()

# Self-hosted cross-encoder reranker singleton + lock. Lazy-loaded on first
# call (downloads ~600MB the first time, then cached in ~/.cache/huggingface).
# The lock prevents two concurrent first-callers from each triggering a
# download — relevant when --parallel-runs has multiple eval threads starting
# simultaneously. After the first load, _get_reranker() is a pure dict lookup.
_reranker = None
_reranker_lock = threading.Lock()

# BM25 encoder singleton + lock. Loaded from disk on first call; the encoder
# is fitted on the full corpus at ingest time and serialised to
# settings.bm25_encoder_path. Same double-checked locking pattern as the
# reranker — parallel eval workers share one loaded instance.
_bm25_encoder = None
_bm25_encoder_lock = threading.Lock()


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


def _select_device() -> str:
    """
    Pick the best available torch device for cross-encoder inference.

    bge-reranker-v2-m3 is a 568M-parameter model — on CPU each (query, doc)
    pair takes 2-4 seconds, which scales to ~16 minutes per --runs 3 eval.
    Hardware acceleration cuts this by 5-10x:

      - "cuda"  → NVIDIA GPU (production servers, Colab)
      - "mps"   → Apple Silicon Metal (M-series Macs — fast and free locally)
      - "cpu"   → fallback when neither is available

    Order matters: prefer cuda over mps over cpu, because cuda kernels are
    more mature for transformer inference. We don't error on missing
    hardware — cpu is always a valid fallback.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def warmup_reranker() -> None:
    """
    Load the cross-encoder AND trigger first-inference kernel compilation
    in the calling thread, so multi-threaded callers don't all race the
    cold-start path.

    Why this exists: MPS (and to a lesser extent CUDA) compiles kernels
    lazily on the first call with a given input shape. On Apple Silicon,
    that first compilation takes 5-15 seconds — long enough that, with
    `--parallel-runs`, multiple worker threads all try to do it
    simultaneously and either deadlock or serialise into a long stall.
    Calling this from the main thread BEFORE spawning workers means:
      1. Model load (~30s) happens once, visibly, in main.
      2. MPS kernel compilation happens once, in main, on a dummy input
         with realistic shape (query + medium doc).
      3. Worker threads then hit a fully-warm singleton — no cold path,
         no kernel compilation, no contention on first-call.

    Idempotent: if the singleton is already loaded, this just runs a
    cheap dummy predict to keep semantics simple. The cost when already
    warm is ~100ms.
    """
    import numpy as np
    reranker = _get_reranker()
    # Dummy pair with realistic-length text so the kernel that gets
    # compiled is the same one used in production. Predict + sigmoid =
    # the exact code path retrieve_and_rerank takes.
    pairs = [("warmup query about pregnancy nutrition",
              "Pregnant women should consume 60 mg of elemental iron and "
              "500 mcg of folic acid daily from the second trimester.")]
    raw = reranker.predict(pairs)
    _ = 1.0 / (1.0 + np.exp(-np.asarray(raw)))  # exercise the sigmoid path too


def _get_reranker():
    """
    Return the cached cross-encoder reranker, loading on first call.

    Lazy-loaded so importing this module is cheap (tests, ingest scripts,
    eval driver all import it without needing the reranker). Thread-safe via
    a double-checked lock so `--parallel-runs` doesn't race two concurrent
    first-call downloads of the 568M-param / ~600MB model.

    First call: ~10-30s (model load + device placement).
    Subsequent calls: instant (singleton lookup).
    Per-rerank latency for a 15-candidate pool: ~50-200ms on CUDA/MPS,
    ~2-5s on CPU.
    """
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is None:  # double-check after acquiring
            # Import inside the function so the heavy sentence-transformers /
            # torch import only happens at first rerank, not at module load.
            from sentence_transformers import CrossEncoder
            device = _select_device()
            print(f"[reranker] Loading {settings.reranker_model} on {device}...")
            # max_length pinned explicitly to 8192. Why: most cross-encoders
            # (including the older BGE v1 family and ms-marco-MiniLM) have a
            # tokenizer.model_max_length of 512. bge-reranker-v2-m3 is the
            # long-context variant ("m3" = multi-lingual, multi-functionality,
            # multi-granularity) and supports 8192. If RERANKER_MODEL is ever
            # swapped to a 512-token model, this pin makes the regression
            # visible: either sentence-transformers will clamp + warn, or the
            # tokenizer call will error — either way, no silent truncation.
            _reranker = CrossEncoder(
                settings.reranker_model,
                device=device,
                max_length=8192,
            )
            # Assert the loaded model actually supports the pinned length, so
            # a future BAAI re-publish that quietly drops max_length to 512
            # fails loud at startup instead of degrading retrieval silently.
            assert _reranker.tokenizer.model_max_length >= 8192, (
                f"Reranker {settings.reranker_model} has "
                f"tokenizer.model_max_length={_reranker.tokenizer.model_max_length}, "
                f"but we configured max_length=8192. The model is no longer "
                f"long-context — pick a different reranker or lower the pin."
            )
    return _reranker


def _get_bm25_encoder():
    """
    Return the cached BM25Encoder, loading from disk on first call.

    The encoder is fitted on the full corpus at ingest time
    (scripts/ingest.py) and saved to settings.bm25_encoder_path. Loading is
    lazy so importing this module is cheap (ingest scripts, tests, and the
    eval driver all import retriever without needing the encoder).

    Fails loud if the encoder file doesn't exist — this means hybrid ingest
    hasn't been run yet, and silently falling back to dense-only would produce
    misleading retrieval behaviour without any observable signal.

    Thread-safe via double-checked lock: same pattern as _get_reranker().
    """
    global _bm25_encoder
    if _bm25_encoder is not None:
        return _bm25_encoder
    with _bm25_encoder_lock:
        if _bm25_encoder is None:
            from pinecone_text.sparse import BM25Encoder
            encoder_path = PROJECT_ROOT / settings.bm25_encoder_path
            if not encoder_path.exists():
                raise RuntimeError(
                    f"BM25 encoder not found at '{settings.bm25_encoder_path}'. "
                    f"The index must be re-created and re-ingested with hybrid "
                    f"sparse vectors. Run:\n"
                    f"    python -m scripts.ingest --recreate-index"
                )
            print(f"[retriever] Loading BM25 encoder from {settings.bm25_encoder_path}...")
            _bm25_encoder = BM25Encoder().load(str(encoder_path))
    return _bm25_encoder


def get_index():
    """
    Lazily initialise and return the Pinecone index handle.

    Creates the index on first call for a fresh project. For an existing
    index, validates that the metric is 'dotproduct' — required for hybrid
    sparse+dense retrieval. For normalised dense vectors (OpenAI
    text-embedding-3-small outputs unit vectors), dotproduct == cosine
    similarity, so this is a transparent change for the dense channel.

    Fails loud if the existing index uses a different metric rather than
    silently degrading: hybrid queries sent to a cosine index return
    unexpected results because Pinecone applies metric-specific score
    normalisation that conflicts with our client-side alpha scaling.
    """
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    with _pinecone_index_lock:
        # Re-check inside the lock: another thread may have initialised the
        # index while we waited (double-checked locking).
        if _pinecone_index is not None:
            return _pinecone_index

        pc = _get_client()
        existing_names = pc.list_indexes().names()

        if settings.pinecone_index_name not in existing_names:
            # Fresh project — create with dotproduct from the start.
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=EMBEDDING_DIMENSIONS,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            # Existing index — verify metric is dotproduct.
            info = pc.describe_index(settings.pinecone_index_name)
            if info.metric != "dotproduct":
                raise RuntimeError(
                    f"Index '{settings.pinecone_index_name}' uses metric='{info.metric}' "
                    f"but hybrid search requires 'dotproduct'. Recreate it by running:\n"
                    f"    python -m scripts.ingest --recreate-index"
                )

        _pinecone_index = pc.Index(settings.pinecone_index_name)
    return _pinecone_index


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """
    Upsert chunk embeddings + sparse vectors + metadata into Pinecone, batched.

    Each record carries:
      - values:         dense embedding (OpenAI text-embedding-3-small, 1536-dim)
      - sparse_values:  BM25 sparse vector (indices + values from the corpus-
                        fitted BM25Encoder loaded from settings.bm25_encoder_path)
      - metadata:       full Chunk fields for filtering and display

    The BM25Encoder must be fitted and saved before this is called — the ingest
    script handles that in a separate pass over the full corpus before upserting.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
            f"must have equal length"
        )

    # Encode all chunk texts to sparse BM25 vectors in one batch.
    # _get_bm25_encoder() loads the corpus-fitted encoder from disk (lazy,
    # cached). encode_documents returns List[{"indices": [...], "values": [...]}].
    bm25 = _get_bm25_encoder()
    sparse_vecs = bm25.encode_documents([c.text for c in chunks])

    index = get_index()
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": emb,
            "sparse_values": sparse_vec,
            "metadata": chunk.model_dump(),
        }
        for chunk, emb, sparse_vec in zip(chunks, embeddings, sparse_vecs)
    ]

    for i in range(0, len(vectors), _UPSERT_BATCH_SIZE):
        index.upsert(vectors=vectors[i : i + _UPSERT_BATCH_SIZE])

    print(f"Upserted {len(vectors)} vectors to Pinecone.")


def _query_source(
    source_name: str,
    embedding: list[float],
    bm25_query: str,
) -> list[RetrievedChunk]:
    """
    Stage 1 recall: hybrid query Pinecone for one source.

    Combines a dense (semantic) channel and a sparse (BM25 keyword) channel
    in a single Pinecone query. Client-side alpha scaling:
      - dense vector  × alpha       (e.g. 0.75)
      - sparse values × (1−alpha)   (e.g. 0.25)

    This is the standard approach for Pinecone's classic Index API, which has
    no server-side alpha parameter. For normalised dense vectors (OpenAI
    text-embedding-3-small outputs unit vectors), dot product = cosine
    similarity, so the dense channel is semantically equivalent to before.

    bm25_query must be the original user query (pre-diet-hint-augmentation).
    Feeding "[Diet: Vegetarian]" to BM25 would match unrelated chunks that
    happen to mention vegetarian diets; the diet hint is for dense only.

    similarity_threshold note: the combined hybrid score is in a different
    range than pure cosine (dense is scaled by alpha; sparse adds additional
    signal). Recalibrate SIMILARITY_THRESHOLD if you see unexpected fallbacks.
    """
    alpha = settings.hybrid_alpha

    # Dense channel: scale by alpha so the two channels contribute proportionally.
    scaled_dense = [v * alpha for v in embedding]

    # Sparse channel: encode query, scale values by (1-alpha).
    bm25 = _get_bm25_encoder()
    raw_sparse = bm25.encode_queries(bm25_query)
    scaled_sparse = {
        "indices": raw_sparse["indices"],
        "values": [v * (1 - alpha) for v in raw_sparse["values"]],
    }

    results = get_index().query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
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


def _build_reranker_query(base_query: str, profile) -> str:
    """
    Build a natural-language reranker query that embeds profile constraints
    as prose so the cross-encoder's attention mechanism can contrast them
    directly against chunk content word by word.

    Why a different format for each retrieval channel:
      - Dense embedding:  augmented with "[Diet: X]" tag → steers the vector
                          toward diet-relevant regions of embedding space.
      - BM25:             raw query only → avoids spurious keyword matches on
                          the diet tag itself (e.g. "Vegetarian" hitting chunks
                          that discuss vegetarian diets for unrelated reasons).
      - Cross-encoder:    full prose description → gives the attention mechanism
                          explicit constraint tokens ("no meat", "no poultry")
                          to attend against chunk content ("beef liver",
                          "chicken curry"). A two-word tag like "[Diet: Vegetarian]"
                          is too weak to override a strong topical match on
                          "iron sources" from a meat-focused chunk.

    Example output for a vegetarian user with no conditions in week 20:
      "vegetarian (no meat, poultry, fish, or eggs) pregnant woman with no
       medical conditions (second trimester): iron food sources"

    This makes the chunk about beef liver actively conflict with the query
    at the token level, which the cross-encoder correctly penalises.
    """
    if profile is None:
        return base_query

    diet_phrase = {
        "vegetarian": "vegetarian (no meat, poultry, fish, or eggs)",
        "ovo_vegetarian": "ovo-vegetarian (no meat, poultry, or fish; eggs allowed)",
        "non_vegetarian": "non-vegetarian",
    }[profile.diet_type.name]

    week = profile.pregnancy_week
    if week <= 12:
        trimester = "first trimester"
    elif week <= 26:
        trimester = "second trimester"
    else:
        trimester = "third trimester"

    if profile.medical_conditions:
        conditions = " and ".join(c.value.lower() for c in profile.medical_conditions)
        health_phrase = f"with {conditions}"
    else:
        health_phrase = "with no medical conditions"

    return f"{diet_phrase} pregnant woman {health_phrase} ({trimester}): {base_query}"


def _rerank(query: str, texts: list[str]) -> np.ndarray:
    """
    Score each text against the query with the bge-reranker-v2-m3 cross-encoder.

    Returns a numpy array of relevance scores in [0, 1] aligned to `texts` order
    (higher = more relevant). Two interchangeable backends
    (settings.reranker_backend), both the same model so ordering is preserved:

      - "local":    self-hosted sentence-transformers on this machine's CPU/MPS.
      - "pinecone": Pinecone Inference's hosted copy — no local model, RAM, or
                    torch, so the deployed backend fits a free tier.
    """
    if settings.reranker_backend == "pinecone":
        return _rerank_pinecone(query, texts)
    return _rerank_local(query, texts)


def _rerank_local(query: str, texts: list[str]) -> np.ndarray:
    """
    Self-hosted cross-encoder. predict() returns raw logits in input order; we
    sigmoid-normalise to a 0-1 score so the numbers in eval reports and Langfuse
    traces are comparable to Pinecone's hosted output of the same model.

    Lock around .predict() because PyTorch MPS (Apple Silicon Metal) is NOT
    reliably thread-safe under concurrent forward passes — it can deadlock when
    multiple threads call .predict() on the same model instance. The lock
    serialises just the model call, not the surrounding pipeline, so OpenAI /
    Pinecone I/O still parallelises normally.
    """
    with record_stage("rerank_load"):
        reranker = _get_reranker()
    pairs = [(query, t) for t in texts]
    with record_stage("rerank_infer"):
        with _reranker_lock:
            raw_scores = reranker.predict(pairs)
        return 1.0 / (1.0 + np.exp(-np.asarray(raw_scores)))  # sigmoid


def _rerank_pinecone(query: str, texts: list[str]) -> np.ndarray:
    """
    Hosted rerank via Pinecone Inference (same bge-reranker-v2-m3 weights). The
    API returns already-normalised relevance scores sorted by relevance; we map
    them back to input order so the downstream argsort/top_k logic is unchanged.
    No local model, so rerank_load is ~free — only the API call is timed.
    """
    with record_stage("rerank_infer"):
        result = _get_client().inference.rerank(
            model=settings.pinecone_rerank_model,
            query=query,
            documents=texts,
            top_n=len(texts),
            return_documents=False,
        )
    scores = np.zeros(len(texts), dtype=float)
    for item in result.data:
        # Items expose both attribute and mapping access depending on SDK
        # version; support both so a minor bump doesn't silently break scoring.
        idx = item["index"] if isinstance(item, dict) else item.index
        score = item["score"] if isinstance(item, dict) else item.score
        scores[int(idx)] = float(score)
    return scores


@observe(name="retrieve_and_rerank")
def retrieve_and_rerank(query: str, profile=None) -> list[RetrievedChunk]:
    """
    Two-stage retrieval: recall from all sources, then cross-encoder rerank.

    Stage 0 — Query embedding (with optional HyDE transformation):
      If settings.hyde_enabled is True AND a profile was passed in, run
      HyDE: a small LLM generates a hypothetical answer to the query
      (personalised by profile), and we embed THAT answer instead of the
      raw query. The hypothetical answer is in the same prose register as
      the answer chunks, which substantially improves embedding match for
      natural-language queries. See backend/app/rag/hyde.py for details.
      Falls back to raw-query embedding when HyDE is disabled or profile
      is None (e.g. internal callers that don't have profile context).

    Stage 1 — Recall:
      Query all sources simultaneously. Pool and deduplicate by text content.
      Every source gets a fair shot at the reranker — no hard waterfall.

    Stage 2 — Rerank:
      Self-hosted bge-reranker-v2-m3 cross-encoder scores each
      (query, candidate) pair jointly. Selection is pure semantic relevance;
      the cross-encoder is significantly more precise than cosine similarity.

    Stage 3 — Order:
      Sort selected chunks by (doc_reference_order ASC, reranker_score DESC).
      Source priority is expressed as position in the LLM context window
      (primacy effect), not as an additive score nudge — no coefficients to tune.

    Returns an empty list if no candidates pass the similarity noise floor,
    which the pipeline treats as the "no_results" fallback.

    profile: UserProfile | None — only used for HyDE personalisation when
    HyDE is enabled. Annotated loosely as `None` default so callers without
    profile (tests, internal tools) still work without conditional imports.
    """
    update_current_span(input={
        "query": query,
        "hyde_enabled": settings.hyde_enabled and profile is not None,
        "hybrid_alpha": settings.hybrid_alpha,
    })

    # BM25 must see the original user query, not the diet-hint suffix appended
    # by augment_query (e.g. "Is amla safe? [Diet: Vegetarian]"). The "[Diet: X]"
    # tag steers dense embedding toward diet-relevant chunks — BM25 doesn't
    # benefit from it and would spuriously match chunks mentioning that diet.
    bm25_query = query.split(" [Diet:")[0]

    # Stage 0: HyDE transformation, opt-in via settings + requires profile.
    if settings.hyde_enabled and profile is not None:
        # Import inside the conditional so retriever has no hard dep on
        # hyde.py when HyDE is disabled (and to avoid a circular import if
        # hyde ever needs the embedder).
        from backend.app.rag.hyde import generate_hypothetical_answer
        with record_stage("hyde"):
            text_to_embed = generate_hypothetical_answer(query, profile)
    else:
        text_to_embed = query

    with record_stage("embed"):
        embedding = embed_query(text_to_embed)

    # Stage 1: recall from all sources, pool, deduplicate.
    all_candidates: list[RetrievedChunk] = []
    sources_hit: dict[str, int] = {}
    with record_stage("pinecone"):
        # Fan the per-source hybrid queries out across threads: each is a
        # network-bound Pinecone call, so concurrency overlaps the waits.
        # ThreadPoolExecutor.map preserves input order and we re-pool in
        # priority order below, so all_candidates — and therefore dedup +
        # rerank — is identical to the sequential version no matter which
        # query returns first. This changes latency only, not results.
        sources = priority_order()
        with ThreadPoolExecutor(max_workers=len(sources)) as pool:
            per_source = list(
                pool.map(lambda s: _query_source(s, embedding, bm25_query), sources)
            )
        for source_name, source_chunks in zip(sources, per_source):
            sources_hit[source_name] = len(source_chunks)
            all_candidates.extend(source_chunks)

    all_candidates = _dedup_by_text(all_candidates)

    if not all_candidates:
        update_current_span(output={"chunks_returned": 0, "sources_hit": sources_hit})
        return []

    # Stage 2: rerank with the bge-reranker-v2-m3 cross-encoder. Two
    # interchangeable backends (settings.reranker_backend) — same model weights,
    # so ordering is preserved; see _rerank(). Both return a 0-1 relevance score
    # per candidate in input order.
    #
    # Build a profile-aware reranker query in natural prose so the cross-encoder
    # can attend dietary/medical constraints directly against chunk tokens.
    # Uses bm25_query (the clean, tag-stripped query) as the base — the [Diet: X]
    # tag was for the embedding channel only, not for cross-encoder input.
    reranker_query = _build_reranker_query(bm25_query, profile)
    scores = _rerank(reranker_query, [c.text for c in all_candidates])

    # Pick the top_k candidates by reranker score. The returned indices point
    # back into all_candidates so we can preserve full metadata.
    top_indices = np.argsort(-scores)[: settings.top_k]

    # Reconstruct RetrievedChunks from the top picks. The reranker order is
    # intentionally discarded here — Stage 3 re-sorts by (source priority,
    # reranker score) for the final output ordering.
    ranked: list[RetrievedChunk] = []
    rank_by_org = priority_rank_by_org()
    for idx in top_indices:
        original = all_candidates[int(idx)]
        ranked.append(RetrievedChunk(
            text=original.text,
            org_display_name=original.org_display_name,
            doc_title=original.doc_title,
            page_number=original.page_number,
            year_published=original.year_published,
            score=float(scores[idx]),  # reranker score replaces cosine similarity
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
            # Full chunk text + per-chunk metadata so the trace UI shows
            # what the LLM actually ingested. Without this, debugging "why
            # did the LLM answer about vitamin A?" required re-running
            # retrieval offline — this puts the answer one click away in
            # the Langfuse span. Verbose (~5KB per case) but worth it.
            "chunks": [
                {
                    "source": c.org_display_name,
                    "page": c.page_number,
                    "doc_title": c.doc_title,
                    "score": c.score,
                    "text": c.text,
                }
                for c in ranked
            ],
        },
    )
    return ranked
