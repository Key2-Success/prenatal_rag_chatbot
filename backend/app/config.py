"""
config.py — Project paths and environment-driven settings.

Two kinds of configuration live here, kept deliberately separate:

  1. Module constants (PROJECT_ROOT, DATA_DIR)
     Derived from the codebase layout, not from the environment. Putting
     these on Settings would imply they're env-tunable, which they aren't.

  2. Settings (env vars + tunable runtime knobs)
     Everything that can or must vary across environments — secrets,
     model choices, retrieval thresholds. Validated once at startup.

Why walk up to find PROJECT_ROOT instead of Path(__file__).parents[N]:
  parents[N] silently breaks if this file moves; counting is fragile.
  Walking up to the nearest pyproject.toml is location-independent.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root() -> Path:
    """Walk up from this file to the nearest pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError(
        "Could not locate project root. "
        "pyproject.toml must exist at the repository root."
    )


# --- Path constants (not env-configurable) -----------------------------------

PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"


# --- Settings (env-configurable) ---------------------------------------------

class Settings(BaseSettings):
    """
    All values can be overridden via .env or process env vars.

    The retrieval / LLM knobs default to production-sane values. Surfacing
    them here (instead of as constants in retriever.py / pipeline.py) means
    a tuning run can override one with `SIMILARITY_THRESHOLD=0.55 ...`
    without touching code.
    """

    # --- Required secrets ---
    openai_api_key: str
    pinecone_api_key: str

    # --- Optional secrets ---
    # Used only by the RAGAS judge, which is cross-vendor (Claude judging GPT)
    # to avoid same-family bias. Optional so the rest of the app runs without
    # an Anthropic key — eval/ragas_eval.py validates this at score time.
    anthropic_api_key: str | None = None
    # Used only at ingestion time (`python -m scripts.ingest`). The runtime
    # app reads from Pinecone and never calls LlamaParse, so this key is not
    # required to start the server.
    llama_cloud_api_key: str | None = None

    # --- App ---
    pinecone_index_name: str = "poshan-saathi"
    app_env: str = "development"

    # --- Retrieval knobs ---
    # Recall-phase noise floor: chunks scoring below this are excluded from the
    # reranker input (applied BEFORE reranking).
    # Recalibrated to 0.05 for HYBRID retrieval: with hybrid_alpha=0.75 the score
    # is a client-side-scaled dense+sparse blend, not a pure cosine, so it lands
    # in a much lower range than the old 0.3 (which was calibrated for pure cosine
    # and over-filtered under hybrid). 0.05 is a low-but-nonzero floor — per R30,
    # recalibrate the filter, never zero it.
    # Override per-run via env, e.g. `SIMILARITY_THRESHOLD=0.1 python -m eval.ragas_eval`.
    similarity_threshold: float = 0.05
    # Final number of chunks returned to the LLM context window.
    top_k: int = 3
    # Candidates fetched per source during the recall phase (Stage 1).
    # All sources are queried and pooled before the reranker sees them.
    # More → better recall input for the reranker; fewer → cheaper + faster.
    # Start conservative at 5 per source (15 pool max → reranked to top_k=5).
    reranker_candidate_k: int = 3
    # HuggingFace model ID for the cross-encoder reranker (Stage 2).
    # bge-reranker-v2-m3 is a strong multilingual cross-encoder, self-hosted
    # via sentence-transformers — no per-call API rate limit. The model is
    # downloaded once to ~/.cache/huggingface (~600MB) and cached across runs.
    # Override per-run via env, e.g. RERANKER_MODEL=BAAI/bge-reranker-large.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Reranker backend:
    #   "local"    → self-hosted sentence-transformers on this machine's CPU/MPS.
    #                No per-call cost, but needs ~2GB RAM + torch in the image.
    #   "pinecone" → Pinecone Inference's hosted copy of the SAME model. No local
    #                model/RAM/torch, so the deployed backend fits a free tier.
    #                Free tier caps at 500 reranks/month (fine behind our rate
    #                limits). Same weights → ordering preserved.
    # Local is the dev default; set RERANKER_BACKEND=pinecone for deployment.
    reranker_backend: str = "local"
    # Model id for the hosted (Pinecone Inference) reranker. Same weights as the
    # local BAAI/bge-reranker-v2-m3; Pinecone returns already-normalised
    # relevance scores (no client-side sigmoid needed).
    pinecone_rerank_model: str = "bge-reranker-v2-m3"

    # --- LLM knobs ---
    # Answer model. Switched nano → mini after an A/B (2026-06-27): faithfulness
    # was identical across both (~0.82 mean, variance-limited — see eval reports
    # eval_20260627_173136 / _173619), but mini gave a reliable answer_relevancy
    # gain (0.88 → 0.93) with notably tighter run-to-run variance. Cost is ~5×
    # nano per answer but negligible in absolute terms at this app's volume.
    # NOTE: this value also drives the validate_and_fix rewriter, so the
    # opener/hedge/cadence corrections now run on mini too.
    # Override per-run: LLM_MODEL=gpt-4.1-nano python -m eval.ragas_eval.
    llm_model: str = "gpt-4.1-mini"
    # 0 for reproducibility. History: a 2026-05 experiment set this to 0.1 and
    # reported faithfulness/context_precision got WORSE, concluding the problem
    # was grounding (not sampling) and reverting to 0.3. That comparison was a
    # SINGLE eval run each — and we have since established (see the per-case
    # faithfulness matrix, 2026-06-27) that single-run faithfulness is variance-
    # dominated: the same code scores calcium 0.167 and 1.0 on consecutive runs.
    # A single 0.1-vs-0.3 run therefore couldn't detect a real temperature effect
    # at all; it measured noise. At temp 0.3 the answer LLM regenerates a
    # different answer every request, so a stray ungrounded rider appears in a
    # fraction of runs and swings the tiny-denominator score 0.5↔1.0. Setting
    # temperature 0 fixes the answer text across runs, removing the answer-side
    # regeneration noise so faithfulness reflects the grounding logic, not the
    # sampler. Verified with 3× repeat eval runs (grounding is still enforced by
    # the post-generation review + deterministic strippers, not by temperature).
    llm_temperature: float = 0.0

    # --- Answer review knobs (faithfulness) ---
    # Hard post-generation review. The answer model's system prompt has
    # FORBIDDEN rules against ungrounded claims, but those are soft constraints
    # it bypasses ~regularly — the same situation diet and deflective-opener
    # rules were in before they moved to the deterministic validator.
    # review_answer decomposes the answer into atomic claims, verifies each
    # against the retrieved context, and drops the unsupported ones (one call).
    # Unlike the diet/opener checks there's no cheap regex/profile gate, so this
    # is an always-on LLM call per answer (acceptable for an eval suite).
    # Disable to A/B in eval: VALIDATOR_GROUNDING_ENABLED=false.
    validator_grounding_enabled: bool = True
    # Judge model. At least as strong as the answer model (llm_model, now
    # gpt-4.1-mini): detecting an ungrounded claim is at least as hard as
    # generating one, so the judge must never be the weaker model. Override
    # per-run: VALIDATOR_GROUNDING_MODEL=gpt-4.1-nano (cost parity A/B).
    validator_grounding_model: str = "gpt-4.1-mini"
    # Answerability routing gate. Answerability is now a DETERMINISTIC regex
    # check (validator.check_answerability), not an LLM verdict — it routes a
    # quantity question answered with no quantity to no_results. This flag
    # controls only whether that check is allowed to force no_results; set
    # VALIDATOR_ANSWERABILITY_ENABLED=false to A/B with the gate disabled.
    validator_answerability_enabled: bool = True

    # --- HyDE knobs ---
    # HyDE (Hypothetical Document Embeddings) transforms the query before
    # retrieval: a small LLM generates a plausible hypothetical answer, and
    # we embed THAT instead of the raw question. The hypothetical answer is
    # semantically closer to actual answer chunks than a question is, closing
    # the prose↔question gap that hurts retrieval on natural-language queries.
    # Default ON for production — A/B in eval via `--no-hyde` to compare.
    # Cost: one extra LLM call per query (~50-200ms, ~$0.0001 per query).
    hyde_enabled: bool = False
    # Model for the HyDE generation step. Cheap+fast is the right tradeoff —
    # the hypothetical answer is for embedding, not user-visible output.
    hyde_model: str = "gpt-4.1-nano"
    # temperature=0 for reproducibility; same hypothetical answer for the
    # same query enables apples-to-apples eval comparisons.
    hyde_temperature: float = 0.0

    # --- Chunking knobs ---
    # SemanticChunker (langchain_experimental) groups consecutive sentences into
    # a chunk, cutting only where the embedding distance between neighbouring
    # sentence groups exceeds the threshold.
    # "percentile" cuts at the Nth percentile of observed distances — higher N
    # means fewer cuts (longer, more coherent chunks). Lowered from 95 → 85
    # after LlamaParse re-migration: cleaner markdown from LlamaParse means
    # pages with many numbered sub-sections (e.g. 1.3.2.1–1.3.2.8) can land
    # in a single chunk at 95th percentile because sub-section boundaries don't
    # produce top-5% distance jumps. 85th percentile catches those section
    # boundaries and keeps chunks topically focused.
    # Override per-run: SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT=90
    semantic_breakpoint_threshold_type: str = "percentile"
    semantic_breakpoint_threshold_amount: float = 85.0
    # Hard token cap applied AFTER SemanticChunker as a backstop. Any chunk
    # exceeding this limit is split further with RecursiveCharacterTextSplitter
    # (token-aware, sentence-boundary-respecting). Set to 512 based on the
    # observed p90=502 token distribution — catches the top ~10% of outlier
    # chunks without fragmenting the median (203 tokens) chunks.
    # Override per-run: CHUNK_MAX_TOKENS=400
    chunk_max_tokens: int = 512

    # --- Hybrid search knobs ---
    # Sparse (BM25) + dense (OpenAI embeddings) hybrid retrieval.
    # The alpha parameter weights the two channels; scaling is applied
    # client-side before the Pinecone query (the classic Index.query API has
    # no server-side alpha parameter).
    #   alpha=1.0 → pure dense (cosine via dotproduct on normalised vectors)
    #   alpha=0.0 → pure sparse (BM25 keyword matching only)
    #   alpha=0.75 → Pinecone's recommended default for RAG workloads
    # Tunable without code change: HYBRID_ALPHA=0.5 python -m eval.ragas_eval
    # NOTE: similarity_threshold is calibrated for cosine [0,1]. After hybrid,
    # scores combine dense + sparse — a value near 0 may be needed if sparse
    # signal on exact-match queries inflates previously-below-threshold chunks.
    hybrid_alpha: float = 0.75
    # Path (relative to PROJECT_ROOT) where the fitted BM25Encoder is saved
    # at ingest time and loaded at query time. Fitted on the full corpus so
    # IDF weights reflect this domain (pregnancy nutrition guidelines).
    bm25_encoder_path: str = "data/bm25_encoder.json"

    # --- Classifier knobs ---
    # Triage LLM that labels each incoming message as in_scope / emergency /
    # out_of_scope before retrieval. Kept separate from llm_model so we can
    # use the cheapest viable model for triage and a stronger one for answers.
    classifier_model: str = "gpt-4.1-nano"
    # Triage is a routing decision — we want the same input to always get the
    # same label, so default to deterministic.
    classifier_temperature: float = 0.0

    # --- Security / rate limiting (public deployment) ---
    # CORS allowed origins, comma-separated. Dev default is the local Next.js
    # frontend; in prod set CORS_ALLOW_ORIGINS=https://your-app.vercel.app.
    cors_allow_origins: str = "http://localhost:3000"
    # Per-IP rate limits on /chat (slowapi). The burst cap (per minute) stops a
    # tight loop; the daily cap bounds one visitor's total — no legitimate user
    # needs 50 nutrition questions a day.
    rate_limit_per_minute: int = 10
    rate_limit_per_day: int = 50
    # Global daily request budget across ALL clients — the hard cost ceiling a
    # botnet (many IPs, each under the per-IP caps) can't evade. When exceeded,
    # /chat returns 503 until UTC midnight. In-memory + single-instance: fine
    # for one container; use a shared counter (Redis) if you scale out.
    daily_request_budget: int = 500
    # Reject request bodies larger than this many bytes before parsing — a cheap
    # backstop to the message max_length=1000 already on ChatRequest (a huge
    # medical_conditions list or malformed body is rejected before any work).
    max_body_bytes: int = 16384

    @property
    def cors_allow_origins_list(self) -> list[str]:
        """CORS origins as a list (env stores them comma-separated)."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    # --- Observability (Langfuse) ---
    # Optional. When both keys are set, the OpenAI client is auto-wrapped
    # so every embedding / chat / parse call shows up in the Langfuse trace
    # tree. Without keys, we fall back to the plain OpenAI client and the
    # @observe decorators silently no-op — code paths are identical.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
    )


settings = Settings()
