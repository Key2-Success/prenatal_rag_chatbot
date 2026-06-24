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
    # Pinecone cosine scores typically land in [0.0, 1.0] for related text.
    # Lower → more recall, more noise. Higher → more precision, more fallbacks.
    # Applies during the recall phase (before reranking) as a noise floor —
    # chunks that don't clear this threshold are excluded from the reranker input.
    # Override per-run via env, e.g. `SIMILARITY_THRESHOLD=0.55 python -m eval.run_eval`.
    similarity_threshold: float = 0.3
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

    # --- LLM knobs ---
    llm_model: str = "gpt-4.1-nano"
    # Lower temperature = more consistent, factual answers (good for medical).
    # Tried 0.1 to reduce "creative gap-filling," but the eval showed it made
    # BOTH faithfulness AND context_precision worse — so 0.3 stays. The
    # hallucination problem is a grounding problem, not a sampling-randomness
    # problem; it needs a hard post-generation grounding check, not a lower
    # temperature. Keep at 0.3.
    llm_temperature: float = 0.3

    # --- Answer review knobs (faithfulness + answerability) ---
    # Hard post-generation review. The answer model's system prompt has
    # FORBIDDEN rules against ungrounded claims, but those are soft constraints
    # it bypasses ~regularly — the same situation diet and deflective-opener
    # rules were in before they moved to the deterministic validator.
    # review_answer decomposes the answer into atomic claims, verifies each
    # against the retrieved context, drops the unsupported ones, AND judges
    # whether the surviving answer actually addresses the question (one call).
    # Unlike the diet/opener checks there's no cheap regex/profile gate, so this
    # is an always-on LLM call per answer (acceptable for an eval suite).
    # Disable to A/B in eval: VALIDATOR_GROUNDING_ENABLED=false (gates the whole
    # review call — both faithfulness and answerability detection).
    validator_grounding_enabled: bool = True
    # Judge model. Deliberately STRONGER than llm_model (gpt-4.1-nano):
    # detecting an ungrounded claim is harder than generating one, and the
    # nano model grading its own output is the weakest possible judge.
    # Override per-run: VALIDATOR_GROUNDING_MODEL=gpt-4.1-nano (cost parity A/B).
    validator_grounding_model: str = "gpt-4.1-mini"
    # Answerability routing gate. The review call ALWAYS produces an
    # answers_question verdict (negligible marginal cost). This flag controls
    # only whether a FALSE verdict ROUTES the response to no_results. Keeping it
    # separate lets us A/B "does answerability help?" without prompt surgery:
    # VALIDATOR_ANSWERABILITY_ENABLED=false keeps the verdict in traces but
    # stops it from forcing no_results.
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
