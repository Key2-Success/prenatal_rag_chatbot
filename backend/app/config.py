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
    reranker_candidate_k: int = 5
    # Pinecone Inference model used for cross-encoder reranking (Stage 2).
    # bge-reranker-v2-m3 is a strong multilingual cross-encoder; no extra API
    # key required — uses the existing PINECONE_API_KEY.
    reranker_model: str = "bge-reranker-v2-m3"

    # --- LLM knobs ---
    llm_model: str = "gpt-4.1-mini"
    # Lower temperature = more consistent, factual answers (good for medical).
    llm_temperature: float = 0.3

    # --- Chunking knobs ---
    # SemanticChunker (langchain_experimental) groups consecutive sentences into
    # a chunk, cutting only where the embedding distance between neighbouring
    # sentence groups exceeds the threshold.
    # "percentile" cuts at the Nth percentile of observed distances — higher N
    # means fewer cuts (longer, more coherent chunks). 95 = cut only at the
    # top 5% largest topic shifts; reasonable starting point for structured
    # clinical guidelines. Override per-run: SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT=90
    semantic_breakpoint_threshold_type: str = "percentile"
    semantic_breakpoint_threshold_amount: float = 95.0

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
