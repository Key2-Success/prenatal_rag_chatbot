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

    # --- App ---
    pinecone_index_name: str = "poshan-saathi"
    app_env: str = "development"

    # --- Retrieval knobs ---
    # Pinecone cosine scores typically land in [0.0, 1.0] for related text.
    # Lower → more recall, more noise. Higher → more precision, more fallbacks.
    # Override per-run via env, e.g. `SIMILARITY_THRESHOLD=0.55 python -m eval.run_eval`.
    similarity_threshold: float = 0.3
    # Candidates fetched per source before threshold filtering.
    top_k: int = 5

    # --- LLM knobs ---
    llm_model: str = "gpt-4.1-nano"
    # Lower temperature = more consistent, factual answers (good for medical).
    llm_temperature: float = 0.3

    # --- Classifier knobs ---
    # Triage LLM that labels each incoming message as in_scope / emergency /
    # out_of_scope before retrieval. Kept separate from llm_model so we can
    # use the cheapest viable model for triage and a stronger one for answers.
    classifier_model: str = "gpt-4.1-nano"
    # Triage is a routing decision — we want the same input to always get the
    # same label, so default to deterministic.
    classifier_temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
    )


settings = Settings()
