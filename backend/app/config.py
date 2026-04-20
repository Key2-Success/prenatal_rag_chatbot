"""
config.py — Single source of truth for project paths and environment variables.

Why pydantic-settings instead of plain os.environ:
  - Validates all required env vars at startup, with a clear error if any are missing
  - Type-coerces automatically (e.g. "true" → True for booleans)
  - One place to see every config knob the app uses

Why define PROJECT_ROOT here:
  - Every other file imports `settings` and uses `settings.data_dir`, etc.
  - If the project structure changes, we fix the root definition ONCE here,
    not hunting down parents[N] across multiple files.
"""

from pathlib import Path
from pydantic_settings import BaseSettings

# config.py lives at: backend/app/config.py
# parents[0] = backend/app
# parents[1] = backend
# parents[2] = prenatal_rag_chatbot  ← project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # --- Required (must be in .env) ---
    openai_api_key: str
    pinecone_api_key: str

    # --- Optional with defaults ---
    pinecone_index_name: str = "poshan-saathi"
    app_env: str = "development"

    # --- Derived paths (computed, not from .env) ---
    data_dir: Path = PROJECT_ROOT / "data"

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "extra": "ignore"}


settings = Settings()
