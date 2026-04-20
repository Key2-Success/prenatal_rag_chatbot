"""
config.py — Single source of truth for project paths and environment variables.

Why pydantic-settings instead of plain os.environ:
  - Validates all required env vars at startup, with a clear error if any are missing
  - Type-coerces automatically (e.g. "true" → True for booleans)
  - One place to see every config knob the app uses

Why walk up to find PROJECT_ROOT instead of parents[N]:
  - parents[N] breaks silently if this file ever moves — wrong count, wrong path
  - Walking up to the nearest pyproject.toml is location-independent: this file
    can live anywhere in the tree and it will always find the true project root
"""

from pathlib import Path
from pydantic_settings import BaseSettings


def _find_project_root() -> Path:
    """
    Walk up the directory tree from this file until we find pyproject.toml.
    That file sits at the project root and acts as an unambiguous marker.
    Raises RuntimeError if we reach the filesystem root without finding it.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:       # stop at filesystem root
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError(
        "Could not locate project root. "
        "Make sure pyproject.toml exists at the root of the repository."
    )


PROJECT_ROOT = _find_project_root()


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
