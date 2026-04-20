"""
sources.py — Single loader for data/sources.json.

Both chunker.py (at ingestion time) and retriever.py (at query time)
need information from sources.json. Putting the loader here means:
  - One file owns the schema of sources.json
  - Priority order is derived from the file, not duplicated in code
  - Adding/removing/reordering sources is a one-file edit

lru_cache means the JSON is parsed once per process.
"""

import json
from functools import lru_cache

from backend.app.config import settings


@lru_cache(maxsize=1)
def load_sources() -> list[dict]:
    """
    Returns every source dict from sources.json,
    sorted by doc_reference_order (1 = highest priority).
    """
    with open(settings.data_dir / "sources.json") as f:
        sources = json.load(f)
    return sorted(sources, key=lambda s: s["doc_reference_order"])


@lru_cache(maxsize=1)
def sources_by_filename() -> dict[str, dict]:
    """Sources keyed by file_name — used at chunking time to look up metadata."""
    return {s["file_name"]: s for s in load_sources()}


@lru_cache(maxsize=1)
def priority_order() -> list[str]:
    """
    org_display_names in priority order.
    Derived from sources.json so retrieval order follows the data, not hardcoded.
    e.g. ["MoHFW", "FOGSI", "WHO"]
    """
    return [s["org_display_name"] for s in load_sources()]
