"""
sources.py — Typed loader for data/sources.json.

Both chunker.py (at ingestion time) and retriever.py (at query time) need
information from sources.json. This module owns:

  - The schema (Source) — declared once, validated at load
  - The single load (lru_cache) — the JSON is parsed exactly once per process
  - Derived views (sources_by_filename, priority_order)

Adding/removing/reordering sources is a one-file edit (sources.json).
Renaming a JSON field will fail loudly here, at startup, instead of
silently breaking a downstream KeyError at retrieval time.
"""

import json
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

from backend.app.config import DATA_DIR


class Source(BaseModel):
    """One row of data/sources.json."""

    # Strict: any unexpected key in sources.json is almost certainly a typo
    # we want to hear about, not silently ignore.
    model_config = ConfigDict(extra="forbid")

    doc_id: int
    file_name: str
    file_type: str
    doc_title: str
    doc_language: str
    org_geographic_scope: str
    org_official_name: str
    org_display_name: str
    doc_source: str
    doc_year_published: int
    doc_num_pages: int
    # Lower number = higher priority (1 wins over 2). Validated > 0 to make
    # zero/negative values impossible — they'd silently sort to the front.
    doc_reference_order: int = Field(..., gt=0)
    doc_description: str
    doc_intended_use: str


@lru_cache(maxsize=1)
def load_sources() -> tuple[Source, ...]:
    """
    Parse sources.json and return all sources sorted by priority.

    Returns a tuple (not a list) so callers can't accidentally mutate the
    cached value and corrupt every subsequent call.
    """
    with open(DATA_DIR / "sources.json") as f:
        raw = json.load(f)
    sources = [Source(**row) for row in raw]
    sources.sort(key=lambda s: s.doc_reference_order)
    return tuple(sources)


@lru_cache(maxsize=1)
def sources_by_filename() -> dict[str, Source]:
    """Sources keyed by file_name — used at chunking time to look up metadata."""
    return {s.file_name: s for s in load_sources()}


@lru_cache(maxsize=1)
def priority_order() -> tuple[str, ...]:
    """
    Org display names in priority order, e.g. ("MoHFW", "FOGSI", "WHO").

    Tuple (not list) so callers can't mutate the cached order.
    """
    return tuple(s.org_display_name for s in load_sources())


@lru_cache(maxsize=1)
def priority_rank_by_org() -> dict[str, int]:
    """
    Map org_display_name → doc_reference_order (lower = higher priority).

    Used by the retriever to sort selected chunks so the LLM sees
    higher-priority source content first in the context window, without
    any additive score nudges.

    Example: {"MoHFW": 1, "FOGSI": 2, "WHO": 3}
    """
    return {s.org_display_name: s.doc_reference_order for s in load_sources()}
