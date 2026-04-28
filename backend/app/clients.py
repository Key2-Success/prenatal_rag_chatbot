"""
clients.py — Lazily-initialized singleton clients for external services.

Why one place:
  Both embedder.py (ingestion-time) and pipeline.py (query-time) talk to
  OpenAI. Without this, each module would carry its own `_client` global
  and `get_client()` accessor — three lines of duplicated bookkeeping per
  module that all need to do the same thing. Centralizing means there's
  exactly one OpenAI client per process, and exactly one place to change
  if we ever add retry/timeout config.

Why lazy:
  Importing this module shouldn't open network handles. Tests can monkey-
  patch `settings.openai_api_key` before the first call without OpenAI
  raising on a missing key at import.
"""

from openai import OpenAI

from backend.app.config import settings

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Return the process-wide OpenAI client, creating it on first call."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client
