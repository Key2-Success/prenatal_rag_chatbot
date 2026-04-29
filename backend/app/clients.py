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

Langfuse wrapping:
  When `settings.langfuse_enabled`, we return Langfuse's drop-in OpenAI
  subclass (`langfuse.openai.OpenAI`) instead of the vanilla one. It's a
  transparent proxy — same API surface, same return types — that emits
  one Langfuse generation per call (capturing prompts, responses, latency,
  token counts, cost). Without keys, we return the vanilla client and
  no Langfuse data is produced. Callers never branch on which they got.
"""

from openai import OpenAI

from backend.app.config import settings

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Return the process-wide OpenAI client, creating it on first call."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if settings.langfuse_enabled:
        # Drop-in subclass — preserves the OpenAI API surface, adds tracing.
        # Imported lazily so the langfuse package is only loaded when used.
        from langfuse.openai import OpenAI as LangfuseOpenAI
        _openai_client = LangfuseOpenAI(api_key=settings.openai_api_key)
    else:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client
