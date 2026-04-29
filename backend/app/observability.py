"""
observability.py — Thin Langfuse v4 shim used by pipeline / classifier / retriever.

Why a shim:
  Without one, every instrumented file would need to either (a) import
  langfuse unconditionally (hard dep, breaks if uninstalled) or (b) wrap
  every call site in `if settings.langfuse_enabled` (visual noise that
  buries the actual logic). This module hides that branch in one place.

API exposed (stable whether Langfuse is enabled or not):

  observe(name=..., as_type=...)
    Decorator. Wraps the function in a Langfuse span when enabled; an
    identity decorator otherwise.

  update_current_span(input=..., output=..., metadata=...)
    Updates the active observation (the closest enclosing @observe span).
    Use to set EXPLICIT input/output instead of the default behaviour of
    capturing every function argument — see Langfuse skill:
    "Not explicitly setting input with @observe: All function args become
     trace input (including API keys, configs)".

  propagate_attributes(session_id=..., user_id=..., tags=...)
    Context manager that attaches trace-level attributes (session_id,
    user_id, tags) to all observations created inside the `with` block.
    Used in main.py to thread the request_id into traces as session_id.

  flush()
    Force-send any buffered events. Required at the end of short-lived
    scripts (eval runner, ingestion) — uvicorn keeps the process alive
    long enough for the background flusher to drain on its own.

Design choice: this shim deliberately exposes a NARROWER API than Langfuse
itself. We forward only what the pipeline actually uses. Adding a feature
means adding it here intentionally — easier to reason about than a wide
import surface.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from backend.app.config import settings

if settings.langfuse_enabled:
    # The langfuse SDK reads its credentials from os.environ. Pydantic-settings
    # has already loaded them from .env into our `settings` object, but the
    # SDK reads os.environ directly — so we mirror them across before import.
    # IMPORTANT: this MUST happen before `from langfuse import ...` so the
    # client picks up the right credentials.
    import os

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key or "")
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host or "")

    from langfuse import (  # noqa: E402
        get_client,
        observe as _observe,
        propagate_attributes as _propagate_attributes,
    )

    _client = get_client()

    def observe(name: str | None = None, as_type: str | None = None) -> Callable:
        """Forward to langfuse.observe — only pass kwargs that were explicitly set.

        Passing `as_type=None` (or any None default) to Langfuse's `observe`
        confuses the SDK's introspection — it can fail to nest spans under
        the active OTel context. Forwarding only what the caller specified
        keeps the SDK on its documented happy path.
        """
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if as_type is not None:
            kwargs["as_type"] = as_type
        return _observe(**kwargs)

    def update_current_span(**kwargs: Any) -> None:
        """Forward to langfuse.update_current_span on the active observation."""
        _client.update_current_span(**kwargs)

    @contextmanager
    def propagate_attributes(**kwargs: Any) -> Iterator[None]:
        """Pass trace-level attrs (session_id, user_id, tags) to all enclosed observations."""
        with _propagate_attributes(**kwargs):
            yield

    def flush() -> None:
        """Drain buffered events. Call at the end of scripts."""
        _client.flush()

else:
    # No-op fallback. The decorator returns the function untouched so there
    # is zero runtime overhead when Langfuse is disabled (no extra frames,
    # no dict lookups, nothing).
    def observe(name: str | None = None, as_type: str | None = None) -> Callable:
        def decorator(fn: Callable) -> Callable:
            return fn
        return decorator

    def update_current_span(**kwargs: Any) -> None:
        return None

    @contextmanager
    def propagate_attributes(**kwargs: Any) -> Iterator[None]:
        yield

    def flush() -> None:
        return None
