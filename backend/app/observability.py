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

  score_current_trace(name=..., value=..., data_type=...)
    Attach a filterable/chartable score to the current trace. Used for
    outcome dimensions known only at the end of the pipeline (response_type)
    that propagate_attributes — applied at entry — can't capture.

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
        """Decorator that creates a named Langfuse span around a function.

        Implementation uses `client.start_as_current_observation(name=...)`
        directly rather than the SDK's `@observe` because in Langfuse v4.6.x,
        `@observe(name="x")` silently drops the name attribute — observations
        come back with name=None, the UI shows blank/unlabelled spans, and
        the trace tree view becomes unreadable. Verified by direct API
        query against our Langfuse instance (May 2026). The lower-level
        `start_as_current_observation` accepts name as a REQUIRED kwarg
        and stores it correctly on the OTel span.

        Tradeoff: we lose @observe's auto-capture of function inputs/outputs.
        That's fine for this project — every instrumented function already
        calls update_current_span(input=..., output=...) explicitly, which
        is the recommended pattern anyway (the auto-capture would dump full
        ChatRequest objects including the user profile into the trace input,
        whereas the explicit calls trim to just the relevant fields).
        """
        from functools import wraps

        obs_type = as_type or "span"

        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapped(*args: Any, **kw: Any) -> Any:
                # Fall back to function name if no explicit name given.
                # Matches @observe's default behaviour for that case.
                span_name = name or fn.__name__
                with _client.start_as_current_observation(
                    name=span_name,
                    as_type=obs_type,
                ):
                    return fn(*args, **kw)
            return wrapped

        return decorator

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

    def score_current_trace(**kwargs: Any) -> None:
        """Attach a score to the current trace (numeric / categorical / boolean).

        Used for filterable outcome dimensions known only at the END of the
        pipeline — e.g. response_type — which propagate_attributes (applied at
        request entry) cannot capture. Scores are filterable and chartable in
        the Langfuse UI, so they power "how many emergencies / no_results /
        answers" breakdowns.
        """
        _client.score_current_trace(**kwargs)

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

    def score_current_trace(**kwargs: Any) -> None:
        return None
