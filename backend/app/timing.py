"""
timing.py — opt-in, per-stage latency collection for the chat pipeline.

The pipeline spans multiple modules (pipeline.py → retriever.py), so rather
than thread a timings dict through every call signature, we use a context
variable: a benchmark activates a collector with `collect_timings()`, and any
`record_stage(name)` block executing inside that context (in the same thread)
adds its elapsed seconds to the collector.

Production overhead is negligible: when no collector is active — the only case
in production, since nothing calls collect_timings() there — `record_stage`
does a single ContextVar read and yields without touching the clock.

Why a ContextVar and not a global: it's the same isolation model the Langfuse
`@observe` spans use, so concurrent requests (or parallel eval workers) never
cross-contaminate each other's timings.
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

# Active collector for the current context, or None in production.
_collector: ContextVar[dict[str, float] | None] = ContextVar(
    "stage_timings", default=None
)


@contextmanager
def collect_timings() -> Iterator[dict[str, float]]:
    """
    Activate stage-timing collection for the duration of the block and yield
    the dict that `record_stage` writes into (stage name → summed seconds).

    Repeated stage names accumulate, so a stage that runs twice in one request
    reports its combined time.
    """
    collector: dict[str, float] = {}
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)


@contextmanager
def record_stage(name: str) -> Iterator[None]:
    """
    Time the wrapped block under `name` when a collector is active; otherwise
    do nothing measurable (production path — one ContextVar read, no timing).
    """
    collector = _collector.get()
    if collector is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        collector[name] = collector.get(name, 0.0) + (time.perf_counter() - start)
