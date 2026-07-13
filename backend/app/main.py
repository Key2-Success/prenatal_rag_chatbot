"""
main.py — FastAPI application entry point.

Endpoints:
  GET  /health  → liveness check
  POST /chat    → main RAG chat endpoint

Local dev:
    uvicorn backend.app.main:app --reload
    open http://localhost:8000/docs   # interactive Swagger UI
"""

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.app.chat.pipeline import run_chat
from backend.app.config import settings
from backend.app.models.schemas import ChatRequest, ChatResponse, UserProfile
from backend.app.observability import propagate_attributes
from backend.app.rag.retriever import warmup_reranker

logger = logging.getLogger(__name__)


def _client_ip(request: Request) -> str:
    """
    Real client IP for rate-limit bucketing.

    Behind a proxy / platform load balancer (Render, Fly, Cloudflare), the
    socket peer is the proxy, so every request would share one key and get
    limited together. Prefer the first hop of X-Forwarded-For, which the
    platform sets; fall back to the socket address for direct/local runs.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


# Per-IP rate limiter. Limits are applied per-route via @limiter.limit (see
# /chat). In-memory storage — fine for a single instance.
limiter = Limiter(key_func=_client_ip)


# Global daily budget: a hard ceiling on total /chat calls across ALL clients,
# so distributed abuse (many IPs, each under the per-IP caps) still can't run
# up the OpenAI bill. In-memory + single-instance; resets at UTC midnight.
# Locked because sync endpoints run in a threadpool.
_budget_lock = threading.Lock()
_budget_day: date | None = None
_budget_count = 0


def _enforce_daily_budget() -> None:
    """Count today's served request; raise 503 once over the daily budget."""
    global _budget_day, _budget_count
    today = datetime.now(timezone.utc).date()
    with _budget_lock:
        if today != _budget_day:
            _budget_day = today
            _budget_count = 0
        if _budget_count >= settings.daily_request_budget:
            raise HTTPException(
                status_code=503,
                detail="The service has reached its daily capacity. "
                "Please try again tomorrow.",
            )
        _budget_count += 1


def _trimester(week: int) -> str:
    """Map pregnancy week → trimester bucket ("1" | "2" | "3")."""
    if week <= 13:
        return "1"
    if week <= 27:
        return "2"
    return "3"


def _trace_attrs(profile: UserProfile) -> dict:
    """
    Profile-derived tags + metadata for Langfuse trace filtering.

    These are known at request entry, so they ride on propagate_attributes
    (the Langfuse v4 idiom for correlating attributes — they apply to the
    trace and every child observation). Outcome dimensions known only later,
    like response_type, are attached as scores inside the pipeline instead.
    """
    diet = profile.diet_type.value
    trimester = _trimester(profile.pregnancy_week)
    conditions = [c.value for c in profile.medical_conditions]
    tags = [f"diet:{diet}", f"trimester:{trimester}"]
    tags += [f"condition:{c}" for c in conditions] or ["condition:none"]
    metadata = {
        "diet_type": diet,
        "trimester": trimester,
        "pregnancy_week": str(profile.pregnancy_week),
        "medical_conditions": ", ".join(conditions) or "none",
    }
    return {"tags": tags, "metadata": metadata}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown hook. Warm the cross-encoder reranker before the server
    accepts traffic: the first inference otherwise pays a ~20s model-load +
    MPS kernel-compile cost that would land on the first real user. Warming
    at boot moves that cost to where it belongs.
    """
    logger.info("Warming up reranker at startup...")
    warmup_reranker()
    logger.info("Reranker warm; ready to serve.")
    yield


app = FastAPI(
    title="Poshan Saathi API",
    description="Prenatal nutrition RAG chatbot for women in India.",
    version="0.1.0",
    lifespan=lifespan,
)

# Wire the rate limiter: register it on the app and return a clean 429 (not a
# 500) when a per-IP limit is exceeded.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: env-driven. Dev defaults to the local Next.js frontend; set
# CORS_ALLOW_ORIGINS=https://your-app.vercel.app in prod. Kept to an explicit
# allow-list (never "*") so a malicious page can't drive the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """
    Reject oversized bodies before parsing — a cheap DoS / token-bomb guard on
    top of the message max_length already enforced by the ChatRequest schema.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > settings.max_body_bytes:
                return JSONResponse(
                    status_code=413, content={"detail": "Request body too large."}
                )
        except ValueError:
            pass  # malformed header — let downstream parsing reject it
    return await call_next(request)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Attach a per-request UUID for log correlation and Langfuse session_id.

    Read from inbound `X-Request-ID` if a frontend / load balancer set one;
    otherwise generate. Echoed back in the response so clients can quote
    it in bug reports and we can find the matching trace + log lines.
    """
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health", tags=["Meta"])
def health():
    """Liveness check. 200 means the process is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
@limiter.limit(f"{settings.rate_limit_per_day}/day")
def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.

    FastAPI auto-validates the request body against `ChatRequest` (Pydantic)
    and returns 422 with field-level errors if anything is missing or wrong.

    Note: the Starlette `Request` is named `request` (not `http_request`) so
    slowapi's @limiter.limit can locate it; the JSON body is `body`.
    """
    request_id = request.state.request_id
    # Global daily cost ceiling. Checked AFTER per-IP rate limiting — slowapi
    # rejects over-limit requests before this runs, so they don't consume budget.
    _enforce_daily_budget()
    try:
        # `propagate_attributes` is the Langfuse v4 idiom for trace-level
        # attrs (session_id, tags, metadata). It threads them through every
        # observation created inside the `with` block, including the parent
        # @observe span on run_chat. Keeps run_chat free of infra concerns.
        with propagate_attributes(session_id=request_id, **_trace_attrs(body.user_profile)):
            return run_chat(body)
    except Exception:
        # Log internally with traceback — never echo internals to clients.
        # request_id flows out via the middleware header, so support can
        # cross-reference logs and Langfuse without a stack trace leak.
        logger.exception("run_chat failed (request_id=%s)", request_id)
        raise HTTPException(status_code=500, detail="Internal server error")
