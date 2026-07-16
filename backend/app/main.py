"""
main.py — FastAPI application entry point.

Endpoints:
  GET  /health  → liveness check
  POST /chat    → main RAG chat endpoint

Local dev:
    uvicorn backend.app.main:app --reload
    open http://localhost:8000/docs   # interactive Swagger UI
"""

import ipaddress
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone

import httpx
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


def _incr_daily_budget_memory(today: date) -> int:
    """In-memory daily counter (resets on restart). Returns the new count."""
    global _budget_day, _budget_count
    with _budget_lock:
        if today != _budget_day:
            _budget_day = today
            _budget_count = 0
        _budget_count += 1
        return _budget_count


def _incr_daily_budget_redis(day_key: str) -> int:
    """
    Increment today's counter in Upstash Redis (REST) and return the new count.
    Persists across restarts so Render's spin-down can't reset the cap; sets a
    48h TTL on the first hit of the day so keys self-clean.
    """
    base = settings.upstash_redis_rest_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.upstash_redis_rest_token}"}
    with httpx.Client(timeout=3.0) as client:
        r = client.post(base, headers=headers, json=["INCR", day_key])
        r.raise_for_status()
        count = int(r.json()["result"])
        if count == 1:
            client.post(base, headers=headers, json=["EXPIRE", day_key, "172800"])
    return count


def _enforce_daily_budget() -> None:
    """
    Count today's served request across ALL clients; raise 503 once over the
    daily budget. Uses Upstash Redis when configured (survives restarts), else
    an in-memory counter. Redis failures fall back to memory — a flaky counter
    must never take the app down, and the OpenAI hard cap is the true backstop.
    """
    today = datetime.now(timezone.utc).date()
    if settings.upstash_redis_rest_url and settings.upstash_redis_rest_token:
        try:
            count = _incr_daily_budget_redis(f"budget:{today.isoformat()}")
        except Exception as e:  # noqa: BLE001 — never let the counter break /chat
            logger.warning("Upstash budget check failed (%s); using in-memory.", e)
            count = _incr_daily_budget_memory(today)
    else:
        count = _incr_daily_budget_memory(today)
    if count > settings.daily_request_budget:
        raise HTTPException(
            status_code=503,
            detail="The service has reached its daily capacity. "
            "Please try again tomorrow.",
        )


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


def _parse_user_agent(ua: str) -> dict[str, str]:
    """Lightweight User-Agent → device/os/browser (no external dependency)."""
    u = ua.lower()
    if "ipad" in u or ("tablet" in u and "mobile" not in u):
        device = "tablet"
    elif "mobi" in u or "iphone" in u or "android" in u:
        device = "mobile"
    else:
        device = "desktop" if ua else "unknown"
    if "iphone" in u or "ipad" in u:
        os_name = "iOS"
    elif "android" in u:
        os_name = "Android"
    elif "windows" in u:
        os_name = "Windows"
    elif "mac os x" in u or "macintosh" in u:
        os_name = "macOS"
    elif "linux" in u:
        os_name = "Linux"
    else:
        os_name = "unknown"
    # Order matters: Edge/Chrome UAs also contain "safari"/"chrome" tokens.
    if "edg/" in u or "edgios" in u or "edga" in u:
        browser = "Edge"
    elif "firefox" in u or "fxios" in u:
        browser = "Firefox"
    elif "chrome" in u or "crios" in u:
        browser = "Chrome"
    elif "safari" in u:
        browser = "Safari"
    else:
        browser = "unknown"
    return {"device_type": device, "os": os_name, "browser": browser}


# Country lookups are cached per IP (only successful ones) and resolved in a
# background thread, so geo never adds latency to the request path.
_country_cache: dict[str, str] = {}
_country_lock = threading.Lock()


def _lookup_country(ip: str) -> None:
    """Best-effort IP → country code, cached. Runs in a background thread."""
    try:
        r = httpx.get(f"https://ipwho.is/{ip}", params={"fields": "country_code"}, timeout=2.5)
        cc = (r.json() or {}).get("country_code")
    except Exception:  # noqa: BLE001 — geo is best-effort, never fatal
        cc = None
    if cc:
        with _country_lock:
            _country_cache[ip] = cc


def _client_country(ip: str) -> str:
    """
    Coarse country code for analytics. Never blocks: returns a cached value, or
    "pending" while a background thread fills the cache for this IP's *next*
    request. Private/loopback IPs (local dev) return "local". Country only — no
    city, and we never store the raw IP itself.
    """
    try:
        addr = ipaddress.ip_address(ip)
        if addr.is_private or addr.is_loopback:
            return "local"
    except ValueError:
        return "unknown"
    with _country_lock:
        if ip in _country_cache:
            return _country_cache[ip]
    threading.Thread(target=_lookup_country, args=(ip,), daemon=True).start()
    return "pending"


def _client_attrs(request: Request) -> dict:
    """
    Anonymous, header-derived trace attributes for audience analytics in
    Langfuse. There's no auth, so user_id is a client-generated anonymous id
    (a UUID the frontend keeps in localStorage) — enough to count unique
    visitors and group their sessions without collecting any real identity.
    Device/OS/browser come from the User-Agent, language from Accept-Language,
    and a coarse **country** code from a cached, non-blocking geo-IP lookup.
    We keep the footprint coarse and anonymous — country only (no city), and the
    raw IP is used transiently for the lookup, never logged.
    """
    metadata = _parse_user_agent(request.headers.get("user-agent", ""))
    lang = request.headers.get("accept-language", "")
    if lang:
        metadata["language"] = lang.split(",")[0].split(";")[0].strip()
    metadata["country"] = _client_country(_client_ip(request))
    attrs: dict = {"metadata": metadata}
    anon_id = request.headers.get("x-anon-id")
    if anon_id:
        attrs["user_id"] = anon_id[:64]  # defensive length cap
    return attrs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown hook.

    For the LOCAL reranker backend, warm the cross-encoder before the server
    accepts traffic — the first inference otherwise pays a ~20s model-load +
    kernel-compile cost that would land on the first real user.

    For the hosted (pinecone) backend there is no local model to warm — and the
    lean production image deliberately omits sentence-transformers / torch — so
    warming is skipped. Calling warmup_reranker() there would import
    sentence-transformers and crash the container on boot.
    """
    if settings.reranker_backend == "local":
        logger.info("Warming up local reranker at startup...")
        warmup_reranker()
        logger.info("Reranker warm; ready to serve.")
    else:
        logger.info(
            "Reranker backend '%s' is hosted — skipping local warmup.",
            settings.reranker_backend,
        )
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


@app.get("/", tags=["Meta"], include_in_schema=False)
def root():
    """
    Friendly root. The API has no web UI of its own (the frontend is a separate
    Next.js app), so point anyone who pokes the bare URL at the docs + health
    instead of returning a bare 404.
    """
    return {
        "service": "Poshan Saathi API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


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

    # Trace attributes: profile-derived (diet/trimester/conditions) + anonymous
    # client analytics (user_id, device/os/browser, language). Merged into one
    # propagate_attributes call so they land on the trace and every child span.
    profile_attrs = _trace_attrs(body.user_profile)
    client = _client_attrs(request)
    pa_kwargs = {
        "session_id": request_id,
        "tags": profile_attrs["tags"],
        "metadata": {**profile_attrs["metadata"], **client["metadata"]},
    }
    if "user_id" in client:
        pa_kwargs["user_id"] = client["user_id"]
    try:
        # `propagate_attributes` is the Langfuse v4 idiom for trace-level attrs.
        # It threads them through every observation created inside the block,
        # including the parent @observe span on run_chat.
        with propagate_attributes(**pa_kwargs):
            return run_chat(body)
    except Exception:
        # Log internally with traceback — never echo internals to clients.
        # request_id flows out via the middleware header, so support can
        # cross-reference logs and Langfuse without a stack trace leak.
        logger.exception("run_chat failed (request_id=%s)", request_id)
        raise HTTPException(status_code=500, detail="Internal server error")
