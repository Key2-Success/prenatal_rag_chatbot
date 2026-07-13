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
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.chat.pipeline import run_chat
from backend.app.models.schemas import ChatRequest, ChatResponse, UserProfile
from backend.app.observability import propagate_attributes
from backend.app.rag.retriever import warmup_reranker

logger = logging.getLogger(__name__)


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

# CORS: allow the Next.js frontend (localhost:3000 in dev, your Vercel
# domain in prod). Tighten allow_origins before going live.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
def chat(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint.

    FastAPI auto-validates the request body against `ChatRequest` (Pydantic)
    and returns 422 with field-level errors if anything is missing or wrong.
    """
    request_id = http_request.state.request_id
    try:
        # `propagate_attributes` is the Langfuse v4 idiom for trace-level
        # attrs (session_id, user_id, tags). It threads them through every
        # observation created inside the `with` block, including the parent
        # @observe span on run_chat. Keeps run_chat free of infra concerns.
        with propagate_attributes(session_id=request_id, **_trace_attrs(request.user_profile)):
            return run_chat(request)
    except Exception:
        # Log internally with traceback — never echo internals to clients.
        # request_id flows out via the middleware header, so support can
        # cross-reference logs and Langfuse without a stack trace leak.
        logger.exception("run_chat failed (request_id=%s)", request_id)
        raise HTTPException(status_code=500, detail="Internal server error")
