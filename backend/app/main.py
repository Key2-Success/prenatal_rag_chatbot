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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.chat.pipeline import run_chat
from backend.app.models.schemas import ChatRequest, ChatResponse
from backend.app.observability import propagate_attributes

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Poshan Saathi API",
    description="Prenatal nutrition RAG chatbot for women in India.",
    version="0.1.0",
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
        with propagate_attributes(session_id=request_id):
            return run_chat(request)
    except Exception:
        # Log internally with traceback — never echo internals to clients.
        # request_id flows out via the middleware header, so support can
        # cross-reference logs and Langfuse without a stack trace leak.
        logger.exception("run_chat failed (request_id=%s)", request_id)
        raise HTTPException(status_code=500, detail="Internal server error")
