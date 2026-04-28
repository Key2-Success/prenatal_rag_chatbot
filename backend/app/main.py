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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.chat.pipeline import run_chat
from backend.app.models.schemas import ChatRequest, ChatResponse

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


@app.get("/health", tags=["Meta"])
def health():
    """Liveness check. 200 means the process is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Main chat endpoint.

    FastAPI auto-validates the request body against `ChatRequest` (Pydantic)
    and returns 422 with field-level errors if anything is missing or wrong.
    """
    try:
        return run_chat(request)
    except Exception:
        # Log internally with traceback — never echo internals to clients.
        # Client gets a generic message; the request id (added by your
        # middleware in prod) makes server logs cross-referenceable.
        logger.exception("run_chat failed")
        raise HTTPException(status_code=500, detail="Internal server error")
