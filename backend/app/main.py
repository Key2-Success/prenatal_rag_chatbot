"""
main.py — FastAPI application entry point.

Endpoints:
  GET  /health          → liveness check
  POST /chat            → main RAG chat endpoint

To run locally:
    uvicorn backend.app.main:app --reload

Then open http://localhost:8000/docs for the interactive Swagger UI —
you can test every endpoint directly in the browser, no extra tools needed.
"""

from dotenv import load_dotenv
load_dotenv()  # must be called before any os.environ access

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.models.schemas import ChatRequest, ChatResponse
from backend.app.chat.pipeline import run_chat

app = FastAPI(
    title="Poshan Saathi API",
    description="Prenatal nutrition RAG chatbot for women in India.",
    version="0.1.0",
)

# CORS: allows the Next.js frontend (running on localhost:3000 in dev,
# or your Vercel domain in prod) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # tighten this to your domain in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Meta"])
def health():
    """Quick liveness check. Returns 200 if the server is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Accepts a message + full user profile, runs the RAG pipeline,
    and returns an answer with source citations.

    FastAPI automatically validates the request body against ChatRequest
    (Pydantic) and returns a 422 with clear error messages if anything
    is missing or the wrong type — no manual validation needed.
    """
    try:
        return run_chat(request)
    except Exception as e:
        # In production you'd log this properly (e.g. with structlog or Sentry)
        raise HTTPException(status_code=500, detail=str(e))
