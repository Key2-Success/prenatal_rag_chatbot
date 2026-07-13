# Backend image for deployment (FastAPI + hosted Pinecone rerank).
#
# Deliberately lean: with RERANKER_BACKEND=pinecone (set in the host env) there
# is no torch / sentence-transformers / 600MB model download, so this image fits
# a free-tier container. Ingest code and its heavy deps are excluded.

FROM python:3.12-slim

WORKDIR /app

# Only runtime deps — see requirements-prod.txt for what's excluded and why.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Runtime code + the two data files the query path needs at run time:
#   - data/bm25_encoder.json  (BM25 sparse encoder, loaded as a singleton)
#   - data/sources.json       (source priority ordering)
# Ingest scripts, PDFs, eval, tests, and the frontend are excluded (.dockerignore).
COPY backend/ backend/
COPY data/bm25_encoder.json data/sources.json data/
COPY pyproject.toml .

# config.py walks up to pyproject.toml for PROJECT_ROOT; PYTHONPATH makes the
# `backend.app...` absolute imports resolve when uvicorn runs from /app.
ENV PYTHONPATH=/app

# Hosts (Render/Railway/Fly) inject $PORT; default 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
