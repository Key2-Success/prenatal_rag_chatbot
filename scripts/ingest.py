"""
ingest.py — One-time script to parse PDFs, embed chunks, and upsert to Pinecone.

Prerequisite (one-time):
    pip install -e .          # makes `backend.app.*` importable from anywhere

Run:
    python -m scripts.ingest

Re-run only when the source PDFs change. Think of it as "building the
knowledge base" before the app goes live.

Note on idempotency: every chunk gets a fresh UUID at upsert time, so running
this script twice doubles your index. If you need to re-ingest cleanly, delete
the Pinecone index from the Pinecone console first (or we can add a --reset
flag later).
"""

from backend.app.rag.chunker import chunk_all_pdfs
from backend.app.rag.embedder import embed_texts
from backend.app.rag.retriever import upsert_chunks


def main():
    print("=== Poshan Saathi — PDF Ingestion ===\n")

    print("Step 1/3: Chunking PDFs...")
    chunks = chunk_all_pdfs()

    print(f"\nStep 2/3: Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    print(f"  → Got {len(embeddings)} embeddings")

    print("\nStep 3/3: Upserting to Pinecone...")
    upsert_chunks(chunks, embeddings)

    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    main()
