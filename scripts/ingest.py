"""
ingest.py — One-time script to parse PDFs, embed chunks, and upsert to Pinecone.

Run this ONCE after setting up your .env file:
    python -m scripts.ingest

You don't run this on every server start — only when the source PDFs change.
Think of it as "building the knowledge base" before the app goes live.
"""

import sys
from pathlib import Path

# Allow running from the project root: python -m scripts.ingest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
