"""
ingest.py — One-time script to parse PDFs, embed chunks, and upsert to Pinecone.

Prerequisite (one-time):
    pip install -e .          # makes `backend.app.*` importable from anywhere

Run:
    python -m scripts.ingest                # incremental upsert (will double if run twice)
    python -m scripts.ingest --reset        # delete all vectors first, then re-ingest cleanly

Re-run only when the source PDFs change OR when the chunking / parsing pipeline
itself changes (e.g. pypdf → LlamaParse, RecursiveCharacterTextSplitter →
SemanticChunker). Think of it as "rebuilding the knowledge base" — every chunk
gets a fresh UUID at upsert time, so without --reset a second run leaves stale
duplicates that pollute retrieval scores.
"""

import argparse
import sys

from backend.app.config import settings
from backend.app.observability import flush as flush_traces
from backend.app.rag.chunker import chunk_all_pdfs
from backend.app.rag.embedder import embed_texts
from backend.app.rag.retriever import get_index, upsert_chunks


def _reset_index() -> None:
    """
    Delete every vector in the configured Pinecone index.

    Used by --reset before re-ingestion so we don't end up with stale chunks
    from a previous parsing pipeline alongside the new ones. Pinecone's
    delete(delete_all=True) clears the default namespace in one call; we
    don't use named namespaces in this project so this covers everything.

    Why not delete and recreate the index? Index recreation can take a
    minute on serverless tiers and serial creation hits a stricter rate
    limit. Vector-level delete is instant and equivalent for our purposes.
    """
    index = get_index()
    print(f"  → Deleting all vectors in '{settings.pinecone_index_name}'...")
    index.delete(delete_all=True)
    print("  → Reset complete.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse PDFs with LlamaParse, embed, and upsert to Pinecone."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete every existing vector in the Pinecone index before "
            "upserting. Use this when the parsing / chunking pipeline has "
            "changed and stale chunks from the previous pipeline would "
            "otherwise pollute retrieval results."
        ),
    )
    args = parser.parse_args()

    print("=== Poshan Saathi — PDF Ingestion ===\n")

    if args.reset:
        print("Step 0/3: Resetting Pinecone index...")
        _reset_index()
        print()

    print("Step 1/3: Chunking PDFs (LlamaParse → SemanticChunker)...")
    chunks = chunk_all_pdfs()

    if not chunks:
        print("\n⚠ No chunks produced — aborting before embedding/upsert.")
        return 1

    print(f"\nStep 2/3: Embedding {len(chunks)} chunks...")
    # Chunk is a Pydantic model — attribute access, not dict subscript.
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    print(f"  → Got {len(embeddings)} embeddings")

    print("\nStep 3/3: Upserting to Pinecone...")
    upsert_chunks(chunks, embeddings)

    # Drain Langfuse buffer before exit. No-op when Langfuse is disabled.
    # Without this, the embedding generations may never reach the server
    # because the process exits before the background flusher runs.
    flush_traces()

    print("\n✓ Ingestion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
