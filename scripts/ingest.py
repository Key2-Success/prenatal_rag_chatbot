"""
ingest.py — One-time script to parse PDFs, embed chunks, and upsert to Pinecone.

Prerequisite (one-time):
    pip install -e .          # makes `backend.app.*` importable from anywhere

Run:
    python -m scripts.ingest                # incremental upsert (doubles vectors if run twice)
    python -m scripts.ingest --reset        # delete all vectors first, then re-ingest cleanly
    python -m scripts.ingest --only mohfw   # process only files whose key contains "mohfw"

Re-run only when the source PDFs change OR when the chunking / parsing pipeline
itself changes (e.g. pypdf → LlamaParse). Think of it as "rebuilding the knowledge base."

Per-file commit design:
    Each PDF is parsed → chunked → embedded → upserted as a complete unit BEFORE
    the next file starts. If LlamaParse times out, network drops, or the user
    hits Ctrl+C, every PDF that completed already has its vectors in Pinecone —
    only the in-flight file's work is lost. The previous all-or-nothing design
    (chunk_all_pdfs → embed_all → upsert_all) lost EVERY file's progress on
    any interruption. We hit that bug on the LlamaParse migration when MoHFW
    and FOGSI never finished while WHO had already been chunked.

Per-file try/except:
    A LlamaParse error on one PDF (rate limit, unsupported format, scan with
    no OCR) shouldn't kill the entire ingest. Failed files are reported in
    the summary so the user can re-run just those with --only.
"""

import argparse
import sys
import time

from backend.app.config import settings
from backend.app.observability import flush as flush_traces
from backend.app.rag.chunker import chunk_pdf
from backend.app.rag.embedder import embed_texts
from backend.app.rag.retriever import get_index, upsert_chunks
from backend.app.sources import sources_by_filename


def _reset_index() -> None:
    """
    Delete every vector in the configured Pinecone index.

    Used by --reset before re-ingestion so we don't end up with stale chunks
    from a previous parsing pipeline alongside the new ones.

    Empty-namespace edge case: Pinecone creates the default namespace
    LAZILY on first upsert. If the index has never had vectors (fresh
    index, or previously cleared from the console), the namespace doesn't
    exist and delete_all=True returns 404 "Namespace not found". That's
    already the desired state — treat it as success and continue.
    """
    from pinecone.exceptions import NotFoundException

    index = get_index()
    print(f"  → Deleting all vectors in '{settings.pinecone_index_name}'...")
    try:
        index.delete(delete_all=True)
        print("  → Reset complete.")
    except NotFoundException as e:
        if "Namespace not found" in str(e):
            print("  → Index already empty (default namespace not yet created); "
                  "nothing to delete.")
        else:
            raise


def _ingest_one(file_name: str, idx: int, total: int) -> tuple[bool, int, float, str]:
    """
    Run the full parse → chunk → embed → upsert pipeline for ONE PDF.

    Returns (succeeded, chunk_count, elapsed_seconds, error_message).
    Catches and reports exceptions instead of propagating them so a single
    file's failure can't abort the rest of the ingest.
    """
    t0 = time.perf_counter()
    print(f"\n[{idx}/{total}] Ingesting {file_name}...")
    try:
        chunks = chunk_pdf(file_name)
        print(f"  → {len(chunks)} chunks produced by chunker")

        if not chunks:
            elapsed = time.perf_counter() - t0
            return False, 0, elapsed, "chunker returned 0 chunks (PDF empty or LlamaParse silently failed?)"

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        print(f"  → {len(embeddings)} embeddings generated")

        upsert_chunks(chunks, embeddings)
        elapsed = time.perf_counter() - t0
        print(f"  ✓ [{idx}/{total}] {file_name} committed to Pinecone ({elapsed:.1f}s)")
        return True, len(chunks), elapsed, ""
    except Exception as e:
        elapsed = time.perf_counter() - t0
        # Show the error inline but don't re-raise — other files can still proceed.
        # Specifically catches LlamaParse rate-limit / API errors, OpenAI embedding
        # errors, Pinecone upsert errors.
        msg = f"{type(e).__name__}: {e}"
        print(f"  ✗ [{idx}/{total}] {file_name} FAILED after {elapsed:.1f}s — {msg}")
        return False, 0, elapsed, msg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse PDFs with LlamaParse, embed, and upsert to Pinecone."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete every existing vector in the Pinecone index before "
            "upserting. Use when the parsing / chunking pipeline has "
            "changed and stale chunks would otherwise pollute retrieval."
        ),
    )
    parser.add_argument(
        "--only",
        help=(
            "Process only files whose name contains this substring "
            "(case-insensitive). Useful for re-ingesting a single source "
            "after a partial failure without touching the others. Skips "
            "--reset for non-matching files automatically. Example: "
            "--only mohfw → reingest just anc_guidelines_india_mohfw.pdf."
        ),
    )
    args = parser.parse_args()

    print("=== Poshan Saathi — PDF Ingestion ===")

    if args.reset:
        if args.only:
            print(f"\n⚠ --reset + --only would wipe everything and only re-ingest "
                  f"'{args.only}'. That's almost certainly a mistake. Refusing.")
            return 2
        print()
        _reset_index()

    # Resolve which files to process
    all_sources = sources_by_filename()
    if args.only:
        needle = args.only.lower()
        files = [f for f in all_sources if needle in f.lower()]
        if not files:
            print(f"\n✗ No source files match '--only {args.only}'. "
                  f"Available: {list(all_sources.keys())}")
            return 1
        print(f"\nProcessing {len(files)} file(s) matching '--only {args.only}'.")
    else:
        files = list(all_sources.keys())
        print(f"\nProcessing all {len(files)} declared source(s).")

    # Per-file ingest loop. Each file is parsed → chunked → embedded → upserted
    # as a complete unit so partial progress survives interruptions.
    results: list[tuple[str, bool, int, float, str]] = []
    overall_t0 = time.perf_counter()
    for i, file_name in enumerate(files, start=1):
        ok, n_chunks, elapsed, err = _ingest_one(file_name, i, len(files))
        results.append((file_name, ok, n_chunks, elapsed, err))

    overall_elapsed = time.perf_counter() - overall_t0

    # Final summary — surfaces partial-success / partial-failure clearly.
    print()
    print("=" * 72)
    print("  Ingestion summary")
    print("=" * 72)
    total_chunks = sum(n for _, ok, n, _, _ in results if ok)
    n_ok = sum(1 for _, ok, *_ in results if ok)
    n_fail = sum(1 for _, ok, *_ in results if not ok)
    for file_name, ok, n, elapsed, err in results:
        mark = "✓" if ok else "✗"
        line = f"  {mark} {file_name:<40} {n:>5} chunks  ({elapsed:.1f}s)"
        if not ok:
            line += f"  ← {err}"
        print(line)
    print(f"\n  {n_ok}/{len(results)} files succeeded  |  "
          f"{total_chunks} total chunks  |  {overall_elapsed:.1f}s wall-clock")

    if n_fail > 0:
        print(f"\n  ⚠ {n_fail} file(s) failed. To retry just those, run:")
        for file_name, ok, *_ in results:
            if not ok:
                # Recommend a substring distinctive enough to match only that file
                keyword = file_name.split("_")[-1]  # e.g. "mohfw" from "anc_guidelines_india_mohfw"
                print(f"      python -m scripts.ingest --only {keyword}")

    # Drain Langfuse buffer before exit. No-op when Langfuse is disabled.
    # Without this, the embedding generations may never reach the server
    # because the process exits before the background flusher runs.
    flush_traces()

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
