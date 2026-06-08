"""
ingest.py — Parse PDFs, embed chunks, and upsert to Pinecone with hybrid vectors.

Prerequisite (one-time):
    pip install -e .          # makes `backend.app.*` importable from anywhere

Run:
    python -m scripts.ingest --recreate-index   # FIRST-TIME hybrid setup (or after metric change)
    python -m scripts.ingest --reset            # re-ingest without recreating the index
    python -m scripts.ingest --only mohfw       # re-ingest one source (reuses existing BM25 encoder)

--recreate-index vs --reset:
    --recreate-index  Deletes the entire Pinecone index and recreates it with
                      metric='dotproduct' (required for hybrid sparse+dense).
                      Also re-fits the BM25 encoder. Use when: first hybrid
                      setup, or if the index metric is wrong.
    --reset           Deletes all vectors but keeps the index schema. Re-fits
                      the BM25 encoder. Use when: re-ingesting a changed corpus
                      without touching the index configuration.

Two-pass ingest design (required for corpus-fitted BM25):
    BM25 IDF weights must be computed over the FULL corpus before any chunk
    can be encoded as a sparse vector. This forces a two-pass approach:

    Pass 1: parse + chunk all PDFs → collect all Chunk objects in memory
    Fit:    BM25Encoder.fit(all_texts) → save to settings.bm25_encoder_path
    Pass 2: embed + upsert per file (error isolation: one file's failure
            doesn't discard other files' work)

    With --only (partial re-ingest): Pass 1 still chunks just the target file,
    but BM25 fitting is SKIPPED — the existing encoder (from the last full
    ingest) is reused. This keeps IDF weights stable across partial updates
    and avoids re-fitting on an incomplete corpus.
"""

import argparse
import sys
import time

from backend.app.config import PROJECT_ROOT, settings
from backend.app.observability import flush as flush_traces
from backend.app.rag.chunker import Chunk, chunk_pdf
from backend.app.rag.embedder import embed_texts
from backend.app.rag.retriever import get_index, upsert_chunks
from backend.app.sources import sources_by_filename


def _recreate_index() -> None:
    """
    Delete the existing Pinecone index and recreate it with metric='dotproduct'.

    Required for hybrid search: Pinecone's classic Index.query API supports
    a sparse_vector parameter only on indexes using dotproduct metric.
    For OpenAI text-embedding-3-small (unit-normalised vectors), dotproduct
    and cosine are mathematically equivalent — this is a schema change, not
    a retrieval quality change.

    Also invalidates the cached index singleton so the next get_index() call
    returns the freshly created index.
    """
    from pinecone import Pinecone, ServerlessSpec
    from backend.app.rag.embedder import EMBEDDING_DIMENSIONS

    pc = Pinecone(api_key=settings.pinecone_api_key)
    name = settings.pinecone_index_name

    existing = pc.list_indexes().names()
    if name in existing:
        print(f"  → Deleting index '{name}'...")
        pc.delete_index(name)
        # Brief pause — Pinecone is eventually consistent; the old index must
        # fully deregister before create succeeds.
        time.sleep(3)
        print(f"  → Deleted.")

    print(f"  → Creating index '{name}' with metric='dotproduct'...")
    pc.create_index(
        name=name,
        dimension=EMBEDDING_DIMENSIONS,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"  ✓ Index '{name}' ready (metric=dotproduct).")

    # Invalidate the cached index handle so get_index() returns the new one.
    import backend.app.rag.retriever as _retriever
    _retriever._pinecone_index = None


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


def _fit_bm25_encoder(chunks: list[Chunk]) -> None:
    """
    Fit BM25Encoder on all corpus chunk texts and save to disk.

    Fitting on the full corpus (rather than using the pre-built MS-MARCO
    default encoder) gives domain-appropriate IDF weights for pregnancy
    nutrition terms ('elemental iron', 'folic acid', 'amla', etc.) that
    appear frequently in our guidelines but rarely in general web text.

    Must be called AFTER all PDFs are chunked so IDF is computed over the
    complete vocabulary. The saved encoder is loaded as a singleton at query
    time by _get_bm25_encoder() in retriever.py.

    Also invalidates the in-memory BM25 singleton so the next retrieval call
    reloads the freshly-fitted encoder rather than any stale cached version.
    """
    from pinecone_text.sparse import BM25Encoder

    texts = [c.text for c in chunks]
    print(f"\nFitting BM25 encoder on {len(texts)} corpus chunks...")
    encoder = BM25Encoder()
    encoder.fit(texts)

    encoder_path = PROJECT_ROOT / settings.bm25_encoder_path
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.dump(str(encoder_path))
    print(f"  ✓ BM25 encoder saved → {settings.bm25_encoder_path}")

    # Invalidate in-memory singleton so the next query uses the fresh encoder.
    import backend.app.rag.retriever as _retriever
    _retriever._bm25_encoder = None


def _parse_and_chunk(
    file_name: str, idx: int, total: int,
) -> tuple[list[Chunk] | None, float, str]:
    """
    Parse one PDF with LlamaParse and apply the semantic chunker.

    Returns (chunks, elapsed_seconds, error_message). Returns (None, ..., ...)
    on failure so the caller can skip embed/upsert for this file without
    aborting the rest of the ingest.
    """
    t0 = time.perf_counter()
    print(f"\n[{idx}/{total}] Parsing and chunking {file_name}...")
    try:
        chunks = chunk_pdf(file_name)
        elapsed = time.perf_counter() - t0
        if not chunks:
            return None, elapsed, "chunker returned 0 chunks (PDF empty or LlamaParse silently failed?)"
        print(f"  → {len(chunks)} chunks ({elapsed:.1f}s)")
        return chunks, elapsed, ""
    except Exception as e:
        elapsed = time.perf_counter() - t0
        msg = f"{type(e).__name__}: {e}"
        print(f"  ✗ FAILED after {elapsed:.1f}s — {msg}")
        return None, elapsed, msg


def _embed_and_upsert(
    chunks: list[Chunk], file_name: str, idx: int, total: int,
) -> tuple[bool, int, float, str]:
    """
    Embed chunks and upsert dense + sparse vectors to Pinecone.

    Assumes the BM25 encoder is already fitted and saved (either from this
    run's _fit_bm25_encoder call, or from a previous full ingest). upsert_chunks
    loads the encoder lazily from disk the first time it's called.

    Returns (succeeded, chunk_count, elapsed_seconds, error_message).
    """
    t0 = time.perf_counter()
    print(f"[{idx}/{total}] Embedding and upserting {file_name}...")
    try:
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        upsert_chunks(chunks, embeddings)
        elapsed = time.perf_counter() - t0
        print(f"  ✓ [{idx}/{total}] {file_name} committed ({elapsed:.1f}s)")
        return True, len(chunks), elapsed, ""
    except Exception as e:
        elapsed = time.perf_counter() - t0
        msg = f"{type(e).__name__}: {e}"
        print(f"  ✗ [{idx}/{total}] {file_name} FAILED after {elapsed:.1f}s — {msg}")
        return False, 0, elapsed, msg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse PDFs with LlamaParse, embed, and upsert to Pinecone with hybrid vectors."
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help=(
            "Delete the existing Pinecone index and recreate it with "
            "metric='dotproduct' (required for hybrid sparse+dense retrieval). "
            "Then re-ingest all files and re-fit the BM25 encoder. "
            "Use for first-time hybrid setup or when the index metric is wrong."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete every existing vector in the Pinecone index before "
            "upserting. Use when the parsing / chunking pipeline has "
            "changed and stale chunks would otherwise pollute retrieval. "
            "Keeps the index schema (metric) unchanged."
        ),
    )
    parser.add_argument(
        "--only",
        help=(
            "Process only files whose name contains this substring "
            "(case-insensitive). Reuses the existing BM25 encoder — "
            "re-fitting is skipped to keep IDF weights stable. Example: "
            "--only mohfw → reingest just anc_guidelines_india_mohfw.pdf."
        ),
    )
    args = parser.parse_args()

    # Guard against flag combinations that would produce a broken state.
    if args.recreate_index and args.only:
        print("✗ --recreate-index + --only would recreate the full index "
              "then only re-ingest one source, leaving the rest empty. Refusing.")
        return 2
    if args.reset and args.only:
        print("✗ --reset + --only would wipe everything and only re-ingest "
              f"'{args.only}'. That's almost certainly a mistake. Refusing.")
        return 2

    print("=== Poshan Saathi — PDF Ingestion ===")

    if args.recreate_index:
        print()
        print("Recreating Pinecone index with metric='dotproduct'...")
        _recreate_index()
    elif args.reset:
        print()
        _reset_index()

    # Resolve which files to process.
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

    overall_t0 = time.perf_counter()

    # ── Pass 1: parse + chunk all target files ────────────────────────────────
    print("\nPass 1 of 2 — Parsing and chunking PDFs...")
    chunks_by_file: dict[str, list[Chunk]] = {}
    parse_results: list[tuple[str, bool, int, float, str]] = []

    for i, file_name in enumerate(files, start=1):
        chunks, elapsed, err = _parse_and_chunk(file_name, i, len(files))
        ok = chunks is not None
        n = len(chunks) if chunks else 0
        parse_results.append((file_name, ok, n, elapsed, err))
        if ok:
            chunks_by_file[file_name] = chunks

    # ── BM25 fitting (full ingest only) ───────────────────────────────────────
    if not args.only:
        # Full ingest: re-fit on all newly parsed chunks so IDF reflects the
        # complete corpus. Chunks from files that failed parsing are excluded —
        # those files need manual inspection and re-run with --only.
        all_chunks_flat = [c for cs in chunks_by_file.values() for c in cs]
        if all_chunks_flat:
            _fit_bm25_encoder(all_chunks_flat)
        else:
            print("\n⚠ No chunks produced — skipping BM25 fitting.")
            return 1
    else:
        # --only mode: validate the existing encoder is present before trying
        # to upsert (upsert_chunks will fail loud if it's missing, but better
        # to surface this before the embed API calls).
        encoder_path = PROJECT_ROOT / settings.bm25_encoder_path
        if not encoder_path.exists():
            print(f"\n✗ BM25 encoder not found at '{settings.bm25_encoder_path}'.")
            print("  Run a full re-ingest first:")
            print("      python -m scripts.ingest --recreate-index")
            return 2

    # ── Pass 2: embed + upsert per file ───────────────────────────────────────
    print("\nPass 2 of 2 — Embedding and upserting to Pinecone...")
    embed_results: list[tuple[str, bool, int, float, str]] = []

    for i, file_name in enumerate(files, start=1):
        if file_name not in chunks_by_file:
            embed_results.append((file_name, False, 0, 0.0, "skipped (parse failed)"))
            continue
        ok, n, elapsed, err = _embed_and_upsert(
            chunks_by_file[file_name], file_name, i, len(files),
        )
        embed_results.append((file_name, ok, n, elapsed, err))

    overall_elapsed = time.perf_counter() - overall_t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  Ingestion summary")
    print("=" * 72)

    # Combine parse + embed results for a single-row-per-file display.
    parse_map = {f: (ok, n, e, err) for f, ok, n, e, err in parse_results}
    embed_map = {f: (ok, n, e, err) for f, ok, n, e, err in embed_results}

    total_chunks = 0
    n_ok = 0
    n_fail = 0
    failed_files: list[str] = []
    for file_name in files:
        p_ok, p_n, p_elapsed, p_err = parse_map.get(file_name, (False, 0, 0.0, ""))
        e_ok, e_n, e_elapsed, e_err = embed_map.get(file_name, (False, 0, 0.0, ""))
        ok = p_ok and e_ok
        err = p_err or e_err
        n = e_n if e_ok else p_n
        elapsed = p_elapsed + e_elapsed
        mark = "✓" if ok else "✗"
        line = f"  {mark} {file_name:<40} {n:>5} chunks  ({elapsed:.1f}s)"
        if not ok:
            line += f"  ← {err}"
        print(line)
        if ok:
            total_chunks += n
            n_ok += 1
        else:
            n_fail += 1
            failed_files.append(file_name)

    print(f"\n  {n_ok}/{len(files)} files succeeded  |  "
          f"{total_chunks} total chunks  |  {overall_elapsed:.1f}s wall-clock")

    if n_fail > 0:
        print(f"\n  ⚠ {n_fail} file(s) failed. To retry just those, run:")
        for file_name in failed_files:
            keyword = file_name.split("_")[-1]
            print(f"      python -m scripts.ingest --only {keyword}")

    # Drain Langfuse buffer before exit.
    flush_traces()

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
