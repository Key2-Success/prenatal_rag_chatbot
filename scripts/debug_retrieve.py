"""
debug_retrieve.py — Inspect what Pinecone returns for a query.

Diagnostic tool, not production. Use cases:
  - A relevant query is triggering fallback → find out why
  - Picking a similarity_threshold → see score distribution
  - Verifying pypdf didn't mangle a table or section

Usage:
    python -m scripts.debug_retrieve "How much iron should I be eating?"
    python -m scripts.debug_retrieve "..." --no-augment
    python -m scripts.debug_retrieve "..." --diet Non-Vegetarian --top-k 15
    python -m scripts.debug_retrieve "..." --no-audit      # skip PDF re-read
"""

import argparse
from functools import lru_cache

from backend.app.config import settings
from backend.app.rag.chunker import Chunk, chunk_all_pdfs
from backend.app.rag.embedder import embed_query
from backend.app.rag.retriever import get_index
from backend.app.sources import priority_order

DEBUG_TOP_K = 10
THRESHOLD_SWEEP = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)

# Words to strip before picking the keyword to audit. Not rigorous — just
# enough to pick a content word for the PDF text search.
STOPWORDS = frozenset({
    "how", "much", "should", "i", "be", "am", "are", "is", "was", "were",
    "the", "a", "an", "what", "when", "where", "why", "who", "which",
    "can", "do", "does", "did", "have", "has", "had", "take", "taking",
    "eat", "eating", "to", "for", "of", "in", "on", "at", "by", "my",
    "you", "your", "me", "it", "this", "that", "get", "getting",
})


def _query_source(source_name: str, query_text: str, top_k: int) -> list[dict]:
    """Query Pinecone for one source, no threshold — return raw matches."""
    results = get_index().query(
        vector=embed_query(query_text),
        top_k=top_k,
        filter={"org_display_name": {"$eq": source_name}},
        include_metadata=True,
    )
    return list(results["matches"])


def _extract_keyword(query: str) -> str | None:
    """Pick the longest non-stopword from the query as the keyword to audit."""
    words = (w.strip("?.,!;:\"'").lower() for w in query.split())
    content = [w for w in words if w and w not in STOPWORDS]
    return max(content, key=len) if content else None


@lru_cache(maxsize=1)
def _all_chunks() -> list[Chunk]:
    """Re-chunk the PDFs in-process so we can grep them locally."""
    print("(Re-chunking PDFs for keyword audit — first run only...)")
    return chunk_all_pdfs()


def _audit_keyword(keyword: str) -> None:
    """Count chunks per source that contain the keyword (case-insensitive)."""
    needle = keyword.lower()
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for c in _all_chunks():
        if needle in c.text.lower():
            counts[c.org_display_name] = counts.get(c.org_display_name, 0) + 1
            examples.setdefault(
                c.org_display_name,
                f"p.{c.page_number}: {c.text[:120]}",
            )

    if not counts:
        print(f"  ⚠ NO chunks mention '{keyword}' in any PDF.")
        print("    → pypdf may have failed to extract that content, or the PDFs")
        print("      don't cover this topic. Try a different keyword or inspect the PDFs.")
        return

    for source in priority_order():
        n = counts.get(source, 0)
        print(f"  {source}: {n} chunks mention '{keyword}'")
        if source in examples:
            print(f"    example → {examples[source]}")


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_per_source_matches(all_results: dict[str, list[dict]], top_k: int) -> None:
    _print_header(f"TOP {top_k} MATCHES PER SOURCE (no threshold)")
    for source_name, matches in all_results.items():
        print(f"\n[{source_name}]")
        if not matches:
            print("  (no results)")
            continue
        for m in matches:
            preview = m["metadata"]["text"][:90].replace("\n", " ")
            print(f"  {m['score']:.3f}  p.{m['metadata']['page_number']:<3}  {preview}...")


def _print_threshold_sweep(all_results: dict[str, list[dict]]) -> None:
    _print_header("THRESHOLD SWEEP — chunks passing each threshold per source")
    sources = priority_order()
    header = "  threshold   " + "  ".join(f"{s:<8}" for s in sources)
    print(header)
    print("  " + "-" * (len(header) - 2))
    current = settings.similarity_threshold
    for t in THRESHOLD_SWEEP:
        counts = [sum(1 for m in all_results[s] if m["score"] >= t) for s in sources]
        marker = "  ← current threshold" if abs(t - current) < 1e-6 else ""
        print(f"  {t:<11} " + "  ".join(f"{c:<8}" for c in counts) + marker)


def run(query: str, use_augment: bool, diet: str, top_k: int, keyword_audit: bool) -> None:
    # Mirror the augmentation pipeline.py uses, so debug results match prod.
    search_text = f"{query} [Diet: {diet}]" if use_augment else query

    _print_header("QUERY")
    print(f"  Raw:       {query!r}")
    print(f"  Augmented: {search_text!r}")
    print(f"  Sending:   {'augmented' if use_augment else 'raw'}")

    all_results = {s: _query_source(s, search_text, top_k) for s in priority_order()}
    _print_per_source_matches(all_results, top_k)
    _print_threshold_sweep(all_results)

    if keyword_audit:
        kw = _extract_keyword(query)
        if kw:
            _print_header(f"KEYWORD AUDIT — PDFs containing '{kw}' (case-insensitive)")
            _audit_keyword(kw)
        else:
            _print_header("KEYWORD AUDIT — skipped (no content word found in query)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose a RAG retrieval by inspecting per-source scores and PDF content."
    )
    parser.add_argument("query", help="The query to debug (in quotes)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Don't append [Diet: X] — test the raw query only")
    parser.add_argument("--diet", default="Vegetarian",
                        help="Diet to append when augmenting (default: Vegetarian)")
    parser.add_argument("--top-k", type=int, default=DEBUG_TOP_K,
                        help=f"Candidates per source (default: {DEBUG_TOP_K})")
    parser.add_argument("--no-audit", action="store_true",
                        help="Skip PDF keyword audit (faster, no PDF re-read)")
    args = parser.parse_args()

    run(
        query=args.query,
        use_augment=not args.no_augment,
        diet=args.diet,
        top_k=args.top_k,
        keyword_audit=not args.no_audit,
    )


if __name__ == "__main__":
    main()
