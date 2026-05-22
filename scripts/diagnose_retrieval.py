"""
diagnose_retrieval.py — One-shot diagnostic for "everything is returning no_results."

Walks a single query through every stage of the retrieval pipeline and prints
what comes back, so we can see exactly where things break:

  1. Pinecone index health     — vector count, sources represented
  2. Query embedding            — sanity-check the embedder is alive
  3. Per-source recall          — raw cosine scores from each org
  4. Threshold filter           — what survives similarity_threshold
  5. Reranker output            — what comes out of bge-reranker-v2-m3

Run:
    python -m scripts.diagnose_retrieval
    python -m scripts.diagnose_retrieval --query "how much paneer should i eat"
"""

import argparse
import sys

from backend.app.config import settings
from backend.app.rag.embedder import embed_query
from backend.app.rag.retriever import (
    _get_reranker, _query_source, get_index,
)
from backend.app.sources import priority_order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="how much iron should i take during pregnancy",
        help="Query to trace through the pipeline (default: a canonical iron-intake question)",
    )
    args = parser.parse_args()
    query = args.query

    print("=" * 72)
    print("  Retrieval pipeline diagnostic")
    print("=" * 72)
    print()

    # --- Stage 0: Index health ---
    print("── Stage 0: Pinecone index health ──")
    index = get_index()
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0)
    print(f"  Index name:           {settings.pinecone_index_name}")
    print(f"  Total vector count:   {total}")
    if total == 0:
        print("  ✗ INDEX IS EMPTY — ingestion didn't populate vectors.")
        print("    Re-run: python -m scripts.ingest --reset")
        return 1
    namespaces = stats.get("namespaces", {})
    for ns_name, ns_stats in namespaces.items():
        label = ns_name or "<default>"
        print(f"  Namespace '{label}': {ns_stats.get('vector_count', '?')} vectors")
    print()

    # --- Stage 0b: Per-source population check ---
    # Probe each source with a top-k=1 query just to confirm at least one
    # vector with that org_display_name exists. This catches the case where
    # ingestion only got partway through and one source is missing entirely.
    print("── Stage 0b: Sources represented in index ──")
    dummy_vec = embed_query("anything")
    for source_name in priority_order():
        try:
            result = index.query(
                vector=dummy_vec,
                top_k=1,
                filter={"org_display_name": {"$eq": source_name}},
                include_metadata=False,
            )
            n = len(result.get("matches", []))
            mark = "✓" if n > 0 else "✗"
            print(f"  {mark} {source_name:<12} {n} vector(s) found")
        except Exception as e:
            print(f"  ✗ {source_name:<12} query failed: {e}")
    print()

    # --- Stage 1: Query embedding ---
    print(f"── Stage 1: Query embedding ──")
    print(f"  Query: {query!r}")
    embedding = embed_query(query)
    print(f"  Embedding dimensions: {len(embedding)}")
    print(f"  First 5 values: {[round(v, 3) for v in embedding[:5]]}")
    print()

    # --- Stage 2: Per-source recall (raw cosine scores) ---
    # Bypass the threshold filter so we can SEE what cosine scores Pinecone
    # is actually returning — even if they're all below the threshold.
    print("── Stage 2: Per-source recall (cosine scores from Pinecone) ──")
    print(f"  similarity_threshold = {settings.similarity_threshold}")
    print(f"  reranker_candidate_k = {settings.reranker_candidate_k}")
    print()

    all_above_threshold = 0
    all_below_threshold = 0
    for source_name in priority_order():
        print(f"  [{source_name}]")
        raw_results = index.query(
            vector=embedding,
            top_k=settings.reranker_candidate_k,
            filter={"org_display_name": {"$eq": source_name}},
            include_metadata=True,
        )
        matches = raw_results.get("matches", [])
        if not matches:
            print(f"    (no matches returned at all)")
            continue
        for m in matches:
            score = m["score"]
            survives = score >= settings.similarity_threshold
            mark = "✓" if survives else "✗"
            preview = m["metadata"]["text"][:80].replace("\n", " ")
            if survives:
                all_above_threshold += 1
            else:
                all_below_threshold += 1
            print(f"    {mark} score={score:.3f}  p.{m['metadata'].get('page_number', '?')}  {preview}…")
        print()

    print(f"  Across all sources: {all_above_threshold} pass threshold, "
          f"{all_below_threshold} below threshold")
    print()

    if all_above_threshold == 0:
        print("  ✗ DIAGNOSIS: every chunk's cosine score is below "
              f"similarity_threshold={settings.similarity_threshold}.")
        print("    Either (a) the threshold is too high for the new LlamaParse "
              "chunks, or (b) the chunks genuinely don't match this query.")
        print("    Suggested next steps:")
        print("      - Try the same diagnostic with --query on a known-easy "
              "topic to confirm the index has good chunks.")
        print("      - If scores are consistently 0.1-0.3 even for canonical "
              "queries, lower SIMILARITY_THRESHOLD (try 0.15-0.2).")
        return 0

    # --- Stage 3: Reranker (only if anything survived threshold) ---
    print("── Stage 3: Reranker (bge-reranker-v2-m3, self-hosted) ──")
    import numpy as np
    reranker = _get_reranker()
    # Gather all candidates that survived threshold across all sources
    candidates = []
    for source_name in priority_order():
        chunks = _query_source(source_name, embedding)
        candidates.extend(chunks)
    if not candidates:
        print("  (no candidates after dedup — see stage 2 above)")
        return 0
    pairs = [(query, c.text) for c in candidates]
    raw = reranker.predict(pairs)
    scores = 1.0 / (1.0 + np.exp(-np.asarray(raw)))
    order = np.argsort(-scores)[: settings.top_k]
    for rank, idx in enumerate(order, start=1):
        c = candidates[idx]
        preview = c.text[:80].replace("\n", " ")
        print(f"  #{rank}  rerank={scores[idx]:.3f}  {c.org_display_name} p.{c.page_number}  {preview}…")
    print()
    print("  ✓ Reranker working. Top results above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
