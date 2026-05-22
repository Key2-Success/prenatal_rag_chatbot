"""
diagnose_index.py — Why only WHO is in the index. Four-way audit.

Compares:
  1. PDFs declared in sources.json       (what we EXPECT to ingest)
  2. PDFs actually on disk in data/      (what's available to ingest)
  3. Org names found in Pinecone vectors (what actually got ingested)
  4. Chunk counts per source             (was it complete or partial?)

Run:
    .venv/bin/python -m scripts.diagnose_index
"""

from collections import Counter

from backend.app.config import DATA_DIR, settings
from backend.app.rag.embedder import embed_query
from backend.app.rag.retriever import get_index
from backend.app.sources import priority_order, sources_by_filename


def main() -> int:
    print("=" * 72)
    print("  Index population diagnostic")
    print("=" * 72)

    # 1. What sources.json declares
    print("\n── 1. sources.json declarations ──")
    sources = sources_by_filename()
    for file_name, source in sources.items():
        pdf_path = DATA_DIR / f"{file_name}.pdf"
        exists = "✓" if pdf_path.exists() else "✗ MISSING"
        size_mb = (pdf_path.stat().st_size / 1024 / 1024) if pdf_path.exists() else 0
        print(f"  {exists}  {source.org_display_name:<10} {file_name}.pdf "
              f"({size_mb:.1f} MB)  org_display_name='{source.org_display_name}'")

    # 2. What's actually in data/
    print("\n── 2. PDFs on disk in data/ ──")
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"  ✗ NO PDFs found in {DATA_DIR}")
    for pdf in pdfs:
        declared = pdf.stem in sources
        mark = "✓" if declared else "?"
        size_mb = pdf.stat().st_size / 1024 / 1024
        print(f"  {mark}  {pdf.name} ({size_mb:.1f} MB)  "
              f"{'declared in sources.json' if declared else 'NOT in sources.json'}")

    # 3. What's actually in Pinecone — sample 100 vectors and tally org names
    print("\n── 3. org_display_name values actually in the Pinecone index ──")
    index = get_index()
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0)
    print(f"  Total vectors in index: {total}")
    if total == 0:
        print("  ✗ Index is empty.")
        return 1

    # Use a random-ish vector to fetch a broad sample
    sample_vec = embed_query("nutrition pregnancy")
    sample = index.query(
        vector=sample_vec,
        top_k=min(100, total),  # ask for up to 100 to see the distribution
        include_metadata=True,
    )
    matches = sample.get("matches", [])
    org_counts = Counter()
    source_file_counts = Counter()
    for m in matches:
        meta = m.get("metadata", {})
        org_counts[meta.get("org_display_name", "<missing>")] += 1
        source_file_counts[meta.get("source_file", "<missing>")] += 1
    print(f"  Sampled {len(matches)} vectors. Org distribution:")
    for org, count in org_counts.most_common():
        pct = count / len(matches) * 100 if matches else 0
        print(f"    {org:<20} {count:>3} ({pct:.0f}%)")

    print(f"\n  source_file distribution:")
    for sf, count in source_file_counts.most_common():
        pct = count / len(matches) * 100 if matches else 0
        print(f"    {sf:<40} {count:>3} ({pct:.0f}%)")

    # 4. Per-expected-source presence check (no filter, just count)
    print("\n── 4. Per-source vector counts (exact filter on org_display_name) ──")
    for source_name in priority_order():
        # Use a high top_k to actually count vectors with that org
        r = index.query(
            vector=sample_vec,
            top_k=1000,
            filter={"org_display_name": {"$eq": source_name}},
            include_metadata=False,
        )
        n = len(r.get("matches", []))
        mark = "✓" if n > 0 else "✗"
        expected = "expected" if any(s.org_display_name == source_name for s in sources.values()) else "not declared"
        print(f"  {mark}  {source_name:<12} {n:>4} vectors  ({expected})")

    print("\n── Diagnosis ──")
    declared_orgs = {s.org_display_name for s in sources.values()}
    found_orgs = set(org_counts.keys())
    missing = declared_orgs - found_orgs
    extra = found_orgs - declared_orgs

    if missing:
        print(f"  ✗ Sources declared but NOT in index: {sorted(missing)}")
        print(f"    Possible causes:")
        print(f"      a) PDF missing from data/ — check section 1 above for '✗ MISSING'")
        print(f"      b) LlamaParse failed on those PDFs during the last ingest")
        print(f"         (check the output of `python -m scripts.ingest --reset`)")
        print(f"      c) Ingestion was interrupted before processing those files")
    if extra:
        print(f"  ⚠ Extra orgs in index not declared in sources.json: {sorted(extra)}")
    if not missing and not extra:
        print(f"  ✓ All declared sources are represented in the index.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
