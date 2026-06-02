"""
chunker.py — PDF loading (via pypdf) and semantic text chunking.

Two-stage ingestion:

  Stage 1 — Parse (pypdf):
    Basic text extraction. pypdf reads the PDF's text stream in roughly
    reading order and returns a string per page. Tables get flattened to
    space-separated text, multi-column layouts get interleaved row-by-row,
    and images are dropped (the captions stay).

    Reverted from LlamaParse (May 2026) after A/B testing: LlamaParse
    cleanly extracted tables as markdown but the resulting clean-prose
    chunks shifted the answer-LLM's register toward bureaucratic
    "the guidelines recommend..." prose, broke the diet-filter rule
    (vegetarian users got non-veg sources because cleaner chunks bundled
    all foods together), and dropped RAGAS answer_relevancy 0.868 → 0.715.
    pypdf's noisier extraction forces the LLM to synthesise across chunks,
    which (counterintuitively) produces more fluent, more personalised
    user-facing answers in this corpus. Engineering trumps tool collection.

  Stage 2 — Chunk (SemanticChunker):
    Splits each page's text into chunks at the largest topic shifts
    (top 5% of inter-sentence embedding-distance jumps). Same chunker
    we used with LlamaParse; only the parser changed back.

Design decisions:
  - One splitter instance per chunk_pdf() call, shared across all pages.
    SemanticChunker embeds sentences via the OpenAI API on every
    split_text() call; sharing avoids repeated client init.
  - Per-page chunking is retained. The PDF page is the natural unit that
    preserves page_number metadata; chunking across page boundaries would
    lose that provenance.
  - breakpoint_threshold_type="percentile", amount=95 by default.
    Only cut where the similarity drop is in the top 5% of all observed
    drops on that page — i.e., only on genuine topic shifts. Tunable at
    runtime via SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT.
  - api_key passed directly to OpenAIEmbeddings — no os.environ side
    effects. langchain_openai.OpenAIEmbeddings accepts api_key= as a
    constructor param, so we pass settings.openai_api_key directly.

Token audit:
  After chunking, _report_token_stats prints min/median/p95/max chunk
  lengths in tokens — surfaces silent-truncation risk early if chunks
  ever creep above the 8191/8192 caps for embeddings/reranker.
"""

from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from pypdf import PdfReader

from backend.app.config import DATA_DIR, settings
from backend.app.sources import Source, sources_by_filename

# Drop chunks shorter than this — they're almost always page numbers,
# headers, or extraction noise that hurt retrieval signal.
MIN_CHUNK_CHARS = 50


class Chunk(BaseModel):
    """One unit of text ready for embedding + Pinecone upsert."""
    text: str
    source_file: str
    org_display_name: str
    doc_title: str
    doc_reference_order: int
    year_published: int
    page_number: int  # page where this chunk begins


class _Page(BaseModel):
    """Internal: extracted text for a single PDF page."""
    text: str
    page_number: int


def _extract_pages(pdf_path: Path) -> list[_Page]:
    """Extract non-empty pages from a PDF, preserving 1-based page numbers."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(_Page(text=text, page_number=i))
    return pages


def _build_splitter() -> SemanticChunker:
    """
    Build a SemanticChunker backed by the project's embedding model.

    Uses the same model (text-embedding-3-small) as the query embedder for
    consistency — semantic proximity at ingest time matches semantic
    proximity at retrieval time. The api_key is passed directly rather
    than relying on os.environ so this module has no env-mutation side
    effects.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
    )
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=settings.semantic_breakpoint_threshold_type,
        breakpoint_threshold_amount=settings.semantic_breakpoint_threshold_amount,
    )


def _chunks_for_page(page: _Page, source: Source, splitter: SemanticChunker) -> list[Chunk]:
    """Split one page's text into Chunks stamped with source metadata."""
    chunks: list[Chunk] = []
    for raw in splitter.split_text(page.text):
        text = raw.strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        chunks.append(Chunk(
            text=text,
            source_file=source.file_name,
            org_display_name=source.org_display_name,
            doc_title=source.doc_title,
            doc_reference_order=source.doc_reference_order,
            year_published=source.doc_year_published,
            page_number=page.page_number,
        ))
    return chunks


def chunk_pdf(file_name: str) -> list[Chunk]:
    """Chunk a single PDF declared in sources.json."""
    source = sources_by_filename()[file_name]
    pdf_path = DATA_DIR / f"{file_name}.pdf"
    splitter = _build_splitter()

    chunks: list[Chunk] = []
    for page in _extract_pages(pdf_path):
        chunks.extend(_chunks_for_page(page, source, splitter))
    return chunks


def _report_token_stats(chunks: list[Chunk]) -> None:
    """
    Print min / median / p95 / max token counts across all chunks.

    Why we measure: SemanticChunker produces variable-length chunks, which
    is good for semantic coherence but exposes us to silent-truncation
    bugs. The two downstream caps that matter:
      - text-embedding-3-small accepts up to 8191 tokens; anything longer
        is truncated FROM THE END (you lose the tail of the chunk
        silently).
      - bge-reranker-v2-m3 accepts up to 8192 tokens per (query + doc)
        pair; same truncation behaviour.

    What "safe" looks like for this corpus: max should comfortably sit
    under ~2000 tokens (typical PDF pages are 300-1200 tokens, and we
    chunk WITHIN pages). If max approaches 8000, we've found a giant
    uninterrupted block (probably a long table or unbreakable
    monolithic paragraph) and need to either lower the SemanticChunker
    breakpoint percentile or add a hard-cap splitter as a backstop.

    Uses tiktoken with the cl100k_base encoding — same tokeniser used by
    text-embedding-3-small. Approximate for the reranker (which uses
    sentencepiece) but accurate to within ~10-20%, plenty for a sanity
    audit.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    lengths = sorted(len(enc.encode(c.text)) for c in chunks)
    n = len(lengths)
    p95 = lengths[int(n * 0.95)] if n else 0
    median = lengths[n // 2] if n else 0
    longest = lengths[-1] if n else 0

    print()
    print("Token-length audit (chunk size in tokens):")
    print(f"  min      = {lengths[0] if n else 0}")
    print(f"  median   = {median}")
    print(f"  p95      = {p95}")
    print(f"  max      = {longest}")
    print(f"  caps     = 8191 (embedding) / 8192 (reranker)")
    if longest > 8000:
        print(f"  ⚠ WARNING: longest chunk ({longest} tokens) is at or above "
              f"the embedding/reranker cap. The tail will be truncated silently.")
    elif longest > 4000:
        print(f"  ⚠ NOTE: longest chunk ({longest} tokens) is unusually long "
              f"for this corpus — investigate whether a single chunk should "
              f"be split.")


def chunk_all_pdfs() -> list[Chunk]:
    """Chunk every PDF declared in sources.json."""
    all_chunks: list[Chunk] = []
    for file_name in sources_by_filename():
        print(f"Chunking {file_name}...")
        chunks = chunk_pdf(file_name)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
    _report_token_stats(all_chunks)
    return all_chunks
