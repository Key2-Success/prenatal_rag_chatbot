"""
chunker.py — PDF loading and semantic text chunking.

Upgrade from RecursiveCharacterTextSplitter:
  - The old splitter cut every 600 characters, regardless of topic.
    A paragraph discussing iron dosage and a sentence about folic acid
    could land in the same chunk, diluting the cross-encoder score for
    any query about either topic.
  - SemanticChunker embeds each sentence, computes cosine distance between
    consecutive sentence groups, and only cuts where the distance exceeds a
    percentile threshold — so a chunk ends when the topic demonstrably shifts,
    not when a character counter trips.

Design decisions:
  - One splitter instance per chunk_pdf() call, shared across all pages.
    SemanticChunker embeds sentences via the OpenAI API on every split_text()
    call; building a new instance per page would re-initialise nothing but is
    still cleaner to share.
  - Per-page chunking is retained. The PDF page is the natural unit that
    preserves page_number metadata; chunking across page boundaries would lose
    that provenance.
  - breakpoint_threshold_type="percentile", amount=95 by default.
    This means: only cut where the similarity drop is in the top 5% of all
    observed drops on that page — i.e., only on genuine topic shifts.
    Tunable at runtime: SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT=90 python -m scripts.ingest
  - api_key passed directly to OpenAIEmbeddings — no os.environ side effect.
    langchain_openai.OpenAIEmbeddings accepts api_key= as a constructor param
    (alias for openai_api_key field), so we pass settings.openai_api_key
    directly without needing to mirror it into the environment.
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
    consistency — semantic proximity at ingest time matches semantic proximity
    at retrieval time. The api_key is passed directly rather than relying on
    os.environ so this module has no env-mutation side effects.
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


def chunk_all_pdfs() -> list[Chunk]:
    """Chunk every PDF declared in sources.json."""
    all_chunks: list[Chunk] = []
    for file_name in sources_by_filename():
        print(f"Chunking {file_name}...")
        chunks = chunk_pdf(file_name)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks
