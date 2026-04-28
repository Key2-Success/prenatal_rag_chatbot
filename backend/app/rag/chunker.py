"""
chunker.py — PDF loading and text chunking.

Upgrade from the original notebook:
  - Notebook: PDFReader gave one document per PAGE — pages are arbitrary
    splits, a topic can start mid-page and end on the next.
  - Here: RecursiveCharacterTextSplitter respects natural text boundaries
    (paragraphs → sentences → words) before falling back to character
    splits. chunk_overlap ensures context isn't lost at boundaries.

Outputs are `Chunk` Pydantic models, not dicts — every downstream consumer
(embedder, retriever upsert) gets typed access instead of `c["text"]`.
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader

from backend.app.config import DATA_DIR
from backend.app.sources import Source, sources_by_filename

# Tuned for ANC guidelines. chunk_size/overlap are CHARACTERS
# (RecursiveCharacterTextSplitter's default length_function is len()).
# ~600 chars ≈ 100–120 English words — enough for a full recommendation
# with context, while staying well under the embedding model's token limit.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Drop chunks shorter than this — they're almost always page numbers,
# headers, or extraction noise that hurt retrieval signal.
MIN_CHUNK_CHARS = 50

# Splitter separators in priority order: try the largest natural boundary
# first, fall back to single chars only if nothing else fits.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


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


def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )


def _chunks_for_page(page: _Page, source: Source, splitter) -> list[Chunk]:
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
