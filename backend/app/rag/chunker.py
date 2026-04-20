"""
chunker.py — PDF loading and text chunking.

Upgrade from the notebook:
  - Notebook: PDFReader gave one document per PAGE (80 docs for an 80-page PDF).
    Pages are arbitrary splits — a topic can start mid-page and end on the next.
  - Here: RecursiveCharacterTextSplitter respects natural text boundaries
    (paragraphs → sentences → words) before falling back to character splits.
    chunk_overlap ensures context isn't lost at boundaries.
"""

from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.config import settings
from backend.app.sources import sources_by_filename

# Tuned for ANC guidelines. chunk_size/overlap are measured in CHARACTERS
# (RecursiveCharacterTextSplitter's default length_function is len()).
# ~600 chars ≈ 100–120 English words — enough for a full recommendation
# with context, while staying well under the embedding model's token limit.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def extract_text_with_pages(pdf_path: Path) -> list[dict]:
    """
    Extracts text page-by-page from a PDF.
    Returns a list of {text, page_number} dicts.
    We preserve page_number so citations still point to the right page.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "page_number": i})
    return pages


def chunk_pdf(file_name: str) -> list[dict]:
    """
    Loads a PDF, splits it into overlapping chunks, and stamps each chunk
    with full source metadata from sources.json.

    Returns a list of chunk dicts ready for embedding + Pinecone upsert:
    {
        "text": str,
        "source_file": str,
        "org_display_name": str,
        "doc_title": str,
        "doc_reference_order": int,
        "year_published": int,
        "page_number": int,   # page where this chunk begins
    }
    """
    source_meta = sources_by_filename()[file_name]
    pdf_path = settings.data_dir / f"{file_name}.pdf"
    pages = extract_text_with_pages(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        for chunk_text in splitter.split_text(page["text"]):
            cleaned = chunk_text.strip()
            if len(cleaned) < 50:  # skip noise (headers, page numbers)
                continue
            chunks.append({
                "text": cleaned,
                "source_file": file_name,
                "org_display_name": source_meta["org_display_name"],
                "doc_title": source_meta["doc_title"],
                "doc_reference_order": source_meta["doc_reference_order"],
                "year_published": source_meta["doc_year_published"],
                "page_number": page["page_number"],
            })

    return chunks


def chunk_all_pdfs() -> list[dict]:
    """Chunks every PDF declared in sources.json."""
    all_chunks = []
    for file_name in sources_by_filename():
        print(f"Chunking {file_name}...")
        chunks = chunk_pdf(file_name)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks
