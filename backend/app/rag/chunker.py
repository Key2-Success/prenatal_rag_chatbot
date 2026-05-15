"""
chunker.py — PDF loading (via LlamaParse) and semantic text chunking.

Two-stage ingestion:

  Stage 1 — Parse (LlamaParse, hosted):
    pypdf was the previous parser. It flattens tables into space-separated
    soup, merges multi-column layouts row-by-row across columns, and drops
    figures entirely. For a corpus that has nutrition intake tables, dietary
    schedules, and multi-column anaemia guidelines, that's a data-quality
    defect at the parse step that no downstream chunker can recover from.

    LlamaParse is the LlamaIndex team's hosted parser, currently SOTA for
    structured-PDF extraction in production RAG. Tables come out as proper
    markdown tables (| col | col | rows | preserved); multi-column layouts
    are linearised correctly; figures and equations get descriptive captions
    instead of being silently dropped. result_type="markdown" gives us text
    the embedding model and the LLM both read more cleanly than pypdf's raw
    extraction.

    Cost: 1000 pages/day free tier; ~$3/1000 pages paid. Our corpus is well
    under that.

  Stage 2 — Chunk (SemanticChunker):
    Unchanged. Splits each page's markdown into chunks at the largest topic
    shifts (top 5% of inter-sentence embedding-distance jumps). Operates on
    markdown text from LlamaParse, which gives cleaner sentence boundaries
    than the soup pypdf produced — same chunker, better input.

Design decisions:
  - LlamaParse called via the synchronous `load_data()` path. The async path
    streams pages back as they're parsed; we don't need that latency win for
    a one-time ingestion script. Sync is simpler and the wait is bounded.
  - One LlamaParse client per chunk_pdf() call. The client holds an HTTP
    session; sharing across files isn't critical at this scale but the
    re-init cost is negligible.
  - Per-page chunking is retained. LlamaParse returns one Document per page
    by default, which lines up with our `page_number` metadata. The page is
    still the provenance unit; chunks never cross page boundaries.
  - api_key passed directly via constructor — no os.environ side effects,
    same pattern as OpenAIEmbeddings.
"""

from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from llama_cloud_services import LlamaParse
from pydantic import BaseModel

from backend.app.config import DATA_DIR, settings
from backend.app.sources import Source, sources_by_filename

# Drop chunks shorter than this — they're almost always page numbers,
# headers, or extraction noise that hurt retrieval signal. LlamaParse
# produces cleaner output than pypdf so this filter triggers less often,
# but it still catches "Page 12 of 47" footers etc.
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
    """Internal: extracted markdown for a single PDF page."""
    text: str
    page_number: int


def _build_parser() -> LlamaParse:
    """
    Build a LlamaParse client configured for markdown output.

    result_type="markdown" is the key knob: tables come back as | col | col |
    rows, headings as #/##, lists as bullets. This is what makes the parse
    a step-change over pypdf — the LLM and the embedding model both
    understand markdown structure natively, where pypdf's space-separated
    table soup is ambiguous to both.

    The API key is required at this point. Fail loud here rather than at the
    first network call so the user gets a clear setup error if they forgot
    LLAMA_CLOUD_API_KEY.
    """
    if not settings.llama_cloud_api_key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set in .env — required for "
            "ingestion. Sign up at https://cloud.llamaindex.ai (free tier "
            "includes 1000 pages/day, more than enough for this project)."
        )
    return LlamaParse(
        api_key=settings.llama_cloud_api_key,
        result_type="markdown",
        # Verbose tells the user which page is being parsed — useful for a
        # one-time ingest, where the user is watching the script run.
        verbose=True,
    )


def _extract_pages(pdf_path: Path, parser: LlamaParse) -> list[_Page]:
    """
    Parse a PDF into a list of per-page markdown blobs.

    LlamaParse.load_data returns one Document per page by default, with a
    "page" key in metadata (1-based). We unwrap into our _Page shape so the
    rest of the chunker doesn't have to know about LlamaIndex types.

    Empty pages (some PDFs have blank separator pages) get dropped here so
    the downstream chunker doesn't waste an embedding call on whitespace.
    """
    documents = parser.load_data(str(pdf_path))
    pages: list[_Page] = []
    for doc in documents:
        text = (doc.text or "").strip()
        if not text:
            continue
        # LlamaParse stores 1-based page number in metadata["page"]. Fall
        # back to enumeration order if the key isn't there (shouldn't happen
        # but defensive — we don't want to crash ingestion on a metadata edge
        # case from a future LlamaParse version).
        page_number = doc.metadata.get("page", len(pages) + 1)
        pages.append(_Page(text=text, page_number=page_number))
    return pages


def _build_splitter() -> SemanticChunker:
    """
    Build a SemanticChunker backed by the project's embedding model.

    Uses the same model (text-embedding-3-small) as the query embedder for
    consistency — semantic proximity at ingest time matches semantic
    proximity at retrieval time. The api_key is passed directly rather than
    relying on os.environ so this module has no env-mutation side effects.
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
    """Split one page's markdown into Chunks stamped with source metadata."""
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

    parser = _build_parser()
    splitter = _build_splitter()

    chunks: list[Chunk] = []
    for page in _extract_pages(pdf_path, parser):
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
