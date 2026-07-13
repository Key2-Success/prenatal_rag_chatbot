"""
models.py — lightweight RAG data models shared by ingest and runtime.

`Chunk` lives here (not in chunker.py) so the query-time path can import it
without pulling chunker.py's ingest-only, heavyweight dependencies
(LlamaParse / llama-index, langchain SemanticChunker, tiktoken). retriever.py
imports Chunk from here; the deployed backend therefore needs none of that
machinery, which keeps its image small enough for a free tier.
"""

from pydantic import BaseModel


class Chunk(BaseModel):
    """One unit of text ready for embedding + Pinecone upsert."""

    text: str
    source_file: str
    org_display_name: str
    doc_title: str
    doc_reference_order: int
    year_published: int
    page_number: int  # page where this chunk begins
    section_heading: str = ""  # deepest markdown header enclosing this chunk
