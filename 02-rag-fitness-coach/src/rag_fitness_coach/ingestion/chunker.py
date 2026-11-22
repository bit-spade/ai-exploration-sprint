"""
rag_fitness_coach.ingestion.chunker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Text chunking strategies for the RAG pipeline.

Strategy used: RecursiveCharacterTextSplitter
  - Tries to split on paragraphs → sentences → words → chars (in that order).
  - Preserves natural language boundaries better than fixed-size splitting.
  - Chunk size and overlap are configurable via Settings.

Each output chunk inherits ALL metadata from its parent Document and gains:
  - chunk_index  : position within the parent document
  - char_count   : length of the chunk content
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_fitness_coach.config import settings

logger = logging.getLogger(__name__)


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter with the given (or default) params.

    Args:
        chunk_size:    Max characters per chunk. Defaults to settings.chunk_size.
        chunk_overlap: Overlap characters. Defaults to settings.chunk_overlap.

    Returns:
        Configured splitter instance.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Split hierarchy: paragraphs → sentences → words → characters
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks.

    Metadata is forwarded from each parent Document and enriched with
    chunk_index and char_count fields.

    Args:
        documents:     Source Documents from the loader.
        chunk_size:    Override chunk size (chars).
        chunk_overlap: Override overlap (chars).

    Returns:
        List of chunk Documents ready for embedding.
    """
    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks: list[Document] = []
    for doc in documents:
        chunks = splitter.split_documents([doc])

        for idx, chunk in enumerate(chunks):
            # Enrich metadata
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["char_count"] = len(chunk.page_content)
            # Ensure these fields are always present (may be missing in web docs)
            chunk.metadata.setdefault("page", None)

        all_chunks.extend(chunks)

    logger.info(
        "Chunked %d documents → %d chunks (size=%s, overlap=%s)",
        len(documents),
        len(all_chunks),
        chunk_size or settings.chunk_size,
        chunk_overlap or settings.chunk_overlap,
    )
    return all_chunks


def chunk_stats(chunks: list[Document]) -> dict[str, Any]:
    """
    Compute descriptive statistics over a list of chunks.
    Useful for the chunking experiments notebook.

    Returns:
        Dict with min/max/mean/median char counts and pillar breakdown.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "min_chars": 0,
            "max_chars": 0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "pillar_breakdown": {},
        }

    import statistics

    lengths = [len(c.page_content) for c in chunks]
    pillar_counts: dict[str, int] = {}
    for chunk in chunks:
        pillar = chunk.metadata.get("pillar", "unknown")
        pillar_counts[pillar] = pillar_counts.get(pillar, 0) + 1

    return {
        "total_chunks": len(chunks),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_chars": round(statistics.mean(lengths), 1),
        "median_chars": round(statistics.median(lengths), 1),
        "pillar_breakdown": pillar_counts,
    }