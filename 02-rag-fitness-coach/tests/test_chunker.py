"""Tests for the chunker module."""

import pytest
from langchain_core.documents import Document

from rag_fitness_coach.ingestion.chunker import chunk_documents, chunk_stats, build_splitter


SAMPLE_DOC = Document(
    page_content=(
        "Resistance training is the systematic use of progressive overload to stimulate "
        "muscular adaptations. Compound movements such as the squat, deadlift, and bench press "
        "recruit multiple muscle groups simultaneously and should form the foundation of any "
        "strength programme.\n\n"
        "Isolation exercises supplement compounds by targeting specific muscles. For example, "
        "bicep curls isolate the biceps brachii and are often used as accessory work after "
        "compound pulling movements.\n\n"
        "Recovery between sessions is essential. The muscle protein synthesis window remains "
        "elevated for approximately 24-48 hours after a resistance training bout."
    ),
    metadata={"source": "test.pdf", "source_type": "pdf", "pillar": "exercise_library", "title": "Test Doc"},
)


def test_build_splitter_defaults():
    splitter = build_splitter()
    assert splitter._chunk_size > 0


def test_build_splitter_custom():
    splitter = build_splitter(chunk_size=100, chunk_overlap=10)
    assert splitter._chunk_size == 100
    assert splitter._chunk_overlap == 10


def test_chunk_documents_returns_list():
    chunks = chunk_documents([SAMPLE_DOC], chunk_size=200, chunk_overlap=20)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_metadata_preserved():
    chunks = chunk_documents([SAMPLE_DOC], chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["pillar"] == "exercise_library"
        assert chunk.metadata["title"] == "Test Doc"


def test_chunk_metadata_enriched():
    chunks = chunk_documents([SAMPLE_DOC], chunk_size=200, chunk_overlap=20)
    for idx, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata
        assert "char_count" in chunk.metadata
        assert chunk.metadata["char_count"] == len(chunk.page_content)


def test_chunk_size_respected():
    chunk_size = 150
    chunks = chunk_documents([SAMPLE_DOC], chunk_size=chunk_size, chunk_overlap=0)
    for chunk in chunks:
        # Allow small overshoot at word boundaries
        assert len(chunk.page_content) <= chunk_size + 20


def test_chunk_stats_structure():
    chunks = chunk_documents([SAMPLE_DOC], chunk_size=200, chunk_overlap=20)
    stats = chunk_stats(chunks)
    assert "total_chunks" in stats
    assert "min_chars" in stats
    assert "max_chars" in stats
    assert "mean_chars" in stats
    assert "median_chars" in stats
    assert "pillar_breakdown" in stats
    assert stats["pillar_breakdown"]["exercise_library"] >= 1


def test_empty_document_list():
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_stats_empty():
    stats = chunk_stats([])
    assert stats["total_chunks"] == 0
