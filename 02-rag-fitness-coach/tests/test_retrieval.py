"""Unit and integration tests for retrieval and reranking."""

import pytest
from langchain_core.documents import Document

from rag_fitness_coach.retrieval.retriever import get_fitness_retriever
from rag_fitness_coach.retrieval.reranker import rerank_documents


def test_qdrant_fitness_retriever():
    retriever = get_fitness_retriever(top_k=4)
    docs = retriever.invoke("muscle hypertrophy training frequency")
    assert isinstance(docs, list)
    assert len(docs) <= 4
    for doc in docs:
        assert isinstance(doc, Document)
        assert "vector_score" in doc.metadata
        assert doc.metadata["vector_score"] > 0


def test_qdrant_fitness_retriever_with_filter():
    retriever = get_fitness_retriever(top_k=3, pillar_filter="nutrition")
    docs = retriever.invoke("protein intake for recovery")
    assert len(docs) <= 3
    for doc in docs:
        assert doc.metadata.get("pillar") == "nutrition"


def test_cross_encoder_reranker():
    query = "How much protein should an athlete consume?"
    docs = [
        Document(
            page_content="Aerobic endurance running increases left ventricular volume.",
            metadata={"title": "Aerobic Adaptations"},
        ),
        Document(
            page_content="Strength athletes need around 1.6 to 2.2 grams of protein per kilogram of body weight.",
            metadata={"title": "Sports Nutrition"},
        ),
        Document(
            page_content="Static stretching before lifting can slightly reduce peak power output.",
            metadata={"title": "Warmup Protocol"},
        ),
    ]

    reranked = rerank_documents(query=query, documents=docs, top_k=2)
    assert len(reranked) == 2
    # The protein document must be ranked #1
    top_doc, top_score = reranked[0]
    assert "protein" in top_doc.page_content
    assert top_doc.metadata["rerank_score"] == top_score
    # Score for relevant document should be substantially higher than irrelevant
    assert top_score > reranked[1][1]
