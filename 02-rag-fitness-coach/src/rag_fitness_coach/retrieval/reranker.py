"""
rag_fitness_coach.retrieval.reranker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cross-encoder reranking using ms-marco-MiniLM-L-6-v2.
Scores (query, document) pairs directly through a joint transformer,
significantly improving precision over vector-only similarity.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from rag_fitness_coach.config import settings

logger = logging.getLogger(__name__)

_RERANKER_MODEL = None


def get_reranker_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Singleton loader for SentenceTransformers CrossEncoder model.
    """
    global _RERANKER_MODEL
    if _RERANKER_MODEL is not None:
        return _RERANKER_MODEL

    from sentence_transformers import CrossEncoder

    logger.info("Initializing CrossEncoder reranker '%s'", model_name)
    try:
        _RERANKER_MODEL = CrossEncoder(model_name, local_files_only=True)
    except Exception:
        _RERANKER_MODEL = CrossEncoder(model_name)

    return _RERANKER_MODEL


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int | None = None,
) -> List[Tuple[Document, float]]:
    """
    Reranks a list of candidate Documents against the query using CrossEncoder.

    Args:
        query: The user prompt / search query.
        documents: Initial candidate Documents from retriever.
        top_k: Number of highest-ranked documents to return.

    Returns:
        List of (Document, cross_encoder_score) tuples sorted descending by score.
    """
    if not documents:
        return []

    k = top_k or settings.reranker_top_k
    model = get_reranker_model()

    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = list(zip(documents, [float(s) for s in scores]))
    # Sort by cross-encoder score descending
    scored_docs.sort(key=lambda item: item[1], reverse=True)

    ranked_top = scored_docs[:k]

    for doc, score in ranked_top:
        doc.metadata["rerank_score"] = score

    logger.info(
        "Reranked %d candidates down to %d top documents",
        len(documents),
        len(ranked_top),
    )
    return ranked_top
