"""
rag_fitness_coach.api.routes.health
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Health check and readiness route.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter

from rag_fitness_coach import __version__
from rag_fitness_coach.api.schemas import HealthStatus
from rag_fitness_coach.config import settings
from rag_fitness_coach.vectorstore.client import get_collection_info

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthStatus)
def get_health() -> HealthStatus:
    """
    Check health status of API, Qdrant vector store, and model configurations.
    """
    try:
        col_info = get_collection_info()
        q_status = col_info.get("status", "unknown")
        vectors_count = col_info.get("vectors_count", 0)
    except Exception as e:
        logger.error("Failed to fetch Qdrant collection info: %s", e)
        q_status = f"error: {e}"
        vectors_count = 0

    llm_ready = bool(settings.google_api_key and settings.google_api_key.strip())

    return HealthStatus(
        status="healthy",
        version=__version__,
        qdrant_status=q_status,
        qdrant_collection=settings.qdrant_collection,
        indexed_vectors=vectors_count,
        embedding_model=settings.embedding_model,
        llm_configured=llm_ready,
    )
