"""
rag_fitness_coach.vectorstore.client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Qdrant client singleton and collection initialization.
Supports:
  - Local embedded disk storage (when settings.qdrant_url is empty)
  - Remote / Docker Qdrant server (when settings.qdrant_url is provided)
"""

from __future__ import annotations

import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_fitness_coach.config import settings

logger = logging.getLogger(__name__)

_CLIENT: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Return a reusable QdrantClient instance based on current settings.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    if settings.qdrant_url and settings.qdrant_url.strip():
        logger.info("Connecting to Qdrant server at %s", settings.qdrant_url)
        _CLIENT = QdrantClient(url=settings.qdrant_url)
    else:
        qdrant_dir = Path(settings.qdrant_path)
        qdrant_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing embedded local Qdrant at %s", qdrant_dir)
        _CLIENT = QdrantClient(path=str(qdrant_dir))

    return _CLIENT


def init_collection(
    collection_name: str | None = None,
    vector_size: int | None = None,
    recreate: bool = False,
) -> bool:
    """
    Ensure the Qdrant collection exists with cosine distance.

    Args:
        collection_name: Name of the collection. Defaults to settings.qdrant_collection.
        vector_size: Dimension of embeddings. Defaults to settings.embedding_dim.
        recreate: If True, delete and recreate the collection.

    Returns:
        True if collection is ready.
    """
    client = get_qdrant_client()
    name = collection_name or settings.qdrant_collection
    dim = vector_size or settings.embedding_dim

    collections = [c.name for c in client.get_collections().collections]

    if name in collections:
        if recreate:
            logger.warning("Recreating Qdrant collection '%s' (dim=%d)", name, dim)
            client.delete_collection(name)
        else:
            logger.info("Qdrant collection '%s' already exists", name)
            return True

    logger.info("Creating Qdrant collection '%s' (dim=%d, metric=COSINE)", name, dim)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
        ),
    )
    return True


def get_collection_info(collection_name: str | None = None) -> dict:
    """
    Fetch collection statistics (vector count, status, segments).
    """
    client = get_qdrant_client()
    name = collection_name or settings.qdrant_collection
    info = client.get_collection(collection_name=name)
    return {
        "name": name,
        "status": str(info.status),
        "vectors_count": info.points_count or 0,
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
    }
