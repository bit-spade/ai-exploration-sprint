"""
rag_fitness_coach.vectorstore.indexer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Batch embedding and upsert pipeline for document chunks into Qdrant.
Includes:
  - Deduplication via deterministic point ID (UUID5 from content + source)
  - Batching to keep memory usage low and leverage vector SIMD / MPS acceleration
  - Vector similarity search helper
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from qdrant_client.http import models

from rag_fitness_coach.config import settings
from rag_fitness_coach.vectorstore.client import get_qdrant_client, init_collection
from rag_fitness_coach.vectorstore.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def generate_point_id(content: str, source: str | None = None) -> str:
    """
    Generate a deterministic UUID string based on content and source.
    Ensures idempotency and deduplication upon re-indexing.
    """
    key = f"{source or ''}::{content.strip()}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Use uuid from hash prefix
    return str(uuid.UUID(bytes=digest[:16], version=4))


def index_documents(
    documents: List[Document],
    batch_size: int = 64,
    recreate_collection: bool = False,
    collection_name: str | None = None,
) -> int:
    """
    Embed and upsert documents into Qdrant in batches.

    Args:
        documents: List of LangChain Documents (chunks) to index.
        batch_size: Number of documents to embed and upsert per batch.
        recreate_collection: If True, clear collection before indexing.
        collection_name: Optional collection name override.

    Returns:
        Total number of vectors upserted.
    """
    if not documents:
        logger.warning("No documents provided for indexing.")
        return 0

    target_collection = collection_name or settings.qdrant_collection
    init_collection(
        collection_name=target_collection,
        vector_size=settings.embedding_dim,
        recreate=recreate_collection,
    )

    client = get_qdrant_client()
    embedder = get_embedding_model()

    total_docs = len(documents)
    logger.info(
        "Beginning indexing of %d chunks into collection '%s' (batch_size=%d)",
        total_docs,
        target_collection,
        batch_size,
    )

    upserted_count = 0

    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        texts = [doc.page_content for doc in batch]

        # Generate embeddings for batch
        embeddings = embedder.embed_documents(texts)

        points = []
        for doc, vector in zip(batch, embeddings):
            point_id = generate_point_id(
                content=doc.page_content,
                source=doc.metadata.get("source"),
            )
            payload = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "title": doc.metadata.get("title", ""),
                "pillar": doc.metadata.get("pillar", "general"),
                "source": doc.metadata.get("source", ""),
                "source_type": doc.metadata.get("source_type", "unknown"),
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=target_collection,
            points=points,
            wait=True,
        )
        upserted_count += len(points)
        logger.info("Upserted %d / %d chunks", upserted_count, total_docs)

    logger.info("Successfully indexed %d chunks into Qdrant", upserted_count)
    return upserted_count


def index_from_jsonl(
    jsonl_path: str | Path,
    batch_size: int = 64,
    recreate_collection: bool = False,
    collection_name: str | None = None,
) -> int:
    """
    Load chunks from a processed JSONL file and index them.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed chunks file not found: {path}")

    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            docs.append(
                Document(
                    page_content=item["page_content"],
                    metadata=item.get("metadata", {}),
                )
            )

    return index_documents(
        documents=docs,
        batch_size=batch_size,
        recreate_collection=recreate_collection,
        collection_name=collection_name,
    )


def similarity_search(
    query: str,
    top_k: int | None = None,
    pillar_filter: str | None = None,
    collection_name: str | None = None,
) -> List[Tuple[Document, float]]:
    """
    Query Qdrant for the most similar chunks.

    Args:
        query: User text query.
        top_k: Number of candidates to return (defaults to settings.retriever_top_k).
        pillar_filter: Optional filter by pillar ('exercise_library', 'health_issues', 'nutrition').
        collection_name: Optional collection name override.

    Returns:
        List of (Document, cosine_similarity_score) tuples.
    """
    target_collection = collection_name or settings.qdrant_collection
    k = top_k or settings.retriever_top_k

    client = get_qdrant_client()
    embedder = get_embedding_model()

    query_vector = embedder.embed_query(query)

    query_filter = None
    if pillar_filter:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="pillar",
                    match=models.MatchValue(value=pillar_filter),
                )
            ]
        )

    search_result = client.query_points(
        collection_name=target_collection,
        query=query_vector,
        query_filter=query_filter,
        limit=k,
        with_payload=True,
    )

    results: List[Tuple[Document, float]] = []
    for hit in search_result.points:
        payload = hit.payload or {}
        content = payload.get("page_content", "")
        meta = payload.get("metadata", {})
        results.append((Document(page_content=content, metadata=meta), float(hit.score)))

    return results
