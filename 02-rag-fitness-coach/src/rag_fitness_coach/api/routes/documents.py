"""
rag_fitness_coach.api.routes.documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Inspection routes for the curated knowledge base and indexed sources.
"""

from __future__ import annotations

from fastapi import APIRouter

from rag_fitness_coach.api.schemas import DocumentItem, DocumentListResponse
from rag_fitness_coach.ingestion.loader import SOURCE_REGISTRY

router = APIRouter(tags=["Documents"])


@router.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """
    List all curated reference documents in the knowledge base across the 3 pillars.
    """
    items = []
    for entry in SOURCE_REGISTRY:
        url_or_path = entry.get("url") or entry.get("path", "")
        items.append(
            DocumentItem(
                title=entry.get("title", "Unknown"),
                pillar=entry.get("pillar", "general"),
                source_type=entry.get("type", "web"),
                url_or_path=url_or_path,
                status="indexed",
            )
        )

    return DocumentListResponse(
        total_documents=len(items),
        documents=items,
    )
