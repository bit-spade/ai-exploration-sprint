"""
rag_fitness_coach.api.schemas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic schemas for the FastAPI REST & streaming interface.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


PillarType = Literal["exercise_library", "health_issues", "nutrition"]


class SourceDoc(BaseModel):
    """Metadata and excerpt of a cited document passage."""
    index: int = Field(..., description="Citation index corresponding to [Doc N]")
    title: str = Field(..., description="Document title")
    pillar: str = Field(..., description="Knowledge pillar")
    source: str = Field("", description="Source URL or file path")
    vector_score: Optional[float] = Field(None, description="Initial dense cosine score")
    rerank_score: Optional[float] = Field(None, description="Cross-encoder relevance score")
    content: str = Field(..., description="Text excerpt used as context")


class ChatRequest(BaseModel):
    """Payload for submitting a question to the coach."""
    message: str = Field(..., min_length=1, description="The user's question or statement")
    session_id: str = Field("default", description="Conversation session ID for memory")
    pillar: Optional[PillarType] = Field(None, description="Optional domain filter")
    stream: bool = Field(False, description="Whether to stream the response as Server-Sent Events (SSE)")


class ChatResponse(BaseModel):
    """Synchronous response containing the answer and supporting citations."""
    answer: str = Field(..., description="The coach's response with [Doc N] citations")
    session_id: str = Field(..., description="Session ID")
    sources: List[SourceDoc] = Field(default_factory=list, description="Cited reference documents")


class DocumentItem(BaseModel):
    """Summary of an indexed document or knowledge base entry."""
    title: str
    pillar: str
    source_type: str
    url_or_path: str
    status: str = "indexed"


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentItem]


class HealthStatus(BaseModel):
    status: str
    version: str
    qdrant_status: str
    qdrant_collection: str
    indexed_vectors: int
    embedding_model: str
    llm_configured: bool
