"""
rag_fitness_coach.api.routes.chat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chat endpoints supporting:
  - Standard JSON request/response (POST /chat with stream=false)
  - Server-Sent Events (SSE) streaming tokens (POST /chat with stream=true)
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from rag_fitness_coach.api.schemas import ChatRequest, ChatResponse, SourceDoc
from rag_fitness_coach.chain.rag_chain import FitnessRAGChain

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=None)
async def chat_endpoint(request: ChatRequest):
    """
    Submit a question to the RAG Fitness Coach.
    Returns standard JSON or an SSE stream when stream=true.
    """
    chain = FitnessRAGChain(pillar_filter=request.pillar)

    if not request.stream:
        # Synchronous execution
        res = chain.ask(question=request.message, session_id=request.session_id)
        return ChatResponse(
            answer=res["answer"],
            session_id=request.session_id,
            sources=[SourceDoc(**src) for src in res["sources"]],
        )

    # Streaming mode via Server-Sent Events (SSE)
    async def event_generator() -> AsyncIterator[dict]:
        # First event: emit retrieved and reranked sources
        docs = chain.retrieve_and_rerank(request.message)
        sources_payload = [
            {
                "index": i + 1,
                "title": doc.metadata.get("title", "Unknown"),
                "pillar": doc.metadata.get("pillar", "general"),
                "source": doc.metadata.get("source", ""),
                "vector_score": doc.metadata.get("vector_score"),
                "rerank_score": doc.metadata.get("rerank_score"),
                "content": doc.page_content,
            }
            for i, doc in enumerate(docs)
        ]
        yield {
            "event": "sources",
            "data": json.dumps({"sources": sources_payload}),
        }

        # Subsequent events: stream tokens
        async for token in chain.astream_answer(request.message, session_id=request.session_id):
            yield {
                "event": "token",
                "data": json.dumps({"token": token}),
            }

        yield {
            "event": "done",
            "data": json.dumps({"session_id": request.session_id}),
        }

    return EventSourceResponse(event_generator())
