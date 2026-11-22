"""Unit and integration tests for the FastAPI backend routes."""

import pytest
from httpx import ASGITransport, AsyncClient

from rag_fitness_coach.api.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "indexed_vectors" in data
    assert data["indexed_vectors"] > 0
    assert data["qdrant_collection"] == "fitness_docs"


@pytest.mark.asyncio
async def test_documents_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] >= 7
    assert len(data["documents"]) == data["total_documents"]
    pillars = {d["pillar"] for d in data["documents"]}
    assert "exercise_library" in pillars
    assert "health_issues" in pillars
    assert "nutrition" in pillars


@pytest.mark.asyncio
async def test_chat_sync_endpoint():
    payload = {
        "message": "What is the recommended daily protein intake for resistance training?",
        "session_id": "test_api_session",
        "pillar": "nutrition",
        "stream": False,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert data["session_id"] == "test_api_session"
    assert len(data["sources"]) > 0
    first_source = data["sources"][0]
    assert first_source["index"] == 1
    assert first_source["pillar"] == "nutrition"
    assert "content" in first_source


@pytest.mark.asyncio
async def test_chat_streaming_endpoint():
    payload = {
        "message": "How many sets per week for hypertrophy?",
        "session_id": "test_stream_session",
        "stream": True,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/chat", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "event: sources" in content
    assert "event: token" in content
    assert "event: done" in content
