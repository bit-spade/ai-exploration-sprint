"""Unit and integration tests for the full LCEL RAG chain and conversational memory."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_fitness_coach.chain.memory import ConversationMemoryManager, get_session_memory
from rag_fitness_coach.chain.prompts import format_docs_for_prompt
from rag_fitness_coach.chain.rag_chain import FitnessRAGChain
from langchain_core.documents import Document


def test_conversation_memory_window():
    mem = ConversationMemoryManager(window_k=2)
    # Add 3 turns (6 messages)
    mem.add_user_message("Turn 1 user")
    mem.add_ai_message("Turn 1 AI")
    mem.add_user_message("Turn 2 user")
    mem.add_ai_message("Turn 2 AI")
    mem.add_user_message("Turn 3 user")
    mem.add_ai_message("Turn 3 AI")

    msgs = mem.get_messages()
    # Window k=2 keeps at most 4 messages (turns 2 and 3)
    assert len(msgs) == 4
    assert msgs[0].content == "Turn 2 user"
    assert msgs[-1].content == "Turn 3 AI"


def test_format_docs_for_prompt():
    docs = [
        Document(
            page_content="Progressive mechanical tension drives hypertrophy.",
            metadata={"title": "Hypertrophy", "pillar": "exercise_library", "source": "https://example.com/1"},
        ),
        Document(
            page_content="Protein synthesis requires leucine triggers.",
            metadata={"title": "Nutrition", "pillar": "nutrition"},
        ),
    ]
    formatted = format_docs_for_prompt(docs)
    assert "[Doc 1]" in formatted
    assert "[Doc 2]" in formatted
    assert "Title: Hypertrophy" in formatted
    assert "Progressive mechanical tension" in formatted


def test_rag_chain_execution():
    fake_llm = FakeListChatModel(
        responses=[
            "According to [Doc 1], progressive overload is key to hypertrophy, and [Doc 2] confirms protein intake needs."
        ]
    )
    chain = FitnessRAGChain(
        llm=fake_llm,
        retriever_top_k=4,
        reranker_top_k=2,
    )

    response = chain.ask(
        question="How to build muscle effectively?",
        session_id="pytest_session",
    )

    assert "answer" in response
    assert "[Doc 1]" in response["answer"]
    assert len(response["sources"]) == 2
    assert response["sources"][0]["index"] == 1
    assert "rerank_score" in response["sources"][0]

    # Verify memory was updated for this session
    mem = get_session_memory("pytest_session")
    history = mem.get_messages()
    assert len(history) == 2
    assert history[0].content == "How to build muscle effectively?"
