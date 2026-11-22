"""Unit and integration tests for the vectorstore module."""

import pytest
from langchain_core.documents import Document

from rag_fitness_coach.vectorstore.client import (
    get_qdrant_client,
    init_collection,
    get_collection_info,
)
from rag_fitness_coach.vectorstore.embeddings import get_embedding_model
from rag_fitness_coach.vectorstore.indexer import (
    generate_point_id,
    index_documents,
    similarity_search,
)


@pytest.fixture
def test_collection_name():
    return "test_fitness_docs_pytest"


def test_embedding_model_shape():
    embedder = get_embedding_model()
    vec = embedder.embed_query("Bench press exercise targets the pectoralis major.")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


def test_generate_point_id_deterministic():
    id1 = generate_point_id("Some text", "source1")
    id2 = generate_point_id("Some text", "source1")
    id3 = generate_point_id("Different text", "source1")
    assert id1 == id2
    assert id1 != id3


def test_init_and_get_collection(test_collection_name):
    created = init_collection(
        collection_name=test_collection_name,
        vector_size=384,
        recreate=True,
    )
    assert created is True
    info = get_collection_info(test_collection_name)
    assert info["name"] == test_collection_name
    assert info["vectors_count"] == 0


def test_index_and_similarity_search(test_collection_name):
    docs = [
        Document(
            page_content="Squats and lunges recruit the quadriceps, glutes, and hamstrings for lower body strength.",
            metadata={"title": "Leg Training", "pillar": "exercise_library", "source": "test_guide"},
        ),
        Document(
            page_content="Chronic non-specific low back pain can be managed with core stabilization and the McGill Big 3 exercises.",
            metadata={"title": "Back Rehab", "pillar": "health_issues", "source": "test_rehab"},
        ),
        Document(
            page_content="Consuming 1.6 to 2.2 grams of protein per kilogram of body weight supports muscle protein synthesis.",
            metadata={"title": "Protein Requirements", "pillar": "nutrition", "source": "test_nutrition"},
        ),
    ]

    indexed_count = index_documents(
        documents=docs,
        batch_size=2,
        recreate_collection=True,
        collection_name=test_collection_name,
    )
    assert indexed_count == 3

    # Search for nutrition query
    results = similarity_search(
        query="How much protein should I eat per day for building muscle?",
        top_k=2,
        collection_name=test_collection_name,
    )
    assert len(results) >= 1
    top_doc, top_score = results[0]
    assert top_doc.metadata["pillar"] == "nutrition"
    assert "protein" in top_doc.page_content.lower()
    assert top_score > 0.5

    # Filtered search for exercise library
    filtered_results = similarity_search(
        query="rehab exercises for back",
        top_k=2,
        pillar_filter="health_issues",
        collection_name=test_collection_name,
    )
    assert len(filtered_results) == 1
    assert filtered_results[0][0].metadata["pillar"] == "health_issues"


def test_cleanup_test_collection(test_collection_name):
    client = get_qdrant_client()
    client.delete_collection(test_collection_name)
    collections = [c.name for c in client.get_collections().collections]
    assert test_collection_name not in collections
