"""
rag_fitness_coach.retrieval.retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangChain-compatible Retriever wrapping the Qdrant vector store.
Fetches initial candidates based on dense cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_fitness_coach.config import settings
from rag_fitness_coach.vectorstore.indexer import similarity_search

logger = logging.getLogger(__name__)


class QdrantFitnessRetriever(BaseRetriever):
    """
    Retriever that queries the Qdrant fitness_docs collection.
    Allows optional pillar filtering ('exercise_library', 'health_issues', 'nutrition').
    """

    top_k: int = settings.retriever_top_k
    pillar_filter: Optional[str] = None
    collection_name: Optional[str] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = similarity_search(
            query=query,
            top_k=self.top_k,
            pillar_filter=self.pillar_filter,
            collection_name=self.collection_name,
        )
        # Attach similarity score into metadata for downstream inspection
        documents: List[Document] = []
        for doc, score in results:
            doc.metadata["vector_score"] = score
            documents.append(doc)

        logger.info(
            "Retrieved %d candidate documents from Qdrant (top_k=%d, filter=%s)",
            len(documents),
            self.top_k,
            self.pillar_filter,
        )
        return documents


def get_fitness_retriever(
    top_k: int | None = None,
    pillar_filter: str | None = None,
    collection_name: str | None = None,
) -> QdrantFitnessRetriever:
    """
    Factory helper to instantiate a configured QdrantFitnessRetriever.
    """
    return QdrantFitnessRetriever(
        top_k=top_k or settings.retriever_top_k,
        pillar_filter=pillar_filter,
        collection_name=collection_name,
    )
