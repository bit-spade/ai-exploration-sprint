"""
rag_fitness_coach.chain.rag_chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LCEL (LangChain Expression Language) RAG Chain.
Pipeline:
  1. Receive query & session_id
  2. Retrieve candidate documents (top_k=8) from Qdrant
  3. Rerank candidates with CrossEncoder (top_k=3)
  4. Format retrieved context into prompt with [Doc N] references
  5. Stream or invoke LLM (Google Gemini or Mock fallback for testing)
  6. Store turns into conversational memory
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser

from rag_fitness_coach.config import settings
from rag_fitness_coach.chain.memory import get_session_memory
from rag_fitness_coach.chain.prompts import format_docs_for_prompt, get_rag_prompt_template
from rag_fitness_coach.retrieval.reranker import rerank_documents
from rag_fitness_coach.retrieval.retriever import get_fitness_retriever

logger = logging.getLogger(__name__)


def get_llm() -> BaseChatModel:
    """
    Instantiate the chat LLM. Uses ChatGoogleGenerativeAI if GOOGLE_API_KEY is present,
    or falls back to a clean mock message generator for testing without an API key.
    """
    if settings.google_api_key and settings.google_api_key.strip():
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("Initializing ChatGoogleGenerativeAI (%s)", settings.gemini_model)
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
            streaming=True,
        )

    # Clean fallback when no API key is set yet
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    logger.warning("GOOGLE_API_KEY is empty. Using simulated LLM output for demonstration.")
    return FakeListChatModel(
        responses=[
            "Based on the sports science evidence in [Doc 1], resistance training with progressive overload stimulates muscle hypertrophy effectively. Nutrition guidelines in [Doc 2] recommend 1.6 to 2.2 g/kg daily protein."
        ]
    )


class FitnessRAGChain:
    """
    Encapsulates the end-to-end RAG workflow.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        pillar_filter: Optional[str] = None,
        retriever_top_k: Optional[int] = None,
        reranker_top_k: Optional[int] = None,
    ):
        self.llm = llm or get_llm()
        self.pillar_filter = pillar_filter
        self.retriever = get_fitness_retriever(
            top_k=retriever_top_k or settings.retriever_top_k,
            pillar_filter=pillar_filter,
        )
        self.reranker_top_k = reranker_top_k or settings.reranker_top_k
        self.prompt_template = get_rag_prompt_template()
        self.output_parser = StrOutputParser()

    def retrieve_and_rerank(self, query: str) -> List[Document]:
        """
        Runs stage 1 (vector retrieval) and stage 2 (cross-encoder reranking).
        """
        candidates = self.retriever.invoke(query)
        scored_pairs = rerank_documents(
            query=query,
            documents=candidates,
            top_k=self.reranker_top_k,
        )
        return [doc for doc, _ in scored_pairs]

    def ask(
        self,
        question: str,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Synchronous Q&A returning the full answer and source documents.
        """
        memory = get_session_memory(session_id)
        history = memory.get_messages()

        docs = self.retrieve_and_rerank(question)
        context_str = format_docs_for_prompt(docs)

        # Build prompt messages
        prompt_val = self.prompt_template.invoke(
            {
                "context": context_str,
                "history": history,
                "question": question,
            }
        )

        response = self.llm.invoke(prompt_val)
        answer = self.output_parser.invoke(response)

        # Update conversational memory
        memory.add_user_message(question)
        memory.add_ai_message(answer)

        return {
            "answer": answer,
            "sources": [
                {
                    "index": i + 1,
                    "title": doc.metadata.get("title", "Unknown"),
                    "pillar": doc.metadata.get("pillar", "general"),
                    "source": doc.metadata.get("source", ""),
                    "rerank_score": doc.metadata.get("rerank_score"),
                    "vector_score": doc.metadata.get("vector_score"),
                    "content": doc.page_content,
                }
                for i, doc in enumerate(docs)
            ],
        }

    async def astream_answer(
        self,
        question: str,
        session_id: str = "default",
    ) -> AsyncIterator[str]:
        """
        Asynchronous token generator for streaming answers.
        """
        memory = get_session_memory(session_id)
        history = memory.get_messages()

        docs = self.retrieve_and_rerank(question)
        context_str = format_docs_for_prompt(docs)

        prompt_val = self.prompt_template.invoke(
            {
                "context": context_str,
                "history": history,
                "question": question,
            }
        )

        accumulated_chunks = []
        async for chunk in self.llm.astream(prompt_val):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            accumulated_chunks.append(token)
            yield token

        full_answer = "".join(accumulated_chunks)
        memory.add_user_message(question)
        memory.add_ai_message(full_answer)
