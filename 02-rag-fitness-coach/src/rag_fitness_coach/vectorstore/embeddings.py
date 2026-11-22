"""
rag_fitness_coach.vectorstore.embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Embeddings factory providing LangChain-compatible Embeddings objects.
Supports:
  - Local SentenceTransformers (e.g. BAAI/bge-small-en-v1.5) with PyTorch MPS acceleration
  - Ollama embeddings (e.g. nomic-embed-text)
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.embeddings import Embeddings

from rag_fitness_coach.config import settings

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain Embeddings implementation using local sentence-transformers on MPS/CPU."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(
            "Loading local embedding model '%s' on device '%s'",
            self.model_name,
            self.device,
        )
        try:
            self.client = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=True,
            )
        except Exception:
            self.client = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.client.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.client.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()


def get_embedding_model() -> Embeddings:
    """
    Factory function returning the configured LangChain Embeddings instance.
    """
    provider = settings.embedding_provider.lower().strip()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        logger.info(
            "Using Ollama embeddings '%s' via %s",
            settings.embedding_model,
            settings.ollama_base_url,
        )
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    # Default: local sentence-transformers
    return LocalSentenceTransformerEmbeddings(model_name=settings.embedding_model)
