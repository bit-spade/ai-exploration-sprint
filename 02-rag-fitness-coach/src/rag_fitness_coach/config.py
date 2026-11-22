"""
rag_fitness_coach.config
~~~~~~~~~~~~~~~~~~~~~~~~
Central configuration loaded from environment variables / .env file.
All modules import from here — never read os.environ directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Typed, validated configuration via Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────
    google_api_key: str = Field("", description="Gemini API key (required for LLM inference)")
    gemini_model: str = Field("gemini-1.5-flash", description="Gemini model name")

    # ── Embeddings (Local SentenceTransformers or Ollama) ────
    embedding_provider: str = Field("sentence-transformers", description="Embedding provider: sentence-transformers or ollama")
    embedding_model: str = Field("BAAI/bge-small-en-v1.5", description="Model name (HuggingFace id or Ollama model)")
    embedding_dim: int = Field(384, description="Embedding vector dimension (384 for bge-small, 768 for nomic)")
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama server URL if using ollama")

    # ── Vector Store (Qdrant) ─────────────────────────────────
    qdrant_url: str = Field("", description="Qdrant server URL (e.g. http://localhost:6333); if empty, uses local disk storage")
    qdrant_path: str = Field("data/qdrant_storage", description="Path to local Qdrant directory if qdrant_url is empty")
    qdrant_collection: str = Field("fitness_docs", description="Qdrant collection name")

    # ── Retrieval ─────────────────────────────────────────────
    retriever_top_k: int = Field(8, description="Candidates fetched from Qdrant")
    reranker_top_k: int = Field(3, description="Chunks kept after cross-encoder reranking")

    # ── Chunking ──────────────────────────────────────────────
    chunk_size: int = Field(512, description="Max chars per chunk")
    chunk_overlap: int = Field(64, description="Overlap chars between adjacent chunks")

    # ── Memory ────────────────────────────────────────────────
    memory_window_k: int = Field(6, description="Recent conversation turns to keep")

    # ── API ───────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(8000, description="FastAPI bind port")


# Singleton — import this everywhere
settings = Settings()
