"""
rag_fitness_coach.api.main
~~~~~~~~~~~~~~~~~~~~~~~~~~
FastAPI application factory with CORS middleware, lifespan setup, and route registration.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_fitness_coach import __version__
from rag_fitness_coach.api.routes.chat import router as chat_router
from rag_fitness_coach.api.routes.documents import router as documents_router
from rag_fitness_coach.api.routes.health import router as health_router
from rag_fitness_coach.config import settings
from rag_fitness_coach.vectorstore.client import init_collection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown hook.
    Ensures vector store collection is ready upon launch.
    """
    logger.info("Initializing RAG Fitness Coach API v%s...", __version__)
    try:
        init_collection()
        logger.info("Vector store collection verified.")
    except Exception as e:
        logger.warning("Could not pre-initialize vector collection: %s", e)
    yield
    logger.info("Shutting down RAG Fitness Coach API...")


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.
    """
    app = FastAPI(
        title="RAG Fitness Coach API",
        description="Evidence-based fitness coaching assistant grounded in sports science literature.",
        version=__version__,
        lifespan=lifespan,
    )

    # Enable CORS for Streamlit / web frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chat_router)

    return app


app = create_app()
