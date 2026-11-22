"""
rag_fitness_coach.ingestion.loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Document loaders for PDF files and web pages.

Supported sources:
  - Local PDF files  →  PyMuPDFLoader  (fast, preserves page metadata)
  - Web URLs         →  WebBaseLoader  (BeautifulSoup HTML → plain text)

All loaders return a list of LangChain Document objects with enriched metadata:
  {source, source_type, pillar, title, page (PDFs only)}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

Pillar = Literal["exercise_library", "health_issues", "nutrition", "general"]


# ── PDF loader ────────────────────────────────────────────────────────────────

def load_pdf(
    path: str | Path,
    pillar: Pillar = "general",
    title: str | None = None,
) -> list[Document]:
    """
    Load a local PDF and return one Document per page.

    Args:
        path:   Absolute or relative path to the PDF file.
        pillar: Knowledge base pillar this document belongs to.
        title:  Human-readable document title (defaults to filename stem).

    Returns:
        List of LangChain Documents, one per page.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    title = title or path.stem.replace("_", " ").replace("-", " ").title()
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()

    for doc in docs:
        doc.metadata.update(
            {
                "source": str(path),
                "source_type": "pdf",
                "pillar": pillar,
                "title": title,
            }
        )

    logger.info("Loaded PDF '%s': %d pages", title, len(docs))
    return docs


# ── Web loader ────────────────────────────────────────────────────────────────

def load_web(
    url: str,
    pillar: Pillar = "general",
    title: str | None = None,
) -> list[Document]:
    """
    Load a web page and return it as a single Document.

    Args:
        url:    The full URL to fetch.
        pillar: Knowledge base pillar this document belongs to.
        title:  Human-readable title (defaults to the URL).

    Returns:
        List with a single LangChain Document containing the page text.
    """
    title = title or url
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {
        "timeout": 20,
        "headers": {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        },
    }
    docs = loader.load()

    for doc in docs:
        doc.metadata.update(
            {
                "source": url,
                "source_type": "web",
                "pillar": pillar,
                "title": title,
            }
        )

    logger.info("Loaded web page '%s': %d chars", title, sum(len(d.page_content) for d in docs))
    return docs


# ── Convenience: load from the sources registry ───────────────────────────────

# Each entry: (type, identifier, pillar, title)
SOURCE_REGISTRY: list[dict] = [
    # ── Exercise Library ────────────────────────────────────────────────
    {
        "type": "web",
        "url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
        "pillar": "exercise_library",
        "title": "WHO Guidelines on Physical Activity",
    },
    {
        "type": "web",
        "url": "https://en.wikipedia.org/wiki/Strength_training",
        "pillar": "exercise_library",
        "title": "Strength and Resistance Training — Principles & Physiology",
    },
    {
        "type": "web",
        "url": "https://en.wikipedia.org/wiki/Exercise_physiology",
        "pillar": "exercise_library",
        "title": "Exercise Physiology — Muscular & Cardiovascular Adaptations",
    },
    # ── Health Issues ────────────────────────────────────────────────────
    {
        "type": "web",
        "url": "https://en.wikipedia.org/wiki/Low_back_pain",
        "pillar": "health_issues",
        "title": "Low Back Pain — Exercise, Prevention and Rehabilitation",
    },
    {
        "type": "web",
        "url": "https://en.wikipedia.org/wiki/Aerobic_exercise",
        "pillar": "health_issues",
        "title": "Aerobic Exercise — Cardiovascular Health & Metabolic Benefits",
    },
    # ── Nutrition ────────────────────────────────────────────────────────
    {
        "type": "web",
        "url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        "pillar": "nutrition",
        "title": "WHO — Healthy Diet & Micronutrient Guidance",
    },
    {
        "type": "web",
        "url": "https://en.wikipedia.org/wiki/Sports_nutrition",
        "pillar": "nutrition",
        "title": "Sports Nutrition — Macronutrients, Protein, Hydration & Timing",
    },
]


def load_all_sources(base_dir: str | Path = ".") -> list[Document]:
    """
    Load every document in SOURCE_REGISTRY.
    PDFs that don't exist locally are skipped with a warning.

    Args:
        base_dir: Project root directory (used to resolve relative PDF paths).

    Returns:
        Combined list of all loaded Documents.
    """
    base_dir = Path(base_dir)
    all_docs: list[Document] = []

    for entry in SOURCE_REGISTRY:
        try:
            if entry["type"] == "pdf":
                pdf_path = base_dir / entry["path"]
                if not pdf_path.exists():
                    logger.warning("PDF not found, skipping: %s", pdf_path)
                    continue
                docs = load_pdf(pdf_path, pillar=entry["pillar"], title=entry["title"])
            elif entry["type"] == "web":
                docs = load_web(entry["url"], pillar=entry["pillar"], title=entry["title"])
            else:
                logger.warning("Unknown source type '%s', skipping.", entry["type"])
                continue

            all_docs.extend(docs)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load '%s': %s", entry.get("url") or entry.get("path"), exc)

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs