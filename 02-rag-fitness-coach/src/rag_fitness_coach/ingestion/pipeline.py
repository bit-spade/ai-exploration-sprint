"""
rag_fitness_coach.ingestion.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI ingestion pipeline: load → chunk → save as JSONL.

Usage:
    # From the project root (with venv active):
    python -m rag_fitness_coach.ingestion.pipeline ingest

    # Override chunk parameters:
    python -m rag_fitness_coach.ingestion.pipeline ingest --chunk-size 256 --overlap 32

    # Ingest only web sources (skip PDFs):
    python -m rag_fitness_coach.ingestion.pipeline ingest --web-only

    # Show stats only (dry-run):
    python -m rag_fitness_coach.ingestion.pipeline stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Ensure the src/ directory is importable when run as __main__
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_fitness_coach.ingestion.loader import SOURCE_REGISTRY, load_all_sources, load_web, load_pdf
from rag_fitness_coach.ingestion.chunker import chunk_documents, chunk_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
console = Console()

# ── Output paths ──────────────────────────────────────────────────────────────

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_chunks_jsonl(chunks, output_path: Path) -> None:
    """Persist chunks as newline-delimited JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"[green]✓[/green] Saved {len(chunks)} chunks → [bold]{output_path}[/bold]")


def print_stats_table(stats: dict) -> None:
    table = Table(title="Chunk Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total chunks", str(stats["total_chunks"]))
    table.add_row("Min chars", str(stats["min_chars"]))
    table.add_row("Max chars", str(stats["max_chars"]))
    table.add_row("Mean chars", str(stats["mean_chars"]))
    table.add_row("Median chars", str(stats["median_chars"]))
    console.print(table)

    pillar_table = Table(title="Chunks by Pillar", show_header=True, header_style="bold magenta")
    pillar_table.add_column("Pillar")
    pillar_table.add_column("Chunks", justify="right")
    for pillar, count in stats["pillar_breakdown"].items():
        pillar_table.add_row(pillar, str(count))
    console.print(pillar_table)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    """Load all sources, chunk them, and save to data/processed/."""
    console.rule("[bold blue]RAG Fitness Coach — Ingestion Pipeline[/bold blue]")

    # 1. Load
    console.print("\n[bold]Step 1/3:[/bold] Loading source documents...")
    docs = load_all_sources(base_dir=PROJECT_ROOT)
    if not docs:
        console.print("[red]No documents loaded. Check that PDFs exist in data/raw/ and web sources are reachable.[/red]")
        sys.exit(1)
    console.print(f"  Loaded [bold]{len(docs)}[/bold] document pages/sections.")

    # 2. Chunk
    console.print("\n[bold]Step 2/3:[/bold] Chunking documents...")
    chunks = chunk_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )
    stats = chunk_stats(chunks)
    print_stats_table(stats)

    # 3. Save
    if not args.dry_run:
        console.print("\n[bold]Step 3/3:[/bold] Saving chunks...")
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        output_path = PROCESSED_DIR / f"chunks_{timestamp}.jsonl"
        save_chunks_jsonl(chunks, output_path)

        # Also save a "latest" symlink for easy reference
        latest_link = PROCESSED_DIR / "chunks_latest.jsonl"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(output_path.name)
        console.print(f"[dim]  Symlink: chunks_latest.jsonl → {output_path.name}[/dim]")
    else:
        console.print("\n[yellow]Dry-run mode — skipping save.[/yellow]")

    console.rule("[bold green]Done![/bold green]")


def cmd_stats(args: argparse.Namespace) -> None:
    """Load latest chunks and print statistics."""
    latest = PROCESSED_DIR / "chunks_latest.jsonl"
    if not latest.exists():
        console.print("[red]No processed chunks found. Run 'ingest' first.[/red]")
        sys.exit(1)

    chunks_data = [json.loads(line) for line in latest.open()]
    from langchain_core.documents import Document
    chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in chunks_data]
    stats = chunk_stats(chunks)
    print_stats_table(stats)


def cmd_index(args: argparse.Namespace) -> None:
    """Read processed JSONL chunks, embed them, and index into Qdrant."""
    from rag_fitness_coach.vectorstore.indexer import index_from_jsonl
    from rag_fitness_coach.vectorstore.client import get_collection_info

    jsonl_path = Path(args.file) if args.file else PROCESSED_DIR / "chunks_latest.jsonl"
    if not jsonl_path.exists():
        console.print(f"[red]Chunks file not found at {jsonl_path}. Run 'ingest' first.[/red]")
        sys.exit(1)

    console.rule("[bold cyan]RAG Fitness Coach — Vector Store Indexing[/bold cyan]")
    console.print(f"Reading chunks from: [bold]{jsonl_path}[/bold]")

    indexed_count = index_from_jsonl(
        jsonl_path=jsonl_path,
        batch_size=args.batch_size,
        recreate_collection=args.recreate,
    )
    info = get_collection_info()
    console.print(f"\n[green]✓ Successfully indexed {indexed_count} vectors into collection '{info['name']}'![/green]")
    console.print(f"Total collection points: [bold]{info['vectors_count']}[/bold]")
    console.rule("[bold green]Done![/bold green]")


def cmd_search(args: argparse.Namespace) -> None:
    """Query Qdrant for semantically relevant chunks."""
    from rag_fitness_coach.vectorstore.indexer import similarity_search

    console.rule(f"[bold magenta]Query: {args.query}[/bold magenta]")
    results = similarity_search(
        query=args.query,
        top_k=args.top_k,
        pillar_filter=args.pillar,
    )

    if not results:
        console.print("[yellow]No matching documents found.[/yellow]")
        return

    for idx, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        title = meta.get("title", "Unknown")
        pillar = meta.get("pillar", "general")
        source = meta.get("source", "")
        console.print(f"\n[bold green]Result #{idx}[/bold green] (Cosine Score: [cyan]{score:.4f}[/cyan])")
        console.print(f"[bold]Title:[/bold] {title} | [bold]Pillar:[/bold] {pillar}")
        console.print(f"[dim]Source: {source}[/dim]")
        preview = doc.page_content.strip()
        if len(preview) > 300:
            preview = preview[:300] + "..."
        console.print(f"[white]{preview}[/white]")


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask the RAG Fitness Coach a question through the full retrieval + reranking + LLM chain."""
    from rag_fitness_coach.chain.rag_chain import FitnessRAGChain

    console.rule(f"[bold cyan]Coach Query: {args.question}[/bold cyan]")
    chain = FitnessRAGChain(pillar_filter=args.pillar)

    with console.status("[bold green]Retrieving, reranking evidence and generating response...[/bold green]"):
        result = chain.ask(question=args.question, session_id=args.session_id)

    console.print("\n[bold green]Coach Response:[/bold green]")
    console.print(result["answer"])

    console.print("\n[bold magenta]Retrieved & Cited Sources:[/bold magenta]")
    for src in result["sources"]:
        v_score = f"{src['vector_score']:.3f}" if src.get("vector_score") else "N/A"
        r_score = f"{src['rerank_score']:.3f}" if src.get("rerank_score") is not None else "N/A"
        console.print(
            f"  [bold]• [Doc {src['index']}][/bold] {src['title']} ({src['pillar']}) "
            f"[dim][Cosine: {v_score}, Rerank: {r_score}][/dim]"
        )
        if src.get("source"):
            console.print(f"    [dim underline]{src['source']}[/dim underline]")


# ── CLI entry point ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_fitness_coach.ingestion.pipeline",
        description="RAG Fitness Coach ingestion pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_p = sub.add_parser("ingest", help="Load, chunk, and save documents")
    ingest_p.add_argument("--chunk-size", type=int, default=None, help="Override chunk size (chars)")
    ingest_p.add_argument("--overlap", type=int, default=None, help="Override chunk overlap (chars)")
    ingest_p.add_argument("--dry-run", action="store_true", help="Run without saving output")

    # stats
    sub.add_parser("stats", help="Print statistics for the latest chunk file")

    # index
    index_p = sub.add_parser("index", help="Embed and index processed chunks into Qdrant")
    index_p.add_argument("--file", type=str, default=None, help="Path to JSONL file (defaults to chunks_latest.jsonl)")
    index_p.add_argument("--batch-size", type=int, default=64, help="Embedding and upsert batch size")
    index_p.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection from scratch")

    # search
    search_p = sub.add_parser("search", help="Perform vector similarity search against Qdrant")
    search_p.add_argument("query", type=str, help="Search query string")
    search_p.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    search_p.add_argument("--pillar", type=str, default=None, choices=["exercise_library", "health_issues", "nutrition"], help="Filter by pillar")

    # ask
    ask_p = sub.add_parser("ask", help="Ask a question through the full RAG chain")
    ask_p.add_argument("question", type=str, help="Fitness question to ask the coach")
    ask_p.add_argument("--pillar", type=str, default=None, choices=["exercise_library", "health_issues", "nutrition"], help="Filter by pillar")
    ask_p.add_argument("--session-id", type=str, default="cli_user", help="Session ID for conversation memory")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "ask":
        cmd_ask(args)


if __name__ == "__main__":
    main()