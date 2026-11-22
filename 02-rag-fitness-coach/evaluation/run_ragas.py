#!/usr/bin/env python3
"""
evaluation/run_ragas.py
~~~~~~~~~~~~~~~~~~~~~~~
RAGAS evaluation harness for RAG Fitness Coach.

Metrics evaluated:
  - Faithfulness       : Is the answer grounded in the retrieved context?
  - Answer Relevancy   : Is the answer relevant to the question?
  - Context Precision  : Is the retrieved context mostly relevant?
  - Context Recall     : Does the context cover the ground-truth information?

Usage:
    python evaluation/run_ragas.py [--limit N] [--output-dir PATH]

Requirements:
    - GOOGLE_API_KEY must be set in .env (RAGAS uses an LLM judge)
    - The Qdrant fitness_docs collection must be populated (run: pipeline index)
    - Install: pip install ragas==0.1.21 datasets

The script will:
    1. Load evaluation/test_set.json
    2. For each question: retrieve + rerank + generate answer via FitnessRAGChain
    3. Collect {question, answer, contexts, ground_truth} for each sample
    4. Run ragas.evaluate() with the four metrics
    5. Save results to evaluation/results/baseline_<timestamp>.csv
    6. Print a summary table to stdout
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# ── Make sure the src/ package is importable ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Load .env before touching config ─────────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ragas")

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_SET_PATH = PROJECT_ROOT / "evaluation" / "test_set.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_test_set(path: Path) -> List[dict]:
    """Load and validate the JSON test set."""
    with open(path) as fh:
        data = json.load(fh)
    required_keys = {"question", "ground_truth", "relevant_pillar"}
    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Item {i} is missing keys: {missing}")
    logger.info("Loaded %d test samples from %s", len(data), path.name)
    return data


def get_ragas_llm():
    """
    Return a LangChain LLM wrapped for RAGAS.

    RAGAS 0.1.x requires a real LLM to score faithfulness / relevancy.
    We use ChatGoogleGenerativeAI (free tier via Google AI Studio).
    If GOOGLE_API_KEY is absent, we raise a clear error rather than silently
    producing meaningless scores.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  RAGAS requires a real LLM judge to compute metrics.        ║\n"
            "║  Please set GOOGLE_API_KEY in your .env file.               ║\n"
            "║  Get a free key at: https://aistudio.google.com/apikey      ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLM

    logger.info("Using ChatGoogleGenerativeAI as RAGAS LLM judge")
    lc_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.0,
        convert_system_message_to_human=True,
    )
    return LangchainLLM(llm=lc_llm)


def get_ragas_embeddings():
    """
    Return a RAGAS-compatible embedding wrapper using our local BGE model.
    This is used by answer_relevancy and context metrics.
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from rag_fitness_coach.vectorstore.embeddings import get_embedding_model

    logger.info("Using local BAAI/bge-small-en-v1.5 for RAGAS embeddings")
    return LangchainEmbeddingsWrapper(get_embedding_model())


def build_chain_responses(test_items: List[dict]) -> List[dict]:
    """
    Call FitnessRAGChain.ask() for every test question.

    Returns a list of dicts with keys:
        question, answer, contexts, ground_truth, pillar
    """
    from rag_fitness_coach.chain.rag_chain import FitnessRAGChain

    chain = FitnessRAGChain()
    records: List[dict] = []

    for idx, item in enumerate(test_items, 1):
        q = item["question"]
        gt = item["ground_truth"]
        pillar = item["relevant_pillar"]

        logger.info("[%d/%d] %-60s", idx, len(test_items), q[:60])

        try:
            result = chain.ask(question=q, session_id=f"ragas_eval_{idx}")
            answer = result["answer"]
            contexts = [src["content"] for src in result.get("sources", [])]

            if not contexts:
                logger.warning("  → No contexts retrieved for question %d", idx)

            records.append(
                {
                    "question": q,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": gt,
                    "pillar": pillar,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("  → Error for question %d: %s", idx, exc)
            # Include a placeholder so dataset row count stays aligned
            records.append(
                {
                    "question": q,
                    "answer": f"ERROR: {exc}",
                    "contexts": [],
                    "ground_truth": gt,
                    "pillar": pillar,
                }
            )

    return records


def run_evaluation(
    records: List[dict],
    ragas_llm,
    ragas_embeddings,
) -> dict:
    """
    Build a HuggingFace Dataset and call ragas.evaluate().
    Returns the result dict (metric → score).
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    ds = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"] if r["contexts"] else ["No context retrieved"],
                "ground_truth": r["ground_truth"],
            }
            for r in records
        ]
    )

    logger.info("Running RAGAS evaluation on %d samples …", len(ds))
    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )
    return result


def save_results(result, records: List[dict], output_dir: Path) -> Path:
    """
    Save evaluation results to a CSV file.
    Includes per-sample metric scores plus aggregate mean row.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"baseline_{timestamp}.csv"

    # RAGAS result can be converted to a pandas DataFrame
    try:
        df = result.to_pandas()
        # Add pillar column
        df.insert(0, "pillar", [r["pillar"] for r in records[: len(df)]])
        df.to_csv(csv_path, index=False)
        logger.info("Saved per-sample scores to %s", csv_path)
    except Exception:  # noqa: BLE001
        # Fallback: write aggregate scores only
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["metric", "score"])
            writer.writeheader()
            for m in metric_cols:
                writer.writerow({"metric": m, "score": result.get(m, "N/A")})
        logger.info("Saved aggregate scores to %s", csv_path)

    return csv_path


def print_summary(result) -> None:
    """Pretty-print the aggregate metric scores."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "=" * 60)
    print("  RAG FITNESS COACH — RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    for metric in metrics:
        score = result.get(metric, None)
        if score is None:
            score_str = "N/A"
            bar = ""
        else:
            score_str = f"{score:.4f}"
            filled = int(score * 30)
            bar = "█" * filled + "░" * (30 - filled)
        label = metric.replace("_", " ").title()
        print(f"  {label:<22} {score_str}  {bar}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the RAG Fitness Coach pipeline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only evaluate the first N questions (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        metavar="PATH",
        help=f"Directory for CSV result files (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        default=False,
        help="Skip local embedding model for RAGAS (answer_relevancy may be less accurate).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load test set ─────────────────────────────────────────────────────
    test_items = load_test_set(TEST_SET_PATH)
    if args.limit:
        test_items = test_items[: args.limit]
        logger.info("Limiting evaluation to first %d samples", args.limit)

    # 2. Validate that a real LLM is available ─────────────────────────────
    ragas_llm = get_ragas_llm()  # Raises ValueError with friendly msg if key missing

    # 3. Prepare embeddings ────────────────────────────────────────────────
    ragas_embeddings = None if args.no_embeddings else get_ragas_embeddings()

    # 4. Generate chain responses ──────────────────────────────────────────
    logger.info("Generating answers from FitnessRAGChain …")
    records = build_chain_responses(test_items)
    logger.info("Collected %d records", len(records))

    # 5. Run RAGAS evaluation ──────────────────────────────────────────────
    result = run_evaluation(records, ragas_llm, ragas_embeddings)

    # 6. Print summary ─────────────────────────────────────────────────────
    print_summary(result)

    # 7. Save CSV ──────────────────────────────────────────────────────────
    csv_path = save_results(result, records, args.output_dir)
    print(f"  Results saved → {csv_path}\n")


if __name__ == "__main__":
    main()
