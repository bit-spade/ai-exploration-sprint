# RAG Fitness Coach 🏋️

> **Project 02 of the AI Exploration Sprint** — A RAG-powered fitness coaching assistant grounded in real sports science documents. It retrieves relevant evidence, cites its sources, and refuses to hallucinate.

---

## Architecture

```
User → Streamlit UI → FitnessRAGChain (LangChain LCEL)
                              ↓
              Qdrant (local embedded disk mode)
              BAAI/bge-small-en-v1.5 (sentence-transformers, MPS)
              ms-marco-MiniLM-L-6-v2 (cross-encoder reranker)
              Gemini 1.5 Flash (Google AI Studio free API)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | LangChain + LCEL |
| LLM | Gemini 1.5 Flash (Google AI Studio free tier) |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers (local, MPS-accelerated) |
| Vector DB | Qdrant (local embedded disk mode — no Docker required) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Backend | FastAPI + SSE streaming |
| Frontend | Streamlit |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision, context recall) |

## Knowledge Base

Three pillars — all open-access, CC-licensed sources:

- **Exercise library** — muscle groups, exercise purpose & form
- **Health-issue exercises** — rehab, chronic conditions, prevention
- **Nutrition** — macro guidance, meal timing, evidence-based diet

960 chunks → 958 unique vectors ingested into Qdrant.

## Setup

### Prerequisites
- Python 3.11+ (tested on 3.13.5)
- pip
- Free [Gemini API key](https://aistudio.google.com/apikey) from Google AI Studio

### 1. Clone & install dependencies

```bash
cd 02-rag-fitness-coach
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Ingest documents & index into Qdrant

```bash
python -m rag_fitness_coach.ingestion.pipeline ingest
python -m rag_fitness_coach.ingestion.pipeline index
```

### 4. Run the API + UI

```bash
# Terminal 1
uvicorn rag_fitness_coach.api.main:app --reload

# Terminal 2
streamlit run src/rag_fitness_coach/ui/app.py
```

### 5. (Optional) Run RAGAS evaluation

```bash
# Requires GOOGLE_API_KEY to be set
python evaluation/run_ragas.py

# Then open the analysis notebook
jupyter notebook notebooks/03_ragas_analysis.ipynb
```

## Project Structure

```
02-rag-fitness-coach/
├── data/raw/              # Source PDFs (gitignored)
├── data/processed/        # Chunked JSONL (gitignored)
├── data/qdrant_storage/   # Embedded Qdrant DB (gitignored)
├── docs/sources.md        # Document index with licenses
├── evaluation/
│   ├── test_set.json      # 20 Q&A benchmark samples (3 pillars)
│   ├── run_ragas.py       # RAGAS evaluation harness
│   └── results/           # Timestamped CSV result files
├── notebooks/
│   ├── 01_chunking_experiments.ipynb
│   ├── 02_retrieval_debug.ipynb
│   └── 03_ragas_analysis.ipynb
├── src/rag_fitness_coach/
│   ├── config.py          # Pydantic Settings
│   ├── ingestion/         # Loader, chunker, pipeline CLI
│   ├── vectorstore/       # Qdrant client + indexer
│   ├── retrieval/         # Retriever + cross-encoder reranker
│   ├── chain/             # LCEL RAG chain + memory
│   ├── api/               # FastAPI backend (SSE streaming)
│   └── ui/                # Streamlit chat frontend
└── tests/                 # 25 passing tests
```

## Milestones

- [x] **M1** — Project setup + ingestion pipeline (960 chunks, 3 pillars)
- [x] **M2** — Embedding + Qdrant vector store (958 unique vectors)
- [x] **M3** — Retrieval chain + reranking (BGE + ms-marco cross-encoder)
- [x] **M4** — FastAPI streaming backend (SSE)
- [x] **M5** — Streamlit chat UI with citation expanders
- [x] **M6** — RAGAS evaluation harness (faithfulness / relevancy / precision / recall)
- [ ] **M7** — Personal coaching via push-up tracker data

## Evaluation

Run the full evaluation after setting `GOOGLE_API_KEY`:

```bash
python evaluation/run_ragas.py
# Outputs: evaluation/results/baseline_<timestamp>.csv
```

| Metric | Description | Target |
|--------|-------------|--------|
| **Faithfulness** | Are answers grounded in retrieved context? | ≥ 0.8 |
| **Answer Relevancy** | Does the answer address the question? | ≥ 0.8 |
| **Context Precision** | What fraction of retrieved chunks were relevant? | ≥ 0.7 |
| **Context Recall** | Does context cover ground-truth information? | ≥ 0.7 |

*Results visualised in `notebooks/03_ragas_analysis.ipynb`.*

## Tests

```bash
.venv/bin/pytest tests/ -v
# 25 tests passing (M1–M5 coverage)
```

## Related

- [Project 01 — Real-time Push-up Tracker](../01-computer-vision-tracker/)
