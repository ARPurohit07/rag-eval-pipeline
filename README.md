# RAG Pipeline with RAGAS Evaluation

A retrieval-augmented Q&A system with a measured evaluation harness, not just a demo.
Given a corpus of technical documents, it answers questions by combining **hybrid
BM25 + semantic search** over a **pgvector** store, generating answers with a local
**Ollama** model, and scoring the whole pipeline with **RAGAS** (faithfulness, answer
relevancy, context precision, context recall) so quality claims are backed by numbers
instead of vibes.

## Why this exists

Most RAG tutorials stop at "it retrieves something and the LLM answers." The harder
and more interesting problem is: *how do you know if it's actually good, and which
part of the pipeline is responsible when it isn't?* This project treats retrieval and
generation as separately measurable stages, and includes the eval harness as a
first-class part of the system rather than a one-off notebook.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              INGESTION (offline)          │
                    │                                            │
  data/corpus/*.md ─▶  chunk (recursive, token-aware, overlap)   │
                    │        │                                   │
                    │        ▼                                   │
                    │  embed (sentence-transformers, bge-small)  │
                    │        │                                   │
                    │        ▼                                   │
                    │  store in Postgres (pgvector, HNSW index)  │
                    └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │               QUERY (online)               │
                    │                                            │
        question ──▶│  ┌────────────┐      ┌──────────────────┐ │
                    │  │ BM25 search │      │ pgvector cosine   │ │
                    │  │ (in-memory) │      │ similarity search │ │
                    │  └──────┬─────┘      └─────────┬────────┘ │
                    │         └─────────┬────────────┘          │
                    │                   ▼                       │
                    │      Reciprocal Rank Fusion (RRF)          │
                    │                   │                        │
                    │                   ▼                        │
                    │      top-k chunks → prompt → Ollama LLM    │
                    │                   │                        │
                    │                   ▼                        │
        answer ◀────┤              generated answer               │
                    └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │            EVALUATION (offline)            │
                    │                                            │
    eval_dataset ──▶│  run pipeline on each Q  →  RAGAS scoring  │
    (Q + reference)  │  (LLM-as-judge over question/answer/      │
                    │   retrieved context/reference)             │
                    │                   │                        │
                    │                   ▼                        │
                    │   faithfulness · answer relevancy ·        │
                    │   context precision · context recall       │
                    └─────────────────────────────────────────┘
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| Vector store | PostgreSQL + `pgvector` (Docker) | One database for vectors + metadata, SQL joins/filters alongside ANN search, no separate vector DB service to run |
| Sparse retrieval | BM25 (`rank_bm25`, in-memory) | Exact-term matching that dense embeddings miss (identifiers, jargon, acronyms) |
| Fusion | Reciprocal Rank Fusion | Combines BM25 + vector rankings without needing to normalize incomparable score scales |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) | Local, free, fast; swappable via `.env` |
| Generation | Ollama (local or [Ollama Cloud](https://ollama.com) models) | Runs fully locally with no API key; same client also proxies to larger cloud-hosted models once signed in |
| Evaluation | [RAGAS](https://github.com/explodinggremlins/ragas) | Decomposes RAG quality into retrieval metrics (precision/recall) vs. generation metrics (faithfulness/relevancy) instead of one opaque end-to-end score |
| Apps | Streamlit (chat + eval dashboard), FastAPI (`/query` endpoint) | Two interfaces over the same `src/pipeline.py` core |

## Corpus

19 original technical documents (`data/corpus/`) covering Transformers, attention,
embeddings, vector databases, RAG, BM25, hybrid search, chunking, fine-tuning vs.
RAG vs. prompting, quantization, RLHF/DPO, LLM evaluation, hallucination, VLMs,
LangGraph, MCP, PostgreSQL internals, and knowledge distillation. Topics were chosen
to include semantically close pairs (BM25 vs. vector search, fine-tuning vs. RAG vs.
prompting, quantization vs. distillation) specifically so retrieval quality is
actually being tested, not just topic classification.

`eval/eval_dataset.json` has 100 hand-written question/reference-answer pairs against
this corpus, including two deliberately out-of-corpus questions to check that the
system says "I don't know" instead of hallucinating.

## Setup

### 1. Prerequisites
- Python 3.11+
- Docker Desktop
- [Ollama](https://ollama.com) installed, with at least one model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```
  For higher-quality evaluation, sign in to Ollama Cloud (`ollama signin`) and use a
  `-cloud` suffixed model (e.g. `gpt-oss:120b-cloud`) as the RAGAS judge — see
  `RAGAS_JUDGE_MODEL` below.

### 2. Install
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # then edit if you want non-default settings
```

### 3. Start Postgres + pgvector
```bash
docker compose up -d
```

### 4. Ingest the corpus
```bash
python -m src.ingest
```
This chunks every file in `data/corpus/`, embeds each chunk, and stores it in
Postgres. Re-running is idempotent — already-ingested documents are skipped. Pass
`--reset` to drop and rebuild the schema from scratch.

### 5. Ask questions
```bash
streamlit run app.py
```
or hit the API directly:
```bash
uvicorn api.main:app --reload
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How does Reciprocal Rank Fusion combine BM25 and vector search?"}'
```

### 6. Run the evaluation
```bash
python -m eval.run_eval
```
This runs the full pipeline over every question in `eval/eval_dataset.json`, scores
the results with RAGAS, and writes `eval/results/summary_<timestamp>.json` plus a
per-question CSV. The Streamlit app's "Evaluation Dashboard" tab reads the latest
summary automatically. Useful flags:
```bash
python -m eval.run_eval --top-k 3       # change retrieved chunk count
python -m eval.run_eval --limit 5       # quick smoke test on a subset
```

## Configuration

All tunables live in `.env` (see `.env.example`): chunk size/overlap, retrieval
`top_k`, BM25 vs. vector fusion weights, embedding model, and which Ollama model
answers questions vs. which one judges RAGAS metrics. Using a stronger model as the
judge than the one being evaluated is a deliberate, documented RAGAS best practice —
judge quality bounds metric reliability.

## Results

Latest run against all 100 eval questions, `top_k=5`, generation model `qwen2.5:7b`
(local), judge model `gpt-oss:120b-cloud` (Ollama Cloud):

| Metric | Score |
|---|---|
| Faithfulness | 0.932 |
| Answer Relevancy | 0.920 |
| Context Precision | 0.928 |
| Context Recall | 0.970 |

Context recall of 0.970 across 100 questions (versus a perfect 1.0 on an earlier, smaller
32-question run) is the more credible number — at 3x the eval set size, a few genuine
retrieval gaps surface that a smaller sample was too small to catch, which is exactly
what a bigger, harder eval set is supposed to reveal. Context precision of 0.928 shows
the top-5 chunks were mostly, but not perfectly, on-topic. Faithfulness of 0.932 means a
small fraction of generated claims weren't directly traceable to the retrieved context
even though the right context was present — still the single most informative failure
mode to chase next, since it's a generation-stage issue rather than a retrieval-stage
one. Re-run `python -m eval.run_eval` after any change to chunking, `top_k`, fusion
weights, or the generation model to see how these four numbers move independently —
that independence is the point of decomposing the metrics this way.

Raw per-question output: `eval/results/ragas_results_20260916T135234Z.csv`.

## Deploying it live

This runs entirely locally for now (Postgres in Docker, Ollama on localhost), which
a public Streamlit Cloud or Hugging Face Space cannot run as-is. To deploy publicly
later:
1. Swap the vector store for a hosted Postgres+pgvector instance (e.g. a free
   Supabase project) and point `POSTGRES_*` in `.env`/Streamlit secrets at it.
2. Swap `src/llm.py`'s Ollama client for a hosted LLM API (Groq's free tier is a
   reasonable default with no billing setup), since the platform can't run Ollama.
3. Push this repo to GitHub and connect it from Streamlit Community Cloud (or create
   a Streamlit-SDK Hugging Face Space), setting the swapped credentials as secrets.

## Project layout

```
rag-eval-pipeline/
├── data/corpus/          # source documents (19 authored .md files)
├── src/
│   ├── config.py         # pydantic-settings, reads .env
│   ├── db.py              # pgvector schema + connection
│   ├── chunking.py        # recursive token-aware text splitting
│   ├── embeddings.py      # sentence-transformers wrapper
│   ├── ingest.py          # corpus -> chunks -> embeddings -> Postgres
│   ├── retrieval.py       # BM25 + pgvector hybrid search, RRF fusion
│   ├── llm.py              # Ollama chat wrapper + grounded system prompt
│   └── pipeline.py        # retrieve -> generate orchestration
├── eval/
│   ├── eval_dataset.json  # 100 hand-written Q/reference-answer pairs
│   ├── run_eval.py         # runs pipeline + scores with RAGAS
│   └── results/            # generated summaries/CSVs (tracked - real eval output)
├── app.py                 # Streamlit: chat + eval dashboard
├── api/main.py             # FastAPI: /query endpoint
├── tests/                  # unit tests (chunking, RRF fusion math)
└── docker-compose.yml      # Postgres + pgvector
```

## Notes on design choices

- **Why RRF instead of weighted score blending**: BM25 and cosine-similarity scores
  live on different, incomparable scales. RRF sidesteps normalization entirely by
  fusing on rank position, which is more robust without a labeled tuning set.
- **Why pgvector over a dedicated vector DB**: this corpus is small enough that
  pgvector's main advantage — one database, SQL joins between vector search and
  relational metadata, existing backup/ops tooling — outweighs the extra scale
  headroom a dedicated vector DB offers.
- **Why decompose evaluation into 4 RAGAS metrics instead of one score**: a RAG
  system can fail at retrieval (low context precision/recall) or at generation
  (low faithfulness/relevancy) independently. Reporting one blended number would
  hide which stage needs work.
