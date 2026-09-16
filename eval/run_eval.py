"""Runs the RAG pipeline over eval/eval_dataset.json and scores it with RAGAS
(faithfulness, answer relevancy, context precision, context recall).

Usage:
    python -m eval.run_eval [--top-k 5] [--limit 10]

Requires: Postgres running with the corpus already ingested (src/ingest.py),
and Ollama running locally with both the answering model and the judge
model (settings.ollama_model / settings.ragas_judge_model) available.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

from src.config import settings
from src.pipeline import RAGPipeline

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_eval_dataset(limit: int | None = None) -> list[dict]:
    data = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))
    return data[:limit] if limit else data


def run_pipeline_over_dataset(top_k: int, limit: int | None) -> list[dict]:
    pipeline = RAGPipeline()
    eval_items = load_eval_dataset(limit)
    samples = []
    for i, item in enumerate(eval_items, start=1):
        print(f"[{i}/{len(eval_items)}] {item['question'][:70]}...")
        result = pipeline.answer(item["question"], top_k=top_k)
        samples.append(
            {
                "user_input": item["question"],
                "response": result.answer,
                "retrieved_contexts": [c.content for c in result.contexts],
                "reference": item["ground_truth"],
            }
        )
    return samples


def score_with_ragas(samples: list[dict]) -> "pd.DataFrame":
    judge_llm = LangchainLLMWrapper(
        ChatOllama(base_url=settings.ollama_host, model=settings.ragas_judge_model, temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    dataset = EvaluationDataset.from_list(samples)
    # Ollama Cloud rate-limits concurrent requests; the ragas default of 16
    # parallel judge calls trips "too many concurrent requests" errors, so
    # cap concurrency and give retries more room to back off.
    run_config = RunConfig(max_workers=4, timeout=180, max_retries=15, max_wait=30)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )
    return result.to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Running RAG pipeline (top_k={args.top_k}) over eval set...")
    samples = run_pipeline_over_dataset(args.top_k, args.limit)

    print(f"Scoring {len(samples)} samples with RAGAS (judge={settings.ragas_judge_model})...")
    df = score_with_ragas(samples)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = RESULTS_DIR / f"ragas_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available_cols = [c for c in metric_cols if c in df.columns]
    summary = {c: round(float(df[c].mean()), 4) for c in available_cols}
    summary["n_samples"] = len(df)
    summary["top_k"] = args.top_k
    summary["judge_model"] = settings.ragas_judge_model
    summary["generation_model"] = settings.ollama_model
    summary["timestamp"] = timestamp

    summary_path = RESULTS_DIR / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest_path = RESULTS_DIR / "latest_summary.json"
    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== RAGAS Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nFull per-question results: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
