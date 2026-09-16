import json
from pathlib import Path

import streamlit as st

from src.config import settings
from src.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Pipeline + Eval", page_icon="🔎", layout="wide")

RESULTS_DIR = Path(__file__).parent / "eval" / "results"


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def render_chat_tab() -> None:
    st.subheader("Ask a question over the corpus")
    st.caption(
        "Corpus: 19 authored technical documents on Transformers, RAG, retrieval, "
        "evaluation, and adjacent ML/LLM-systems topics. Retrieval is hybrid "
        "BM25 + pgvector cosine similarity, fused with Reciprocal Rank Fusion."
    )

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=settings.retrieval_top_k)
        model = st.text_input("Ollama model", value=settings.ollama_model)
        st.caption(f"Ollama host: `{settings.ollama_host}`")

    question = st.text_input("Question", placeholder="Why does RRF not require score normalization?")
    if st.button("Ask", type="primary") and question:
        try:
            pipeline = get_pipeline()
        except Exception as e:
            st.error(f"Could not connect to the retriever/DB: {e}")
            return

        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = pipeline.answer(question, top_k=top_k, model=model)
            except Exception as e:
                st.error(f"Generation failed (is Ollama running with model '{model}'?): {e}")
                return

        st.markdown("### Answer")
        st.write(result.answer)

        st.markdown("### Retrieved context")
        for i, chunk in enumerate(result.contexts, start=1):
            with st.expander(f"[{i}] {chunk.title} (score={chunk.score:.4f}, bm25_rank={chunk.bm25_rank}, vector_rank={chunk.vector_rank})"):
                st.write(chunk.content)


def render_eval_tab() -> None:
    st.subheader("RAGAS Evaluation Results")
    st.caption(
        "Run `python -m eval.run_eval` to score the pipeline against eval/eval_dataset.json "
        "and populate this dashboard."
    )

    latest = RESULTS_DIR / "latest_summary.json"
    if not latest.exists():
        st.info("No evaluation results yet. Run the eval script from the project root first.")
        return

    summary = json.loads(latest.read_text(encoding="utf-8"))

    metric_cols = st.columns(4)
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }
    for col, name in zip(metric_cols, metric_names):
        if name in summary:
            col.metric(labels[name], f"{summary[name]:.3f}")

    st.markdown("---")
    st.markdown(
        f"**Samples evaluated:** {summary.get('n_samples')}  \n"
        f"**top_k:** {summary.get('top_k')}  \n"
        f"**Generation model:** `{summary.get('generation_model')}`  \n"
        f"**Judge model:** `{summary.get('judge_model')}`  \n"
        f"**Run timestamp (UTC):** {summary.get('timestamp')}"
    )

    csv_candidates = sorted(RESULTS_DIR.glob("ragas_results_*.csv"), reverse=True)
    if csv_candidates:
        import pandas as pd

        st.markdown("### Per-question breakdown")
        df = pd.read_csv(csv_candidates[0])
        display_cols = [c for c in ["user_input", *metric_names] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)


st.title("RAG Pipeline with RAGAS Evaluation")

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation Dashboard"])
with tab_chat:
    render_chat_tab()
with tab_eval:
    render_eval_tab()
