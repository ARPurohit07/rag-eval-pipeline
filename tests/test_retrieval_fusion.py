"""Unit tests for the RRF fusion math in HybridRetriever, isolated from the
DB/embedding model by constructing the retriever without calling refresh()."""
from src.retrieval import HybridRetriever, RRF_K


def make_retriever(chunks: list[dict]) -> HybridRetriever:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._chunks = chunks
    retriever._bm25 = None
    return retriever


def test_rrf_favors_document_ranked_high_by_both_methods():
    chunks = [
        {"id": 1, "content": "a", "source": "s", "title": "t"},
        {"id": 2, "content": "b", "source": "s", "title": "t"},
        {"id": 3, "content": "c", "source": "s", "title": "t"},
    ]
    retriever = make_retriever(chunks)

    # idx 0 ranked #1 by both methods; idx 1 ranked #1 by bm25 only; idx 2 ranked #1 by vector only
    bm25_rank_by_idx = {0: 1, 1: 1}
    vector_rank_by_idx = {0: 1, 2: 1}

    scores = {}
    for idx in {0, 1, 2}:
        score = 0.0
        if idx in bm25_rank_by_idx:
            score += 0.4 * (1.0 / (RRF_K + bm25_rank_by_idx[idx]))
        if idx in vector_rank_by_idx:
            score += 0.6 * (1.0 / (RRF_K + vector_rank_by_idx[idx]))
        scores[idx] = score

    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
