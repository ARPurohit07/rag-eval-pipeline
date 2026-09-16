import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.config import settings
from src.db import get_connection
from src.embeddings import embed_query

RRF_K = 60
CANDIDATE_POOL = 20

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: int
    content: str
    source: str
    title: str
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None


class HybridRetriever:
    """BM25 (sparse, in-memory) + pgvector cosine similarity (dense),
    fused with Reciprocal Rank Fusion so the two incomparable score
    scales never need to be normalized against each other."""

    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self.refresh()

    def refresh(self) -> None:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.content, d.source, d.title
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.id
                """
            ).fetchall()
        self._chunks = [
            {"id": r[0], "content": r[1], "source": r[2], "title": r[3]} for r in rows
        ]
        corpus_tokens = [_tokenize(c["content"]) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def _bm25_ranking(self, query: str, pool: int) -> list[tuple[int, float]]:
        """Returns [(chunk_index_in_self._chunks, score), ...] sorted desc."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:pool] if score > 0]

    def _vector_ranking(self, query: str, pool: int) -> list[tuple[int, float]]:
        """Returns [(chunk_id, distance), ...] sorted by ascending cosine distance."""
        query_vec = embed_query(query)
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, embedding <=> %s AS distance
                FROM chunks
                ORDER BY distance ASC
                LIMIT %s
                """,
                (query_vec, pool),
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k
        id_to_idx = {c["id"]: i for i, c in enumerate(self._chunks)}

        bm25_ranked = self._bm25_ranking(query, CANDIDATE_POOL)
        bm25_rank_by_idx = {idx: rank for rank, (idx, _) in enumerate(bm25_ranked, start=1)}

        vector_ranked = self._vector_ranking(query, CANDIDATE_POOL)
        vector_rank_by_idx = {
            id_to_idx[chunk_id]: rank
            for rank, (chunk_id, _) in enumerate(vector_ranked, start=1)
            if chunk_id in id_to_idx
        }

        all_idxs = set(bm25_rank_by_idx) | set(vector_rank_by_idx)
        fused: list[RetrievedChunk] = []
        for idx in all_idxs:
            bm25_rank = bm25_rank_by_idx.get(idx)
            vector_rank = vector_rank_by_idx.get(idx)
            rrf_score = 0.0
            if bm25_rank is not None:
                rrf_score += settings.bm25_weight * (1.0 / (RRF_K + bm25_rank))
            if vector_rank is not None:
                rrf_score += settings.vector_weight * (1.0 / (RRF_K + vector_rank))
            chunk = self._chunks[idx]
            fused.append(
                RetrievedChunk(
                    chunk_id=chunk["id"],
                    content=chunk["content"],
                    source=chunk["source"],
                    title=chunk["title"],
                    score=rrf_score,
                    bm25_rank=bm25_rank,
                    vector_rank=vector_rank,
                )
            )

        fused.sort(key=lambda c: c.score, reverse=True)
        return fused[:top_k]
