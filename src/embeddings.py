from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    """BGE models expect no special prefix for documents but a query
    instruction prefix for queries; see embed_query below."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[0]
