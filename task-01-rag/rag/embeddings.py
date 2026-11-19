"""Embedding model.

all-MiniLM-L6-v2: ~90 MB, CPU-only, milliseconds per chunk, 384 dimensions, and
strong on passage-retrieval benchmarks. Vectors are L2-normalised so that a
FAISS inner-product search is exactly cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once per process (Streamlit reruns the script often)."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed passages -> float32 matrix (n, dim), L2-normalised."""
    if not texts:
        return np.zeros((0, DIMENSION), dtype="float32")
    vectors = get_model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def embed_query(text: str) -> np.ndarray:
    """Embed a question -> float32 vector (dim,), L2-normalised."""
    vector = get_model().encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.astype("float32")[0]
