"""Embedding model.

BAAI/bge-small-en-v1.5 through fastembed: 384 dimensions, ONNX runtime, no
PyTorch. That matters because the app is deployed on a free 512 MB container,
and torch alone needs more than that. bge-small is strong on passage retrieval
and its 512-token window comfortably holds a chunk.

Vectors are L2-normalised, so a FAISS inner-product search is exactly cosine
similarity.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384

# bge models are trained with this instruction on the question side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    """Load the model once per process (Streamlit reruns the script often)."""
    return TextEmbedding(model_name=MODEL_NAME)


def _encode(texts: List[str]) -> np.ndarray:
    """Embed texts -> float32 matrix (n, dim), L2-normalised."""
    vectors = np.array(list(get_model().embed(texts)), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed passages -> float32 matrix (n, dim), L2-normalised."""
    if not texts:
        return np.zeros((0, DIMENSION), dtype="float32")
    return _encode(texts)


def embed_query(text: str) -> np.ndarray:
    """Embed a question -> float32 vector (dim,), L2-normalised."""
    return _encode([QUERY_PREFIX + text])[0]
