"""Vector store.

FAISS `IndexFlatIP` over L2-normalised vectors is an exact cosine-similarity
search: no training step, no approximation, and instant at the scale of a few
PDFs.

The index is rebuilt whenever the corpus changes. Rebuilding instead of
appending keeps chunk indices contiguous and citations stable, which matters
more here than the few milliseconds an append would save.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import faiss
import numpy as np

from .chunking import Chunk
from .embeddings import embed_texts


@dataclass
class RetrievedChunk:
    """A chunk plus its cosine similarity score."""

    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    """In-memory FAISS index with the chunk metadata kept beside it."""

    chunks: List[Chunk] = field(default_factory=list)
    index: object | None = None
    documents: Dict[str, int] = field(default_factory=dict)  # name -> chunk count

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def build(self, chunks: List[Chunk]) -> None:
        """(Re)build the index from scratch for exactly these chunks."""
        self.chunks = chunks
        self.documents = {}
        for chunk in chunks:
            self.documents[chunk.doc_name] = self.documents.get(chunk.doc_name, 0) + 1

        if not chunks:
            self.index = None
            return

        vectors = embed_texts([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index = index

    def add_document(self, new_chunks: List[Chunk]) -> None:
        """Add one document's chunks and refresh the index."""
        self.build(self.chunks + new_chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[RetrievedChunk]:
        """Return the top-k most similar chunks, best first."""
        if self.is_empty:
            return []
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than k results exist
                continue
            results.append(RetrievedChunk(chunk=self.chunks[int(idx)], score=float(score)))
        return results
