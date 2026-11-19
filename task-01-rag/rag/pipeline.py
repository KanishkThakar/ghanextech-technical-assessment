"""The RAG pipeline, end to end.

ingest -> chunk -> embed -> index -> retrieve -> grounded answer.

`RagPipeline` owns the vector store and the loaded-document bookkeeping, so the
Streamlit layer only has to call `add_pdf` and `ask`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_pages
from .embeddings import embed_query
from .ingest import IngestError, ingest_pdf
from .llm import answer_question
from .store import RetrievedChunk, VectorStore


@dataclass
class Answer:
    """One answered question, with the passages it was grounded in."""

    text: str
    sources: List[RetrievedChunk] = field(default_factory=list)


class RagPipeline:
    """Holds the corpus for one browser session."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store = VectorStore()
        self.loaded_documents: List[str] = []
        self.warnings: List[str] = []

    @property
    def has_documents(self) -> bool:
        return not self.store.is_empty

    def add_pdf(self, data: bytes, doc_name: str) -> int:
        """Ingest one PDF and refresh the index. Returns the chunks added."""
        if doc_name in self.loaded_documents:
            raise IngestError(f"{doc_name} is already loaded.")

        result = ingest_pdf(data, doc_name)
        if result.warning:
            self.warnings.append(result.warning)

        chunks = chunk_pages(result.pages, self.chunk_size, self.chunk_overlap)
        self.store.add_document(chunks)
        self.loaded_documents.append(doc_name)
        return len(chunks)

    def ask(self, question: str, top_k: int = 4, history: List[dict] | None = None) -> Answer:
        """Retrieve, then answer from the retrieved passages only."""
        if not self.has_documents:
            raise IngestError("Upload at least one PDF before asking a question.")

        retrieved = self.store.search(embed_query(question), top_k=top_k)
        if not retrieved:
            raise IngestError("Nothing was retrieved for that question.")

        text = answer_question(question, [r.chunk.text for r in retrieved], history)
        return Answer(text=text, sources=retrieved)

    def reset(self) -> None:
        self.store.build([])
        self.loaded_documents = []
        self.warnings = []
