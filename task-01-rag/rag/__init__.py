"""RAG document assistant.

ingest      PDF -> per-page text, with typed errors for the awkward cases
chunking    page text -> overlapping chunks that carry their citation
embeddings  bge-small-en-v1.5, L2-normalised
store       FAISS IndexFlatIP (exact cosine) + chunk metadata
llm         Groq chat completion with a strict "answer only from context" prompt
pipeline    the whole flow, owned by one object the UI talks to
"""
