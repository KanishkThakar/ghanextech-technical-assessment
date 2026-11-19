# Task 1 — Document Assistant with RAG

Upload one or more PDFs, ask questions in plain English, and get answers that
are grounded in those documents only, with page-level citations.

**Live app:** https://ghanex-rag.onrender.com

## Architecture

```
PDF upload (Streamlit file_uploader)
        │
        ▼
rag/ingest.py      pypdf → per-page text, typed errors for
                   scanned / empty / encrypted / non-PDF files
        │
        ▼
rag/chunking.py    recursive character splitting (800 / 150 overlap)
        │
        ▼
rag/embeddings.py  bge-small-en-v1.5 (fastembed), L2-normalised
        │
        ▼
rag/store.py       FAISS IndexFlatIP (exact cosine search) + chunk metadata
        │
        ▼
rag/pipeline.py    query → embed → top-k → numbered context → LLM
        │
        ▼
rag/llm.py         Groq gpt-oss-120b, strict "answer only from context" prompt
        │
        ▼
app.py             chat history, answer, expandable sources with page numbers
```

Each module has one job, so the interview questions ("what happens if a PDF is
encrypted?", "where would you add OCR?") map to a single file.

## Chunking strategy

| Parameter | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 800 characters | ~150–200 English words: a full idea, still well inside the model's 512-token window, so nothing is silently truncated at embedding time. |
| `CHUNK_OVERLAP` | 150 characters | ~20% overlap, so a sentence that straddles a boundary appears whole in one of the two chunks. |
| Separators | `\n\n`, `\n`, `. `, `? `, `! `, `; `, `, `, ` ` | Coarsest boundary first: paragraphs, then lines, then sentences, then words. Mid-word cuts only as a last resort. |

Chunks are created per page, so every chunk carries its document name and page
number and citations stay exact.

## Embedding model

`BAAI/bge-small-en-v1.5` through `fastembed`: 384 dimensions, ONNX runtime, no
PyTorch, milliseconds per chunk, and strong on passage retrieval. Skipping
PyTorch is deliberate: the app runs on a free 512 MB container and torch alone
needs more than that. Vectors are L2-normalised, which makes FAISS inner product
identical to cosine similarity. Questions are embedded with bge's retrieval
instruction prefix, passages without it, as the model was trained.

## Index refresh

`VectorStore.add_document()` rebuilds the index over the whole corpus whenever a
document is added. Rebuilding keeps chunk indices contiguous and citations
stable; at this scale it costs milliseconds.

## Setup

```bash
cd task-01-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your Groq key
streamlit run app.py
```

Get a free Groq key at https://console.groq.com/keys. The key is read from
Streamlit secrets first, then from the `GROQ_API_KEY` environment variable. It is
never committed.

## Error handling

| Case | What the user sees |
|---|---|
| Non-PDF upload | "only PDF files are supported" |
| Password-protected PDF | "this PDF is password-protected" (owner-password PDFs that open with an empty password still work) |
| Scanned / image-only PDF | "no extractable text found … does not run OCR" |
| Partly scanned PDF | warning naming how many pages were skipped |
| Missing / rejected API key | explicit message naming the secret to set |
| Rate limit (429) | "free tier is rate-limited, wait a moment and retry" |

## Known limitations

- No OCR: scanned PDFs are reported, not read. Adding `pytesseract` on pages
  with no text layer is the natural next step.
- No table or figure understanding: `pypdf` returns text only.
- One process-wide embedding model; a multi-user deploy would want a shared
  cache or a hosted embedding API.
- The index lives in memory, so a Streamlit restart clears it.
- Retrieval is pure vector search: no hybrid BM25, no re-ranking.
