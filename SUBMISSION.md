# Internship Assessment Submission - Kanishk Thakar

## Candidate Information

- **Full Name:** Kanishk Thakar
- **Email Address:** kanishkthakar@gmail.com
- **GitHub Profile URL:** https://github.com/KanishkThakar
- **Portfolio:** https://kanishk-thakar-portfolio.vercel.app/
- **Submission Date:** 20 November 2026

All three tasks live in one mono-repo: https://github.com/KanishkThakar/ghanextech-technical-assessment

Hosting note: every app runs on Render's free tier, which sleeps after 15 minutes without traffic. The first visit after a pause can take 30-60 seconds to wake the container; after that it responds normally.

---

## Task Deliverables

### Task 1: AI Document Assistant (RAG)

- **GitHub Repository URL:** https://github.com/KanishkThakar/ghanextech-technical-assessment/tree/main/task-01-rag
- **Live Deployed Application URL:** https://ghanex-rag.onrender.com
- **Primary Technologies Used:** Python 3.12, Streamlit, pypdf, fastembed (BAAI/bge-small-en-v1.5, ONNX), FAISS, Groq API (openai/gpt-oss-120b)

- **Project Summary:**

  The assistant takes one or more PDFs, extracts text page by page with pypdf, splits each page into overlapping chunks, embeds them and stores them in an in-memory FAISS index. A question is embedded the same way, the top-k most similar chunks are retrieved, numbered and passed to the LLM with a strict instruction to answer only from that context and to say so plainly when the documents do not contain the answer. Every answer is shown with its sources (document name, page and chunk) in an expandable panel, and the chat keeps its history across turns.

  Chunking is recursive character splitting at 800 characters with 150 characters of overlap (about 20%). The splitter tries the coarsest boundary first (paragraph, line, sentence, then word), so chunks hold complete ideas and a sentence that crosses a boundary appears whole in at least one chunk. Chunks are created per page so each one carries an exact page number for citation. Adding a document rebuilds the index over the whole corpus, which keeps chunk ids contiguous and citations stable.

  Embeddings use BAAI/bge-small-en-v1.5 through fastembed: 384 dimensions, strong on passage retrieval, and served by ONNX runtime rather than PyTorch. Vectors are L2-normalised so FAISS inner-product search is exact cosine similarity, and queries carry bge's retrieval instruction prefix as the model was trained. Ingestion reports non-PDF uploads, password-protected files and scanned or image-only PDFs with clear messages, and warns when only some pages had extractable text.

- **Key Challenges & Solutions:**
  - *Memory on the free tier.* The first version used sentence-transformers (all-MiniLM-L6-v2), which pulls in PyTorch and exceeds a 512 MB container. Switching to fastembed's ONNX build of bge-small kept retrieval quality and brought the app well inside the limit.
  - *Native wheels on the host's default Python.* Render defaults to a Python version for which faiss-cpu 1.9 has no wheel. Pinning `PYTHON_VERSION=3.12.7` gave reproducible installs.
  - *Grounding.* Early answers occasionally filled gaps from general knowledge. The prompt now numbers each chunk, requires citations by number and gives an explicit fallback sentence when the context lacks the answer.
  - *Cold starts.* A sleeping container can drop the first websocket during upload; the app reloads cleanly and the README documents the behaviour.

---

### Task 2: AI Agent with Tool Calling

- **GitHub Repository URL:** https://github.com/KanishkThakar/ghanextech-technical-assessment/tree/main/task-02-agent
- **Live Deployed Application URL:** https://ghanex-kt-agent.onrender.com
- **Primary Technologies Used:** Python 3.12, Streamlit, Groq chat-completions API with native function calling (openai/gpt-oss-120b), requests, Open-Meteo, Wikipedia REST API, Frankfurter (ECB rates)

- **Implemented Tools:**
  1. **calculator** (`expression: str`): safe arithmetic. The expression is parsed with Python's `ast` and walked with a whitelist of operators, constants and maths functions; no `eval()`, and exponents are capped so input like `9**9**9` cannot hang the process.
  2. **weather** (`location: str`, `days: int = 3`): Open-Meteo geocoding turns the place name into coordinates, then the forecast endpoint returns current conditions and the next days, with WMO weather codes mapped to plain English.
  3. **wikipedia** (`query: str`): searches the public Wikipedia REST API and returns the top article's title, summary and link so the model can cite it.
  4. **currency** (`amount: float`, `from_currency: str`, `to_currency: str`): converts using Frankfurter's free ECB reference rates; unknown currency codes come back as a clear error.

- **Project Summary:**

  The agent is split into three layers. `agent/tools/` holds one module per tool, each a frozen dataclass with its name, the description the model reads, a JSON Schema for its arguments and the function itself, so the schema can never drift from the code. `agent/registry.py` hands the model the list of schemas and dispatches a returned call by name, validating that the tool exists and that the arguments are a JSON object. `agent/loop.py` is the reasoning loop: send the transcript and schemas, and if the reply contains tool calls, run each one, append one `tool` message per result and ask again; when the reply has no tool calls it is the final answer.

  The model decides on its own whether to answer directly, call one tool, or call several in one turn (for example weather and a currency conversion together). The loop is capped at six steps so a model that keeps calling tools cannot spin forever. The Streamlit UI keeps the full multi-turn history in session state and shows, for every turn, which tools ran, with what arguments and what they returned.

  Error handling follows one contract: a tool raises `ToolError` for anything recoverable (place not found, unknown currency, invalid expression), and the registry also catches unexpected exceptions. Either way the error is returned to the model as the tool's result, so it can retry with different arguments or explain the problem to the user. A failing tool never breaks the conversation. Missing API keys and provider rate limits are reported in the UI with the action needed.

- **Key Challenges & Solutions:**
  - *Model availability.* The model originally planned (llama-3.3-70b) was not served for this key. I tested the available Groq models for tool-call and JSON reliability and moved to openai/gpt-oss-120b, configurable through `GROQ_MODEL`.
  - *Unsafe evaluation.* A calculator built on `eval()` would execute arbitrary code; the AST walker with a whitelist removes that risk and still supports functions such as `sqrt` and `log`.
  - *Runaway tool loops.* Without a limit a confused model could call tools indefinitely; the step cap plus returning errors as tool results keeps every turn bounded and explainable.
  - *Secrets.* The key is read from the environment or Streamlit secrets, `.env` and `secrets.toml` are git-ignored, and only `.example` files are committed.

---

### Task 3: GitHub Repository AI Analyzer & Open-Source Contribution

- **GitHub Repository URL:** https://github.com/KanishkThakar/ghanextech-technical-assessment/tree/main/task-03-github-analyzer
- **Live Deployed Application URL:** https://ghanex-task-03-analyzer.onrender.com
- **Primary Technologies Used:** Python 3.12, Streamlit, GitHub REST API, requests, Groq API (openai/gpt-oss-120b, JSON mode)

- **Project Summary:**

  The analyzer accepts `owner/repo` or any GitHub URL (including a trailing `.git` or a deep link). `analyzer/github_client.py` fetches the repository metadata, README, recursive file tree, recent commits, contributors and dependency manifests (requirements.txt, pyproject.toml, package.json, environment.yml, Pipfile, go.mod, Cargo.toml) through the REST API. An optional personal access token raises the limit from 60 to 5,000 requests an hour, and errors are mapped to messages a user can act on: not found or private, rate limit reached with its reset time, unavailable for legal reasons.

  `analyzer/metrics.py` computes everything countable in plain Python, with no network and no LLM: language share by bytes, file types, busiest directories, commit activity, contributor spread, and the presence of tests, CI, licence and docs. The dashboard draws these as charts beside headline numbers (stars, forks, open issues, files, size).

  `analyzer/insights.py` sends a compact digest of those facts to the LLM in JSON mode and validates the result into a fixed shape: a summary, strengths, at least three recommendations, contribution opportunities with where to start and why, and maintenance risks. Because the numbers are computed first and passed in, the model interprets facts it was given rather than inventing them, and each recommendation points at real files from the tree.

- **Open-Source Contribution:**
  - **Target Open-Source Repository:** https://github.com/deepchem/deepchem
  - **Contribution URL (Issue / Pull Request):** https://github.com/deepchem/deepchem/pull/5291
  - **Contribution Summary:** Addresses the wrong-featurizer diagnostic tracked in deepchem issue #2513. Passing ECFP features to `GraphConvModel` used to fail deep inside graph batching with an unhelpful `AttributeError` about `atom_features`. The PR checks for `ConvMol` inputs in the shared batch generator and raises a `ValueError` that names `ConvMolFeaturizer` as the fix, and documents the required feature type and the MoleculeNet option `featurizer='GraphConv'`. While validating it I found that running doctests directly on a file lets DeepChem's internal `lightning` package shadow the installed dependency, so the contributor checklist and coding guide now use `python -m pytest --doctest-modules`, consistent with CI; a TensorFlow `WeaveModel` doctest also now imports its class explicitly. Two regression tests fail on the original code and pass with the fix, and YAPF, Flake8 and `mypy -p deepchem` pass. The PR (4 files, +31/-4) has received positive reviews from community reviewers and is awaiting maintainer merge.

- **Key Challenges & Solutions:**
  - *Rate limits.* Unauthenticated GitHub access allows 60 requests an hour, and a large repository can use several per analysis. The client reads `X-RateLimit-Remaining` and the reset time, stops early with a clear message, accepts an optional token, and limits commit reads to a user-chosen window.
  - *Keeping the LLM honest.* Asking the model to describe a repository directly produced invented numbers. Computing every metric in Python and giving the model only that digest, in JSON mode with a validated schema, fixed it.
  - *Charting on the host runtime.* Streamlit's Altair-based charts failed on the host's default Python 3.14 due to a typing incompatibility. Pinning Python 3.12.7 and drawing the bar charts as lightweight HTML removed the dependency on Altair entirely.

---

## Demonstration Video (Optional)

- **Video Link:** Not included. All three applications are live at the URLs above.
