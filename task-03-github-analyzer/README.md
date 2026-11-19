# Task 3 — GitHub Repository AI Analyzer

Point it at any public repository and it returns a health report: metadata,
language mix, activity, contributor spread, an LLM-written review with strengths
and at least three concrete recommendations, and a list of contribution
opportunities for someone who wants to help.

**Live app:** https://ghanex-task-03-analyzer.onrender.com

## What it does

You paste `owner/repo` or a full GitHub URL. The app:

1. Fetches repository metadata, the README, the file tree and the dependency
   manifests through the GitHub REST API.
2. Computes the numbers locally — no LLM involved in anything countable.
3. Sends a compact digest of those facts to the LLM, which writes the summary,
   the strengths, the recommendations and the contribution opportunities.
4. Draws the numbers as charts beside the prose.

## Architecture

```
app.py                     Streamlit dashboard: input, charts, review
   │
   ├── analyzer/github_client.py   REST calls, rate limits, error mapping
   ├── analyzer/metrics.py         numbers computed from the snapshot
   └── analyzer/insights.py        the LLM half: summary, strengths, advice
```

The split matters: everything factual is computed in Python and shown as it is,
so the LLM can only interpret numbers it was given, never invent them.

### `analyzer/github_client.py`

- `parse_repo_input()` accepts `owner/repo`, `https://github.com/owner/repo`,
  a trailing `.git`, or a deep link, and returns `(owner, repo)`.
- `fetch_snapshot()` makes the calls: `/repos/{owner}/{repo}`, the README
  (`/readme`, base64-decoded), the tree (`/git/trees/{branch}?recursive=1`), and
  the manifests it finds (`requirements.txt`, `pyproject.toml`, `package.json`,
  `environment.yml`, `Pipfile`, `go.mod`, `Cargo.toml`).
- Errors are mapped to messages a user can act on: 404 → "not found, or it is
  private", 403/429 with `X-RateLimit-Remaining: 0` → "rate limit reached,
  resets at HH:MM, add a token to raise it to 5,000 an hour", 451 → "unavailable
  for legal reasons".
- An optional `GITHUB_TOKEN` raises the limit from 60 to 5,000 requests an hour.
  It is read from the environment or Streamlit secrets and never committed.

### `analyzer/metrics.py`

Pure functions over the snapshot, no network and no LLM:

- language mix by file count and by bytes
- stars, forks, watchers, open issues, size
- days since the last push, commits in the last 30/90 days
- contributor count and the top contributors' share
- file-type breakdown, test-file ratio, presence of CI, licence, docs, tests
- dependency count per manifest

Keeping these pure means they are unit-testable and the dashboard can be trusted
independently of the model.

### `analyzer/insights.py`

Builds a compact digest — metadata, the metrics, the README's first ~4,000
characters, the top-level tree, the dependency lists — and asks the model for
strict JSON: `summary`, `strengths[]`, `recommendations[]` (each with a title,
why it matters and how to do it), `contribution_opportunities[]` (each with what
to do and the skill it needs). The response is parsed defensively: a malformed
reply becomes a readable error, not a crash.

## Running it locally

```bash
cd task-03-github-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Groq key, optionally a GitHub token
export $(grep -v '^#' .env | xargs)
streamlit run app.py
```

## Design decisions worth defending

- **Numbers in Python, words in the model.** The LLM never computes a metric; it
  receives them. That is what keeps the report honest.
- **One snapshot, then everything from it.** The tree, README and manifests are
  fetched once and passed around as a dict, so the dashboard is fast and the
  rate limit is spent once per analysis.
- **Errors are mapped, not dumped.** A private repo, a typo and an exhausted
  rate limit each get their own message.
- **Optional token.** The app works with no token at all (60 requests an hour is
  enough for a handful of analyses) and gets better with one.

## Limitations

- Only public repositories; private ones need a token with access, which the app
  does not ask for.
- Commit history is summarised from the repository metadata rather than walked,
  to stay inside the unauthenticated rate limit.
- The README is truncated before it reaches the model.
- Very large trees are sampled rather than sent whole.
