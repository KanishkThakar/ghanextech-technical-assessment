# Task 2 — AI Agent with Tool Calling

A multi-turn chat agent that decides, on its own, whether a question needs a
tool, which one, and with what arguments. Built on Groq's OpenAI-compatible
chat completions API with native function calling.

**Live app:** https://ghanex-agent.onrender.com

## What it does

You ask a question in plain language. The model either answers directly or
emits one or more tool calls. The app runs them, feeds the results back, and
loops until the model produces a final answer. The whole exchange is visible in
the UI: which tool ran, with which arguments, and what it returned.

Example prompts that exercise each path:

| Prompt | What happens |
|---|---|
| `What is 18% of 4,250 plus 96?` | one call to `calculator` |
| `Weather in Mumbai tomorrow, and convert 50 USD to INR` | two calls in one turn, then one answer |
| `Who was Ada Lovelace?` | one call to `wikipedia` |
| `Explain what a vector database is` | no tool, answered from the model |
| `Weather on Mars` | tool raises `ToolError`, the model explains it instead of crashing |

## Architecture

```
app.py                  Streamlit chat UI, transcript in st.session_state
   │
   ▼
agent/loop.py           the reasoning loop: ask → run tools → ask again
   │
   ├── agent/llm.py      Groq chat-completions client (requests, no SDK)
   └── agent/registry.py tool schemas for the model + dispatcher that runs a choice
          │
          ▼
       agent/tools/     one module per tool, all built on tools/base.py
```

Each module has one job, so a question like "what happens if a tool throws?"
maps to a single file.

### `agent/tools/base.py`

A tool is a frozen dataclass: `name`, `description` (what the model reads),
`parameters` (JSON Schema for the arguments) and `func`. `Tool.schema()` renders
the OpenAI-style function schema. Because the schema sits next to the function
that implements it, the model can never be told about an argument the code does
not accept.

`ToolError` is the contract for recoverable failure. Anything raised as a
`ToolError` — or any unexpected exception, caught in the registry — is returned
to the model as the tool's result, so it can retry with different arguments or
tell the user what went wrong. A broken tool never breaks the conversation.

### `agent/registry.py`

Holds `ALL_TOOLS` and does two things: hands the model the list of schemas, and
dispatches a returned call by name, validating that the name exists and that the
arguments are a JSON object. Unknown tool names come back as an error message
for the model, not an exception.

### `agent/loop.py`

The loop, capped at `MAX_STEPS` (6) so a model that keeps calling tools cannot
spin forever:

1. Send the transcript plus the tool schemas.
2. If the reply has no tool calls, it is the answer — return it.
3. Otherwise append the assistant message, run every requested call, append one
   `tool` message per result, and go back to step 1.

The cap is the safety net; the model normally finishes in one or two rounds.

### `agent/llm.py`

A thin `requests` wrapper around `POST /openai/v1/chat/completions`. No SDK, so
the request shape is explicit and easy to explain. The key is read from the
environment first, then Streamlit secrets, and is never committed.

## The four tools

1. **`calculator`** — `expression: str`. Arithmetic without `eval()`: the
   expression is parsed with `ast` and walked, allowing only numbers, a fixed
   operator set, `pi`/`e` and a whitelist of maths functions. Exponents are
   capped at 1000 so `9**9**9` cannot hang the process.
2. **`weather`** — `location: str`, `days: int = 3`. Open-Meteo, no key. Two
   calls: geocoding turns the place name into coordinates, then the forecast
   endpoint returns current conditions and the next few days. WMO weather codes
   are mapped to plain English.
3. **`wikipedia`** — `query: str`. Searches the public REST API, then reads the
   top article's summary. Two steps so the model can see which article was
   chosen and cite it.
4. **`currency`** — `amount: float`, `from_currency: str`, `to_currency: str`.
   Frankfurter, free ECB reference rates, no key. A currency it does not know
   comes back as a `ToolError` naming the problem.

## Running it locally

```bash
cd task-02-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your Groq key in it
export $(grep -v '^#' .env | xargs)
streamlit run app.py
```

Or put the key in `.streamlit/secrets.toml` (see `secrets.toml.example`) and run
`streamlit run app.py` with no environment variables.

Get a free key at https://console.groq.com/keys. The default model is
`openai/gpt-oss-120b`, which supports native tool calling on Groq's free tier;
override it with `GROQ_MODEL`.

## Design decisions worth defending

- **Native tool calling, not prompt-parsed JSON.** The provider validates the
  arguments against the schema before the app sees them, which removes a whole
  class of parsing bugs.
- **Errors are data.** Every failure path returns a readable string to the
  model, so the agent degrades into an explanation instead of a traceback.
- **No SDK.** Two dependencies (`streamlit`, `requests`) and an explicit request
  body is easier to debug and to explain than a wrapper.
- **Stateless server, stateful client.** The transcript lives in
  `st.session_state`; the loop takes it as an argument and returns the new one,
  so the loop is testable without Streamlit.

## Limitations

- The transcript is not persisted: a page reload starts a new conversation.
- No streaming; the answer appears when the loop finishes.
- `MAX_STEPS = 6` is a fixed cap rather than a token budget.
- Tool results are truncated before being sent back, to keep the context small.
