"""LLM client for grounded answering.

Groq's free tier through its OpenAI-compatible chat-completions endpoint. Plain
`requests` keeps the dependency list small and makes the request shape explicit,
which is easier to explain than an SDK wrapper.
"""

from __future__ import annotations

import os
from typing import List

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
TIMEOUT_SECONDS = 60

SYSTEM_PROMPT = """You are a document question-answering assistant.

Rules:
1. Answer ONLY from the numbered context passages provided by the user.
2. Cite the passages you used with their numbers, like [1] or [2][3].
3. If the context does not contain the answer, say exactly: "I could not find
   that in the uploaded documents." Then, if useful, say what the documents do
   cover. Never fill the gap with outside knowledge.
4. Be concise and factual. Do not invent page numbers, figures or names."""


class LLMError(RuntimeError):
    """Raised when the LLM call fails in a way the UI should explain."""


def get_api_key() -> str | None:
    """Read the key from Streamlit secrets, then the environment."""
    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"])
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def build_context_block(passages: List[str]) -> str:
    """Number the passages so the model can cite them by index."""
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))


def answer_question(question: str, passages: List[str], history: List[dict] | None = None) -> str:
    """Ask the model to answer strictly from `passages`."""
    api_key = get_api_key()
    if not api_key:
        raise LLMError(
            "No GROQ_API_KEY configured. Add it to .streamlit/secrets.toml "
            "(local) or the app's Secrets (Streamlit Cloud)."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # A short slice of history keeps follow-ups ("and the second one?")
    # understandable without letting old turns crowd out the retrieved text.
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append(
        {
            "role": "user",
            "content": (
                f"Context passages:\n\n{build_context_block(passages)}\n\n"
                f"Question: {question}"
            ),
        }
    )

    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "temperature": 0.1, "max_tokens": 800},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach the LLM service: {exc}") from exc

    if response.status_code == 401:
        raise LLMError("The GROQ_API_KEY was rejected. Check the key and try again.")
    if response.status_code == 429:
        raise LLMError("The LLM free tier is rate-limited right now. Wait a moment and retry.")
    if response.status_code >= 400:
        raise LLMError(f"LLM request failed ({response.status_code}): {response.text[:200]}")

    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError("The LLM returned an unexpected response.") from exc
