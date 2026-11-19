"""Thin Groq client: one function that asks for the next assistant message.

Groq exposes an OpenAI-compatible chat completions endpoint with native tool
calling, so the request is the standard `tools` / `tool_choice` shape. Keeping
the HTTP call in one place means the agent loop never touches the network.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
TIMEOUT_SECONDS = 60


class LLMError(Exception):
    """Raised when the model cannot be reached or refuses the request."""


def _streamlit_secret(name: str) -> str:
    """Read a Streamlit secret, but only when a secrets file actually exists.

    Touching st.secrets with no file present makes Streamlit print a
    "No secrets found" banner on the page, so check for the file first.
    """
    try:
        from pathlib import Path

        import streamlit as st

        candidates = (
            Path.cwd() / ".streamlit" / "secrets.toml",
            Path.home() / ".streamlit" / "secrets.toml",
        )
        if not any(path.exists() for path in candidates):
            return ""
        return str(st.secrets[name])
    except Exception:  # noqa: BLE001 - not under Streamlit, or no such key
        return ""


def get_api_key() -> str:
    """Environment first (deployments), then Streamlit secrets (local runs).

    The key is never written to the repo.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    return _streamlit_secret("GROQ_API_KEY").strip()


def chat(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None,
         model: str | None = None, temperature: float = 0.2) -> Dict[str, Any]:
    """Send the conversation and return the assistant message object."""
    api_key = get_api_key()
    if not api_key:
        raise LLMError(
            "No GROQ_API_KEY found. Add it to .streamlit/secrets.toml locally, or to the "
            "app's secrets on Streamlit Cloud."
        )

    payload: Dict[str, Any] = {
        "model": model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach the model service: {exc}") from exc

    if response.status_code == 401:
        raise LLMError("The GROQ_API_KEY was rejected. Check the key and try again.")
    if response.status_code == 429:
        raise LLMError("The model service is rate limiting this key. Wait a moment and retry.")
    if response.status_code >= 400:
        raise LLMError(f"The model service returned {response.status_code}: {response.text[:300]}")

    try:
        data = response.json()
        return data["choices"][0]["message"]
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError("The model service returned an unexpected response.") from exc
