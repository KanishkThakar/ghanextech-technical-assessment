"""The LLM half of the analyzer: summary, strengths, recommendations, ideas.

The model is given a compact text summary of the repository (never the whole
tree) and must answer with JSON, so the dashboard can render each section
separately instead of parsing prose.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import requests

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
TIMEOUT_SECONDS = 90

SYSTEM_PROMPT = (
    "You are a senior engineer reviewing a GitHub repository for a new contributor. "
    "You are given metadata, the file tree, manifests and the README. "
    "Answer with a single JSON object and nothing else, using exactly these keys:\n"
    '{"summary": string, "strengths": [string], "recommendations": [string], '
    '"contribution_opportunities": [{"title": string, "why": string, "where": string}], '
    '"risks": [string]}\n'
    "Rules: summary is 3-5 sentences on what the project is and how it is built. "
    "strengths: 3-5 specific things done well, each naming a file or practice you can see. "
    "recommendations: at least 3 concrete, actionable improvements, each naming where it applies. "
    "contribution_opportunities: 3-5 tasks a newcomer could pick up, each with the file or area. "
    "risks: 0-4 short notes on maintenance, licensing or security concerns. "
    "Never invent files that are not in the tree. If the repository is nearly empty, say so plainly."
)


class InsightError(Exception):
    """The model could not produce a usable analysis."""


def get_api_key() -> str:
    """Streamlit secrets first, then the environment."""
    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"])
    except Exception:  # noqa: BLE001 - not running under Streamlit
        pass
    return os.environ.get("GROQ_API_KEY", "")


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the JSON object out of a reply that may carry prose or fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise InsightError("The model's analysis was not valid JSON. Try again.")


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def analyse(repo_summary: str, model: str | None = None) -> Dict[str, Any]:
    """Ask the model for the structured review of one repository."""
    api_key = get_api_key()
    if not api_key:
        raise InsightError(
            "No GROQ_API_KEY found. Add it to .streamlit/secrets.toml locally, or to the "
            "app's secrets on Streamlit Cloud."
        )

    payload = {
        "model": model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": repo_summary},
        ],
    }
    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Authorization": "Bearer {}".format(api_key), "Content-Type": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise InsightError("Could not reach the model service: {}".format(exc)) from exc

    if response.status_code == 401:
        raise InsightError("The GROQ_API_KEY was rejected. Check the key and try again.")
    if response.status_code == 429:
        raise InsightError("The model service is rate limiting this key. Wait a moment and retry.")
    if response.status_code >= 400:
        raise InsightError(
            "The model service returned {}: {}".format(response.status_code, response.text[:300])
        )

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        raise InsightError("The model service returned an unexpected response.") from exc

    data = _extract_json(content)
    return {
        "summary": str(data.get("summary", "")).strip(),
        "strengths": [str(item) for item in _as_list(data.get("strengths"))],
        "recommendations": [str(item) for item in _as_list(data.get("recommendations"))],
        "contribution_opportunities": [
            item if isinstance(item, dict) else {"title": str(item), "why": "", "where": ""}
            for item in _as_list(data.get("contribution_opportunities"))
        ],
        "risks": [str(item) for item in _as_list(data.get("risks"))],
    }
