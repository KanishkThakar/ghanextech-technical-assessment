"""Wikipedia tool: search, then read the top article's summary.

The REST API is public and needs no key. Two steps so the model can see that a
search happened and which article was chosen.
"""

from __future__ import annotations

from typing import Any, Dict

import requests

from .base import Tool, ToolError

SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
TIMEOUT_SECONDS = 15
USER_AGENT = "ghanex-assessment-agent/1.0 (educational demo)"


def _get(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        response = requests.get(
            url, params=params, timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
        )
    except requests.RequestException as exc:
        raise ToolError(f"Could not reach Wikipedia: {exc}") from exc
    if response.status_code == 404:
        raise ToolError("No Wikipedia article matched that.")
    if response.status_code >= 400:
        raise ToolError(f"Wikipedia returned {response.status_code}.")
    try:
        return response.json()
    except ValueError as exc:
        raise ToolError("Wikipedia returned something that is not JSON.") from exc


def search_wikipedia(query: str) -> Dict[str, Any]:
    """Search Wikipedia and return the top article's summary and link."""
    query = (query or "").strip()
    if not query:
        raise ToolError("A search query is required.")

    results = _get(SEARCH_URL, {"q": query, "limit": 5}).get("pages") or []
    if not results:
        raise ToolError(f"Nothing on Wikipedia matched '{query}'.")

    top = results[0]
    title = top.get("title") or top.get("key")
    summary = _get(SUMMARY_URL.format(title=requests.utils.quote(title, safe="")))

    return {
        "query": query,
        "title": summary.get("title", title),
        "summary": summary.get("extract", ""),
        "url": (summary.get("content_urls", {}).get("desktop", {}) or {}).get("page"),
        "other_matches": [page.get("title") for page in results[1:4]],
    }


WIKIPEDIA = Tool(
    name="search_wikipedia",
    description=(
        "Look up an encyclopedic fact on Wikipedia and get a short summary with a link. "
        "Use it for people, places, history, science and definitions, not for live data "
        "like prices or news."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up, e.g. 'Ada Lovelace'."}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    func=search_wikipedia,
)
