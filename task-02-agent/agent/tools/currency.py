"""Currency conversion tool: Frankfurter, a free ECB-rates API with no key."""

from __future__ import annotations

from typing import Any, Dict

import requests

from .base import Tool, ToolError

BASE_URL = "https://api.frankfurter.app/latest"
TIMEOUT_SECONDS = 15


def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    """Convert an amount between two currencies at the latest ECB reference rate."""
    source = (from_currency or "").strip().upper()
    target = (to_currency or "").strip().upper()
    if not source or not target:
        raise ToolError("Both from_currency and to_currency are required, e.g. USD and INR.")
    if source == target:
        return {"amount": amount, "from": source, "to": target, "rate": 1.0, "converted": amount}
    try:
        amount = float(amount)
    except (TypeError, ValueError) as exc:
        raise ToolError("amount must be a number.") from exc

    try:
        response = requests.get(
            BASE_URL, params={"amount": amount, "from": source, "to": target}, timeout=TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        raise ToolError(f"Could not reach the exchange-rate service: {exc}") from exc
    if response.status_code >= 400:
        # Frankfurter answers 404 for a currency it does not know.
        raise ToolError(
            f"Could not convert {source} to {target}. Check the three-letter currency codes."
        )
    data = response.json()
    rates = data.get("rates") or {}
    if target not in rates:
        raise ToolError(f"No rate returned for {target}.")
    return {
        "amount": amount,
        "from": source,
        "to": target,
        "rate": rates[target] / amount if amount else None,
        "converted": rates[target],
        "date": data.get("date"),
    }


CURRENCY = Tool(
    name="convert_currency",
    description=(
        "Convert an amount from one currency to another using the latest ECB reference "
        "rates. Use three-letter codes such as USD, INR, EUR, GBP."
    ),
    parameters={
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "How much to convert."},
            "from_currency": {"type": "string", "description": "Source currency code, e.g. USD."},
            "to_currency": {"type": "string", "description": "Target currency code, e.g. INR."},
        },
        "required": ["amount", "from_currency", "to_currency"],
        "additionalProperties": False,
    },
    func=convert_currency,
)
