"""The agent loop: think, call tools, read results, repeat, answer.

One turn of the conversation is one call to `run_turn`. The loop stops when the
model answers without asking for a tool, or when it hits the step limit, which
keeps a confused model from looping forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .llm import LLMError, chat
from .registry import run_tool, tool_schemas

MAX_STEPS = 6

SYSTEM_PROMPT = (
    "You are a helpful assistant that can call tools. "
    "Use a tool whenever the answer depends on a calculation, live data or a fact you are "
    "not certain of, instead of guessing. You may call several tools in one turn, and you "
    "may call a tool again with different arguments if the first attempt failed. "
    "If a tool returns an error, either fix the arguments and retry once, or tell the user "
    "plainly what could not be fetched. "
    "When you answer, be concise and say which tool gave you the facts when it matters. "
    "If no tool is needed, answer directly."
)


@dataclass
class Turn:
    """What happened in one turn: the answer plus every tool call made."""

    answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    stopped_early: bool = False


def run_turn(history: List[Dict[str, Any]], user_message: str) -> Turn:
    """Run one user turn to completion and return the answer and its trace.

    `history` is the prior conversation in OpenAI message format (user and
    assistant messages only). It is not mutated: the caller owns the transcript.
    """
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    steps: List[Dict[str, Any]] = []
    schemas = tool_schemas()

    for _ in range(MAX_STEPS):
        try:
            message = chat(messages, tools=schemas)
        except LLMError as exc:
            return Turn(answer=f"I could not reach the model: {exc}", steps=steps)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return Turn(answer=(message.get("content") or "").strip(), steps=steps)

        # The assistant message must go back into the transcript before the
        # tool results, or the API rejects the follow-up request.
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            }
        )

        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            raw_arguments = function.get("arguments", "{}")
            result = run_tool(name, raw_arguments)
            steps.append(
                {
                    "tool": name,
                    "arguments": _safe_json(raw_arguments),
                    "result": result,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": json.dumps(result, default=str)[:6000],
                }
            )

    return Turn(
        answer=(
            "I stopped after several tool calls without reaching an answer. "
            "Try asking in a more specific way."
        ),
        steps=steps,
        stopped_early=True,
    )

def _safe_json(raw: Any) -> Any:
    """Tool arguments arrive as a JSON string; show them parsed in the trace."""
    if isinstance(raw, str):
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return raw
    return raw
