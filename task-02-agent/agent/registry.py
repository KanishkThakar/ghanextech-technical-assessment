"""Tool registry: the model's menu, and the dispatcher that runs a choice.

The registry is the only place that knows tool names, so a hallucinated name is
rejected here with a message the model can act on rather than a crash.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .tools import ALL_TOOLS, Tool, ToolError

_TOOLS: Dict[str, Tool] = {tool.name: tool for tool in ALL_TOOLS}


def tool_schemas() -> list:
    """The schemas passed to the model on every request."""
    return [tool.schema() for tool in _tools_in_order()]


def _tools_in_order() -> List[Tool]:
    return ALL_TOOLS


def tool_names() -> List[str]:
    return [tool.name for tool in _tools_in_order()]


def describe_tools() -> List[Dict[str, str]]:
    """Name and description per tool, for the UI's sidebar."""
    return [{"name": tool.name, "description": tool.description} for tool in _tools_in_order()]


def run_tool(name: str, arguments: str | Dict[str, Any]) -> Dict[str, Any]:
    """Run one tool call and always return a JSON-serialisable result.

    Errors are results, not exceptions: the model gets `{"error": ...}` and can
    retry, choose another tool, or tell the user what went wrong.
    """
    tool = _TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(tool_names())}."}

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return {"error": f"Arguments were not valid JSON: {exc.msg}."}
    if not isinstance(arguments, dict):
        return {"error": "Arguments must be a JSON object."}

    try:
        result = tool.func(**arguments)
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError as exc:
        # Wrong or missing arguments: the model can read this and correct itself.
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - a tool must never break the chat
        return {"error": f"{name} failed unexpectedly: {exc}"}

    if not isinstance(result, (dict, list)):
        result = {"result": result}
    return result
