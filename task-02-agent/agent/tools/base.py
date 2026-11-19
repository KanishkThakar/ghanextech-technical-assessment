"""The shape every tool follows.

A tool is a name, a description the model reads, a JSON Schema for its
arguments, and a plain Python function. Keeping the schema next to the function
means the model can never be told about an argument the code does not accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


class ToolError(Exception):
    """Raised for a failure the model can read and recover from.

    Anything raised as a ToolError is returned to the model as the tool's
    result, so it can retry with different arguments or explain the problem to
    the user. Unexpected exceptions are caught by the registry and reported the
    same way, so a broken tool never crashes the conversation.
    """


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for the arguments object
    func: Callable[..., Any]

    def schema(self) -> Dict[str, Any]:
        """The OpenAI-style function schema the model is given."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
