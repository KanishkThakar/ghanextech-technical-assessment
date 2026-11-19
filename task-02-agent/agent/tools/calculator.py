"""Calculator tool: arithmetic without eval().

`eval` on model output is a code-execution hole, so the expression is parsed
with `ast` and walked: only numbers and a fixed set of operators are allowed.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict

from .base import Tool, ToolError

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
MAX_EXPONENT = 1000


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ToolError(f"Only numbers are allowed, got {node.value!r}.")
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"Unknown name '{node.id}'. Allowed constants: pi, e.")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError("Exponent too large.")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _FUNCTIONS:
            raise ToolError(f"Unknown function '{node.func.id}'. Allowed: {', '.join(sorted(_FUNCTIONS))}.")
        if node.keywords:
            raise ToolError("Keyword arguments are not supported.")
        return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ToolError("That expression uses something the calculator does not support.")


def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate an arithmetic expression and return the value."""
    expression = (expression or "").strip()
    if not expression:
        raise ToolError("The expression is empty.")
    if len(expression) > 200:
        raise ToolError("The expression is too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Could not parse the expression: {exc.msg}.") from exc
    try:
        value = _evaluate(tree)
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero.") from exc
    except (OverflowError, ValueError) as exc:
        raise ToolError(f"The calculation failed: {exc}.") from exc
    # Show integers as integers: "4" reads better than "4.0" for 2+2.
    if value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return {"expression": expression, "result": value}


CALCULATOR = Tool(
    name="calculate",
    description=(
        "Evaluate an arithmetic expression exactly. Use this for any maths instead of "
        "working it out yourself. Supports + - * / // % **, parentheses, pi and e, and "
        "the functions sqrt, abs, round, log, log10, exp, sin, cos, tan, floor, ceil."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The expression to evaluate, e.g. '(1200 * 1.18) / 3'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    func=calculate,
)
