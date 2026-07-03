"""A single, safe calculator tool shared by both tool-calling agents.

The expression is parsed with the :mod:`ast` module and only arithmetic nodes
are allowed, so arbitrary code execution is not possible.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from langchain_core.tools import tool

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result."""
    try:
        result = _evaluate(ast.parse(expression, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        return f"error: {exc}"
    if result.is_integer():
        return str(int(result))
    return str(result)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression such as '2 + 3 * (4 - 1)'."""
    return calculate(expression)
