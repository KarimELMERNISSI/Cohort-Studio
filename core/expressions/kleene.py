"""
An expression read as a criterion, in three-valued logic (Kleene's, as SQL's WHERE reads it): a comparison or a
membership test on a missing value is unknown, neither True nor False; "not unknown" is unknown, "unknown and False"
is False, "unknown or True" is True; and a criterion fires where its expression is True only. So a missing value never
fires a criterion, written with "!=" or "~" too: "~(df['height_cm'] >= 100)" fired on every missing height, and
"df['status'] != 'withdrawn'" on every missing status. A test inside a call (np.where, a sum, astype) is left as
Python reads it: the call decides what a missing value gives there.
"""

from __future__ import annotations

import ast
import copy
import operator
from collections.abc import Callable
from typing import Any

import pandas as pd

# the comparisons made three-valued, by the name the rewritten tree passes
_COMPARISONS: dict[type[ast.cmpop], tuple[str, Callable[[Any, Any], Any]]] = {
    ast.Eq: ("==", operator.eq),
    ast.NotEq: ("!=", operator.ne),
    ast.Lt: ("<", operator.lt),
    ast.LtE: ("<=", operator.le),
    ast.Gt: (">", operator.gt),
    ast.GtE: (">=", operator.ge),
}
_BY_NAME = dict(_COMPARISONS.values())
# a Series's own tests, unknown where its value is missing: membership, a range, and the tests of its text
_TESTS = frozenset({"isin", "between"})
_TEXT_TESTS = frozenset(
    {
        "contains",
        "startswith",
        "endswith",
        "match",
        "fullmatch",
        "isalpha",
        "isalnum",
        "isdecimal",
        "isdigit",
        "islower",
        "isnumeric",
        "isspace",
        "istitle",
        "isupper",
    }
)
COMPARE, TESTED = "known_compare", "known_test"  # the functions the rewritten tree calls


def _missing(value: Any) -> Any:
    """Where a value is missing: per row for a Series; one answer for a single value."""
    if isinstance(value, pd.Series):
        return value.isna()
    if pd.api.types.is_scalar(value):
        return bool(pd.isna(value))
    return False


def _unknown_where_missing(result: Any, *operands: Any) -> Any:
    """``result`` with unknown (pandas' NA) where an operand is missing; a result other than a Series as it is."""
    if not isinstance(result, pd.Series):
        return result
    unknown = pd.Series(False, index=result.index)
    for operand in operands:
        missing = _missing(operand)
        if isinstance(missing, pd.Series):
            unknown |= missing.reindex(result.index, fill_value=False).astype(bool)
        elif missing:
            unknown[:] = True
    known = result.astype("boolean")
    known[unknown] = pd.NA
    return known


def known_compare(left: Any, symbol: str, right: Any) -> Any:
    """``left <symbol> right``, unknown where either side is missing."""
    return _unknown_where_missing(_BY_NAME[symbol](left, right), left, right)


def known_test(tested: Any, result: Any) -> Any:
    """A Series's own test (``isin``, ``between``, ``str.contains``...), unknown where its value is missing."""
    return _unknown_where_missing(result, tested)


HELPERS: dict[str, Callable[..., Any]] = {COMPARE: known_compare, TESTED: known_test}


def _call(name: str, *args: ast.expr) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])


def _logic(node: ast.expr) -> ast.expr:
    """The tests at the level of the expression's logic made three-valued: the expression itself, the operands of
    &, | and ^, the operand of ~. Anything else is left as it is."""
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARISONS:
        symbol = _COMPARISONS[type(node.ops[0])][0]
        return _call(COMPARE, node.left, ast.Constant(symbol), node.comparators[0])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in _TESTS:
            return _call(TESTED, node.func.value, node)
        text = node.func.value
        if node.func.attr in _TEXT_TESTS and isinstance(text, ast.Attribute) and text.attr == "str":
            return _call(TESTED, text.value, node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd | ast.BitOr | ast.BitXor):
        return ast.BinOp(left=_logic(node.left), op=node.op, right=_logic(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return ast.UnaryOp(op=node.op, operand=_logic(node.operand))
    return node


def three_valued(tree: ast.Expression) -> ast.Expression:
    """A copy of a validated expression whose tests at the level of its logic are three-valued: evaluate it with
    :data:`HELPERS` among its names."""
    body = _logic(copy.deepcopy(tree.body))
    return ast.fix_missing_locations(ast.Expression(body=body))


__all__ = ["COMPARE", "HELPERS", "TESTED", "known_compare", "known_test", "three_valued"]
