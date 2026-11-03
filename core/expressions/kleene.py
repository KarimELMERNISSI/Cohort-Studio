"""
An expression read as a criterion, in three-valued logic (Kleene's, as SQL's WHERE reads it): a comparison or a
membership test on a missing value is unknown, neither True nor False; "not unknown" is unknown, "unknown and False"
is False, "unknown or True" is True; and a criterion fires where its expression is True only. So a missing value never
fires a criterion, written with "!=" or "~" too: "~(df['height_cm'] >= 100)" fired on every missing height, and
"df['status'] != 'withdrawn'" on every missing status.

One reading, however the test is spelt: the methods ``eq ne lt le gt ge`` are the comparisons, ``np.logical_not``,
``logical_and``, ``logical_or`` and ``logical_xor`` are ``~ & | ^``, a test compared with True or False is the test or
its negation, and ``between`` is its two comparisons (unknown where the value or a bound is missing, unless the other
comparison is False). An expression that says itself what a missing
value gives is read as Python reads it: inside a call (np.where, fillna, astype, a sum), a text test's ``na=``, a
missing value in an ``isin`` list.
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
# a Series's comparison methods, read as the operators they stand for
_METHODS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
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
_NUMPY = frozenset({"np", "numpy"})
_NUMPY_LOGIC: dict[str, type[ast.operator]] = {
    "logical_and": ast.BitAnd,
    "logical_or": ast.BitOr,
    "logical_xor": ast.BitXor,
}
# what a list may hold to say that a missing value is among the values looked for
_MISSING_NAMES = frozenset(
    {
        ("np", "nan"),
        ("numpy", "nan"),
        ("pd", "NA"),
        ("pd", "NaT"),
        ("pandas", "NA"),
        ("pandas", "NaT"),
    }
)
# the functions the rewritten tree calls
COMPARE, TESTED, BETWEEN = "known_compare", "known_test", "known_between"
_INCLUSIVE = {
    "both": (">=", "<="),
    "neither": (">", "<"),
    "left": (">=", "<"),
    "right": (">", "<="),
}


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
    """A Series's own test (``isin``, ``str.contains``...), unknown where its value is missing."""
    return _unknown_where_missing(result, tested)


def known_between(tested: Any, low: Any, high: Any, inclusive: str = "both") -> Any:
    """``tested.between(low, high, inclusive)`` as its two comparisons, each unknown where a side is missing, joined
    by Kleene's "and": unknown where the value or a bound is missing, unless the other comparison is False (SQL's
    ``x BETWEEN lo AND NULL`` is unknown where x >= lo, False where x < lo)."""
    if inclusive not in _INCLUSIVE:
        raise ValueError("inclusive has to be either 'both', 'neither', 'left' or 'right'")
    above, below = _INCLUSIVE[inclusive]
    return known_compare(tested, above, low) & known_compare(tested, below, high)


HELPERS: dict[str, Callable[..., Any]] = {
    COMPARE: known_compare,
    TESTED: known_test,
    BETWEEN: known_between,
}


def _call(name: str, *args: ast.expr) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])


def _numpy(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id in _NUMPY


def _flag(node: ast.expr) -> bool | None:
    """True or False written as such; None for anything else."""
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, bool) else None


def _holds_missing(node: ast.expr) -> bool:
    """A list written out (or a tuple, a set) that holds a missing value: None, np.nan, pd.NA, pd.NaT, float('nan')."""
    if not isinstance(node, ast.List | ast.Tuple | ast.Set):
        return False
    for item in node.elts:
        if isinstance(item, ast.Constant) and item.value is None:
            return True
        if (
            isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Name)
            and (item.value.id, item.attr) in _MISSING_NAMES
        ):
            return True
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "float"
            and len(item.args) == 1
            and isinstance(item.args[0], ast.Constant)
            and str(item.args[0].value).strip().lower() == "nan"
        ):
            return True
    return False


def _is_logic(node: ast.expr) -> bool:
    """A test at the level of the logic: a comparison, a test of a value, a negation or a combination of tests."""
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd | ast.BitOr | ast.BitXor):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        name, owner = node.func.attr, node.func.value
        if _numpy(owner):
            return name in {"logical_not", *_NUMPY_LOGIC}
        if name in _METHODS or name in _TESTS:
            return True
        return name in _TEXT_TESTS and isinstance(owner, ast.Attribute) and owner.attr == "str"
    return False


def _negated(node: ast.expr) -> ast.expr:
    return ast.UnaryOp(op=ast.Invert(), operand=_logic(node))


def _method(node: ast.Call, name: str, owner: ast.expr) -> ast.expr | None:
    """A Series's comparison or test, rewritten; None when it says itself what a missing value gives, or is not one."""
    if name in _METHODS:
        if (
            len(node.args) == 1 and not node.keywords
        ):  # fill_value=, level=, axis=: the call decides
            return _call(COMPARE, owner, ast.Constant(_METHODS[name]), node.args[0])
        return None
    if name == "between":
        given = {k.arg: k.value for k in node.keywords}
        if (
            2 <= len(node.args) <= 3
            and set(given) <= {"inclusive"}
            and not (len(node.args) == 3 and given)
        ):
            inclusive = node.args[2] if len(node.args) == 3 else given.get("inclusive")
            return _call(
                BETWEEN, owner, node.args[0], node.args[1], inclusive or ast.Constant("both")
            )
        return _call(TESTED, owner, node)
    if name == "isin":
        if node.args and _holds_missing(node.args[0]):
            return None  # the list says that a missing value is among the values looked for
        return _call(TESTED, owner, node)
    if name in _TEXT_TESTS and isinstance(owner, ast.Attribute) and owner.attr == "str":
        if any(k.arg == "na" for k in node.keywords):
            return None  # na= says what a missing value gives
        return _call(TESTED, owner.value, node)
    return None


def _logic(node: ast.expr) -> ast.expr:
    """The tests at the level of the expression's logic made three-valued: the expression itself, the operands of
    &, | and ^ (and of numpy's logical functions), the operand of ~. Anything else is left as it is."""
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op, left, right = node.ops[0], node.left, node.comparators[0]
        if isinstance(
            op, ast.Eq | ast.NotEq
        ):  # a test compared with True or False: the test, or its negation
            for tested, flag in ((left, _flag(right)), (right, _flag(left))):
                if flag is not None and _is_logic(tested):
                    keeps = flag == isinstance(op, ast.Eq)
                    return _logic(tested) if keeps else _negated(tested)
        if type(op) in _COMPARISONS:
            return _call(COMPARE, left, ast.Constant(_COMPARISONS[type(op)][0]), right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        name, owner = node.func.attr, node.func.value
        if _numpy(owner):
            if name == "logical_not" and len(node.args) == 1 and not node.keywords:
                return _negated(node.args[0])
            if name in _NUMPY_LOGIC and len(node.args) == 2 and not node.keywords:
                return ast.BinOp(
                    left=_logic(node.args[0]), op=_NUMPY_LOGIC[name](), right=_logic(node.args[1])
                )
            return node
        rewritten = _method(node, name, owner)
        if rewritten is not None:
            return rewritten
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd | ast.BitOr | ast.BitXor):
        return ast.BinOp(left=_logic(node.left), op=node.op, right=_logic(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return _negated(node.operand)
    return node


def three_valued(tree: ast.Expression) -> ast.Expression:
    """A copy of a validated expression whose tests at the level of its logic are three-valued: evaluate it with
    :data:`HELPERS` among its names."""
    body = _logic(copy.deepcopy(tree.body))
    return ast.fix_missing_locations(ast.Expression(body=body))


__all__ = [
    "BETWEEN",
    "COMPARE",
    "HELPERS",
    "TESTED",
    "known_between",
    "known_compare",
    "known_test",
    "three_valued",
]
