"""
An expression read as a criterion, in three-valued logic (Kleene's, as SQL's WHERE reads it): a comparison or a
membership test on a missing value is unknown, neither True nor False; "not unknown" is unknown, "unknown and False"
is False, "unknown or True" is True; and a criterion fires where its expression is True only. So a missing value never
fires a criterion, written with "!=" or "~" too: "~(df['height_cm'] >= 100)" fired on every missing height, and
"df['status'] != 'withdrawn'" on every missing status.

One reading, however the test is spelt: the methods ``eq ne lt le gt ge`` are the comparisons (the value by
position or as ``other=``; ``axis=`` changes nothing for a Series), ``np.logical_not``,
``logical_and``, ``logical_or`` and ``logical_xor`` are ``~ & | ^``, a test compared with True or False is the test or
its negation, and ``between`` (its bounds by position or as ``left=``, ``right=``, ``inclusive=``) is its two
comparisons (unknown where the value or a bound is missing, unless the other comparison is False). An expression that says itself what a missing value gives decides it, whatever the
column's dtype: a test's ``fillna(True)`` or ``fillna(False)`` fills the test's unknowns, and an ``isin`` list holding
a missing value (None, np.nan, pd.NA, pd.NaT) finds the missing values (``isin(the others) | isna()``); inside any
other call (np.where, astype, a sum) and in a text test's ``na=`` it is read as Python reads it.
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


def _is_missing(item: ast.expr) -> bool:
    """A missing value written out: None, np.nan, pd.NA, pd.NaT, float('nan')."""
    if isinstance(item, ast.Constant) and item.value is None:
        return True
    if (
        isinstance(item, ast.Attribute)
        and isinstance(item.value, ast.Name)
        and (item.value.id, item.attr) in _MISSING_NAMES
    ):
        return True
    return (
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "float"
        and len(item.args) == 1
        and isinstance(item.args[0], ast.Constant)
        and str(item.args[0].value).strip().lower() == "nan"
    )


def _holds_missing(node: ast.expr) -> bool:
    """A list written out (or a tuple, a set) that holds a missing value."""
    return isinstance(node, ast.List | ast.Tuple | ast.Set) and any(
        _is_missing(item) for item in node.elts
    )


def _argument(
    args: list[ast.expr], given: dict[str | None, ast.expr], names: tuple[str, ...]
) -> list[ast.expr] | None:
    """A call's arguments in the order ``names`` takes them, by position then by keyword (between(1, right=3)); None
    when a keyword is not among ``names``, comes twice, or leaves a gap before a later one."""
    if len(args) > len(names) or any(k not in names for k in given):
        return None
    found: list[ast.expr | None] = [*args, *[None] * (len(names) - len(args))]
    for i, name in enumerate(names):
        if name in given:
            if found[i] is not None:
                return None
            found[i] = given[name]
    filled = [a for a in found if a is not None]
    if found[: len(filled)] != filled:  # a gap: between(1, inclusive="both") lacks its right bound
        return None
    return filled


def _constant(node: ast.expr | None, value: object) -> bool:
    return isinstance(node, ast.Constant) and node.value is value


def _fill_flag(node: ast.Call) -> ast.expr | None:
    """The True or False a ``fillna`` call gives, by position or as ``value=``; ``axis=``, ``inplace=False`` and
    ``limit=None`` change nothing for a Series. None otherwise (a method, a limit: read as written)."""
    given = {k.arg: k.value for k in node.keywords}
    if None in given or ("inplace" in given and not _constant(given["inplace"], False)):
        return None
    if "limit" in given and not _constant(given["limit"], None):
        return None
    rest = {k: v for k, v in given.items() if k not in ("axis", "inplace", "limit")}
    found = _argument(node.args, rest, ("value",))
    if found is None or len(found) != 1:
        return None
    return found[0] if _flag(found[0]) is not None else None


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
        if name == "fillna":  # a test whose unknowns are filled
            return _fill_flag(node) is not None and _is_logic(owner)
        return name in _TEXT_TESTS and isinstance(owner, ast.Attribute) and owner.attr == "str"
    return False


def _negated(node: ast.expr) -> ast.expr:
    return ast.UnaryOp(op=ast.Invert(), operand=_logic(node))


def _method(node: ast.Call, name: str, owner: ast.expr) -> ast.expr | None:
    """A Series's comparison or test, rewritten; None when it says itself what a missing value gives, or is not one."""
    if name in _METHODS:
        # the value compared with, by position or as other=; axis= changes nothing for a Series. fill_value= and
        # level= say themselves what a missing value gives: read as written
        given = {k.arg: k.value for k in node.keywords if k.arg != "axis"}
        other = _argument(node.args, given, ("other",))
        if other is not None and len(node.args) + len(given) == 1:
            return _compared(owner, _METHODS[name], other[0])
        return None
    if name == "between":
        given = {k.arg: k.value for k in node.keywords}
        bounds = _argument(node.args, given, ("left", "right", "inclusive"))
        if bounds is not None and len(bounds) >= 2:
            inclusive = bounds[2] if len(bounds) == 3 else ast.Constant("both")
            return _call(BETWEEN, owner, bounds[0], bounds[1], inclusive)
        return _call(TESTED, owner, node)
    if name == "isin":
        given = {k.arg: k.value for k in node.keywords}
        values = node.args[0] if node.args else given.get("values")
        if values is not None and _holds_missing(values):
            found = _argument(node.args, given, ("values",))  # by position or as values=
            if found is None or len(found) != 1:
                return None
            # the list looks for a missing value too: found by isna(), whatever the column's dtype (pandas matches
            # None, NaN and NA by dtype: np.nan finds a float's missing values, not an Int64's)
            listed = found[0]
            if not isinstance(listed, ast.List | ast.Tuple | ast.Set):
                return None
            others = ast.List(elts=[i for i in listed.elts if not _is_missing(i)], ctx=ast.Load())
            found = ast.Call(
                func=ast.Attribute(value=owner, attr="isin", ctx=ast.Load()),
                args=[others],
                keywords=[],
            )
            missing = ast.Call(
                func=ast.Attribute(value=copy.deepcopy(owner), attr="isna", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
            return ast.BinOp(left=found, op=ast.BitOr(), right=missing)
        return _call(TESTED, owner, node)
    if name == "fillna" and _is_logic(owner):
        value = _fill_flag(node)
        if value is None:
            return None
        # a test's own fill: its unknowns become the True or False given, on every dtype (pandas' comparison of a
        # missing float or text gives False, which a fillna never sees)
        return ast.Call(
            func=ast.Attribute(value=_logic(owner), attr="fillna", ctx=ast.Load()),
            args=[value],
            keywords=[],
        )
    if name in _TEXT_TESTS and isinstance(owner, ast.Attribute) and owner.attr == "str":
        if any(k.arg == "na" for k in node.keywords):
            return None  # na= says what a missing value gives
        return _call(TESTED, owner.value, node)
    return None


def _compared(left: ast.expr, symbol: str, right: ast.expr) -> ast.expr:
    """A comparison made three-valued: unknown where a side is missing. A test compared with True or False is the test
    or its negation, by the operator or by its method ((test).eq(False), (test).ne(True)); a test on either side is
    made three-valued before it is compared, so (a > 1) == (b > 2) is unknown where either test is."""
    if symbol in ("==", "!="):
        for tested, flag in ((left, _flag(right)), (right, _flag(left))):
            if flag is not None and _is_logic(tested):
                return _logic(tested) if flag == (symbol == "==") else _negated(tested)
    sides = [_logic(side) if _is_logic(side) else side for side in (left, right)]
    return _call(COMPARE, sides[0], ast.Constant(symbol), sides[1])


def _logic(node: ast.expr) -> ast.expr:
    """The tests at the level of the expression's logic made three-valued: the expression itself, the operands of
    &, | and ^ (and of numpy's logical functions), the operand of ~. Anything else is left as it is."""
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op, left, right = node.ops[0], node.left, node.comparators[0]
        if type(op) in _COMPARISONS:
            return _compared(left, _COMPARISONS[type(op)][0], right)
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
