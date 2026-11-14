"""
Whitelisting evaluator for pandas/numpy expressions (audit finding E3).

Expressions reach the application from three untrusted sources: free text typed by users, uploaded
configuration or trace JSON, and language-model output. They used to be passed to ``eval`` with the
module globals, which is remote code execution. This module accepts only a small expression grammar:

* names: ``df`` (the data frame), ``np`` and ``pd`` (restricted facades) and caller-provided locals;
* operators: arithmetic, comparison, boolean/bitwise logic, unary minus/not/invert, ``x if c else y``;
* ``df['column']`` / ``df[['a', 'b']]`` / ``df[mask]`` subscripts;
* calls to whitelisted ``np.*`` / ``pd.*`` functions, whitelisted Series/DataFrame methods (including
  ``.str.*`` and ``.dt.*`` accessors) and a few builtins (``abs``, ``round``, ``min``, ``max``, ...);
* literals: numbers, strings, booleans, ``None``, lists and tuples of allowed expressions.

Everything else (attribute names starting with ``_``, lambdas, comprehensions, f-strings, imports,
``getattr``, ``open``, ``.apply``, ``.eval``, ``.to_pickle`` ...) is rejected *before* evaluation, and
evaluation itself runs with empty builtins and facade modules as defence in depth.

What an expression builds is bounded too, since a formula written by a language model is tried on the data before
anyone reads it (``9 ** 9 ** 8`` would keep the server busy for minutes), in two places:

* when it is read, the sizes written as numbers: a constant, a power of constants, an exponent, the size of an array
  built by ``np.full``, ``np.ones`` or ``np.zeros``, a padding width, a text repeated a constant number of times. A
  size is read through ``int``, ``float``, ``abs``, ``round``, ``max`` and ``min`` too (``str.pad(int(1e8))``);
* when it is computed, what each step builds, whatever its sizes are made of: the characters of a text or of a
  column of texts that is repeated (``*``, ``mul``, ``np.multiply``), padded, accumulated or rewritten
  (``str.replace``); the values of an array, with the width a text gives each of them (``np.full``, ``np.where``,
  ``np.select``, ``np.array``); the elements of a repeated list; the intervals asked of ``pd.cut`` and ``pd.qcut``
  as a number; the digits of a whole number raised to a power. Each size passing on its own bounds nothing: two
  products of 10,000 build a hundred million copies, and ``np.ones(len(df) ** 4)`` asks for a size that is no
  number written in the expression. One budget holds for a step: ``MAX_ELEMENTS`` values or characters, or
  ``MAX_PER_ROW`` characters a row in a table of more rows (codes padded to twelve characters are not refused
  in a table of a million rows).

What is not bounded: the time a pattern takes (pandas runs it with an engine that backtracks). An expression nested
too deep to be walked is refused as too complex.
"""

from __future__ import annotations

import ast
import copy
import math
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


class UnsafeExpressionError(ValueError):
    """The expression uses a construct that is not allowed."""


# what an expression may ask for (far beyond any clinical formula, far below what keeps a server busy)
MAX_CONSTANT = 1e15  # a constant, or a power of constants
MAX_EXPONENT = 1_000.0
MAX_ELEMENTS = (
    10_000_000  # what one step builds: the values of an array or of a list, the characters of texts
)
MAX_PER_ROW = (
    100  # or this many characters a row, in a table of more rows than MAX_ELEMENTS / MAX_PER_ROW
)
MAX_WIDTH = 10_000  # a padding width, the times a text is repeated, the intervals asked of pd.cut as a number
MAX_BITS = 1_024  # a whole number computed by a power: beyond the range of a float, nothing is a measurement
_SIZED_ARRAYS = frozenset({"full", "ones", "zeros"})
_WIDTHS = frozenset({"pad", "zfill", "center", "ljust", "rjust"})
_SIZE_KEYWORDS = frozenset({"shape", "width"})  # a size given by its name is a size too
_TOO_DEEP = "Expression too complex (nested too deep to be read)."


def _constant(node: ast.AST) -> float | None:
    """The value of a subtree made of numbers and arithmetic only, computed with floats (None when it uses a name, a
    call or text); UnsafeExpressionError when it is too large to be a measurement."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            return None
        value = float(node.value)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        operand = _constant(node.operand)
        if operand is None:
            return None
        value = -operand if isinstance(node.op, ast.USub) else operand
    elif isinstance(node, ast.BinOp):
        left, right = _constant(node.left), _constant(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Pow):
                if abs(right) > MAX_EXPONENT:
                    raise UnsafeExpressionError(
                        f"An exponent larger than {MAX_EXPONENT:,.0f} is not allowed."
                    )
                value = math.pow(left, right)
            elif isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div | ast.FloorDiv | ast.Mod):
                value = (
                    0.0 if right == 0 else left / right
                )  # the evaluation says what dividing by zero gives
            else:
                return None
        except OverflowError, ValueError:
            raise UnsafeExpressionError(
                "A number this large is not allowed in a formula."
            ) from None
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FOLDED
        and node.args
        and not node.keywords
    ):
        # a number passed through int, round, abs, max or min is a number still: int(1e8) is a width of 100 million
        given = [_constant(arg) for arg in node.args]
        if any(one is None for one in given):
            return None
        try:
            value = float(_FOLDED[node.func.id]([one for one in given if one is not None]))
        except OverflowError, ValueError, TypeError, IndexError:
            return None
    else:
        return None
    if math.isinf(value) or abs(value) > MAX_CONSTANT:
        raise UnsafeExpressionError("A number this large is not allowed in a formula.")
    return value


# what the functions of numbers that an expression may call make of constants (the first for one argument)
_FOLDED: dict[str, Any] = {
    "int": lambda given: math.trunc(given[0]),
    "float": lambda given: given[0],
    "abs": lambda given: abs(given[0]),
    "round": lambda given: round(given[0], int(given[1])) if len(given) > 1 else round(given[0]),
    "max": max,
    "min": min,
}


_NP_FUNCTIONS = frozenset(
    {
        "abs",
        "absolute",
        "add",
        "arccos",
        "arcsin",
        "arctan",
        "array",
        "asarray",
        "cbrt",
        "ceil",
        "clip",
        "cos",
        "cumsum",
        "deg2rad",
        "diff",
        "divide",
        "e",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "fmod",
        "full",
        "hypot",
        "inf",
        "isfinite",
        "isinf",
        "isnan",
        "log",
        "log10",
        "log1p",
        "log2",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "max",
        "maximum",
        "mean",
        "median",
        "min",
        "minimum",
        "mod",
        "multiply",
        "nan",
        "nanmax",
        "nanmean",
        "nanmedian",
        "nanmin",
        "nanpercentile",
        "nanstd",
        "nansum",
        "ones",
        "percentile",
        "pi",
        "power",
        "rad2deg",
        "rint",
        "round",
        "select",
        "sign",
        "sin",
        "sqrt",
        "square",
        "std",
        "subtract",
        "sum",
        "tan",
        "trunc",
        "true_divide",
        "where",
        "zeros",
    }
)
_PD_FUNCTIONS = frozenset(
    {
        "DateOffset",
        "NA",
        "NaT",
        "Timedelta",
        "Timestamp",
        "cut",
        "isna",
        "isnull",
        "notna",
        "notnull",
        "qcut",
        "to_datetime",
        "to_numeric",
        "to_timedelta",
    }
)
_SERIES_METHODS = frozenset(
    {
        "abs",
        "add",
        "all",
        "any",
        "astype",
        "between",
        "clip",
        "count",
        "cummax",
        "cummin",
        "cumsum",
        "diff",
        "div",
        "dt",
        "eq",
        "fillna",
        "floordiv",
        "ge",
        "gt",
        "isin",
        "isna",
        "isnull",
        "le",
        "lt",
        "mask",
        "max",
        "mean",
        "median",
        "min",
        "mod",
        "mul",
        "ne",
        "notna",
        "notnull",
        "nunique",
        "pct_change",
        "pow",
        "quantile",
        "rank",
        "replace",
        "round",
        "shift",
        "std",
        "str",
        "sub",
        "sum",
        "truediv",
        "var",
        "where",
    }
)
_STR_METHODS = frozenset(
    {
        "capitalize",
        "contains",
        "endswith",
        "fullmatch",
        "get",
        "isalnum",
        "isalpha",
        "isdigit",
        "isnumeric",
        "len",
        "lower",
        "lstrip",
        "match",
        "pad",
        "replace",
        "rstrip",
        "slice",
        "split",
        "startswith",
        "strip",
        "title",
        "upper",
        "zfill",
    }
)
_DT_ATTRIBUTES = frozenset(
    {
        "date",
        "day",
        "day_name",
        "dayofweek",
        "dayofyear",
        "days",
        "days_in_month",
        "daysinmonth",
        "floor",
        "hour",
        "is_leap_year",
        "minute",
        "month",
        "month_name",
        "normalize",
        "quarter",
        "second",
        "seconds",
        "strftime",
        "total_seconds",
        "weekday",
        "year",
    }
)
_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "str": str,
}

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Invert,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Subscript,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.List,
    ast.Tuple,
    ast.IfExp,
    ast.Slice,
)
_CONSTANT_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class ExpressionPolicy:
    """What an expression may reference. Instances are immutable; derive variants with ``dataclasses.replace``."""

    max_length: int = 3000
    max_nodes: int = 500
    frame_names: frozenset[str] = frozenset({"df"})
    np_functions: frozenset[str] = _NP_FUNCTIONS
    pd_functions: frozenset[str] = _PD_FUNCTIONS
    series_methods: frozenset[str] = _SERIES_METHODS
    str_methods: frozenset[str] = _STR_METHODS
    dt_attributes: frozenset[str] = _DT_ATTRIBUTES
    builtins: Mapping[str, Any] = field(default_factory=lambda: dict(_BUILTINS))


DEFAULT_POLICY = ExpressionPolicy()


def _np_facade(policy: ExpressionPolicy, rows: int = 0) -> SimpleNamespace:
    """numpy as an expression sees it: the allowed functions, those that build by a size checked when called
    (``rows``: the rows of the table the expression reads, which an array may always have)."""
    allowed = {name: getattr(np, name) for name in policy.np_functions if hasattr(np, name)}
    for name in _SIZED_ARRAYS & allowed.keys():
        allowed[name] = _array_of(allowed[name], rows)
    for name, guard in _NP_GUARDS.items():
        if name in allowed:
            allowed[name] = guard(allowed[name])
    return SimpleNamespace(**allowed)


def _pd_facade(policy: ExpressionPolicy) -> SimpleNamespace:
    allowed = {name: getattr(pd, name) for name in policy.pd_functions if hasattr(pd, name)}
    for name in _BINNED & allowed.keys():
        allowed[name] = _binned(allowed[name])
    return SimpleNamespace(**allowed)


def _size(node: ast.AST) -> float | None:
    """A size as a number: a constant, or the product of a shape's dimensions (None when one is not a constant)."""
    if isinstance(node, ast.Tuple | ast.List):
        total = 1.0
        for item in node.elts:
            one = _constant(item)
            if one is None:
                return None
            total *= one
        return total
    return _constant(node)


def _repeats(node: ast.AST) -> float:
    """How many times a text written in the expression is repeated by ``node``, a product of constants however it
    is built: ``('x' * 10000) * 10000`` repeats it a hundred million times. 0 when ``node`` repeats no such text."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return 1.0
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        for text, times in ((node.left, node.right), (node.right, node.left)):
            inner, count = _repeats(text), _constant(times)
            if inner and count is not None:
                return inner * max(count, 0.0)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return max(_repeats(node.left), _repeats(node.right))
    return 0.0


class _Validator(ast.NodeVisitor):
    def __init__(self, policy: ExpressionPolicy, extra_names: frozenset[str]) -> None:
        self.policy = policy
        self.extra_names = extra_names
        self.count = 0

    # -- generic gate ---------------------------------------------------------------
    def visit(self, node: ast.AST) -> Any:
        self.count += 1
        if self.count > self.policy.max_nodes:
            raise UnsafeExpressionError(
                f"Expression too complex (more than {self.policy.max_nodes} nodes)."
            )
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpressionError(f"'{type(node).__name__}' is not allowed in expressions.")
        return super().visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, _CONSTANT_TYPES):
            raise UnsafeExpressionError(
                f"Literal of type {type(node.value).__name__} is not allowed."
            )

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            raise UnsafeExpressionError("Assignment is not allowed in expressions.")
        allowed = (
            node.id in self.policy.frame_names
            or node.id in ("np", "pd")
            or node.id in self.policy.builtins
            or node.id in self.extra_names
        )
        if not allowed:
            raise UnsafeExpressionError(
                f"Unknown name '{node.id}'. Refer to columns as df['column']."
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr.startswith("_"):
            raise UnsafeExpressionError(f"Attribute '{attr}' is not allowed.")
        base = node.value
        if isinstance(base, ast.Name) and base.id == "np":
            if attr not in self.policy.np_functions:
                raise UnsafeExpressionError(f"np.{attr} is not allowed.")
        elif isinstance(base, ast.Name) and base.id == "pd":
            if attr not in self.policy.pd_functions:
                raise UnsafeExpressionError(f"pd.{attr} is not allowed.")
        elif isinstance(base, ast.Attribute) and base.attr == "str":
            if attr not in self.policy.str_methods:
                raise UnsafeExpressionError(f".str.{attr} is not allowed.")
        elif isinstance(base, ast.Attribute) and base.attr == "dt":
            if attr not in self.policy.dt_attributes:
                raise UnsafeExpressionError(f".dt.{attr} is not allowed.")
        else:
            if attr not in self.policy.series_methods:
                raise UnsafeExpressionError(f"Method or attribute '{attr}' is not allowed on data.")
        self.visit(base)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # the sizes a formula can ask for: a power of constants, an exponent, a text repeated a constant number of times
        _constant(node)
        if isinstance(node.op, ast.Pow):
            exponent = _constant(node.right)
            if exponent is not None and abs(exponent) > MAX_EXPONENT:
                raise UnsafeExpressionError(
                    f"An exponent larger than {MAX_EXPONENT:,.0f} is not allowed."
                )
        if isinstance(node.op, ast.Mult) and _repeats(node) > MAX_WIDTH:
            raise UnsafeExpressionError(
                f"A text repeated more than {MAX_WIDTH:,} times is not allowed."
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in self.policy.builtins and func.id not in self.extra_names:
                raise UnsafeExpressionError(f"Function '{func.id}' is not allowed.")
        elif isinstance(func, ast.Attribute):
            self.visit(func)
            # a size is the first argument, or the one named shape or width; a shape of several dimensions counts
            # their product
            asked = [
                *node.args[:1],
                *(kw.value for kw in node.keywords if kw.arg in _SIZE_KEYWORDS),
            ]
            for size in (_size(given) for given in asked):
                if size is not None and func.attr in _SIZED_ARRAYS and abs(size) > MAX_ELEMENTS:
                    raise UnsafeExpressionError(
                        f"An array of more than {MAX_ELEMENTS:,} values is not allowed."
                    )
                if size is not None and func.attr in _WIDTHS and abs(size) > MAX_WIDTH:
                    raise UnsafeExpressionError(
                        f"A width larger than {MAX_WIDTH:,} characters is not allowed."
                    )
        else:
            raise UnsafeExpressionError("Only named functions and methods may be called.")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.arg is None or kw.arg.startswith("_"):
                raise UnsafeExpressionError(
                    "Keyword unpacking and private keywords are not allowed."
                )
            if kw.arg == "inplace" and not (
                isinstance(kw.value, ast.Constant) and kw.value.value is False
            ):
                # df.replace(97, 60, inplace=True) changed the table a criterion was evaluated on, outside the trace;
                # inplace=False changes nothing, and criteria written with it are stored
                raise UnsafeExpressionError(
                    "An expression reads the data and never changes it: 'inplace' can only be False."
                )
            self.visit(kw.value)

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.visit(child)


def validate_tree(
    tree: ast.Expression,
    *,
    policy: ExpressionPolicy = DEFAULT_POLICY,
    extra_names: frozenset[str] | set[str] | None = None,
) -> ast.Expression:
    """Check an already parsed expression tree; returns it unchanged or raises :class:`UnsafeExpressionError`."""
    if not isinstance(tree, ast.Expression):
        raise UnsafeExpressionError("Only single expressions are allowed.")
    try:
        _Validator(policy, frozenset(extra_names or ())).visit(tree)
    except RecursionError:
        # a few hundred comparisons joined one after the other are a tree too deep to walk: refused as too complex,
        # like an expression of too many nodes, and not as a failure of the application
        raise UnsafeExpressionError(_TOO_DEEP) from None
    return tree


def validate(
    expression: str,
    *,
    policy: ExpressionPolicy = DEFAULT_POLICY,
    extra_names: frozenset[str] | set[str] | None = None,
) -> ast.Expression:
    """Parse and check an expression string; raises :class:`UnsafeExpressionError` (or SyntaxError) if invalid."""
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpressionError("The expression is empty.")
    if len(expression) > policy.max_length:
        raise UnsafeExpressionError(f"Expression longer than {policy.max_length} characters.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Syntax error: {exc.msg} (position {exc.offset}).") from exc
    except RecursionError, MemoryError:
        raise UnsafeExpressionError(_TOO_DEEP) from None
    return validate_tree(tree, policy=policy, extra_names=extra_names)


_TIMES = "__times__"  # the name a product is computed under when one of its sides is not a number written in it
_POWER = "__power__"  # a power, likewise
_METHOD = (
    "__method__"  # a method that repeats, pads, accumulates or rewrites, called through its check
)
_BUILDS = "An expression that builds more than {limit:,} {what} is not allowed."


def _budget(rows: int) -> int:
    """What one step may build: values or characters."""
    return max(MAX_ELEMENTS, MAX_PER_ROW * rows)


def _over(built: float, rows: int, what: str = "characters") -> None:
    if built > _budget(rows):
        raise UnsafeExpressionError(_BUILDS.format(limit=_budget(rows), what=what))


def _is_text(kind: Any) -> bool:
    return bool(pd.api.types.is_string_dtype(kind) or pd.api.types.is_object_dtype(kind))


def _holds_text(value: Any) -> bool:
    """Whether a value is a text, or a column, an array or a table that holds texts."""
    if isinstance(value, str):
        return True
    if isinstance(value, pd.DataFrame):
        return any(_is_text(kind) for kind in value.dtypes)
    kind = getattr(value, "dtype", None)
    if kind is None or not isinstance(value, pd.Series | pd.Index | np.ndarray):
        return False
    return _is_text(kind)


def _rows(value: Any) -> int:
    """The rows of a column, an array or a table; 1 for a single value."""
    if isinstance(value, pd.DataFrame | pd.Series | pd.Index):
        return len(value)
    if isinstance(value, np.ndarray):
        return int(value.shape[0]) if value.ndim else 1
    return 1


def _characters(value: Any) -> tuple[int, int]:
    """The characters of a text, or of the texts of a column, an array or a table: all of them, and the longest."""
    if isinstance(value, str):
        return len(value), len(value)
    if isinstance(value, pd.DataFrame):
        each = [_characters(value[name]) for name, kind in value.dtypes.items() if _is_text(kind)]
        return sum(total for total, _ in each), max((longest for _, longest in each), default=0)
    column = (
        value
        if isinstance(value, pd.Series)
        else pd.Series(np.asarray(value, dtype=object).ravel())
    )
    try:
        lengths = pd.to_numeric(column.str.len(), errors="coerce").fillna(0)
    except AttributeError:  # a column of objects that holds no text
        lengths = column.map(lambda item: len(item) if isinstance(item, str) else 0)
    return (int(lengths.sum()), int(lengths.max())) if len(lengths) else (0, 0)


def _largest(value: Any) -> float:
    """The largest whole number a value holds, as a count of repetitions (0 for anything else)."""
    if isinstance(value, bool | np.bool_):
        return 0.0
    if isinstance(value, int | np.integer):
        return float(abs(int(value)))
    if isinstance(value, pd.DataFrame):
        return max((_largest(value[name]) for name in value.columns), default=0.0)
    kind = getattr(value, "dtype", None)
    counted = kind is not None and isinstance(value, pd.Series | pd.Index | np.ndarray)
    if counted and len(value) and pd.api.types.is_integer_dtype(kind):
        return float(np.nanmax(np.abs(np.asarray(value, dtype="float64"))))
    return 0.0


def _weight(value: Any, seen: dict[int, float] | None = None) -> float:
    """What a list is once laid out: its values, a column or an array counting for its own, a text for its
    characters. A list repeated holds one list many times: each is weighed once."""
    seen = {} if seen is None else seen
    if isinstance(value, list | tuple):
        known = seen.get(id(value))
        if known is None:
            known = 0.0
            for item in value:
                known += _weight(item, seen)
                if known > MAX_ELEMENTS:
                    break
            seen[id(value)] = known
        return known
    if isinstance(value, pd.DataFrame | pd.Series | pd.Index | np.ndarray):
        return float(max(np.size(value), 1))
    return float(max(len(value), 1)) if isinstance(value, str) else 1.0


def _check_times(left: Any, right: Any) -> None:
    """Refuse a product that repeats a text more than ``MAX_WIDTH`` times, or that builds more than the budget:
    ``(df['site'] * 10000) * 10000`` is two products of 10,000 and a hundred million copies."""
    for held, count in ((left, right), (right, left)):
        most = _largest(count)
        if _holds_text(held):
            if most > MAX_WIDTH:
                raise UnsafeExpressionError(
                    f"A text repeated more than {MAX_WIDTH:,} times is not allowed."
                )
            if most > 1:
                rows = max(_rows(held), _rows(count))
                total, _ = _characters(held)
                # a text alone against a column of counts is repeated for each of them
                _over(total * most * (rows if isinstance(held, str) else 1), rows)
        elif isinstance(held, list | tuple) and most > 1:
            _over(_weight(held) * most, 0, "values")


def _times(left: Any, right: Any) -> Any:
    """``left * right``, once :func:`_check_times` has passed it. A text written in the expression is counted when
    the expression is read; a column of texts is known only when it is computed: ``df['site'] * 100000000`` builds
    a hundred million copies of each site's name."""
    _check_times(left, right)
    return left * right


def _whole(value: Any) -> bool:
    return type(value) is int


def _unbounded(value: Any) -> int | None:
    """The largest whole number, in size, among those of ``value`` that Python computes without a limit: a whole
    number itself, or those of a column kept as objects. None for anything else: an array of numbers has a fixed
    width, and a power in it wraps or becomes infinite."""
    if _whole(value):
        return abs(value)
    if isinstance(value, pd.Series | pd.Index | np.ndarray) and pd.api.types.is_object_dtype(
        value.dtype
    ):
        items = np.asarray(value, dtype=object).ravel()
        return max((abs(item) for item in items if _whole(item)), default=None)
    return None


def _check_power(base: Any, exponent: Any) -> None:
    """Refuse a power of whole numbers of more than ``MAX_BITS`` bits: ``len(df) ** len(df) ** len(df)`` never
    ends, and a column of whole numbers kept as objects raised to 1,000 holds thousands of digits a row."""
    largest, raised = _unbounded(base), _unbounded(exponent)
    if largest is None or raised is None or isinstance(exponent, int) and exponent < 0:
        return
    if largest > 1 and raised * largest.bit_length() > MAX_BITS:
        raise UnsafeExpressionError("A number this large is not allowed in a formula.")


def _power(base: Any, exponent: Any) -> Any:
    _check_power(base, exponent)
    return base**exponent


def _check_width(receiver: Any, width: Any) -> None:
    """A padding width known only when it is computed, and what it builds over the rows."""
    if isinstance(width, bool) or not isinstance(width, int | np.integer):
        return
    if abs(int(width)) > MAX_WIDTH:
        raise UnsafeExpressionError(f"A width larger than {MAX_WIDTH:,} characters is not allowed.")
    rows = _rows(receiver)
    _over(rows * abs(int(width)), rows)


def _check_replace(receiver: Any, args: tuple, kwargs: dict) -> None:
    """What ``str.replace`` builds: each match becomes the replacement, and an empty pattern matches between every
    two characters (a text of 10,000 characters put there, twice, is 10,000 times 10,000 times the column)."""
    pattern = args[0] if args else kwargs.get("pat")
    given = args[1] if len(args) > 1 else kwargs.get("repl")
    if not isinstance(pattern, str) or not isinstance(given, str) or len(given) <= 1:
        return
    regex = bool(args[5] if len(args) > 5 else kwargs.get("regex", False))
    rows = _rows(receiver)
    total, longest = _characters(receiver)
    # a replacement that names the match, or a group of it, holds the matched text as often
    named = len(re.findall(r"\\(?:\d|g<)", given)) if regex else 0
    each = len(given) + named * longest
    if rows * (longest + 1) * each <= _budget(
        rows
    ):  # every position a match: within the budget anyway
        return
    flags = args[4] if len(args) > 4 else kwargs.get("flags", 0)
    case = args[3] if len(args) > 3 else kwargs.get("case")
    if case is False:
        flags = int(flags or 0) | re.IGNORECASE
    matches = receiver.str.count(pattern if regex else re.escape(pattern), flags=int(flags or 0))
    _over(total + float(matches.sum()) * each, rows)


def _check_cumsum(receiver: Any) -> None:
    """The cumulative sum of a column of texts holds, row after row, every text before it."""
    if _holds_text(receiver) and not isinstance(receiver, str):
        rows = _rows(receiver)
        total, _ = _characters(receiver)
        _over(total * (rows + 1) / 2, rows)


def _method(receiver: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """``receiver.name(...)`` for the methods that build by a size, checked before they do."""
    if name in ("str.pad", "str.zfill"):
        _check_width(receiver, args[0] if args else kwargs.get("width"))
    elif name == "str.replace":
        if _holds_text(receiver) and isinstance(receiver, pd.Series | pd.Index):
            _check_replace(receiver, args, kwargs)
    elif name == "mul":
        _check_times(receiver, args[0] if args else kwargs.get("other"))
    elif name == "pow":
        _check_power(receiver, args[0] if args else kwargs.get("other"))
    elif name == "cumsum":
        _check_cumsum(receiver)
    target = receiver
    *through, last = name.split(".")
    for part in through:
        target = getattr(target, part)
    return getattr(target, last)(*args, **kwargs)


_GUARDED_METHODS = frozenset({"mul", "pow", "cumsum"})  # of a column or a table
_GUARDED_STR_METHODS = frozenset({"pad", "zfill", "replace"})  # of its texts


def _fixed_text(value: Any) -> int:
    """The longest text among the single values an array is built from: numpy gives every value that width."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list | tuple):
        return max((_fixed_text(item) for item in value[:MAX_WIDTH]), default=0)
    return 0


def _array_of(function: Callable[..., Any], rows: int) -> Callable[..., Any]:
    """``np.full``, ``np.ones`` or ``np.zeros`` with its size checked when it is called: a shape read from the
    data is no number written in the expression (``np.ones(len(df) ** 4)``)."""

    def sized(*args: Any, **kwargs: Any) -> Any:
        shape: Any = args[0] if args else kwargs.get("shape")
        try:
            dimensions = [operator.index(shape)]
        except TypeError:
            try:
                dimensions = [operator.index(one) for one in shape]
            except TypeError:
                dimensions = []  # numpy says what is wrong with it
        asked = math.prod(abs(one) for one in dimensions) if dimensions else 0
        if asked > max(MAX_ELEMENTS, rows):
            raise UnsafeExpressionError(
                f"An array of more than {MAX_ELEMENTS:,} values is not allowed."
            )
        fill = args[1] if len(args) > 1 else kwargs.get("fill_value")
        if isinstance(fill, str):  # every value as wide as the text
            _over(asked * max(len(fill), 1), asked)
        return function(*args, **kwargs)

    return sized


def _chosen(function: Callable[..., Any]) -> Callable[..., Any]:
    """``np.where`` and ``np.select``: a text given as a single value is the width of every value of the result."""

    def chosen(*args: Any, **kwargs: Any) -> Any:
        widest = max((_fixed_text(one) for one in (*args[1:], *kwargs.values())), default=0)
        if widest > MAX_PER_ROW:
            first = args[0] if args else None
            if isinstance(first, list | tuple) and first:  # np.select: a list of conditions
                first = first[0]
            rows = int(np.size(first)) if first is not None else 1
            _over(rows * widest, rows)
        return function(*args, **kwargs)

    return chosen


def _laid_out(function: Callable[..., Any]) -> Callable[..., Any]:
    """``np.array`` and ``np.asarray`` of a list: its values, each as wide as its longest text."""

    def laid_out(*args: Any, **kwargs: Any) -> Any:
        given = args[0] if args else None
        if isinstance(given, list | tuple):
            _over(_weight(given), 0, "values")
            widest = _fixed_text(given)
            if widest > MAX_PER_ROW:
                _over(len(given) * widest, len(given))
        return function(*args, **kwargs)

    return laid_out


def _multiplied(function: Callable[..., Any]) -> Callable[..., Any]:
    def multiplied(*args: Any, **kwargs: Any) -> Any:
        if len(args) >= 2:
            _check_times(args[0], args[1])
        return function(*args, **kwargs)

    return multiplied


def _raised(function: Callable[..., Any]) -> Callable[..., Any]:
    def raised(*args: Any, **kwargs: Any) -> Any:
        if len(args) >= 2:
            _check_power(args[0], args[1])
        return function(*args, **kwargs)

    return raised


def _accumulated(function: Callable[..., Any]) -> Callable[..., Any]:
    def accumulated(*args: Any, **kwargs: Any) -> Any:
        if args:
            _check_cumsum(args[0])
        return function(*args, **kwargs)

    return accumulated


_NP_GUARDS: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]] = {
    "where": _chosen,
    "select": _chosen,
    "array": _laid_out,
    "asarray": _laid_out,
    "multiply": _multiplied,
    "power": _raised,
    "cumsum": _accumulated,
}
_BINNED = frozenset({"cut", "qcut"})


def _binned(function: Callable[..., Any]) -> Callable[..., Any]:
    """``pd.cut`` and ``pd.qcut``: the intervals asked for as a number (edges given as a list are written in the
    expression, and counted there)."""

    def binned(*args: Any, **kwargs: Any) -> Any:
        asked = args[1] if len(args) > 1 else kwargs.get("bins", kwargs.get("q"))
        if (
            isinstance(asked, int | np.integer)
            and not isinstance(asked, bool)
            and abs(int(asked)) > MAX_WIDTH
        ):
            raise UnsafeExpressionError(f"More than {MAX_WIDTH:,} intervals are not allowed.")
        return function(*args, **kwargs)

    return binned


class _Guards(ast.NodeTransformer):
    """The tree as it is computed: each product and each power with a side that is not a number written in the
    expression, and each call of a method that builds by a size, go through their check."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        name = _guard_of(node)
        if name is None:
            return node
        call = ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()), args=[node.left, node.right], keywords=[]
        )
        return ast.copy_location(call, node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        found = _guarded_method(node)
        if found is None:
            return node
        receiver, name = found
        call = ast.Call(
            func=ast.Name(id=_METHOD, ctx=ast.Load()),
            args=[receiver, ast.Constant(value=name), *node.args],
            keywords=node.keywords,
        )
        return ast.copy_location(call, node)


def _both_numbers(node: ast.BinOp) -> bool:
    return all(
        isinstance(side, ast.Constant)
        and isinstance(side.value, int | float)
        and not isinstance(side.value, bool)
        for side in (node.left, node.right)
    )


def _guard_of(node: ast.AST) -> str | None:
    """The check a product or a power is computed through; None for any other node, and for two numbers."""
    if not isinstance(node, ast.BinOp) or _both_numbers(node):
        return None
    if isinstance(node.op, ast.Mult):
        return _TIMES
    return _POWER if isinstance(node.op, ast.Pow) else None


def _guarded_method(node: ast.AST) -> tuple[ast.expr, str] | None:
    """The receiver and the name of a call that goes through :func:`_method`: ``x.mul(...)``, ``x.pow(...)``,
    ``x.cumsum(...)``, and ``x.str.pad(...)``, ``x.str.zfill(...)``, ``x.str.replace(...)``. numpy's and pandas'
    own functions are checked in their facades."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    name, receiver = node.func.attr, node.func.value
    if isinstance(receiver, ast.Name) and receiver.id in ("np", "pd"):
        return None
    if isinstance(receiver, ast.Attribute) and receiver.attr == "str":
        return (receiver.value, f"str.{name}") if name in _GUARDED_STR_METHODS else None
    return (receiver, name) if name in _GUARDED_METHODS else None


def _guarded(tree: ast.Expression) -> ast.Expression:
    """The tree as it is evaluated: a copy with its checks, or the tree itself when it has nothing to check (the
    caller's tree is never changed: it may be evaluated again, and would then not pass the validation)."""
    checked = (
        _guard_of(node) is not None or _guarded_method(node) is not None for node in ast.walk(tree)
    )
    if not any(checked):
        return tree
    try:
        return ast.fix_missing_locations(_Guards().visit(copy.deepcopy(tree)))
    except RecursionError:
        raise UnsafeExpressionError(_TOO_DEEP) from None


def evaluate_tree(
    tree: ast.Expression,
    df: pd.DataFrame | None = None,
    *,
    policy: ExpressionPolicy = DEFAULT_POLICY,
    extra_locals: Mapping[str, Any] | None = None,
) -> Any:
    """Validate and evaluate a tree with empty builtins and restricted ``np`` / ``pd`` facades."""
    extra_locals = dict(extra_locals or {})
    validate_tree(tree, policy=policy, extra_names=frozenset(extra_locals))
    try:
        ast.fix_missing_locations(tree)
        code = compile(_guarded(tree), filename="<expression>", mode="eval")
    except RecursionError:
        raise UnsafeExpressionError(_TOO_DEEP) from None
    namespace: dict[str, Any] = {
        "__builtins__": dict(policy.builtins),
        "np": _np_facade(policy, 0 if df is None else len(df)),
        "pd": _pd_facade(policy),
        _TIMES: _times,
        _POWER: _power,
        _METHOD: _method,
    }
    namespace.update(extra_locals)
    # the expression gets a table of its own, which shares the caller's values until one is written (copy-on-write):
    # whatever a method does in place, the caller's table stays as it was
    own = None if df is None else df.copy(deep=False)
    for name in policy.frame_names:
        namespace[name] = own
    return eval(code, namespace, {})


def evaluate(
    expression: str,
    df: pd.DataFrame | None = None,
    *,
    policy: ExpressionPolicy = DEFAULT_POLICY,
    extra_locals: Mapping[str, Any] | None = None,
) -> Any:
    """Validate and evaluate an expression string against ``df``."""
    tree = validate(expression, policy=policy, extra_names=frozenset(extra_locals or {}))
    return evaluate_tree(tree, df, policy=policy, extra_locals=extra_locals)


def column_references(expression: str, *, policy: ExpressionPolicy = DEFAULT_POLICY) -> set[str]:
    """Column names referenced as ``df['name']`` (used to report missing variables before evaluation)."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in policy.frame_names
        ):
            target = node.slice
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                names.add(target.value)
            elif isinstance(target, (ast.List, ast.Tuple)):
                names.update(
                    e.value
                    for e in target.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )
    return names


__all__ = [
    "DEFAULT_POLICY",
    "ExpressionPolicy",
    "UnsafeExpressionError",
    "column_references",
    "evaluate",
    "evaluate_tree",
    "validate",
    "validate_tree",
]
