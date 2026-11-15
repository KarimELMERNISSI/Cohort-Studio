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
  column of texts that is repeated (``*``, ``mul``, ``np.multiply``), joined to another (``+``, ``add``,
  ``np.add``), padded or rewritten (``str.replace``); a text written into every row (``replace``, ``where``,
  ``mask``, ``fillna``); the values of an array, with the width a text or a fixed-width text type gives each of
  them, an array of such a type included, alone or in a list (``np.full``, ``np.ones``, ``np.zeros``,
  ``np.where``, ``np.select``, ``np.array``, ``astype``, ``+``); the elements of a repeated list; what a selection
  by a computed list of labels selects, a label held several times counted each time (``df[['age'] * 10000]``,
  ``df['age'][[0] * 3500][[0] * 3500]``); the rows of columns and tables combined by an operator, a method or a
  function of two of them, which pandas first aligns by their labels (a label held a times on one side and b
  times on the other is a times b rows: ``df['age'][[0] * 3200] - df['age'][[0] * 3300]``); the intervals asked
  of ``pd.cut`` and ``pd.qcut`` as a number; the digits of a whole number raised to a power; the order of
  ``np.diff``, which is a pass over its values each. A sum of texts (``sum``, ``np.sum``, ``np.nansum``) and a
  cumulative sum of texts (``cumsum``, ``np.cumsum``) are refused: they have no use in a formula, and numpy joins
  texts one by one, in a time that grows as the square of their length. So is a text formatted with ``%``, whose
  width is a number written inside the text (``'%20000000d' % 1``). Each size passing on its own bounds nothing: two
  products of 10,000 build a hundred million copies, twenty padded columns added are twenty times the column, and
  ``np.ones(len(df) ** 4)`` asks for a size that is no number written in the expression. One budget holds for a
  step: ``MAX_ELEMENTS`` values or characters, or ``MAX_PER_ROW`` characters a row in a table of more rows (codes
  padded to twelve characters are not refused in a table of a million rows).

What is not bounded: the time a pattern takes (pandas runs it with an engine that backtracks), what several steps
hold together (each step has the budget: an expression of a few steps holds a few times as much), and what a step
builds by a way these checks do not see. The budget refuses what the checks see before it is built; it is no
memory limit and no time limit, which a process has only from outside (a worker with both is planned). An
expression nested too deep to be walked is refused as too complex.
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

from core.expressions.errors import UnsafeExpressionError
from core.expressions.runner import installed

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
    for name in list(allowed):
        # a function of two values aligns two columns by their labels first (np.subtract, np.maximum, ...), and
        # np.clip calls the column's own clip, which aligns its bounds
        found = getattr(np, name)
        if isinstance(found, np.ufunc) and found.nin == 2 or name == "clip":
            allowed[name] = _aligned_call(allowed[name])
    return SimpleNamespace(**allowed)


def _aligned_call(function: Callable[..., Any]) -> Callable[..., Any]:
    """A numpy function given two columns or tables, counted as pandas aligns them (:func:`_check_aligned`)."""

    def aligned(*args: Any, **kwargs: Any) -> Any:
        _check_aligned(*args, *kwargs.values())
        return function(*args, **kwargs)

    return aligned


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
_PLUS = "__plus__"  # a sum, likewise: of texts, it joins them
_ITEM = "__item__"  # a selection by a key that is computed
_OPERATED = "__operated__"  # any other operator of two values: columns are aligned by their labels
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


_LONG_LIST = 1_000  # a list of more items than this is first read by the types of its items
_SINGLE_VALUES = frozenset({int, float, bool, complex, type(None)})


def _weight(value: Any, seen: dict[int, float] | None = None) -> float:
    """What a list is once laid out: its values, a column or an array counting for its own, a text for its
    characters. A list repeated holds one list many times: each is weighed once."""
    seen = {} if seen is None else seen
    if isinstance(value, list | tuple):
        known = seen.get(id(value))
        if known is None:
            if len(value) > _LONG_LIST and set(map(type, value)) <= _SINGLE_VALUES:
                # a long list of single values weighs its length: weighed one by one, six million zeros took
                # ten seconds of the check that is there to save time
                known = float(len(value))
            else:
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


def _joined(first: pd.Index, second: pd.Index) -> int:
    """
    The labels two columns are aligned on: the same labels in the same order are read position by position;
    otherwise pandas joins them, and a label held a times on one side and b times on the other is held a times b
    times, a label of one side only as often as there.
    """
    if first.equals(second):
        return len(first)
    if first.is_unique and second.is_unique:
        return len(first) + len(second)  # at most
    try:
        left = first.value_counts(dropna=False)
        right = second.value_counts(dropna=False)
        both = left.index.intersection(right.index)
        mine = left.loc[both].to_numpy(dtype="int64")
        theirs = right.loc[both].to_numpy(dtype="int64")
    except TypeError, ValueError, KeyError:
        return len(first) * len(second)  # what a join of these labels can hold at most
    paired = int((mine * theirs).sum())
    return paired + len(first) - int(mine.sum()) + len(second) - int(theirs.sum())


def _aligned_size(first: pd.Series | pd.DataFrame, second: pd.Series | pd.DataFrame) -> int:
    """The values two columns or tables give once aligned: rows by rows, and columns by columns for two tables; a
    column beside a table is aligned on its columns, or on its rows when asked (the larger is counted)."""
    if isinstance(first, pd.Series) and isinstance(second, pd.Series):
        return _joined(first.index, second.index)
    if isinstance(first, pd.DataFrame) and isinstance(second, pd.DataFrame):
        return _joined(first.index, second.index) * _joined(first.columns, second.columns)
    table = first if isinstance(first, pd.DataFrame) else second
    column = second if table is first else first
    if not isinstance(table, pd.DataFrame) or not isinstance(column, pd.Series):
        return len(first) * len(second)
    return max(
        _joined(table.columns, column.index) * len(table),
        _joined(table.index, column.index) * table.shape[1],
    )


def _check_aligned(*sides: Any) -> None:
    """
    Columns and tables combined are first aligned by their labels, whatever they hold: numbers pass every check of
    texts, and ``df['age'][[0] * 3200] - df['age'][[0] * 3300]`` held 10,560,000 values, each selection passing on
    its own (``[0] * 30000`` against ``[0] * 30001``: 900 million). Each column or table given is counted against
    the first. A formula's columns share their labels, and are read position by position.
    """
    labelled = [side for side in sides if isinstance(side, pd.Series | pd.DataFrame)]
    for other in labelled[1:]:
        first = labelled[0]
        _over(_aligned_size(first, other), max(len(first), len(other)), "values")


def _times(left: Any, right: Any) -> Any:
    """``left * right``, once :func:`_check_times` has passed it. A text written in the expression is counted when
    the expression is read; a column of texts is known only when it is computed: ``df['site'] * 100000000`` builds
    a hundred million copies of each site's name."""
    _check_aligned(left, right)
    _check_times(left, right)
    return left * right


_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "Sub": operator.sub,
    "Div": operator.truediv,
    "FloorDiv": operator.floordiv,
    "Mod": operator.mod,
    "BitAnd": operator.and_,
    "BitOr": operator.or_,
    "BitXor": operator.xor,
}
_FORMATTED = "A text formatted with % is not allowed in a formula: join texts with +."


def _operated(left: Any, right: Any, name: str) -> Any:
    """``left - right``, ``/``, ``//``, ``%``, ``&``, ``|`` or ``^``, once the columns it combines are counted as
    aligned (:func:`_check_aligned`). A text formatted with ``%`` is refused: ``'%20000000d' % 1`` is a text of
    twenty million characters, its width a number written inside a text, which no check reads."""
    if name == "Mod" and _holds_text(left):
        raise UnsafeExpressionError(_FORMATTED)
    _check_aligned(left, right)
    return _OPERATORS[name](left, right)


def _array_width(value: Any) -> int:
    """The characters each value of an array of a fixed-width text type holds (``'U100000'``), whatever it holds:
    numpy gives every value of what is built from it at least that width. 0 for anything else."""
    if isinstance(value, np.ndarray):
        return _text_width(value.dtype)
    return 0


def _side_width(side: Any) -> int:
    """The characters a side of a sum gives each value of the result: a text's, an array's fixed width, a column's
    longest text."""
    if isinstance(side, str):
        return len(side)
    width = _array_width(side)
    if width:
        return width
    return _characters(side)[1] if _holds_text(side) else 0


def _check_plus(left: Any, right: Any) -> None:
    """What a sum builds when it joins: two texts, or two columns of texts, row by row (twenty padded columns
    added are twenty times the column, each passing on its own), two lists end to end, and an array of a
    fixed-width text type, whose width every value of the result takes. Numbers add as they did."""
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        _over(_weight(left) + _weight(right), 0, "values")
        return
    if _array_width(left) or _array_width(right):
        rows = max(
            (int(np.size(side)) for side in (left, right) if not isinstance(side, str)), default=1
        )
        _over(rows * (_side_width(left) + _side_width(right)), rows)
        return
    if not (_holds_text(left) and _holds_text(right)):
        return
    rows = max(_rows(left), _rows(right))
    joined = 0.0
    for side in (left, right):
        characters, _ = _characters(side)
        # a text alone is joined to every row of the other side
        joined += characters * (rows if isinstance(side, str) else 1)
    _over(joined, rows)


def _plus(left: Any, right: Any) -> Any:
    """``left + right``, once :func:`_check_plus` has passed it."""
    _check_aligned(left, right)
    _check_plus(left, right)
    return left + right


def _selected(axis: pd.Index, key: Any) -> int | None:
    """How many entries of ``axis`` a list of labels selects, a label that ``axis`` holds several times counted
    each time it is asked for; None for a key that is no list of labels (a mask, a slice, a single label)."""
    if isinstance(key, np.ndarray | pd.Index | pd.Series):
        if key.dtype == bool:
            return None
        labels = key.tolist()
    elif isinstance(key, list | tuple):
        labels = list(key)
    else:
        return None
    if not labels or all(isinstance(one, bool | np.bool_) for one in labels):
        return None  # a mask written as a list
    if axis.is_unique:
        return len(labels)  # each label found once, or pandas says it is missing
    counts = axis.value_counts(dropna=False)
    try:
        return sum(int(counts.get(label, 0)) for label in labels)
    except TypeError:  # a label that is no key: pandas says what is wrong with it
        return None


def _item(receiver: Any, key: Any) -> Any:
    """``receiver[key]``, the key being computed, counted by what it selects. A table asked for more columns than
    it holds is asked for some of them many times (``df[['age'] * 10000]`` is ten thousand copies of a column),
    and a label held several times is selected each time: ``df['age'][[0] * 3500]`` holds 3,500 rows labelled 0,
    and the same selection made of it again holds 12,250,000."""
    if isinstance(receiver, pd.DataFrame):
        chosen = _selected(receiver.columns, key)
        if chosen is not None and chosen > receiver.shape[1]:
            _over(len(receiver) * chosen, len(receiver), "values")
    elif isinstance(receiver, pd.Series):
        chosen = _selected(receiver.index, key)
        if chosen is not None and chosen > len(receiver):
            _over(chosen, len(receiver), "values")
    return receiver[key]


def _widest_text(value: Any, depth: int = 0) -> int:
    """The longest text among the values a method is given to write: a text, or those of a list or of a
    mapping."""
    if isinstance(value, str):
        return len(value)
    if depth > 3:
        return 0
    if isinstance(value, Mapping):
        held = [one for pair in list(value.items())[:MAX_WIDTH] for one in pair]
        return max((_widest_text(one, depth + 1) for one in held), default=0)
    if isinstance(value, list | tuple):
        return max((_widest_text(one, depth + 1) for one in value[:MAX_WIDTH]), default=0)
    return 0


_TEXT_SUM = "A sum of texts is not allowed in a formula: join texts with +."


def _strings_in(value: Any, depth: int = 0) -> bool:
    """Whether a value holds a text: a text, a column, an array or a table holding one, or a list of them (read
    once each, a list repeated holding one column many times)."""
    if isinstance(value, str | bytes):
        return True
    if isinstance(value, pd.DataFrame):
        return any(_strings_in(value.iloc[:, place], depth + 1) for place in range(value.shape[1]))
    if isinstance(value, pd.Series | pd.Index | np.ndarray):
        kind = value.dtype
        if getattr(kind, "kind", "") in "US" or isinstance(kind, pd.StringDtype):
            return True
        if not pd.api.types.is_object_dtype(kind):
            return False
        found = pd.api.types.infer_dtype(np.asarray(value, dtype=object).ravel(), skipna=True)
        return found in ("string", "bytes", "mixed", "mixed-integer")
    if isinstance(value, list | tuple) and depth < 4:
        seen: set[int] = set()
        for item in value[:MAX_WIDTH]:
            if id(item) in seen:
                continue
            seen.add(id(item))
            if _strings_in(item, depth + 1):
                return True
    return False


def _check_sum(receiver: Any, kwargs: dict) -> None:
    """A sum of texts joins them: it has no use in a formula, and numpy joins texts one by one, in a time that
    grows as the square of their length (a list of 1,100 padded columns of ten rows took 40 seconds). A table's
    sum of its numbers only is a sum of numbers."""
    if isinstance(receiver, pd.DataFrame) and kwargs.get("numeric_only") is True:
        return
    if _strings_in(receiver):
        raise UnsafeExpressionError(_TEXT_SUM)


def _check_filled(receiver: Any, args: tuple, kwargs: dict) -> None:
    """A text written into the cells of a column or of a table, at most once in each (``replace``, ``where``,
    ``mask``, ``fillna``): a text of 10,000 characters is ten thousand times the column. A label of ordinary
    length is never refused, whatever the table."""
    widest = max((_widest_text(one) for one in (*args, *kwargs.values())), default=0)
    if widest > MAX_PER_ROW:
        held = isinstance(receiver, pd.DataFrame | pd.Series | pd.Index | np.ndarray)
        cells = int(np.size(receiver)) if held else 1
        _over(cells * widest, _rows(receiver))


def _text_width(kind: Any) -> int:
    """The characters each value of a fixed-width text type holds (``'U100000'``); 0 for any other type, and for
    ``str``, which is as wide as its longest text."""
    if kind is None or kind is str:
        return 0
    try:
        found = np.dtype(kind)
    except TypeError, ValueError:
        return 0
    if found.kind == "U":
        return int(found.itemsize // 4)
    return int(found.itemsize) if found.kind == "S" else 0


def _text_type(kind: Any) -> bool:
    """Whether a type asked of an array is a text type: every value is then as wide as the longest."""
    if kind is None:
        return False
    if kind is str:
        return True
    try:
        return np.dtype(kind).kind in "US"
    except TypeError, ValueError:
        return False


def _check_astype(receiver: Any, args: tuple, kwargs: dict) -> None:
    """An array given a fixed-width text type holds that width for every value (pandas keeps each text at its
    own length, whatever the width asked)."""
    if isinstance(receiver, np.ndarray):
        width = _text_width(args[0] if args else kwargs.get("dtype"))
        if width > MAX_PER_ROW:
            _over(int(receiver.size) * width, _rows(receiver))


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
    _check_aligned(base, exponent)
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


_TEXT_CUMSUM = "A cumulative sum of texts is not allowed in a formula."


def _check_cumsum(receiver: Any) -> None:
    """A cumulative sum of texts holds, row after row, every text before it: refused as a sum of texts is
    (:func:`_check_sum`), a list of columns included (``np.cumsum([df['site'].str.pad(100)] * 100)`` built fifty
    million characters on ten rows, and grows as the square of the list)."""
    if _strings_in(receiver):
        raise UnsafeExpressionError(_TEXT_CUMSUM)


def _method(receiver: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """``receiver.name(...)`` for the methods that build by a size, checked before they do."""
    if name in _ALIGNING:
        _check_aligned(receiver, *args, *kwargs.values())
    if name == "mod" and _holds_text(receiver):
        raise UnsafeExpressionError(_FORMATTED)
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
    elif name == "add":
        _check_plus(receiver, args[0] if args else kwargs.get("other"))
    elif name in _FILLED:
        _check_filled(receiver, args, kwargs)
    elif name == "astype":
        _check_astype(receiver, args, kwargs)
    elif name == "sum":
        _check_sum(receiver, kwargs)
    target = receiver
    *through, last = name.split(".")
    for part in through:
        target = getattr(target, part)
    return getattr(target, last)(*args, **kwargs)


_FILLED = frozenset({"replace", "where", "mask", "fillna"})  # a value written into the rows
# of a column, an array or a table
# the methods that align a column or a table given to them with their receiver
_ALIGNING = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "truediv",
        "floordiv",
        "mod",
        "pow",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    }
    | {"where", "mask", "fillna", "clip"}
)
_GUARDED_METHODS = frozenset({"mul", "pow", "cumsum", "add", "astype", "sum"}) | _FILLED | _ALIGNING
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
        if (
            _array_width(fill) > MAX_PER_ROW
        ):  # or as an array of a wide text type, spread over every value
            _over(asked * _array_width(fill), asked)
        # or as wide as a fixed-width text type says (the third argument of np.full, the second of the others)
        place = 2 if function is np.full else 1
        width = _text_width(kwargs.get("dtype", args[place] if len(args) > place else None))
        if width > MAX_PER_ROW:
            _over(asked * width, asked)
        return function(*args, **kwargs)

    return sized


def _chosen(function: Callable[..., Any]) -> Callable[..., Any]:
    """``np.where`` and ``np.select``: a text given as a single value is the width of every value of the result."""

    def chosen(*args: Any, **kwargs: Any) -> Any:
        given = [*args[1:], *kwargs.values()]
        # an array of a wide text type, given as it is or in np.select's list of choices, is spread too
        arrays = [
            *given,
            *(item for one in given if isinstance(one, list | tuple) for item in one[:MAX_WIDTH]),
        ]
        widest = max(
            [*(_fixed_text(one) for one in given), *(_array_width(one) for one in arrays)],
            default=0,
        )
        if widest > MAX_PER_ROW:
            first = args[0] if args else None
            if isinstance(first, list | tuple) and first:  # np.select: a list of conditions
                first = first[0]
            rows = int(np.size(first)) if first is not None else 1
            _over(rows * widest, rows)
        return function(*args, **kwargs)

    return chosen


def _values_in(value: Any, seen: dict[int, int] | None = None) -> int:
    """The single values a list holds once laid out by numpy: a column or an array for its own, a list repeated
    counted each time it is held and read once."""
    if isinstance(value, list | tuple):
        seen = {} if seen is None else seen
        known = seen.get(id(value))
        if known is None:
            if len(value) > _LONG_LIST and set(map(type, value)) <= _SINGLE_VALUES | {str}:
                known = len(value)
            else:
                known = sum(_values_in(item, seen) for item in value)
            seen[id(value)] = known
        return known
    if isinstance(value, pd.DataFrame | pd.Series | pd.Index | np.ndarray):
        return max(int(np.size(value)), 1)
    return 1


def _widest_in(value: Any, seen: dict[int, int] | None = None) -> int:
    """The width numpy gives each value it lays out from a list: the longest text the list holds, or the widest
    fixed-width text type of the arrays in it (each list read once)."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, np.ndarray):
        return _array_width(value)
    if not isinstance(value, list | tuple):
        return 0
    seen = {} if seen is None else seen
    known = seen.get(id(value))
    if known is None:
        if len(value) > _LONG_LIST and set(map(type, value)) <= _SINGLE_VALUES:
            known = 0
        else:
            known = max((_widest_in(item, seen) for item in value), default=0)
        seen[id(value)] = known
    return known


def _laid_out(function: Callable[..., Any]) -> Callable[..., Any]:
    """``np.array`` and ``np.asarray`` of a list: its values, each as wide as the widest text it holds, or as the
    widest fixed-width text type of its arrays (``[np.array(['x'], dtype='U100000')] * 200`` is 200 values of
    100,000 characters), nested lists counted value by value."""

    def laid_out(*args: Any, **kwargs: Any) -> Any:
        given = args[0] if args else None
        kind = kwargs.get("dtype", args[1] if len(args) > 1 else None)
        if isinstance(given, list | tuple):
            _over(_weight(given), 0, "values")
            values = _values_in(given)
            widest = _widest_in(given)
            if _text_type(kind):
                # a text type: every value as wide as the longest text of the columns and arrays in the list
                widest = max(widest, _text_width(kind), _longest_in(given))
            if widest > MAX_PER_ROW:
                _over(values * widest, values)
        elif isinstance(given, pd.Series | pd.Index | np.ndarray):
            width = _text_width(kind)
            if width > MAX_PER_ROW:
                _over(int(np.size(given)) * width, _rows(given))
        return function(*args, **kwargs)

    return laid_out


def _longest_in(given: Any, seen: set[int] | None = None, depth: int = 0) -> int:
    """The longest text of the columns and arrays a list holds, each read once (a list repeated holds one
    column many times)."""
    seen = set() if seen is None else seen
    if id(given) in seen or depth > 3:
        return 0
    seen.add(id(given))
    if isinstance(given, list | tuple):
        return max((_longest_in(one, seen, depth + 1) for one in given[:MAX_WIDTH]), default=0)
    if isinstance(given, pd.Series | pd.Index | np.ndarray) and _holds_text(given):
        return _characters(given)[1]
    return 0


def _summed(function: Callable[..., Any]) -> Callable[..., Any]:
    def summed(*args: Any, **kwargs: Any) -> Any:
        if len(args) >= 2:
            _check_plus(args[0], args[1])
        return function(*args, **kwargs)

    return summed


def _no_text_sum(function: Callable[..., Any]) -> Callable[..., Any]:
    """``np.sum`` and ``np.nansum``: a sum of texts is refused (:func:`_check_sum`)."""

    def reduced(*args: Any, **kwargs: Any) -> Any:
        if args and _strings_in(args[0]):
            raise UnsafeExpressionError(_TEXT_SUM)
        return function(*args, **kwargs)

    return reduced


def _differenced(function: Callable[..., Any]) -> Callable[..., Any]:
    """``np.diff``: each order is a pass over the values, and an order is a number an expression may write up to
    10^15 (``n=10**9`` would run for minutes on ten rows). The passes are counted with the values."""

    def differenced(*args: Any, **kwargs: Any) -> Any:
        order = args[1] if len(args) > 1 else kwargs.get("n", 1)
        if args and isinstance(order, int | np.integer) and not isinstance(order, bool):
            size = max(int(np.size(args[0])), 1)
            # beyond the number of values a difference is empty; below it, the passes are counted
            if int(order) > 1 and (int(order) > size or int(order) * size > _budget(size)):
                raise UnsafeExpressionError(
                    "A difference of that order is not allowed: each order is a pass over the values."
                )
        return function(*args, **kwargs)

    return differenced


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
    "add": _summed,
    "diff": _differenced,
    "sum": _no_text_sum,
    "nansum": _no_text_sum,
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
    """The tree as it is computed: each product, each power and each sum with a side that is not a number written
    in the expression, each call of a method that builds by a size, and each selection by a computed key go
    through their check."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        name = _guard_of(node)
        if name is None:
            return node
        args: list[ast.expr] = [node.left, node.right]
        if name == _OPERATED:  # which operator, by its name
            args.append(ast.Constant(value=type(node.op).__name__))
        call = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])
        return ast.copy_location(call, node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not _computed_key(node):
            return node
        call = ast.Call(
            func=ast.Name(id=_ITEM, ctx=ast.Load()), args=[node.value, node.slice], keywords=[]
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
    """The check an operator of two values is computed through: a product, a power and a sum their own, any other
    the alignment of the columns it combines; None for any other node, and for two numbers."""
    if not isinstance(node, ast.BinOp) or _both_numbers(node):
        return None
    if isinstance(node.op, ast.Mult):
        return _TIMES
    if isinstance(node.op, ast.Add):
        return _PLUS
    if isinstance(node.op, ast.Pow):
        return _POWER
    return _OPERATED if type(node.op).__name__ in _OPERATORS else None


def _computed_key(node: ast.AST) -> bool:
    """Whether a selection's key is computed (``df[['age'] * 10000]``, ``df[df['age'] > 50]``): neither a name
    or a number written in the expression, nor a list of them (as long as the expression at most), nor a
    slice."""
    if not isinstance(node, ast.Subscript):
        return False
    key = node.slice
    if isinstance(key, ast.Constant | ast.Slice):
        return False
    if isinstance(key, ast.List | ast.Tuple):
        if any(isinstance(item, ast.Slice) for item in key.elts):
            return False  # a slice is no value to pass on
        return not all(isinstance(item, ast.Constant) for item in key.elts)
    return True


def _guarded_method(node: ast.AST) -> tuple[ast.expr, str] | None:
    """The receiver and the name of a call that goes through :func:`_method`: ``x.mul(...)``, ``x.pow(...)``,
    ``x.cumsum(...)``, ``x.add(...)``, ``x.astype(...)``, ``x.replace(...)``, ``x.where(...)``, ``x.mask(...)``,
    ``x.fillna(...)``, and ``x.str.pad(...)``, ``x.str.zfill(...)``, ``x.str.replace(...)``. numpy's and pandas'
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
        _guard_of(node) is not None or _guarded_method(node) is not None or _computed_key(node)
        for node in ast.walk(tree)
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
    """
    Validate and evaluate a tree, through the runner installed (:mod:`core.expressions.runner`): in this process
    when none is (:func:`compute_here`), as always; in the application, in a worker process with a time limit and
    a memory limit. Every computation of an expression comes here.
    """
    runner = installed()
    if runner is None:
        return compute_here(tree, df, policy=policy, extra_locals=extra_locals)
    return runner(tree, df, policy=policy, extra_locals=dict(extra_locals or {}))


def compute_here(
    tree: ast.Expression,
    df: pd.DataFrame | None = None,
    *,
    policy: ExpressionPolicy = DEFAULT_POLICY,
    extra_locals: Mapping[str, Any] | None = None,
) -> Any:
    """Validate and evaluate a tree in this process, with empty builtins and restricted ``np`` / ``pd``
    facades: what a runner, and a worker, compute with."""
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
        _PLUS: _plus,
        _OPERATED: _operated,
        _ITEM: _item,
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
