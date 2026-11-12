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

The sizes an expression can ask for are bounded too, since a formula written by a language model is tried on the data
before anyone reads it: a power of constants, an exponent, a constant, the size of an array built by ``np.full``,
``np.ones`` or ``np.zeros``, a padding width and a repeated text (``9 ** 9 ** 8`` would keep the server busy for
minutes). A size is read where it is written as a number, through ``int``, ``float``, ``abs``, ``round``, ``max`` and
``min`` too (``str.pad(int(1e8))``); a column of texts times a number is known only when it is computed, and is
bounded then (``df['site'] * 100000000``). An expression nested too deep to be walked is refused as too complex.
"""

from __future__ import annotations

import ast
import copy
import math
from collections.abc import Mapping
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
MAX_ELEMENTS = 10_000_000  # an array built from a constant size (np.full, np.ones, np.zeros)
MAX_WIDTH = 10_000  # a padding width, or a text repeated a constant number of times
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


def _np_facade(policy: ExpressionPolicy) -> SimpleNamespace:
    return SimpleNamespace(
        **{name: getattr(np, name) for name in policy.np_functions if hasattr(np, name)}
    )


def _pd_facade(policy: ExpressionPolicy) -> SimpleNamespace:
    return SimpleNamespace(
        **{name: getattr(pd, name) for name in policy.pd_functions if hasattr(pd, name)}
    )


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


def _holds_text(value: Any) -> bool:
    """Whether a value is a text, or a column or an array of texts."""
    if isinstance(value, str):
        return True
    kind = getattr(value, "dtype", None)
    if kind is None or not isinstance(value, pd.Series | pd.Index | np.ndarray):
        return False
    return bool(pd.api.types.is_string_dtype(kind) or pd.api.types.is_object_dtype(kind))


def _largest(value: Any) -> float:
    """The largest whole number a value holds, as a count of repetitions (0 for anything else)."""
    if isinstance(value, bool | np.bool_):
        return 0.0
    if isinstance(value, int | np.integer):
        return float(abs(int(value)))
    kind = getattr(value, "dtype", None)
    counted = kind is not None and isinstance(value, pd.Series | pd.Index | np.ndarray)
    if counted and len(value) and pd.api.types.is_integer_dtype(kind):
        return float(np.nanmax(np.abs(np.asarray(value, dtype="float64"))))
    return 0.0


def _times(left: Any, right: Any) -> Any:
    """
    ``left * right``, unless it repeats a text more than ``MAX_WIDTH`` times. A text written in the expression is
    counted when the expression is read; a column of texts is known only when it is computed:
    ``df['site'] * 100000000`` builds a hundred million copies of each site's name.
    """
    for text, count in ((left, right), (right, left)):
        if _holds_text(text) and _largest(count) > MAX_WIDTH:
            raise UnsafeExpressionError(
                f"A text repeated more than {MAX_WIDTH:,} times is not allowed."
            )
    return left * right


class _Products(ast.NodeTransformer):
    """Each product with a side that is not a number written in the expression, computed by :func:`_times`."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Mult) and not _both_numbers(node):
            call = ast.Call(
                func=ast.Name(id=_TIMES, ctx=ast.Load()), args=[node.left, node.right], keywords=[]
            )
            return ast.copy_location(call, node)
        return node


def _both_numbers(node: ast.BinOp) -> bool:
    return all(
        isinstance(side, ast.Constant)
        and isinstance(side.value, int | float)
        and not isinstance(side.value, bool)
        for side in (node.left, node.right)
    )


def _guarded(tree: ast.Expression) -> ast.Expression:
    """The tree as it is evaluated: a copy with its products guarded, or the tree itself when it has none to guard
    (the caller's tree is never changed: it may be evaluated again, and would then not pass the validation)."""
    products = (
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and not _both_numbers(node)
        for node in ast.walk(tree)
    )
    if not any(products):
        return tree
    try:
        return ast.fix_missing_locations(_Products().visit(copy.deepcopy(tree)))
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
        "np": _np_facade(policy),
        "pd": _pd_facade(policy),
        _TIMES: _times,
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
