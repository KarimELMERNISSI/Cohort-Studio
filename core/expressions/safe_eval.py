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
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


class UnsafeExpressionError(ValueError):
    """The expression uses a construct that is not allowed."""


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

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in self.policy.builtins and func.id not in self.extra_names:
                raise UnsafeExpressionError(f"Function '{func.id}' is not allowed.")
        elif isinstance(func, ast.Attribute):
            self.visit(func)
        else:
            raise UnsafeExpressionError("Only named functions and methods may be called.")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.arg is None or kw.arg.startswith("_"):
                raise UnsafeExpressionError(
                    "Keyword unpacking and private keywords are not allowed."
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
    _Validator(policy, frozenset(extra_names or ())).visit(tree)
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
    return validate_tree(tree, policy=policy, extra_names=extra_names)


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
    ast.fix_missing_locations(tree)
    code = compile(tree, filename="<expression>", mode="eval")
    namespace: dict[str, Any] = {
        "__builtins__": dict(policy.builtins),
        "np": _np_facade(policy),
        "pd": _pd_facade(policy),
    }
    namespace.update(extra_locals)
    for name in policy.frame_names:
        namespace[name] = df
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
