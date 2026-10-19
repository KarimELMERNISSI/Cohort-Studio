"""
Strict evaluation of one study criterion (clinical anomaly rule, inclusion criterion, outlier rule).

A criterion *fires* on a row when its condition holds: for an anomaly or outlier rule the row is
flagged, for an inclusion criterion the row meets it. A missing value never fires (an anomaly is not
flagged on a missing value, an inclusion criterion is not met by one), unless an expression asks for
missing values explicitly (``df['x'].isna()``).

Every problem raises :class:`CriterionError`, so a population built on a broken criterion fails visibly instead
of silently keeping or dropping rows. The Validate checks and the quality dashboard evaluate the rules here too
(``core.population.family_masks``), so every screen counts the same rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from core.expressions import UnsafeExpressionError, evaluate
from core.fingerprint import ranges_to_ids


class CriterionError(ValueError):
    """A criterion could not be evaluated on the data (unknown column, wrong type, invalid expression...)."""


def _as_mask(result: Any, index: pd.Index) -> pd.Series:
    """A True/False Series on ``index`` from the value of an expression (missing values are False)."""
    if isinstance(result, pd.Series):
        series = result
        if not series.index.equals(index):
            if len(series) != len(index):
                raise CriterionError("the expression must give one True/False value per row")
            series = series.set_axis(index)
    elif np.isscalar(result):
        if not isinstance(result, (bool, np.bool_)):
            raise CriterionError("the expression must give True/False per row, not a single value")
        return pd.Series(bool(result), index=index, dtype=bool)
    else:
        values = np.asarray(result)
        if values.ndim != 1 or len(values) != len(index):
            raise CriterionError("the expression must give one True/False value per row")
        series = pd.Series(values, index=index)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    try:
        return series.astype("boolean").fillna(False).astype(bool)
    except (TypeError, ValueError) as exc:
        raise CriterionError(
            f"the expression must give True/False per row (got {series.dtype})"
        ) from exc


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise CriterionError(f"column '{column}' is not in the data")
    series = df[column]
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise CriterionError(f"column '{column}' is not numeric ({series.dtype})")
    return series


def _inside(series: pd.Series, lower: Any, upper: Any) -> pd.Series:
    inside = pd.Series(True, index=series.index)
    if lower is not None:
        inside &= series >= float(lower)
    if upper is not None:
        inside &= series <= float(upper)
    return inside


RULE_KINDS = ("threshold", "row_ids", "bounds", "numeric", "expression")


def rule_kind(name: str, definition: Mapping[str, Any]) -> str:
    """
    The kind of a criterion, one of RULE_KINDS: the only place that decides it, so that the evaluation, the
    columns read and the description agree (they disagreed when each decided in its own order).

    The keys are read in the order of RULE_KINDS. A rule that holds both a numeric range and an expression is
    refused rather than read one way or the other: keep one of them.
    """
    if not isinstance(definition, Mapping):
        raise CriterionError(f"criterion '{name}' has no definition")
    if "threshold" in definition:
        return "threshold"
    if "row_ids" in definition:
        return "row_ids"
    if "bounds" in definition:
        return "bounds"
    expression = str(definition.get("expression") or "").strip()
    if definition.get("numeric"):
        if expression:
            raise CriterionError(
                f"criterion '{name}' holds both a numeric range and an expression: keep one of them"
            )
        return "numeric"
    if expression:
        return "expression"
    raise CriterionError(f"criterion '{name}' has neither an expression nor a numeric range")


def criterion_columns(name: str, definition: Mapping[str, Any]) -> tuple[str, ...]:
    """The columns a criterion reads (empty when they cannot be determined)."""
    from core.expressions import column_references

    try:
        kind = rule_kind(name, definition)
    except CriterionError:
        return ()
    if kind == "threshold":
        members = definition.get("members") or {}
        found = [
            c for member, spec in members.items() for c in criterion_columns(str(member), spec)
        ]
        return tuple(dict.fromkeys(found))
    if kind == "row_ids":  # the outlier rules record the columns they were found on
        return tuple(str(c) for c in definition.get("columns") or ())
    if kind == "bounds":
        return tuple(definition.get("bounds") or {})
    if kind == "numeric":
        return (str(definition.get("column") or name),)
    try:
        return tuple(sorted(column_references(str(definition["expression"]))))
    except Exception:
        return ()


def criterion_mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
    """Rows of ``df`` on which the criterion fires (boolean Series on ``df.index``).

    Supported definitions (the study configuration format), recognised by :func:`rule_kind`:

    - ``{"expression": "df['age'] > 110"}``: a whitelisted boolean expression;
    - ``{"numeric": true, "lower_bound": a, "upper_bound": b, "strategy": "include" | "exclude"}``: a range
      on the column named like the criterion (or ``"column"``): *include* fires inside ``[a, b]``,
      *exclude* outside. With a single bound, the range is open on the other side: ``>= a`` (or ``<= b``)
      fires with *include* or no strategy, ``< a`` (or ``> b``) with *exclude*;
    - ``{"bounds": {"col": [low, high], ...}}``: frozen outlier thresholds, fires when a value of any listed
      column lies outside its range;
    - ``{"row_ids": [[first, last], ...]}``: listed rows (outliers found by a multivariate method);
    - ``{"threshold": k, "members": {name: definition, ...}}``: fires when at least ``k`` members fire.
    """
    kind = rule_kind(name, definition)

    if kind == "threshold":
        members = definition.get("members") or {}
        k = definition.get("threshold")
        if not members or k is None:
            raise CriterionError(
                f"criterion '{name}': a threshold rule needs members and a threshold"
            )
        fired = pd.DataFrame(
            {member: criterion_mask(df, member, spec) for member, spec in members.items()},
            index=df.index,
        )
        return fired.sum(axis=1) >= int(k)

    if kind == "row_ids":
        ids = ranges_to_ids(definition.get("row_ids") or [])
        return pd.Series(df.index.isin(ids), index=df.index)

    if kind == "bounds":
        fires = pd.Series(False, index=df.index)
        for column, (low, high) in (definition.get("bounds") or {}).items():
            series = _numeric_series(df, column)
            fires |= series.notna() & ~_inside(series, low, high)
        return fires

    if kind == "numeric":
        series = _numeric_series(df, str(definition.get("column") or name))
        lower, upper = definition.get("lower_bound"), definition.get("upper_bound")
        if lower is None and upper is None:
            raise CriterionError(
                f"numeric criterion '{name}' has neither a lower nor an upper bound"
            )
        strategy = definition.get("strategy")
        if strategy not in (None, "include", "exclude"):
            raise CriterionError(
                f"numeric criterion '{name}': strategy must be include or exclude, not {strategy!r}"
            )
        inside = _inside(series, lower, upper)
        fires = ~inside if strategy == "exclude" else inside
        return fires & series.notna()

    expression = str(definition.get("expression")).strip()
    try:
        result = evaluate(expression, df)
    except UnsafeExpressionError as exc:
        raise CriterionError(f"criterion '{name}': expression rejected ({exc})") from exc
    except Exception as exc:
        raise CriterionError(f"criterion '{name}': {exc}") from exc
    return _as_mask(result, df.index)


__all__ = ["RULE_KINDS", "CriterionError", "criterion_columns", "criterion_mask", "rule_kind"]
