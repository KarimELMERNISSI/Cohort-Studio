"""
Strict evaluation of one study criterion (clinical anomaly rule, inclusion criterion, outlier rule).

A criterion *fires* on a row when its condition holds: for an anomaly or outlier rule the row is
flagged, for an inclusion criterion the row meets it. A missing value never fires (an anomaly is not
flagged on a missing value, an inclusion criterion is not met by one), unless an expression asks for
missing values explicitly (``df['x'].isna()``).

Unlike the legacy helpers in ``enrich.custom_metrics_and_filters`` (which return ``None`` or an all-False
mask when something goes wrong), every problem raises :class:`CriterionError`, so a population built on a
broken criterion fails visibly instead of silently keeping or dropping rows.
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


def criterion_mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
    """Rows of ``df`` on which the criterion fires (boolean Series on ``df.index``).

    Supported definitions (the study configuration format):

    - ``{"expression": "df['age'] > 110"}``: a whitelisted boolean expression;
    - ``{"numeric": true, "lower_bound": a, "upper_bound": b, "strategy": "include" | "exclude"}``: a range
      on the column named like the criterion (or ``"column"``): *include* fires inside ``[a, b]``,
      *exclude* outside; with a single bound, values at or beyond it (``>= a`` or ``<= b``) fire;
    - ``{"bounds": {"col": [low, high], ...}}``: frozen outlier thresholds, fires when a value of any listed
      column lies outside its range;
    - ``{"row_ids": [[first, last], ...]}``: listed rows (outliers found by a multivariate method);
    - ``{"threshold": k, "members": {name: definition, ...}}``: fires when at least ``k`` members fire.
    """
    if not isinstance(definition, Mapping):
        raise CriterionError(f"criterion '{name}' has no definition")

    if "threshold" in definition:
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

    if "row_ids" in definition:
        ids = ranges_to_ids(definition.get("row_ids") or [])
        return pd.Series(df.index.isin(ids), index=df.index)

    if "bounds" in definition:
        fires = pd.Series(False, index=df.index)
        for column, (low, high) in (definition.get("bounds") or {}).items():
            series = _numeric_series(df, column)
            fires |= series.notna() & ~_inside(series, low, high)
        return fires

    if definition.get("numeric"):
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

    expression = definition.get("expression")
    if not expression or not str(expression).strip():
        raise CriterionError(f"criterion '{name}' has neither an expression nor a numeric range")
    try:
        result = evaluate(str(expression), df)
    except UnsafeExpressionError as exc:
        raise CriterionError(f"criterion '{name}': expression rejected ({exc})") from exc
    except Exception as exc:
        raise CriterionError(f"criterion '{name}': {exc}") from exc
    return _as_mask(result, df.index)


__all__ = ["CriterionError", "criterion_mask"]
