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
    """The columns a criterion reads (empty when they cannot be determined: never an error, whatever the definition)."""
    try:
        return _columns(name, definition)
    except Exception:
        return ()


def _columns(name: str, definition: Mapping[str, Any]) -> tuple[str, ...]:
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

    Raises :class:`CriterionError` and nothing else: a malformed definition (a bound "18 years", a range of three
    numbers) is a criterion that cannot be evaluated, never an error that stops a page (the sidebar counts every
    criterion of the configuration before each page).
    """
    try:
        return _mask(df, name, definition)
    except CriterionError:
        raise
    except Exception as exc:
        raise CriterionError(f"criterion '{name}': {exc}") from exc


def _listed_rows(index: pd.Index, ranges: Any) -> np.ndarray:
    """
    Whether each row id of ``index`` lies in one of the ``[first, last]`` ranges, found on the table's sorted ids
    (searchsorted): the work follows the table and the number of ranges, never the ranges' lengths. Listing every id a
    range spans made the server allocate gigabytes for [[0, 1000000000]]. The rows are those the listed ids matched:
    row labels that are whole numbers (the row ids of a table), whatever the index's type.
    """
    # a malformed pair raises here; bounds past any row id are brought within reach of a float
    pairs = [(max(int(first), -(2**62)), min(int(last), 2**62)) for first, last in ranges or []]
    fires = np.zeros(len(index), dtype=bool)
    if not len(index) or not pairs:
        return fires
    if pd.api.types.is_numeric_dtype(index) and not pd.api.types.is_bool_dtype(index):
        values = np.asarray(index, dtype=float)
        usable = np.isfinite(values) & (values == np.floor(values))
    else:  # labels of another kind: those that are whole numbers
        values = np.array(
            [float(label) if isinstance(label, (int, np.integer)) else np.nan for label in index],
            dtype=float,
        )
        usable = ~np.isnan(values)
    positions = np.flatnonzero(usable)
    if not len(positions):
        return fires
    order = positions[np.argsort(values[positions], kind="stable")]
    ordered = values[order]
    marks = np.zeros(len(order) + 1, dtype=np.int64)
    for first, last in pairs:
        if first > last:
            continue
        start = np.searchsorted(ordered, first, side="left")
        stop = np.searchsorted(ordered, last, side="right")
        if start < stop:
            marks[start] += 1
            marks[stop] -= 1
    fires[order] = np.cumsum(marks[:-1]) > 0
    return fires


def _mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
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
        return pd.Series(_listed_rows(df.index, definition.get("row_ids")), index=df.index)

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


def _number(value: Any) -> bool:
    """Whether ``float()`` reads ``value``, as the mask does: a bound written "18", or true, evaluates."""
    try:
        float(value)
    except TypeError, ValueError, OverflowError:  # an integer of 400 digits overflows
        return False
    return True


def _whole(value: Any) -> bool:
    """Whether ``int()`` reads ``value``, as the mask does: a threshold of 2.5 evaluates, as 2."""
    try:
        int(value)
    except TypeError, ValueError, OverflowError:  # Infinity, which a JSON file may hold, overflows
        return False
    return True


def _numeric_problems(name: str, definition: Mapping[str, Any]) -> list[str]:
    problems = []
    lower, upper = definition.get("lower_bound"), definition.get("upper_bound")
    if lower is None and upper is None:
        problems.append(f"rule '{name}' has neither a lower nor an upper bound")
    for bound in (lower, upper):
        if bound is not None and not _number(bound):
            problems.append(f"rule '{name}': the bound {bound!r} is not a number")
    if definition.get("strategy") not in (None, "include", "exclude"):
        problems.append(
            f"rule '{name}': the strategy must be include or exclude, not {definition.get('strategy')!r}"
        )
    return problems


def _bounds_problems(name: str, definition: Mapping[str, Any]) -> list[str]:
    bounds = definition.get("bounds") or {}  # none: the rule fires on no row, as it evaluates
    sound = isinstance(bounds, Mapping) and all(
        isinstance(pair, (list, tuple))
        and len(pair) == 2
        and all(value is None or _number(value) for value in pair)
        for pair in bounds.values()
    )
    if sound:
        return []
    return [f"rule '{name}': bounds must map each column to [lower, upper], numbers or null"]


def _row_ids_problems(name: str, definition: Mapping[str, Any]) -> list[str]:
    ranges = definition.get("row_ids") or []  # none: the rule fires on no row, as it evaluates
    sound = isinstance(ranges, (list, tuple)) and all(
        isinstance(pair, (list, tuple)) and len(pair) == 2 and all(_whole(value) for value in pair)
        for pair in ranges
    )
    if sound:
        return []
    return [f"rule '{name}': row_ids must list [first, last] pairs of row numbers"]


def _threshold_problems(name: str, definition: Mapping[str, Any]) -> list[str]:
    problems = []
    if not _whole(definition.get("threshold")):
        problems.append(f"rule '{name}': the threshold must be a number of rules")
    members = definition.get("members")
    if not isinstance(members, Mapping) or not members:
        problems.append(
            f"rule '{name}': a threshold rule needs its rules, as {{\"members\": {{name: rule}}}}"
        )
        return problems
    for member, spec in members.items():
        problems += [
            f"rule '{name}': {problem}" for problem in criterion_problems(str(member), spec)
        ]
    return problems


# what makes a rule of each kind unusable on any table (core.population.configuration_problems refuses an upload)
_PROBLEMS = {
    "threshold": _threshold_problems,
    "row_ids": _row_ids_problems,
    "bounds": _bounds_problems,
    "numeric": _numeric_problems,
    "expression": lambda name, definition: [],
}


def criterion_problems(name: str, definition: Any) -> list[str]:
    """
    What makes a criterion unusable on any table, in words (empty when it can be evaluated): a configuration with such
    a rule is refused when it is loaded. Only what fails whatever the data: a bound written "18" and a threshold of
    2.5 evaluate today, and still load.
    """
    try:
        kind = rule_kind(name, definition)
    except CriterionError as exc:
        return [str(exc)]
    try:
        return _PROBLEMS[kind](name, definition)
    except Exception as exc:  # never an error in the upload check itself
        return [f"criterion '{name}': {exc}"]


__all__ = [
    "RULE_KINDS",
    "CriterionError",
    "criterion_columns",
    "criterion_mask",
    "criterion_problems",
    "rule_kind",
]
