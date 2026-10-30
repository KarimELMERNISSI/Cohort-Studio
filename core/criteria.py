"""
Strict evaluation of one study criterion (plausibility rule, inclusion criterion, outlier rule).

A criterion *fires* on a row when its condition holds: for a plausibility or outlier rule the row
is flagged, for an inclusion criterion the row meets it. A missing value never fires (a value is not
flagged implausible when missing, an inclusion criterion is not met by one), unless an expression asks for
missing values explicitly (``df['x'].isna()``).

Every problem raises :class:`CriterionError`, so a population built on a broken criterion fails visibly instead
of silently keeping or dropping rows. The Validate checks and the quality dashboard evaluate the rules here too
(``core.population.family_masks``), so every screen counts the same rows.

Each kind of criterion is one :class:`CriterionKind` in the catalogue :data:`CRITERION_KINDS`: how a definition of
that kind is recognised, evaluated, described and checked. :func:`rule_kind` alone decides a rule's kind; the free
functions below (the mask, the columns, the words, the upload checks, the row lists, the copy for sharing) read it,
and nothing else in core decides a kind from a rule's keys. A stored definition is never rewritten: its fingerprint
keys the population memos and is written in every population block of a trace.
"""

from __future__ import annotations

import ast
import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from biostats.outliers import MULTIVARIATE_METHODS
from core.expressions import UnsafeExpressionError, column_references, evaluate, evaluate_tree
from core.registry import Catalogue

# What a rule selects, as this module reads and evaluates the definitions. A study definition's fingerprint holds it
# (core.protocol): a change to what the same definition selects (how a missing value meets "!=" or "~", say) is a new
# number, which changes the fingerprint of every study with criteria, as their rows may change.
RULES_VERSION = 1


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


# ------------------------------------------------------------------------------------------------------ the catalogue

# How a kind treats a missing value; tests/core_layer/test_criterion_kinds.py checks each kind against its own.
Missing = Literal["never", "asked", "not read", "members"]
NEVER: Missing = "never"  # a missing value never fires it
# an expression: as it asks (.isna() fires on a missing value, > and == do not)
ASKED: Missing = "asked"
NOT_READ: Missing = "not read"  # it reads no value: listed rows fire whatever their values
MEMBERS: Missing = "members"  # as the rules it combines


def _no_problems(name: str, definition: Mapping[str, Any]) -> list[str]:
    return []


# What a key of a definition may hold (CriterionKind.keys): a copy shared with someone else keeps a key only with such
# a value, since a loaded file can hold anything there, row ids too.


def is_text(value: Any) -> bool:
    """A text: an expression, a column's name, a description."""
    return isinstance(value, str)


def is_scalar(value: Any) -> bool:
    """One value: a number, a text, true or false, or nothing (a bound, a threshold, a strategy)."""
    return value is None or isinstance(value, (str, int, float))


def is_flag(value: Any) -> bool:
    """True or false (whether a rule is a numeric range)."""
    return isinstance(value, bool)


def _is_bounds(value: Any) -> bool:
    """Frozen bounds: column names, each with a [lower, upper] pair of single values."""
    return value is None or (
        isinstance(value, Mapping)
        and all(
            isinstance(column, str)
            and isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(is_scalar(bound) for bound in pair)
            for column, pair in value.items()
        )
    )


# the conversions whose missing results are values the rule cannot read (errors="coerce")
_CONVERSIONS = frozenset({"to_datetime", "to_numeric"})


@dataclass(frozen=True)
class RuleCheck:
    """
    One check of a plausibility rule (Domain Validity): the rows it checks, those holding every value it reads, as it
    reads them, and the rows where it fires, always checked. A bounds rule makes one check per column, as a check of
    one field does.
    """

    rule: str
    column: str  # the column of a check made per column (bounds); "" for the whole rule
    checked: pd.Series
    fired: pd.Series

    @property
    def unit(self) -> str:
        """The check's name on the screens: the rule, or "rule: column"."""
        return f"{self.rule}: {self.column}" if self.column else self.rule


@dataclass(frozen=True)
class CriterionKind:
    """
    One kind of criterion. The free functions of this module are the interface: pages and services never call a
    kind's functions, which take the rule's name (a numeric rule reads the column named like it) and its definition.
    A kind that combines rules gives its own checks and columns: the free functions add those of its rules, so it can
    be written in a module of its own, which must not import this one.

    Its id is stored with every rule of its kind, so what a kind selects never changes under the same id: a changed
    selection is a kind of its own, under a new id, and the old one stays registered, marked legacy.
    """

    # stored, never renamed; the five kinds of M5 are implied by their keys, a later kind is stated as "kind"
    id: str
    # its name on the screens that list rules
    label: str
    # the rows where it fires, on df.index
    mask: Callable[[pd.DataFrame, str, Mapping[str, Any]], pd.Series]
    # the columns it reads (listed rows: those they were flagged on); the free function adds its rules'
    columns: Callable[[str, Mapping[str, Any]], tuple[str, ...]]
    # what makes it fire, in words: the flow and the dossier
    describe: Callable[[str, Mapping[str, Any]], str]
    # whether a definition is of this kind, from its keys: it may raise CriterionError for a definition of this kind
    # that cannot be read (a numeric range with an expression); None for a kind that a definition names
    recognise: Callable[[str, Mapping[str, Any]], bool] | None = None
    # what makes a definition unusable on any table: a configuration holding it is refused when loaded; the free
    # function adds its rules' problems
    problems: Callable[[str, Mapping[str, Any]], list[str]] = _no_problems
    # the key under which it keeps the rules it combines, {name: definition}; None when it combines none
    members_key: str | None = None
    # the other keys a definition of this kind holds, each with what it may hold (is_text, is_scalar...): a copy
    # shared with someone else keeps these and no other, and leaves the rule out when a value is of another kind
    keys: Mapping[str, Callable[[Any], bool]] = field(default_factory=dict, hash=False)
    # it lists row ids of this dataset: left out of a configuration shared with someone else
    lists_rows: bool = False
    # how it treats a missing value: NEVER, ASKED, NOT_READ or MEMBERS
    missing: Missing = NEVER
    # its own Domain Validity checks (rule_checks), when they are not the default: the rows holding every column it
    # reads and, for a kind that combines rules, where each of its rules is settled. Called with the table, the name,
    # the definition and its rules' (fires, settled), each a boolean Series; returns, per check, (column or "", the
    # rows holding the values it reads, the rows where it fires): plain Series, so that a kind written in a module of
    # its own needs nothing from this one. A row where it fires is always checked; it may raise as ``mask`` does.
    plausibility_checks: (
        Callable[
            [pd.DataFrame, str, Mapping[str, Any], Mapping[str, tuple[pd.Series, pd.Series]]],
            list[tuple[str, pd.Series, pd.Series]],
        ]
        | None
    ) = None

    def __post_init__(self) -> None:
        # read only, as the rest of the kind
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))


CRITERION_KINDS: Catalogue[CriterionKind] = Catalogue("criterion kind")


# threshold: {"threshold": k, "members": {name: definition, ...}} fires when at least k members fire


def _threshold_mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
    members = definition.get("members") or {}
    k = definition.get("threshold")
    if not members or k is None:
        raise CriterionError(f"criterion '{name}': a threshold rule needs members and a threshold")
    fired = pd.DataFrame(
        {member: criterion_mask(df, member, spec) for member, spec in members.items()},
        index=df.index,
    )
    return fired.sum(axis=1) >= int(k)


def _threshold_text(name: str, definition: Mapping[str, Any]) -> str:
    # each rule's definition: the dossier gives the definition of every criterion
    members = definition.get("members") or {}
    k = int(definition["threshold"])  # as the mask reads it: a threshold of 2.5 is 2
    text = f"at least {k} of: " + "; ".join(
        f"{member} ({describe_definition(str(member), spec)})" for member, spec in members.items()
    )
    if k <= 0:
        return f"{text} (met by every row)"
    if k > len(members):
        return f"{text} (met by no row)"
    return text


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


def _threshold_checks(
    df: pd.DataFrame,
    name: str,
    definition: Mapping[str, Any],
    members: Mapping[str, tuple[pd.Series, pd.Series]],
) -> list[tuple[str, pd.Series, pd.Series]]:
    """Read as Kleene logic reads "at least k": checked where k members fire, or where too few can still fire."""
    fires = _threshold_mask(df, name, definition)  # raises when it cannot be read
    k = int(definition["threshold"])
    fired = pd.Series(0, index=df.index)
    unknown = pd.Series(0, index=df.index)
    for member_fires, known in members.values():
        fired += member_fires.astype(int)
        unknown += (~known).astype(int)
    return [("", fired + unknown < k, fires)]


CRITERION_KINDS.register(
    CriterionKind(
        "threshold",
        "At least k rules",
        mask=_threshold_mask,
        columns=lambda name, definition: (),  # its rules'
        describe=_threshold_text,
        recognise=lambda name, definition: "threshold" in definition,
        problems=_threshold_problems,
        members_key="members",
        keys={"threshold": is_scalar, "description": is_text},
        missing=MEMBERS,
        plausibility_checks=_threshold_checks,
    )
)


# row_ids: {"row_ids": [[first, last], ...]} lists rows (outliers frozen as row ids, with the method that found them)


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


def _row_ids_mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
    return pd.Series(_listed_rows(df.index, definition.get("row_ids")), index=df.index)


def _row_ids_columns(name: str, definition: Mapping[str, Any]) -> tuple[str, ...]:
    # the outlier rules record the columns they were found on; the names of columns only: a loaded file can list
    # anything there
    recorded = definition.get("columns")
    return (
        tuple(c for c in recorded if isinstance(c, str))
        if isinstance(recorded, (list, tuple))
        else ()
    )


def _figure(value: Any) -> str | None:
    """A recorded number as words write it (3.0 reads 3); None for anything else, since a file can hold any text."""
    if isinstance(value, bool) or not _number(value) or not math.isfinite(float(value)):
        return None
    return f"{float(value):g}"


def _zscore_words(params: Mapping[str, Any]) -> str:
    threshold = _figure(params.get("threshold"))
    return f"by |z| > {threshold}" if threshold else "by z-score"


def _iqr_words(params: Mapping[str, Any]) -> str:
    multiplier = _figure(params.get("multiplier"))
    return f"by more than {multiplier} × IQR beyond the quartiles" if multiplier else "by IQR"


def _quantile_words(params: Mapping[str, Any]) -> str:
    lower, upper = _figure(params.get("lower")), _figure(params.get("upper"))
    return f"outside quantiles {lower}–{upper}" if lower and upper else "by quantiles"


# How each outlier method flagged a row ("flagged by |z| > 3"), by the id Validate records (biostats.outliers), with
# the numbers recorded. A method missing here is not named: the words go into the dossier, and a loaded file can hold
# any text there.
METHOD_WORDS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "zscore": _zscore_words,
    "iqr": _iqr_words,
    "quantile": _quantile_words,
    "Local Outlier Factor": lambda params: "by Local Outlier Factor",
    "Isolation Forest": lambda params: "by Isolation Forest",
    "DBSCAN": lambda params: "by DBSCAN",
}
_UNIVARIATE = ("zscore", "iqr", "quantile")  # a value of any one column flags the row


def _listed(columns: list[str], method: str) -> str:
    """
    The columns a method flagged rows on: one of them is enough for a univariate method, a multivariate one reads them
    together, and of a method this version does not know, nothing is said but the columns.
    """
    if len(columns) == 1:
        return columns[0]
    if method in _UNIVARIATE:
        return f"any of {', '.join(columns)}"
    if method in METHOD_WORDS:
        return f"{', '.join(columns[:-1])} and {columns[-1]} together"
    return ", ".join(columns)


def _row_ids_text(name: str, definition: Mapping[str, Any]) -> str:
    # the univariate outlier rules are frozen as rows too: how the rows were flagged, and on which columns
    ranges = definition.get("row_ids") or []
    n = sum(int(b) - int(a) + 1 for a, b in ranges if int(a) <= int(b))  # a reversed range: none
    if n == 0:  # "excluded when no listed row" read as "excluded when not listed"
        return "never met (no listed row)"
    rows = "the 1 listed row" if n == 1 else f"one of the {n:,} listed rows"
    method = definition.get("method")
    if not method:
        return rows
    words = METHOD_WORDS.get(str(method))
    params = definition.get("params")
    if words is None:
        how = "by an unrecognised method"
    else:
        how = words(params if isinstance(params, Mapping) else {})
    recorded = definition.get("columns")
    # the names of columns only: a loaded file can list anything there, row ids too
    columns = (
        [c for c in recorded if isinstance(c, str)] if isinstance(recorded, (list, tuple)) else []
    )
    if columns:
        how += f" on {_listed(columns, str(method))}"
    return f"{rows} (flagged {how})"


def _row_ids_problems(name: str, definition: Mapping[str, Any]) -> list[str]:
    ranges = definition.get("row_ids") or []  # none: the rule fires on no row, as it evaluates
    sound = isinstance(ranges, (list, tuple)) and all(
        isinstance(pair, (list, tuple)) and len(pair) == 2 and all(_whole(value) for value in pair)
        for pair in ranges
    )
    if sound:
        return []
    return [f"rule '{name}': row_ids must list [first, last] pairs of row numbers"]


def _row_ids_checks(
    df: pd.DataFrame,
    name: str,
    definition: Mapping[str, Any],
    members: Mapping[str, tuple[pd.Series, pd.Series]],
) -> list[tuple[str, pd.Series, pd.Series]]:
    """
    The rows the method assessed: those holding a column the rows were flagged on, every one of them for a
    multivariate method (it assesses only complete rows); every row when no column is recorded (older rules); only
    the listed rows when a recorded column is no longer in the table (renamed: which rows it assessed is unknown).
    """
    fires = _row_ids_mask(df, name, definition)
    columns = _row_ids_columns(name, definition)
    if not columns:
        return [("", pd.Series(True, index=df.index), fires)]
    if any(c not in df.columns for c in columns):
        return [("", pd.Series(False, index=df.index), fires)]
    present = df[list(columns)].notna()
    multivariate = definition.get("method") in MULTIVARIATE_METHODS
    return [("", present.all(axis=1) if multivariate else present.any(axis=1), fires)]


CRITERION_KINDS.register(
    CriterionKind(
        "row_ids",
        "Listed rows",
        mask=_row_ids_mask,
        columns=_row_ids_columns,
        describe=_row_ids_text,
        recognise=lambda name, definition: "row_ids" in definition,
        problems=_row_ids_problems,
        keys={
            "row_ids": lambda value: isinstance(value, (list, tuple)),
            "method": is_text,
            "params": lambda value: isinstance(value, Mapping),
            "columns": lambda value: isinstance(value, (list, tuple)),
            "description": is_text,
        },
        lists_rows=True,
        missing=NOT_READ,
        plausibility_checks=_row_ids_checks,
    )
)


# bounds: {"bounds": {"col": [low, high], ...}} frozen outlier thresholds, fires when a value of any listed column lies
# outside its range


def _bounds_mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
    fires = pd.Series(False, index=df.index)
    for column, (low, high) in (definition.get("bounds") or {}).items():
        series = _numeric_series(df, column)
        fires |= series.notna() & ~_inside(series, low, high)
    return fires


def _bounds_columns(name: str, definition: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(definition.get("bounds") or {})


def _bound_text(column: Any, low: Any, high: Any) -> str:
    """Where frozen bounds fire: outside the range, or past its one side ("age > 85", never "outside [None, 85]")."""
    if low is not None and high is not None:
        reversed_ = float(low) > float(high)  # nothing lies within: every recorded value is outside
        return f"{column} < {low} or {column} > {high}" + (
            " (reversed: met by every recorded value)" if reversed_ else ""
        )
    if high is not None:
        return f"{column} > {high}"
    return f"{column} < {low}"


def _bounds_text(name: str, definition: Mapping[str, Any]) -> str:
    # a value outside any column's range: "or" (a list of conditions reads as "and", and inside a threshold rule as
    # several rules); a column without a bound fires on no row
    texts = [
        _bound_text(column, low, high)
        for column, (low, high) in (definition.get("bounds") or {}).items()
        if low is not None or high is not None
    ]
    return " or ".join(texts) if texts else "never met (no bound)"


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


def _bounds_checks(
    df: pd.DataFrame,
    name: str,
    definition: Mapping[str, Any],
    members: Mapping[str, tuple[pd.Series, pd.Series]],
) -> list[tuple[str, pd.Series, pd.Series]]:
    """One check per column, on the rows holding its number."""
    found = []
    for column, (low, high) in (definition.get("bounds") or {}).items():
        series = _numeric_series(df, column)
        held = series.notna()
        found.append((str(column), held, held & ~_inside(series, low, high)))
    return found


CRITERION_KINDS.register(
    CriterionKind(
        "bounds",
        "Frozen bounds",
        mask=_bounds_mask,
        columns=_bounds_columns,
        describe=_bounds_text,
        recognise=lambda name, definition: "bounds" in definition,
        problems=_bounds_problems,
        keys={"bounds": _is_bounds, "description": is_text},
        plausibility_checks=_bounds_checks,
    )
)


# numeric: {"numeric": true, "lower_bound": a, "upper_bound": b, "strategy": "include" | "exclude"} a range on the
# column named like the criterion (or "column")


def _expression_of(definition: Mapping[str, Any]) -> str:
    """The expression a rule holds, stripped ("" when it holds none)."""
    return str(definition.get("expression") or "").strip()


def _is_numeric(name: str, definition: Mapping[str, Any]) -> bool:
    if not definition.get("numeric"):
        return False
    if _expression_of(definition):
        raise CriterionError(
            f"criterion '{name}' holds both a numeric range and an expression: keep one of them"
        )
    return True


def _numeric_mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
    series = _numeric_series(df, str(definition.get("column") or name))
    lower, upper = definition.get("lower_bound"), definition.get("upper_bound")
    if lower is None and upper is None:
        raise CriterionError(f"numeric criterion '{name}' has neither a lower nor an upper bound")
    strategy = definition.get("strategy")
    if strategy not in (None, "include", "exclude"):
        raise CriterionError(
            f"numeric criterion '{name}': strategy must be include or exclude, not {strategy!r}"
        )
    inside = _inside(series, lower, upper)
    fires = ~inside if strategy == "exclude" else inside
    return fires & series.notna()


def _numeric_columns(name: str, definition: Mapping[str, Any]) -> tuple[str, ...]:
    return (str(definition.get("column") or name),)


def _numeric_text(column: Any, low: Any, high: Any, exclude: bool) -> str:
    """
    Where a numeric rule fires, in words: inside the range, or outside it with the *exclude* strategy. Comparisons
    rather than brackets, which leave the reader to know whether a bound itself counts ("18 <= age <= 85").
    """
    if low is not None and high is not None:
        text = (
            f"{column} < {low} or {column} > {high}" if exclude else f"{low} <= {column} <= {high}"
        )
        if float(low) > float(high):  # nothing lies within: every recorded value is outside
            text += (
                " (reversed: met by every recorded value)"
                if exclude
                else " (reversed: met by no value)"
            )
        return text
    if low is not None:
        return f"{column} < {low}" if exclude else f"{column} >= {low}"
    return f"{column} > {high}" if exclude else f"{column} <= {high}"


def _numeric_rule_text(name: str, definition: Mapping[str, Any]) -> str:
    return _numeric_text(
        definition.get("column") or name,
        definition.get("lower_bound"),
        definition.get("upper_bound"),
        definition.get("strategy") == "exclude",
    )


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


def _numeric_checks(
    df: pd.DataFrame,
    name: str,
    definition: Mapping[str, Any],
    members: Mapping[str, tuple[pd.Series, pd.Series]],
) -> list[tuple[str, pd.Series, pd.Series]]:
    """The rows holding the column's number: a text that is no number is a value the rule cannot read."""
    fires = _numeric_mask(df, name, definition)
    return [("", _numeric_series(df, str(definition.get("column") or name)).notna(), fires)]


CRITERION_KINDS.register(
    CriterionKind(
        "numeric",
        "Numeric range",
        mask=_numeric_mask,
        columns=_numeric_columns,
        describe=_numeric_rule_text,
        recognise=_is_numeric,
        problems=_numeric_problems,
        keys={
            "numeric": is_flag,
            "column": is_text,
            "lower_bound": is_scalar,
            "upper_bound": is_scalar,
            "strategy": is_scalar,
            "description": is_text,
        },
        plausibility_checks=_numeric_checks,
    )
)


# expression: {"expression": "df['age'] > 110"} a whitelisted boolean expression (core.expressions)


def _expression_mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
    expression = str(definition.get("expression")).strip()
    try:
        result = evaluate(expression, df)
    except UnsafeExpressionError as exc:
        raise CriterionError(f"criterion '{name}': expression rejected ({exc})") from exc
    except Exception as exc:
        raise CriterionError(f"criterion '{name}': {exc}") from exc
    return _as_mask(result, df.index)


def _expression_columns(name: str, definition: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        return tuple(sorted(column_references(str(definition["expression"]))))
    except Exception:
        return ()


def _expression_checks(
    df: pd.DataFrame,
    name: str,
    definition: Mapping[str, Any],
    members: Mapping[str, tuple[pd.Series, pd.Series]],
) -> list[tuple[str, pd.Series, pd.Series]]:
    """
    The rows holding every column it reads, however it is written (a guard such as ``.notna()`` or ``fillna`` reads no
    missing value either), and where a conversion (``pd.to_datetime``, ``pd.to_numeric``) gives a value: a date that
    cannot be read is not a pass. An expression testing missingness checks the rows it fires on, as every rule.
    """
    fires = _expression_mask(df, name, definition)
    held = pd.Series(True, index=df.index)
    for column in _expression_columns(name, definition):
        if column in df.columns:
            held &= df[column].notna()
    try:
        tree = ast.parse(_expression_of(definition), mode="eval")
    except SyntaxError:
        return [("", held, fires)]
    for node in ast.walk(tree):
        func = node.func if isinstance(node, ast.Call) else None
        if isinstance(func, ast.Attribute) and func.attr in _CONVERSIONS:
            try:
                converted = evaluate_tree(ast.Expression(body=node), df)
            except Exception:
                continue  # a part that cannot be read alone: its columns decide
            if isinstance(converted, pd.Series) and converted.index.equals(df.index):
                held &= converted.notna()
    return [("", held, fires)]


CRITERION_KINDS.register(
    CriterionKind(
        "expression",
        "Expression",
        mask=_expression_mask,
        columns=_expression_columns,
        describe=lambda name, definition: _expression_of(definition),
        recognise=lambda name, definition: bool(_expression_of(definition)),
        keys={"numeric": is_flag, "expression": is_text, "description": is_text},
        missing=ASKED,
        plausibility_checks=_expression_checks,
    )
)


# --------------------------------------------------------------------------------------------------- the interface


def rule_kind(name: str, definition: Mapping[str, Any]) -> CriterionKind:
    """
    The kind of a criterion: the only place that decides it, so that the evaluation, the columns read, the words and
    the upload checks agree (they disagreed when each decided in its own order).

    A definition is read in this order, and never changed:

    1. the five kinds of M5, from their keys, in the order they were registered, whatever "kind" says: one holding
       both "threshold" and "row_ids" is a threshold rule, and a definition that reads today reads the same once a
       kind it names is registered, so that its recorded population still replays;
    2. otherwise a "kind" that names a kind registered without a recogniser (from M6 on, a kind that definitions
       state): that kind, whose definitions hold none of the keys of the five;
    3. nothing read: refused, with the kind it states when it states one (a file written by a later version).

    A rule that holds both a numeric range and an expression is refused rather than read one way or the other: keep
    one of them.
    """
    if not isinstance(definition, Mapping):
        raise CriterionError(f"criterion '{name}' has no definition")
    for kind in CRITERION_KINDS:
        if kind.recognise is not None and kind.recognise(name, definition):
            return kind
    stated = definition.get("kind")
    named = CRITERION_KINDS.find(stated) if isinstance(stated, str) else None  # a list is no id
    if named is not None and named.recognise is None:
        return named
    if stated is not None:
        raise CriterionError(
            f"criterion '{name}': unknown kind {stated!r}; the kinds are "
            + ", ".join(CRITERION_KINDS.ids(legacy=False))
        )
    raise CriterionError(f"criterion '{name}' has neither an expression nor a numeric range")


def _kind_or_none(definition: Mapping[str, Any]) -> CriterionKind | None:
    try:
        return rule_kind("", definition)
    except CriterionError:
        return None


def criterion_columns(name: str, definition: Mapping[str, Any]) -> tuple[str, ...]:
    """
    The columns a criterion reads, those of the rules it combines included (empty when they cannot be determined:
    never an error, whatever the definition).
    """
    try:
        kind = rule_kind(name, definition)
        found = list(kind.columns(name, definition))
        for member, spec in _members(kind, definition).items():
            found += criterion_columns(str(member), spec)
        return tuple(dict.fromkeys(found))
    except Exception:
        return ()


def criterion_mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
    """Rows of ``df`` on which the criterion fires (boolean Series on ``df.index``).

    The kinds are those of :data:`CRITERION_KINDS`, read by :func:`rule_kind`; the five of M5 (the study
    configuration format) are:

    - ``{"expression": "df['age'] > 110"}``: a whitelisted boolean expression;
    - ``{"numeric": true, "lower_bound": a, "upper_bound": b, "strategy": "include" | "exclude"}``: a range
      on the column named like the criterion (or ``"column"``): *include* fires inside ``[a, b]``,
      *exclude* outside. With a single bound, the range is open on the other side: ``>= a`` (or ``<= b``)
      fires with *include* or no strategy, ``< a`` (or ``> b``) with *exclude*;
    - ``{"bounds": {"col": [low, high], ...}}``: frozen outlier thresholds, fires when a value of any listed
      column lies outside its range;
    - ``{"row_ids": [[first, last], ...]}``: listed rows (outliers frozen as row ids);
    - ``{"threshold": k, "members": {name: definition, ...}}``: fires when at least ``k`` members fire.

    Raises :class:`CriterionError` and nothing else: a malformed definition (a bound "18 years", a range of three
    numbers) is a criterion that cannot be evaluated, never an error that stops a page (the sidebar counts every
    criterion of the configuration before each page).
    """
    try:
        return rule_kind(name, definition).mask(df, name, definition)
    except CriterionError:
        raise
    except Exception as exc:
        raise CriterionError(f"criterion '{name}': {exc}") from exc


def rule_checks(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> list[RuleCheck]:
    """
    The Domain Validity checks of a criterion on ``df``: per check, the rows holding every value it reads and the rows
    where it fires, always checked. By default, the rows holding every column its kind reads and, for a kind that
    combines rules, where each of them is settled; a kind gives its own (``plausibility_checks``): a numeric range or
    bounds read the recorded numbers, one check per column for bounds; an expression its columns and its conversions;
    listed rows the rows the method assessed; at least k rules where Kleene logic settles them. A missing value is
    never counted as a pass: Completeness counts it. Raises :class:`CriterionError` and nothing else, as
    :func:`criterion_mask` does.
    """
    try:
        kind = rule_kind(name, definition)
        fires = kind.mask(df, name, definition)
        members = {
            str(member): settled(df, str(member), spec)
            for member, spec in _members(kind, definition).items()
        }
        if kind.plausibility_checks is not None:
            found = kind.plausibility_checks(df, name, definition, members)
            return [RuleCheck(name, column, held | fired, fired) for column, held, fired in found]
        held = pd.Series(True, index=df.index)
        for _, known in members.values():
            held &= known
        for column in kind.columns(name, definition):
            if column in df.columns:
                held &= df[column].notna()
        return [RuleCheck(name, "", held | fires, fires)]
    except CriterionError:
        raise
    except Exception as exc:
        raise CriterionError(f"criterion '{name}': {exc}") from exc


def settled(
    df: pd.DataFrame, name: str, definition: Mapping[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Where a criterion fires, and where its result is known: it fires, or every check it makes is checked."""
    fires = criterion_mask(df, name, definition)
    known = pd.Series(True, index=df.index)
    for check in rule_checks(df, name, definition):
        known &= check.checked
    return fires, fires | known


def describe_definition(name: str, definition: Mapping[str, Any]) -> str:
    """
    A criterion's definition in words (what makes it fire), as the participant flow and the dossier give it. Never an
    error, whatever the definition: a rule that the upload checks refuse reads "cannot be evaluated" and why, never as
    if it were sound ("age <= None", or a threshold rule with a member that cannot be read).
    """
    try:
        problems = criterion_problems(name, definition)
        if problems:
            return f"cannot be evaluated: {'; '.join(problems)}"
        return rule_kind(name, definition).describe(name, definition)
    except Exception as exc:
        return f"cannot be evaluated: {exc}"


def criterion_problems(name: str, definition: Any) -> list[str]:
    """
    What makes a criterion unusable on any table, in words (empty when it can be evaluated): a configuration with such
    a rule is refused when it is loaded. Only what fails whatever the data: a bound written "18" and a threshold of
    2.5 evaluate today, and still load.
    """
    try:
        kind = rule_kind(name, definition)
        problems = list(kind.problems(name, definition))
    except CriterionError as exc:
        return [str(exc)]
    # a kind's own function failed: never an error in the upload check itself
    except Exception as exc:
        return [f"criterion '{name}': {exc}"]
    for member, spec in _members(kind, definition).items():
        problems += [
            f"rule '{name}': {problem}" for problem in criterion_problems(str(member), spec)
        ]
    return problems


def _members(kind: CriterionKind, definition: Mapping[str, Any]) -> Mapping[str, Any]:
    members = definition.get(kind.members_key) if kind.members_key else None
    return members if isinstance(members, Mapping) else {}


def combined_rules(definition: Any) -> Mapping[str, Any]:
    """The rules a rule combines, by name (a threshold rule's members): none for a rule that cannot be read."""
    try:
        kind = _kind_or_none(definition) if isinstance(definition, Mapping) else None
        return _members(kind, definition) if kind is not None else {}
    except Exception:  # a kind's own function failed
        return {}


def holds_row_list(definition: Any) -> bool:
    """
    Whether a rule lists row ids of this dataset: its kind lists rows, or a rule it combines does, at any depth (a
    threshold rule's members). Such a rule only means something for this dataset, and is left out of a configuration
    shared with someone else (core.population.configuration_for_sharing). Each rule is read once, however the rules
    nest, and an error in a kind's own function counts as a row list.
    """
    pending, seen = [definition], set()
    while pending:
        rule = pending.pop()
        if not isinstance(rule, Mapping) or id(rule) in seen:
            continue
        seen.add(id(rule))
        try:
            kind = _kind_or_none(rule)
            if kind is not None and kind.lists_rows:
                return True
            if kind is not None:
                pending.extend(_members(kind, rule).values())
        except Exception:
            return True
    return False


def shared_definition(name: str, definition: Any) -> dict[str, Any] | None:
    """
    A rule as a configuration shared with someone else holds it: the keys its kind holds and no other, each only with
    a value its kind reads there (a loaded file can hold anything, row ids too), and the rules it combines likewise.
    None, and the rule is left out whole, when it lists rows of this dataset (holds_row_list), cannot be read, holds
    another value under one of its keys, or fails the checks of a loaded file (criterion_problems: a bound written
    "row 4512" is a text, as "18" is): the copy never relies on those checks having run.
    """
    try:
        if holds_row_list(definition) or criterion_problems(name, definition):
            return None
        return _shared(name, definition)
    except Exception:  # cannot be read, nested past reach, or a kind's own function failed
        return None


def _shared(name: str, definition: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = rule_kind(name, definition)
    shared: dict[str, Any] = {}
    for key, value in definition.items():  # in the order written
        if key == kind.members_key:
            if not isinstance(value, Mapping):
                return None
            members = {member: _shared(str(member), spec) for member, spec in value.items()}
            if any(rule is None for rule in members.values()):
                return None
            shared[key] = members
        # the catalogue's own key: the kind a definition states
        elif key == "kind" and kind.recognise is None:
            if not is_text(value):
                return None
            shared[key] = value
        elif key in kind.keys:
            if not kind.keys[key](value):
                return None
            shared[key] = copy.deepcopy(value)
    return shared


# ----------------------------------------------------------------------------------------------------- the writers
# Every rule the application writes comes from one of these (tests/test_quality_gates.py), so every definition it
# writes reads as a registered kind and holds only that kind's keys (tests/core_layer/test_criterion_kinds.py).


def numeric_rule(lower: Any, upper: Any, strategy: str | None) -> dict[str, Any]:
    """A range on the column named like the rule (Validate's form): *include* fires inside it, *exclude* outside."""
    return {"numeric": True, "lower_bound": lower, "upper_bound": upper, "strategy": strategy}


def expression_rule(expression: str) -> dict[str, Any]:
    """An expression (Validate's form, the AI assistant's suggestions)."""
    return {"numeric": False, "expression": expression}


def listed_rows_rule(
    row_ids: list[list[int]],
    *,
    method: str,
    params: Mapping[str, Any],
    columns: Sequence[str],
    description: str,
) -> dict[str, Any]:
    """The rows an outlier method flagged, frozen as row ids (Validate's outliers), with how they were found."""
    return {
        "row_ids": row_ids,
        "description": description,
        "method": method,
        "params": dict(params),
        "columns": list(columns),
    }


def threshold_rule(k: Any, members: Mapping[str, Any]) -> dict[str, Any]:
    """At least k of the rules (a family counted with a threshold, core.population.catalogue)."""
    return {"threshold": k, "members": members}


__all__ = [
    "ASKED",
    "CRITERION_KINDS",
    "MEMBERS",
    "METHOD_WORDS",
    "NEVER",
    "NOT_READ",
    "RULES_VERSION",
    "CriterionError",
    "CriterionKind",
    "RuleCheck",
    "combined_rules",
    "criterion_columns",
    "criterion_mask",
    "criterion_problems",
    "describe_definition",
    "expression_rule",
    "holds_row_list",
    "is_flag",
    "is_scalar",
    "rule_checks",
    "settled",
    "is_text",
    "listed_rows_rule",
    "numeric_rule",
    "rule_kind",
    "shared_definition",
    "threshold_rule",
]
