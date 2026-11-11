"""
Design matrices for the regression models, with factor levels that can be frozen.

A covariate that is not numeric is a factor: it is one-hot coded with its first level (sorted) as the
reference, columns named ``<column>_<level>``, numeric columns first, as ``pandas.get_dummies`` does.

Across multiply imputed datasets every analysis must estimate the same terms. ``factor_levels`` reads the
levels once (from the observed data) and ``design_matrix(..., levels=...)`` then builds the same columns for
every completed dataset, even when a level happens to be absent from one of them, and refuses a value
outside the frozen levels instead of silently creating a new term.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def factor_levels(df: pd.DataFrame, covariates: Sequence[str]) -> dict[str, list[str]]:
    """The levels of the non-numeric covariates, sorted (the first one is the reference)."""
    levels: dict[str, list[str]] = {}
    for col in covariates:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            values = sorted(series.dropna().astype(str).unique())
            if values:
                levels[col] = values
    return levels


def design_matrix(
    df: pd.DataFrame,
    covariates: Sequence[str],
    *,
    levels: Mapping[str, Sequence[str]] | None = None,
    intercept: bool = True,
    drop_constant: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    The design matrix of ``covariates`` and the reference level of each factor.

    Without ``levels`` the levels come from ``df`` and constant columns are dropped (one analysis). With
    ``levels`` (frozen across imputations) every level gets its column and nothing is dropped, unless
    ``drop_constant`` asks for it: an analysis run again with the levels it recorded drops a constant numeric
    column as its first run did.
    """
    frozen = levels is not None
    if drop_constant is None:
        drop_constant = not frozen
    used = {
        k: list(v) for k, v in (levels or factor_levels(df, covariates)).items() if k in covariates
    }
    numeric: list[pd.DataFrame] = []
    dummies: list[pd.DataFrame] = []
    absent: dict[str, list[str]] = {}
    for col in covariates:
        if col in used:
            observed = df[col].notna()
            text = df[col].astype(str).where(observed)
            unknown = int((observed & ~text.isin(used[col])).sum())
            if unknown:
                raise ValueError(
                    f"{col} has {unknown} value(s) outside the levels {used[col]} fixed for the analysis"
                )
            values = pd.Categorical(text, categories=used[col])
            counts = pd.Series(values).value_counts()
            missing_levels = [lvl for lvl in used[col] if counts.get(lvl, 0) == 0]
            if missing_levels:
                absent[col] = missing_levels
            coded = pd.get_dummies(values, prefix=col, drop_first=True, dtype=float)
            coded.index = df.index
            dummies.append(coded)
        else:
            numeric.append(pd.to_numeric(df[col], errors="coerce").astype(float).to_frame(col))
    parts = numeric + dummies
    x = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    if frozen and absent:
        # an absent level leaves a column of zeros (or, for the reference, dummies collinear with the
        # intercept): the model cannot be estimated, say which level instead of "singular matrix"
        detail = "; ".join(f"{col}: {', '.join(lvls)}" for col, lvls in absent.items())
        raise ValueError(
            f"level(s) absent from these rows, so their terms cannot be estimated ({detail})"
        )
    if drop_constant:
        x = x.loc[:, x.std(ddof=0) > 0]
    if intercept:
        x = sm.add_constant(x, has_constant="add")
    return x, {col: used[col][0] for col in used}


def level_counts(
    df: pd.DataFrame,
    covariates: Sequence[str],
    events: Sequence[float],
    *,
    levels: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, tuple[int, int, int]]]:
    """
    The rows, the events and the rows with an event at each level of each factor among the rows a model read:
    ``{factor: {level: (rows, events, rows with an event)}}``, the levels in the design's order (the reference
    first). ``events`` gives each row's events (0 or 1, or a count), aligned with ``df``. In a count model the events
    of a level can all be one participant's: the rows with an event say how many people they are.

    A result shared outside the study withholds a term estimated on a few events (the small-cell rule): the counts
    behind each term are recorded with the model for that, and so that a check of the result compares them.
    """
    used = levels if levels is not None else factor_levels(df, covariates)
    happened = np.asarray(list(events), dtype=float)
    out: dict[str, dict[str, tuple[int, int, int]]] = {}
    for col in covariates:
        if col not in used:
            continue
        text = df[col].astype(str).where(df[col].notna()).to_numpy()
        out[col] = {str(level): _cell(text == level, happened) for level in used[col]}
    return out


def _cell(at: np.ndarray, happened: np.ndarray) -> tuple[int, int, int]:
    """The rows of a cell, their events and how many of them have an event."""
    return int(at.sum()), int(happened[at].sum()), int((happened[at] > 0).sum())


def _value_text(value: object) -> str:
    """A cell's value as text: False and True, a whole number without its decimal point, another number as it is."""
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    number = float(value)  # type: ignore[arg-type]
    return str(int(number)) if number.is_integer() else repr(number)


def indicator_counts(
    df: pd.DataFrame,
    covariates: Sequence[str],
    events: Sequence[float],
    *,
    levels: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, tuple[int, int, int]]]:
    """
    The rows, the events and the rows with an event at each value of each numeric or True/False covariate that
    holds two values among the rows a model read (as :func:`level_counts`), the lower value first.

    Such a covariate is an indicator (transplanted, pregnant, coded 0 and 1): its coefficient compares two cells, as
    a factor's does, although the design reads it as a number. With one participant coded 1, the coefficient's sign
    gives that participant's outcome, while the term would carry the whole model's counts: the cells are counted,
    so that a shared result withholds the term as it would a rare level.
    """
    used = levels if levels is not None else factor_levels(df, covariates)
    happened = np.asarray(list(events), dtype=float)
    out: dict[str, dict[str, tuple[int, int, int]]] = {}
    for col in covariates:
        if col in used:
            continue
        values = df[col].to_numpy()
        present = ~pd.isna(values)
        distinct = pd.unique(values[present])
        if len(distinct) != 2:
            continue
        try:
            low, high = sorted(distinct, key=float)
        except TypeError, ValueError:
            continue
        out[col] = {
            _value_text(value): _cell(present & (values == value), happened)
            for value in (low, high)
        }
    return out


__all__ = ["design_matrix", "factor_levels", "indicator_counts", "level_counts"]
