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
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    The design matrix of ``covariates`` and the reference level of each factor.

    Without ``levels`` the levels come from ``df`` and constant columns are dropped (one analysis). With
    ``levels`` (frozen across imputations) every level gets its column and nothing is dropped.
    """
    frozen = levels is not None
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
    if frozen:
        # an absent level leaves a column of zeros (or, for the reference, dummies collinear with the
        # intercept): the model cannot be estimated, say which level instead of "singular matrix"
        if absent:
            detail = "; ".join(f"{col}: {', '.join(lvls)}" for col, lvls in absent.items())
            raise ValueError(
                f"level(s) absent from these rows, so their terms cannot be estimated ({detail})"
            )
    else:
        x = x.loc[:, x.std(ddof=0) > 0]
    if intercept:
        x = sm.add_constant(x, has_constant="add")
    return x, {col: used[col][0] for col in used}


__all__ = ["design_matrix", "factor_levels"]
