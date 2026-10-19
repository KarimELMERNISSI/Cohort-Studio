"""
Outlier detection: the one detector of the Validate page, the quality dashboard and the health score.

:func:`detect_outliers` flags the rows of a table by one method with one set of parameters (``DEFAULT_PARAMS``
fills what a caller leaves out), so the three screens flag the same rows. Before, each screen had its own copy and
they disagreed: the quantile bounds were interpolated differently, Isolation Forest was seeded differently, Local
Outlier Factor used a different number of neighbours, and Validate skipped the nullable integer columns.

- **Univariate methods** (``zscore``, ``iqr``, ``quantile``) work column by column on the continuous columns
  (numeric, not boolean, more than two distinct values: a 0/1 column has no outliers). A value outside the
  column's bounds is flagged; the same bounds clip the values when the outliers are clipped, so a clip previewed
  on the page is the clip recorded in the trace and replayed.
- **Multivariate methods** (Local Outlier Factor, Isolation Forest, DBSCAN) look at the rows that are complete on
  the chosen columns: numeric columns standardised, text columns one-hot encoded, identifier-like text columns
  (nearly one value per row) left out. A row with a missing value is not assessed, and the result says how many;
  it is never imputed first, which would let imputed values decide the flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from biostats._types import RANDOM_STATE

UNIVARIATE_METHODS = ("zscore", "iqr", "quantile")
MULTIVARIATE_METHODS = ("Local Outlier Factor", "Isolation Forest", "DBSCAN")
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "zscore": {"threshold": 3.0},
    "iqr": {"multiplier": 1.5},
    "quantile": {"lower": 0.01, "upper": 0.99},
    "Local Outlier Factor": {"n_neighbors": 50, "contamination": "auto"},
    "Isolation Forest": {"n_estimators": 100, "max_samples": "auto", "contamination": "auto"},
    "DBSCAN": {"eps": 0.5, "min_samples": 5},
}
IDENTIFIER_SHARE = (
    0.3  # a text column with more distinct values than this share of its rows is an identifier
)


def is_continuous(series: pd.Series) -> bool:
    """Numeric (any width, nullable included), not boolean, and more than two distinct values."""
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and series.nunique(dropna=True) > 2
    )


def univariate_bounds(
    series: pd.Series, method: str, params: dict[str, Any] | None = None
) -> tuple[float, float] | None:
    """
    [low, high] of a continuous column: a value outside is an outlier (and is clipped to the nearest bound). None
    when the column cannot have outliers by this method (not continuous, no spread).
    """
    if not is_continuous(series):
        return None
    params = {**DEFAULT_PARAMS[method], **(params or {})}
    values = pd.Series(pd.to_numeric(series, errors="coerce"), dtype=float)
    if method == "zscore":
        mean, sd, t = float(values.mean()), float(values.std()), float(params["threshold"])
        return None if not sd or np.isnan(sd) else (mean - t * sd, mean + t * sd)
    if method == "iqr":
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        m = abs(float(params["multiplier"]))
        return q1 - m * (q3 - q1), q3 + m * (q3 - q1)
    if method == "quantile":
        return float(values.quantile(float(params["lower"]))), float(
            values.quantile(float(params["upper"]))
        )
    raise ValueError(
        f"{method!r} is not a univariate method: use one of {', '.join(UNIVARIATE_METHODS)}"
    )


@dataclass
class OutlierDetection:
    """What one method flags in a table."""

    method: str
    params: dict[str, Any]
    columns: list[str]  # the columns analysed
    flags: pd.DataFrame  # univariate: one column per analysed column; multivariate: "overall"
    bounds: dict[str, list[float]] = field(
        default_factory=dict
    )  # univariate: [low, high] per analysed column
    scores: pd.Series | None = (
        None  # multivariate: the method's score (NaN for the rows not assessed)
    )
    skipped: list[str] = field(
        default_factory=list
    )  # columns not analysed (not continuous, identifier-like)
    not_assessed: int = 0  # multivariate: rows with a missing value in the columns used

    @property
    def flagged(self) -> pd.Index:
        """The rows flagged by any column (univariate) or by the method (multivariate)."""
        return self.flags.index[self.flags.any(axis=1)]


def _univariate(
    df: pd.DataFrame, columns: list[str], method: str, params: dict[str, Any]
) -> OutlierDetection:
    flags = pd.DataFrame(index=df.index)
    bounds: dict[str, list[float]] = {}
    skipped: list[str] = []
    for col in columns:
        limits = univariate_bounds(df[col], method, params)
        if limits is None:
            skipped.append(str(col))
            continue
        values = pd.to_numeric(df[col], errors="coerce").astype(float)
        low, high = float(limits[0]), float(limits[1])
        flags[col] = ((values < low) | (values > high)).fillna(False).astype(bool)
        bounds[str(col)] = [low, high]
    return OutlierDetection(
        method, params, [str(c) for c in flags.columns], flags, bounds, skipped=skipped
    )


def _multivariate(
    df: pd.DataFrame, columns: list[str], method: str, params: dict[str, Any]
) -> OutlierDetection:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric, text, skipped = [], [], []
    for col in columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric.append(col)
        elif series.nunique(dropna=True) <= max(2, IDENTIFIER_SHARE * series.notna().sum()):
            text.append(col)
        else:
            skipped.append(str(col))
    used = numeric + text
    complete = df[used].dropna()
    flags = pd.DataFrame({"overall": False}, index=df.index)
    scores = pd.Series(np.nan, index=df.index, dtype=float)
    result = OutlierDetection(
        method,
        params,
        [str(c) for c in used],
        flags,
        scores=scores,
        skipped=skipped,
        not_assessed=int(len(df) - len(complete)),
    )
    if not used or len(complete) < 3:
        return result
    encode = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                text,
            ),
        ],
        remainder="drop",
    )
    prepared = complete.copy()
    prepared[numeric] = prepared[numeric].astype(float)
    x = encode.fit_transform(prepared)
    if method == "Isolation Forest":
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            n_estimators=int(params["n_estimators"]),
            max_samples=params["max_samples"],
            contamination=params["contamination"],
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(x)
        values = model.decision_function(x)  # the lower, the more abnormal
    elif method == "Local Outlier Factor":
        from sklearn.neighbors import LocalOutlierFactor

        model = LocalOutlierFactor(
            n_neighbors=min(int(params["n_neighbors"]), len(complete) - 1),
            contamination=params["contamination"],
        )
        labels = model.fit_predict(x)
        values = -model.negative_outlier_factor_  # the higher, the more abnormal
    else:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(
            eps=float(params["eps"]), min_samples=int(params["min_samples"])
        ).fit_predict(x)
        values = labels.astype(float)  # the cluster of each row; -1 is noise, the outliers
    flags.loc[complete.index, "overall"] = labels == -1
    scores.loc[complete.index] = values
    return result


def detect_outliers(
    df: pd.DataFrame, columns: list[str], method: str, params: dict[str, Any] | None = None
) -> OutlierDetection:
    """The rows of ``df`` that ``method`` flags on ``columns``; the parameters it leaves out come from DEFAULT_PARAMS."""
    if method not in DEFAULT_PARAMS:
        raise ValueError(
            f"unknown outlier method {method!r}: use one of {', '.join(DEFAULT_PARAMS)}"
        )
    merged = {**DEFAULT_PARAMS[method], **(params or {})}
    if method in UNIVARIATE_METHODS:
        return _univariate(df, list(columns), method, merged)
    return _multivariate(df, list(columns), method, merged)


__all__ = [
    "DEFAULT_PARAMS",
    "MULTIVARIATE_METHODS",
    "UNIVARIATE_METHODS",
    "OutlierDetection",
    "detect_outliers",
    "is_continuous",
    "univariate_bounds",
]
