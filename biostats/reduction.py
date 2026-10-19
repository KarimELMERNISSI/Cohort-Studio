"""
Dimensionality reduction of a cohort table (M5): which columns a PCA or a FAMD can take, how FAMD types them, and how
much of the variance the components explain. Enrich (the preview and the traced transformation), Validate's outlier
projection and Data insight's clustering share these rules.

FAMD, the factor analysis of mixed data (Pagès 2004), analyses quantitative and qualitative variables together: each
quantitative variable adds one dimension, and each qualitative one its number of categories minus one. As in
FactoMineR, a numeric column is quantitative and any other column qualitative. prince's own rule (float columns only)
analysed an integer column, such as the age in years, as one category per value.

prince is imported where it is used (the optional 'ml' group), so the package imports without it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PCA, FAMD = "PCA", "FAMD"
# text whose values are mostly distinct describes rows, not groups (an identifier, free text)
IDENTIFIER_SHARE = 0.5
# the variance profile computes at most this many components (a FAMD can have hundreds of dimensions)
MAX_PROFILE_COMPONENTS = 50
RANDOM_STATE = 42


def _is_quantitative(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _is_temporal(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series)


def famd_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(quantitative, qualitative) columns, typed as FactoMineR types them: the numbers, and everything else."""
    quantitative = [str(name) for name, series in frame.items() if _is_quantitative(series)]
    return quantitative, [str(name) for name in frame.columns if str(name) not in quantitative]


def famd_dimensions(frame: pd.DataFrame) -> int:
    """One dimension per quantitative variable, and per qualitative one its categories minus one."""
    return sum(
        1 if _is_quantitative(series) else max(series.nunique() - 1, 0)
        for _, series in frame.items()
    )


def reduction_columns(
    frame: pd.DataFrame, method: str, *, date_columns: Iterable[str] = ()
) -> tuple[list[str], dict[str, str]]:
    """
    The columns a PCA (numbers) or a FAMD (numbers and categories) can take, and the others with the reason. Dates
    (date columns, and ``date_columns``: text the caller recognised as dates) and text with mostly distinct values
    would add one dimension per value; a column with a single value explains nothing.
    """
    dates = {str(c) for c in date_columns}
    usable: list[str] = []
    excluded: dict[str, str] = {}
    for col, series in frame.items():
        name = str(col)
        distinct = series.nunique(dropna=True)
        if name in dates or _is_temporal(series):
            excluded[name] = (
                "a date: derive a number from it first (e.g. the days since a reference date)"
            )
        elif distinct <= 1:
            excluded[name] = "a single value: it explains no variance"
        elif _is_quantitative(series):
            usable.append(name)
        elif method == PCA:
            excluded[name] = "not a number (FAMD takes categories)"
        elif distinct > IDENTIFIER_SHARE * series.notna().sum():
            excluded[name] = (
                "mostly distinct values (an identifier or free text): one dimension per value"
            )
        else:
            usable.append(name)
    return usable, excluded


def check_famd_input(frame: pd.DataFrame) -> None:
    """Raises ValueError, with what to do, for a table a FAMD cannot analyse."""
    temporal = [str(name) for name, series in frame.items() if _is_temporal(series)]
    if temporal:
        raise ValueError(
            f"{', '.join(temporal)}: a date cannot enter a FAMD; derive a number from it first (e.g. the days since "
            "a reference date)."
        )
    quantitative, qualitative = famd_types(frame)
    if not quantitative:
        raise ValueError(
            "A FAMD needs at least one numeric column; with categories only, analyse their counts."
        )
    if not qualitative:
        raise ValueError(
            "A FAMD needs at least one categorical column; with numbers only, use PCA."
        )


def fit_famd(frame: pd.DataFrame, n_components: int, *, random_state: int = RANDOM_STATE) -> Any:
    """A prince FAMD fitted on ``frame``, its columns typed as FactoMineR types them (``famd_types``)."""
    import prince

    check_famd_input(frame)
    _, qualitative = famd_types(frame)
    model = prince.FAMD(
        n_components=n_components,
        n_iter=3,
        random_state=random_state,
        engine="sklearn",
        categorical_columns=qualitative,
    )
    return model.fit(frame)


def fit_pca(frame: pd.DataFrame, n_components: int, *, random_state: int = RANDOM_STATE) -> Any:
    """A prince PCA of the standardised columns (each variable weighs the same, whatever its unit)."""
    import prince

    model = prince.PCA(
        n_components=n_components,
        rescale_with_mean=True,
        rescale_with_std=True,
        random_state=random_state,
        engine="sklearn",
    )
    return model.fit(frame)


@dataclass(frozen=True)
class VarianceProfile:
    """The share of the variance (%) each computed component explains, out of every dimension of the data."""

    percent: tuple[float, ...]
    n_dimensions: int
    contributions: pd.DataFrame | None = (
        None  # variable x component: correlations (PCA) or shares (FAMD, 0-1)
    )

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(np.asarray(self.percent, dtype=float))

    @property
    def complete(self) -> bool:
        """Every dimension was computed, so the cumulative share reaches 100%."""
        return len(self.percent) >= self.n_dimensions


def variance_profile(
    frame: pd.DataFrame, method: str, *, random_state: int = RANDOM_STATE
) -> VarianceProfile:
    """
    How the variance of ``frame`` spreads over the components: all of them for a PCA; for a FAMD, as many as the data
    has dimensions, up to MAX_PROFILE_COMPONENTS.
    """
    if method == PCA:
        model = fit_pca(frame, frame.shape[1], random_state=random_state)
        return VarianceProfile(
            tuple(float(p) for p in model.percentage_of_variance_),
            frame.shape[1],
            model.column_correlations,
        )
    dimensions = famd_dimensions(frame)
    model = fit_famd(
        frame,
        max(1, min(dimensions, len(frame) - 1, MAX_PROFILE_COMPONENTS)),
        random_state=random_state,
    )
    shares = model.variable_contributions_
    return VarianceProfile(
        tuple(float(p) for p in model.percentage_of_variance_),
        dimensions,
        shares / shares.sum(axis=0),
    )


__all__ = [
    "FAMD",
    "IDENTIFIER_SHARE",
    "MAX_PROFILE_COMPONENTS",
    "PCA",
    "VarianceProfile",
    "check_famd_input",
    "famd_dimensions",
    "famd_types",
    "fit_famd",
    "fit_pca",
    "reduction_columns",
    "variance_profile",
]
