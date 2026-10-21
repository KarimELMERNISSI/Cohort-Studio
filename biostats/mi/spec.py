"""
What an imputation does: the variables of the model, a method per variable, the predictors, m, the
iterations and the seed. The specification is serialisable (it is recorded in the trace and replayed).

Default methods follow mice: ``pmm`` for numeric variables, ``logreg`` for two categories, ``polyreg``
for more. ``""`` leaves a variable unimputed (it is then not used as a predictor while incomplete).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from biostats._types import RANDOM_STATE
from biostats.references import REFERENCES

REFERENCES.setdefault(
    "white2011",
    "White IR, Royston P, Wood AM (2011). Multiple imputation using chained equations: issues and guidance "
    "for practice. Stat Med 30:377-399.",
)
REFERENCES.setdefault(
    "white2009",
    "White IR, Royston P (2009). Imputing missing covariate values for the Cox model. Stat Med 28:1982-1998.",
)

METHODS = ("pmm", "norm", "logreg", "polyreg", "rf", "")
# a variable left out of the chains and computed in each completed dataset by its formula (passive imputation): set
# by the application after the imputation, never asked of the engine
DERIVED = "derived"
NUMERIC_METHODS = {"pmm", "norm", "rf"}
CATEGORICAL_METHODS = {"logreg", "polyreg", "rf"}


class ImputationSpec(BaseModel):
    """
    ``columns``: the variables of the imputation model (imputed when incomplete, predictors otherwise).
    ``methods`` and ``predictors`` override the defaults per variable. ``by`` imputes separately within the
    levels of a complete variable. ``survival`` = (time, event) of a Cox analysis adds the Nelson-Aalen H(T).
    """

    model_config = ConfigDict(frozen=True)

    columns: list[str]
    methods: dict[str, str] = Field(default_factory=dict)
    predictors: dict[str, list[str]] = Field(default_factory=dict)
    m: int = Field(default=20, ge=2, le=500)
    maxit: int = Field(default=10, ge=1, le=200)
    seed: int = RANDOM_STATE
    donors: int = Field(default=5, ge=1, le=50)
    ridge: float = Field(default=1e-5, ge=0)
    rf_trees: int = Field(default=10, ge=1, le=500)
    by: str | None = None
    survival: tuple[str, str] | None = None


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def default_method(series: pd.Series) -> str:
    """mice's defaults: pmm for numbers, logreg for two categories, polyreg for more."""
    if is_numeric(series):
        return "pmm"
    return "logreg" if series.dropna().astype(str).nunique() <= 2 else "polyreg"


def suggest_m(df: pd.DataFrame, columns: Sequence[str]) -> int:
    """
    White, Royston & Wood (2011): m at least the percentage of incomplete cases (minimum 5, maximum 100).
    """
    if len(df) == 0:
        return 5
    incomplete = df[list(columns)].isna().any(axis=1).mean()
    return int(min(100, max(5, math.ceil(100 * incomplete))))


__all__ = [
    "CATEGORICAL_METHODS",
    "DERIVED",
    "METHODS",
    "NUMERIC_METHODS",
    "ImputationSpec",
    "default_method",
    "is_numeric",
    "suggest_m",
]
