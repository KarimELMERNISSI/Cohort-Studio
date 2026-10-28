"""
What the multiple-imputation panel shows: the missingness of the candidate variables, the default
imputation model, and the methods paragraph of a report (Sterne et al. 2009 reporting items).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from biostats.mi.spec import DERIVED, default_method, is_numeric
from biostats.references import REFERENCES

REFERENCES.setdefault(
    "sterne2009",
    "Sterne JAC, White IR, Carlin JB, et al. (2009). Multiple imputation for missing data in epidemiological and "
    "clinical research: potential and pitfalls. BMJ 338:b2393.",
)

METHOD_NAMES = {
    "pmm": "predictive mean matching",
    "norm": "Bayesian linear regression",
    "logreg": "logistic regression",
    "polyreg": "multinomial logistic regression",
    "rf": "random-forest donors",
    # a declared derived variable, left out of the chains and computed in each completed dataset (passive imputation)
    DERIVED: "its formula, computed in each imputed dataset from the imputed values (passive imputation)",
}


def _identifier(series: pd.Series) -> bool:
    """A column whose values are all distinct (a patient or row identifier)."""
    values = series.dropna()
    if len(values) < 2 or values.nunique() < len(values):
        return False
    if not is_numeric(series):
        return True
    as_float = values.astype(float)
    return bool(
        (as_float == as_float.round()).all()
        and (as_float.is_monotonic_increasing or as_float.is_monotonic_decreasing)
    )


def default_model_columns(df: pd.DataFrame, *, max_levels: int = 20) -> list[str]:
    """
    Variables of the default imputation model: numbers and categories with at most ``max_levels`` levels; dates,
    identifiers, constant columns and free text are left out.
    """
    keep = []
    for col in df.columns:
        series = df[col]
        if (
            pd.api.types.is_datetime64_any_dtype(series)
            or int(series.dropna().nunique()) < 2
            or _identifier(series)
        ):
            continue
        if not is_numeric(series) and int(series.dropna().astype(str).nunique()) > max_levels:
            continue
        keep.append(str(col))
    return keep


def missingness_summary(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Per incomplete column: missing count, share, kind and default method (most incomplete first)."""
    rows = []
    for col in columns:
        n = int(df[col].isna().sum())
        if n == 0:
            continue
        numeric = is_numeric(df[col])
        levels = None if numeric else int(df[col].dropna().astype(str).nunique())
        rows.append(
            {
                "column": col,
                "missing": n,
                "share": n / len(df) if len(df) else 0.0,
                "kind": "number" if numeric else f"category ({levels} levels)",
                "method": default_method(df[col]),
            }
        )
    frame = pd.DataFrame(rows, columns=["column", "missing", "share", "kind", "method"])
    return frame.sort_values("missing", ascending=False, kind="stable").reset_index(drop=True)


def incomplete_share(df: pd.DataFrame, columns: Sequence[str]) -> float:
    """Share of rows with at least one missing value among ``columns``."""
    if len(df) == 0 or not columns:
        return 0.0
    return float(df[list(columns)].isna().any(axis=1).mean())


def methods_paragraph(
    binding: Any, *, n_rows: int, incomplete: float | None, info: dict[str, Any] | None = None
) -> str:
    """
    The methods paragraph of a report for an imputation (biostats.mi.binding.MIBinding, or anything with its
    ``spec``, ``methods`` and ``columns``), covering the reporting items of Sterne et al. (2009).
    """
    spec = binding.spec
    methods = binding.methods
    # derived variables: left out of the chains, computed in each completed dataset (a sentence of their own)
    derived = [col for col, method in methods.items() if method == DERIVED]
    groups: dict[str, list[str]] = {}
    for col, method in methods.items():
        if method != DERIVED:
            groups.setdefault(method, []).append(col)
    donors = spec.get("donors", 5)
    how = "; ".join(
        f"{', '.join(cols)} by {METHOD_NAMES.get(method, method)}"
        + (f" ({donors} donors)" if method == "pmm" else "")
        for method, cols in groups.items()
    )
    model = [c for c in binding.columns if c not in derived]
    parts = [
        "Missing data were handled by multiple imputation by chained equations (fully conditional specification), "
        "under the assumption that the data are missing at random given the variables of the imputation model.",
        f"Of the {n_rows:,} participants of the analysis population, {incomplete:.1%} had at least one missing value "
        "in these variables."
        if incomplete is not None
        else f"The analysis population had {n_rows:,} participants.",
        f"The imputation model included {len(model)} variables ({', '.join(model)}); the outcome of the analyses "
        + (
            "was included as a predictor, unless it is a derived variable, computed as described below."
            if derived
            else "was included as a predictor."
        ),
        f"Missing values were imputed as follows: {how}." if how else "",
    ]
    if derived:
        block = (info or {}).get("derived") or {}
        formulas = {str(s.get("target")): str(s.get("formula")) for s in block.get("steps") or []}
        named = "; ".join(f"{c} = {formulas[c]}" if c in formulas else c for c in derived)
        one = len(derived) == 1
        names = derived[0] if one else ", ".join(derived[:-1]) + f" and {derived[-1]}"
        parts.append(
            f"{names} {'was' if one else 'were'} not in the imputation model: {'its' if one else 'their'} missing "
            f"values were computed in each imputed dataset by {'its formula' if one else 'their formulas'} from the "
            f'imputed values ({named}; "impute, then transform", also called passive imputation; von Hippel 2009).'
        )
        left = block.get("left") or {}
        if left:
            parts.append(
                "Where the formula gives no value, missing values remained: "
                + ", ".join(f"{n:,} of {c}" for c, n in left.items())
                + "."
            )
    parts = [part for part in parts if part]
    if spec.get("by"):
        parts.append(f"The imputation was performed separately within the levels of {spec['by']}.")
    if spec.get("survival"):
        time_col, event_col = spec["survival"]
        parts.append(
            f"For the time-to-event analyses, the event indicator ({event_col}) and the Nelson-Aalen estimate of the "
            f"cumulative hazard at {time_col} were included as predictors instead of the follow-up time "
            "(White & Royston 2009)."
        )
    engine = (info or {}).get("engine", "CohortStudio")
    parts.append(
        f"We created {spec['m']} imputed datasets, each after {spec['maxit']} iterations (seed {spec['seed']}; "
        f"{engine}, Python)."
    )
    parts.append(
        "Each analysis was run on every imputed dataset and the results were combined with Rubin's rules, with "
        "Barnard-Rubin degrees of freedom; ratios were combined on the log scale (van Buuren 2018; Sterne et al. 2009)."
    )
    return " ".join(parts)


__all__ = [
    "METHOD_NAMES",
    "default_model_columns",
    "incomplete_share",
    "methods_paragraph",
    "missingness_summary",
]
