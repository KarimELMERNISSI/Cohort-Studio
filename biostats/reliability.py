"""Inter-rater and test-retest reliability (audit S11): Cohen's kappa with CI and intraclass correlation coefficients."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.inter_rater import cohens_kappa as _cohens_kappa

from biostats._types import CI
from biostats.references import REFERENCES

REFERENCES.setdefault(
    "cohen1960",
    "Cohen J (1960). A coefficient of agreement for nominal scales. Educ Psychol Meas 20:37-46.",
)
REFERENCES.setdefault(
    "shrout1979",
    "Shrout PE, Fleiss JL (1979). Intraclass correlations: uses in assessing rater reliability. Psychol Bull 86:420-428.",
)
REFERENCES.setdefault(
    "koo2016",
    "Koo TK, Li MY (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. J Chiropr Med 15:155-163.",
)


@dataclass
class KappaResult:
    kappa: float
    ci: CI
    se: float
    observed_agreement: float
    expected_agreement: float
    weighted: bool
    n: int
    interpretation: str
    references: list[str] = field(default_factory=list)


def landis_koch(k: float) -> str:
    """The word Landis and Koch (1977) give a kappa: a benchmark for an unweighted kappa, to be read on its interval
    as well as on its estimate. A kappa without a value has no word."""
    if not np.isfinite(k):
        return "no value"
    if k < 0:
        return "poor (less than chance)"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


_landis_koch = landis_koch


def category(value: object) -> str:
    """
    A rating as the category it is: a number by its value, whatever its type (2, 2.0 and "2.0" are the category
    "2"), anything else by its text. Two raters' columns are often read as whole numbers for one and as decimals for
    the other (a missing rating makes a column of decimals): compared as text they share no category.
    """
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
    else:
        text = str(value)
        try:
            number = float(text)
        except ValueError:
            return text
        if not np.isfinite(number):
            return text
    return str(int(number)) if number.is_integer() else repr(number)


def category_order(categories: Sequence[str]) -> list[str]:
    """The order categories are proposed in: by value when every one is a number, otherwise by name. For a weighted
    kappa the order is the scale's own, which only its user knows: this is where it starts."""
    try:
        return sorted(categories, key=float)
    except ValueError:
        return sorted(categories, key=lambda name: (name.casefold(), name))


def rater_table(
    first: pd.Series, second: pd.Series, *, order: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    The cross-table of two raters' categories over the participants rated by both: rows the first rater, columns the
    second, the same categories in the same order on both sides (:func:`category`: values are compared as values).

    ``order`` gives the categories from the lowest to the highest, as a weighted kappa reads them; it must hold every
    category either rater used. Without it they are put in the order of :func:`category_order`: fit for an unweighted
    kappa, where the order does not matter, and a first proposal otherwise (sorted as text, "high, low, medium, none"
    is no scale, and grades from 1 to 10 read 1, 10, 2).
    """
    both = first.notna().to_numpy() & second.notna().to_numpy()
    a = [category(v) for v in first[both]]
    b = [category(v) for v in second[both]]
    used = set(a) | set(b)
    if order is None:
        levels = category_order(list(used))
    else:
        levels = [str(name) for name in order]
        missing = sorted(used - set(levels))
        if missing or len(set(levels)) != len(levels):
            raise ValueError(
                "the order must hold every category once"
                + (f" (it lacks {len(missing)})" if missing else "")
            )
    table = pd.crosstab(
        pd.Categorical(a, categories=levels), pd.Categorical(b, categories=levels), dropna=False
    )
    table.index.name, table.columns.name = str(first.name), str(second.name)
    return table


def cohens_kappa(
    table: Sequence[Sequence[int]] | pd.DataFrame,
    *,
    weights: str | None = None,
    alpha: float = 0.05,
) -> KappaResult:
    """Cohen's kappa (unweighted, or linear/quadratic weights for ordered categories) with a normal-approximation CI."""
    arr = np.asarray(table, dtype=float)
    if arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError("a square table (raters agree on the categories) is required")
    wt = {None: None, "linear": "linear", "quadratic": "quadratic"}[weights]
    res = _cohens_kappa(arr, weights=None, wt=wt, return_results=True)
    z = stats.norm.ppf(1 - alpha / 2)
    kappa, se = float(res.kappa), float(res.var_kappa**0.5)
    n = int(arr.sum())
    po = float(np.trace(arr) / n) if n else float("nan")
    marg_r, marg_c = arr.sum(axis=1) / n, arr.sum(axis=0) / n
    pe = float((marg_r * marg_c).sum())
    return KappaResult(
        kappa=kappa,
        ci=CI(
            # kappa lies between -1 and 1: 19 of 20 in the same category gave an interval up to 1.09
            lower=max(-1.0, kappa - z * se),
            upper=min(1.0, kappa + z * se),
            level=1 - alpha,
            method="normal approximation",
        ),
        se=se,
        observed_agreement=po,
        expected_agreement=pe,
        weighted=weights is not None,
        n=n,
        interpretation=_landis_koch(kappa) + " agreement (Landis & Koch)",
        references=[REFERENCES["cohen1960"]],
    )


@dataclass
class ICCResult:
    icc: float
    ci: CI
    model: str  # ICC(2,1) two-way random, absolute agreement | ICC(3,1) two-way mixed, consistency
    f_statistic: float
    df1: float
    df2: float
    p_value: float
    n_subjects: int
    n_raters: int
    interpretation: str
    references: list[str] = field(default_factory=list)


def koo_li(v: float) -> str:
    """The word Koo and Li (2016) give an ICC; its authors read it on the interval, not on the estimate alone. An
    ICC without a value has no word (a missing value compared with 0.9 read "excellent")."""
    if not np.isfinite(v):
        return "no value"
    if v < 0.5:
        return "poor"
    if v < 0.75:
        return "moderate"
    if v < 0.9:
        return "good"
    return "excellent"


_koo_li = koo_li


def icc(
    ratings: pd.DataFrame | np.ndarray, *, model: str = "ICC(2,1)", alpha: float = 0.05
) -> ICCResult:
    """
    Single-measure intraclass correlation from a subjects x raters matrix (complete cases).
    ICC(2,1): two-way random effects, absolute agreement; ICC(3,1): two-way mixed effects, consistency (Shrout & Fleiss 1979).
    """
    data = np.asarray(pd.DataFrame(ratings).dropna(), dtype=float)
    n, k = data.shape
    if n < 3 or k < 2:
        raise ValueError("at least 3 subjects and 2 raters are required")
    grand = data.mean()
    ss_rows = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((data - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ms_r, ms_c, ms_e = ss_rows / (n - 1), ss_cols / (k - 1), ss_err / ((n - 1) * (k - 1))
    if model == "ICC(2,1)":
        value = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
        f = ms_r / ms_e
        # Shrout-Fleiss CI for ICC(2,1)
        a = k * value / (n * (1 - value))
        b = 1 + k * value * (n - 1) / (n * (1 - value))
        v = (a * ms_c + b * ms_e) ** 2 / (
            (a * ms_c) ** 2 / (k - 1) + (b * ms_e) ** 2 / ((n - 1) * (k - 1))
        )
        f_lo = stats.f.ppf(1 - alpha / 2, n - 1, v)
        f_hi = stats.f.ppf(1 - alpha / 2, v, n - 1)
        lo = n * (ms_r - f_lo * ms_e) / (f_lo * (k * ms_c + (k * n - k - n) * ms_e) + n * ms_r)
        hi = n * (f_hi * ms_r - ms_e) / (k * ms_c + (k * n - k - n) * ms_e + n * f_hi * ms_r)
        df1, df2 = n - 1, (n - 1) * (k - 1)
        label = "ICC(2,1) two-way random, absolute agreement"
    elif model == "ICC(3,1)":
        value = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)
        f = ms_r / ms_e
        df1, df2 = n - 1, (n - 1) * (k - 1)
        f_lo = f / stats.f.ppf(1 - alpha / 2, df1, df2)
        f_hi = f * stats.f.ppf(1 - alpha / 2, df2, df1)
        lo = (f_lo - 1) / (f_lo + k - 1)
        hi = (f_hi - 1) / (f_hi + k - 1)
        label = "ICC(3,1) two-way mixed, consistency"
    else:
        raise ValueError("model must be 'ICC(2,1)' or 'ICC(3,1)'")
    p = float(stats.f.sf(f, df1, df2))
    return ICCResult(
        icc=float(value),
        ci=CI(lower=float(lo), upper=float(hi), level=1 - alpha, method="Shrout-Fleiss F"),
        model=label,
        f_statistic=float(f),
        df1=float(df1),
        df2=float(df2),
        p_value=p,
        n_subjects=n,
        n_raters=k,
        interpretation=_koo_li(float(value)) + " reliability (Koo & Li)",
        references=[REFERENCES["shrout1979"], REFERENCES["koo2016"]],
    )


__all__ = [
    "ICCResult",
    "KappaResult",
    "category",
    "category_order",
    "cohens_kappa",
    "icc",
    "koo_li",
    "landis_koch",
    "rater_table",
]
