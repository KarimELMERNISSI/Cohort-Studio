"""
Pooling multiply imputed analyses: Rubin's rules and multi-parameter tests.

The formulas follow R ``mice::pool`` (van Buuren & Groothuis-Oudshoorn 2011) and ``mitml::testModels``
so that a pooled table can be checked against R to 1e-10.

One estimand (a coefficient, a difference, a log ratio), m imputations:

* Q-bar is the mean of the m estimates, U-bar the mean of their variances and B the variance of the
  estimates; the total variance is T = U-bar + (1 + 1/m) B.
* riv = (1 + 1/m) B / U-bar (relative increase in variance), lambda = (1 + 1/m) B / T (proportion of the
  variance due to the missing data) and fmi = (riv + 2 / (df + 3)) / (riv + 1).
* df: Barnard & Rubin (1999) with nu_com, the degrees of freedom of the complete-data analysis (the
  residual df n - k of a regression); lambda is floored at 1e-4 for the df only, as in mice.

Several estimands at once (all the dummy terms of a factor): the D1 Wald test (Li, Raghunathan & Rubin
1991, with Reiter's 2007 small-sample df when nu_com is known). m chi-square statistics (log-rank,
r x c chi-square): the D2 test (Li, Meng, Raghunathan & Rubin 1991), an approximation that can be
conservative or liberal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats

from biostats._types import CI
from biostats.mi.scales import from_scale, get_scale, to_scale
from biostats.references import REFERENCES

REFERENCES.setdefault(
    "rubin1987", "Rubin DB (1987). Multiple Imputation for Nonresponse in Surveys. Wiley."
)
REFERENCES.setdefault(
    "barnard1999",
    "Barnard J, Rubin DB (1999). Small-sample degrees of freedom with multiple imputation. Biometrika 86:948-955.",
)
REFERENCES.setdefault(
    "vanbuuren2011",
    "van Buuren S, Groothuis-Oudshoorn K (2011). mice: multivariate imputation by chained equations in R. "
    "J Stat Softw 45(3).",
)
REFERENCES.setdefault(
    "li1991_d1",
    "Li KH, Raghunathan TE, Rubin DB (1991). Large-sample significance levels from multiply imputed data using "
    "moment-based statistics and an F reference distribution. JASA 86:1065-1073.",
)
REFERENCES.setdefault(
    "reiter2007",
    "Reiter JP (2007). Small-sample degrees of freedom for multi-component significance tests with multiple "
    "imputation for missing data. Biometrika 94:502-508.",
)
REFERENCES.setdefault(
    "li1991_d2",
    "Li KH, Meng XL, Raghunathan TE, Rubin DB (1991). Significance levels from repeated p-values with "
    "multiply-imputed data. Stat Sin 1:65-92.",
)
REFERENCES.setdefault(
    "harel2009",
    "Harel O (2009). The estimation of R2 and adjusted R2 in incomplete data sets using multiple imputation. "
    "J Appl Stat 36:1109-1118.",
)
REFERENCES.setdefault(
    "marshall2009",
    "Marshall A, Altman DG, Holder RL, Royston P (2009). Combining estimates of interest in prognostic modelling "
    "studies after multiple imputation: current practice and guidelines. BMC Med Res Methodol 9:57.",
)

LAMBDA_FLOOR = 1e-4  # mice::barnard.rubin


class TermMismatchError(ValueError):
    """The completed-data analyses did not estimate the same terms (a factor level absent from one imputation)."""


class PooledTerm(BaseModel):
    """One pooled estimand with every quantity mice reports."""

    model_config = ConfigDict(frozen=True)

    term: str = ""
    estimate: float  # Q-bar, on the pooling scale
    se: float  # sqrt(T)
    statistic: float  # Q-bar / se
    df: float  # Barnard-Rubin
    p_value: float
    ci: CI  # on the pooling scale
    ubar: float
    b: float
    t: float
    riv: float
    lam: float
    fmi: float
    dfcom: float
    m: int
    scale: str = "identity"
    natural_estimate: float | None = None  # back-transformed when scale is not identity
    natural_ci: CI | None = None


class MultiParameterTest(BaseModel):
    """A pooled test of several parameters (D1) or of m chi-square statistics (D2)."""

    model_config = ConfigDict(frozen=True)

    method: Literal["D1", "D2"]
    terms: list[str]
    statistic: float  # F statistic
    df1: float
    df2: float
    p_value: float
    riv: float
    m: int
    note: str = ""
    references: list[str]


class PooledTable(BaseModel):
    """Pooled terms of one analysis, with the per-imputation estimates kept for Wald tests of term subsets."""

    model_config = ConfigDict(frozen=True)

    terms: list[PooledTerm]
    m: int
    dfcom: float
    alpha: float
    estimates: list[list[float]]  # m x k, pooling scale
    covariances: list[list[list[float]]] | None = None  # m x k x k, pooling scale
    references: list[str]

    @property
    def names(self) -> list[str]:
        return [t.term for t in self.terms]

    def term(self, name: str) -> PooledTerm:
        for t in self.terms:
            if t.term == name:
                return t
        raise KeyError(name)

    def wald(
        self, terms: Sequence[str], *, null: Sequence[float] | None = None
    ) -> MultiParameterTest:
        """D1 test that the given terms are jointly equal to ``null`` (zero by default)."""
        if self.covariances is None:
            raise ValueError(
                "the covariance matrices of the completed-data analyses are needed for a Wald test"
            )
        idx = [self.names.index(t) for t in terms]
        q = np.asarray(self.estimates, dtype=float)[:, idx]
        u = np.asarray(self.covariances, dtype=float)[:, idx][:, :, idx]
        return d1_wald(q, u, null=null, dfcom=self.dfcom, terms=list(terms))

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for t in self.terms:
            rows.append(
                {
                    "term": t.term,
                    "estimate": t.estimate,
                    "se": t.se,
                    "statistic": t.statistic,
                    "df": t.df,
                    "p_value": t.p_value,
                    "ci_low": t.ci.lower,
                    "ci_high": t.ci.upper,
                    "riv": t.riv,
                    "lambda": t.lam,
                    "fmi": t.fmi,
                    "ubar": t.ubar,
                    "b": t.b,
                    "t": t.t,
                    "dfcom": t.dfcom,
                    "m": t.m,
                    "scale": t.scale,
                    "natural_estimate": t.natural_estimate,
                    "natural_ci_low": None if t.natural_ci is None else t.natural_ci.lower,
                    "natural_ci_high": None if t.natural_ci is None else t.natural_ci.upper,
                }
            )
        return pd.DataFrame(rows)


class PooledRSquared(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimate: float
    ci: CI | None = None
    m: int
    references: list[str]


# --- distributions with infinite degrees of freedom -------------------------------------------------


def _t_sf(x: float, df: float) -> float:
    return float(stats.norm.sf(x) if math.isinf(df) else stats.t.sf(x, df))


def _t_ppf(q: float, df: float) -> float:
    return float(stats.norm.ppf(q) if math.isinf(df) else stats.t.ppf(q, df))


def _f_sf(x: float, df1: float, df2: float) -> float:
    if math.isinf(df2):
        return float(stats.chi2.sf(x * df1, df1))
    return float(stats.f.sf(x, df1, df2))


# --- scalar pooling ---------------------------------------------------------------------------------


def barnard_rubin_df(m: int, b: float, t: float, dfcom: float = math.inf) -> float:
    """mice::barnard.rubin: Barnard & Rubin (1999) degrees of freedom, lambda floored at 1e-4."""
    lam = (1.0 + 1.0 / m) * b / t if t > 0 else 0.0
    lam = max(lam, LAMBDA_FLOOR)
    dfold = (m - 1) / lam**2
    if math.isinf(dfcom):
        return float(dfold)
    dfobs = (dfcom + 1.0) / (dfcom + 3.0) * dfcom * (1.0 - lam)
    return float(dfold * dfobs / (dfold + dfobs))


def _validate(
    estimates: Sequence[float], variances: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    if q.ndim != 1 or u.shape != q.shape:
        raise ValueError("estimates and variances must be two sequences of the same length")
    if len(q) < 2:
        raise ValueError("at least two imputations are required to pool")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(u))):
        raise ValueError(
            "an estimate or a variance is missing or infinite in at least one imputation"
        )
    if np.any(u < 0):
        raise ValueError("a variance is negative")
    return q, u


def pool_scalar(
    estimates: Sequence[float],
    variances: Sequence[float],
    *,
    dfcom: float = math.inf,
    alpha: float = 0.05,
    term: str = "",
    scale: str = "identity",
    from_natural: bool = False,
) -> PooledTerm:
    """
    Rubin's rules for one estimand.

    ``scale`` names the pooling scale (see :mod:`biostats.mi.scales`). By default the estimates and variances
    are already on that scale (a log odds ratio and its variance); with ``from_natural=True`` they are on the
    natural scale (an AUC) and are transformed first, with the delta method for the variances. The pooled
    estimate and interval are also reported back on the natural scale.
    """
    get_scale(scale)
    if from_natural:
        pairs = [to_scale(q, u, scale) for q, u in zip(estimates, variances, strict=True)]
        estimates, variances = [p[0] for p in pairs], [p[1] for p in pairs]
    q, u = _validate(estimates, variances)
    m = len(q)
    if dfcom is None or (not math.isinf(dfcom) and dfcom < 1):
        dfcom = max(float(dfcom or 1.0), 1.0)
    qbar = float(q.mean())
    ubar = float(u.mean())
    b = float(q.var(ddof=1))
    t = ubar + (1.0 + 1.0 / m) * b
    if ubar > 0:
        riv = (1.0 + 1.0 / m) * b / ubar
    else:
        riv = math.inf if b > 0 else 0.0
    lam = (1.0 + 1.0 / m) * b / t if t > 0 else 0.0
    df = barnard_rubin_df(m, b, t, dfcom)
    fmi = 1.0 if math.isinf(riv) else (riv + 2.0 / (df + 3.0)) / (riv + 1.0)
    se = math.sqrt(t)
    if se > 0:
        statistic = qbar / se
        p_value = 2.0 * _t_sf(abs(statistic), df)
        half = _t_ppf(1.0 - alpha / 2.0, df) * se
    else:
        statistic, p_value, half = math.nan, math.nan, 0.0
    ci = CI(
        lower=qbar - half,
        upper=qbar + half,
        level=1.0 - alpha,
        method="Rubin's rules, Barnard-Rubin df",
    )
    natural_estimate: float | None = None
    natural_ci: CI | None = None
    if scale != "identity":
        natural_estimate, natural_ci = from_scale(qbar, ci, scale)
    return PooledTerm(
        term=term,
        estimate=qbar,
        se=se,
        statistic=statistic,
        df=df,
        p_value=p_value,
        ci=ci,
        ubar=ubar,
        b=b,
        t=t,
        riv=riv,
        lam=lam,
        fmi=fmi,
        dfcom=float(dfcom),
        m=m,
        scale=scale,
        natural_estimate=natural_estimate,
        natural_ci=natural_ci,
    )


# --- tables of terms --------------------------------------------------------------------------------


def _as_matrix(
    estimates: Sequence[Any], terms: Sequence[str] | None
) -> tuple[np.ndarray, list[str]]:
    """m x k estimates aligned on the term names (mappings) or taken as given (arrays with ``terms``)."""
    if isinstance(estimates[0], Mapping | pd.Series):
        rows = [{str(key): value for key, value in e.items()} for e in estimates]
        names = list(terms) if terms is not None else list(rows[0])
        missing: dict[int, list[str]] = {}
        extra: dict[int, list[str]] = {}
        for i, row in enumerate(rows):
            if set(row) != set(names):
                missing[i + 1] = [n for n in names if n not in row]
                extra[i + 1] = [k for k in row if k not in names]
        if missing or extra:
            detail = "; ".join(
                f"imputation {i}: missing {missing.get(i) or []}, extra {extra.get(i) or []}"
                for i in sorted(set(missing) | set(extra))
            )
            raise TermMismatchError(
                "the analyses of the imputed datasets did not estimate the same terms "
                f"(fix the factor levels before fitting): {detail}"
            )
        matrix = np.array([[float(row[n]) for n in names] for row in rows], dtype=float)
        return matrix, names
    matrix: np.ndarray = np.asarray(estimates, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("estimates must be m mappings or an m x k array")
    k = int(matrix.shape[-1])
    names = list(terms) if terms is not None else [f"term{j + 1}" for j in range(k)]
    if len(names) != k:
        raise ValueError("terms do not match the number of estimates")
    return matrix, names


def _as_covariances(covariances: Sequence[Any], names: list[str]) -> np.ndarray:
    mats = []
    for c in covariances:
        if isinstance(c, pd.DataFrame):
            c = c.loc[names, names]
        mats.append(np.asarray(c, dtype=float))
    arr = np.asarray(mats, dtype=float)
    k = len(names)
    if arr.shape[1:] != (k, k):
        raise ValueError(f"each covariance matrix must be {k} x {k}")
    return arr


def pool_terms(
    estimates: Sequence[Any],
    *,
    covariances: Sequence[Any] | None = None,
    variances: Sequence[Any] | None = None,
    terms: Sequence[str] | None = None,
    dfcom: float = math.inf,
    alpha: float = 0.05,
    scales: str | Mapping[str, str] | None = None,
) -> PooledTable:
    """
    Pool every term of m completed-data analyses (a regression table), as ``mice::pool``.

    ``estimates`` are m mappings term -> estimate (pandas Series from ``fit.params``) or an m x k array with
    ``terms``. Give either the m covariance matrices (``fit.cov_params()``; needed for Wald tests of several
    terms) or the m vectors of variances. Every analysis must estimate the same terms.
    """
    q, names = _as_matrix(estimates, terms)
    m, k = q.shape
    cov: np.ndarray | None = None
    if covariances is not None:
        cov = _as_covariances(covariances, names)
        u = np.array([np.diag(c) for c in cov])
    elif variances is not None:
        u_list = []
        for v in variances:
            if isinstance(v, Mapping | pd.Series):
                by_name = {str(key): value for key, value in v.items()}
                v = [float(by_name[n]) for n in names]
            u_list.append(np.asarray(v, dtype=float))
        u = np.asarray(u_list, dtype=float)
    else:
        raise ValueError(
            "give the covariance matrices or the variances of the completed-data estimates"
        )
    if u.shape != (m, k):
        raise ValueError("variances do not match the estimates")
    per_term: dict[str, str] = {}
    if isinstance(scales, str):
        per_term = dict.fromkeys(names, scales)
    elif scales is not None:
        per_term = {n: scales.get(n, "identity") for n in names}
    pooled = [
        pool_scalar(
            q[:, j],
            u[:, j],
            dfcom=dfcom,
            alpha=alpha,
            term=names[j],
            scale=per_term.get(names[j], "identity"),
        )
        for j in range(k)
    ]
    return PooledTable(
        terms=pooled,
        m=m,
        dfcom=float(dfcom),
        alpha=alpha,
        estimates=q.tolist(),
        covariances=None if cov is None else cov.tolist(),
        references=[
            REFERENCES["rubin1987"],
            REFERENCES["barnard1999"],
            REFERENCES["vanbuuren2011"],
        ],
    )


# --- multi-parameter tests --------------------------------------------------------------------------


def d1_wald(
    estimates: Any,
    covariances: Any,
    *,
    null: Sequence[float] | None = None,
    dfcom: float = math.inf,
    terms: Sequence[str] = (),
) -> MultiParameterTest:
    """
    D1: pooled Wald test that k parameters equal ``null`` (mitml::testModels, method "D1").

    The denominator df is Li, Raghunathan & Rubin's (1991) when the complete-data df is unknown and Reiter's
    (2007) small-sample df otherwise.
    """
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(covariances, dtype=float)
    if q.ndim != 2 or u.ndim != 3 or u.shape != (q.shape[0], q.shape[1], q.shape[1]):
        raise ValueError("D1 needs m x k estimates and m x k x k covariance matrices")
    m, k = q.shape
    if m < 2:
        raise ValueError("at least two imputations are required to pool")
    qbar = q.mean(axis=0)
    ubar = u.mean(axis=0)
    b = np.atleast_2d(np.cov(q, rowvar=False, ddof=1))
    diff = qbar - (np.zeros(k) if null is None else np.asarray(null, dtype=float))
    try:
        r = (1.0 + 1.0 / m) * float(np.trace(np.linalg.solve(ubar, b))) / k
        statistic = float(diff @ np.linalg.solve(ubar, diff)) / (k * (1.0 + r))
    except np.linalg.LinAlgError:
        raise ValueError("the pooled within-imputation covariance matrix is singular") from None
    t = k * (m - 1)
    note = ""
    if r <= 0:
        df2 = math.inf
    elif math.isinf(dfcom) or t <= 4:
        df2 = (
            4 + (t - 4) * (1 + (1 - 2 / t) / r) ** 2
            if t > 4
            else t * (1 + 1 / k) * (1 + 1 / r) ** 2 / 2
        )
        if not math.isinf(dfcom):
            note = "Too few imputations for Reiter's small-sample df (k(m - 1) <= 4); large-sample df used."
    else:
        a = r * t / (t - 2)
        vstar = (dfcom + 1) / (dfcom + 3) * dfcom
        c0 = 1 / (t - 4)
        c1 = vstar - 2 * (1 + a)
        c2 = vstar - 4 * (1 + a)
        z = 1 / c2 + c0 * (a**2 * c1 / ((1 + a) ** 2 * c2))
        df2 = 4 + 1 / z
    refs = [REFERENCES["li1991_d1"]] + ([] if math.isinf(dfcom) else [REFERENCES["reiter2007"]])
    return MultiParameterTest(
        method="D1",
        terms=list(terms),
        statistic=statistic,
        df1=float(k),
        df2=float(df2),
        p_value=_f_sf(statistic, k, df2),
        riv=r,
        m=m,
        note=note,
        references=refs,
    )


def d2_chisq(
    statistics: Sequence[float], df: int, *, terms: Sequence[str] = ()
) -> MultiParameterTest:
    """
    D2: combine m chi-square statistics with ``df`` degrees of freedom (mitml::testModels, method "D2").

    An approximation for tests that give no estimate with a variance (log-rank, r x c chi-square); it can be
    conservative or liberal, so a D1 test of the corresponding model terms is preferred when one exists.
    """
    d = np.asarray(statistics, dtype=float)
    m = len(d)
    if m < 2:
        raise ValueError("at least two imputations are required to pool")
    if np.any(d < 0) or not np.all(np.isfinite(d)):
        raise ValueError("chi-square statistics must be finite and non-negative")
    k = float(df)
    r = (1.0 + 1.0 / m) * float(np.var(np.sqrt(d), ddof=1))
    statistic = (float(d.mean()) / k - (m + 1.0) / (m - 1.0) * r) / (1.0 + r)
    df2 = math.inf if r == 0 else k ** (-3.0 / m) * (m - 1) * (1 + 1 / r) ** 2
    return MultiParameterTest(
        method="D2",
        terms=list(terms),
        statistic=statistic,
        df1=k,
        df2=float(df2),
        p_value=1.0 if statistic <= 0 else _f_sf(statistic, k, df2),
        riv=r,
        m=m,
        note="D2 combines chi-square statistics; it is approximate and can be conservative or liberal.",
        references=[REFERENCES["li1991_d2"]],
    )


def pool_r_squared(
    values: Sequence[float], *, n: int | None = None, alpha: float = 0.05
) -> PooledRSquared:
    """
    Harel's (2009) pooled R^2: average the Fisher z of sqrt(R^2) and square the back-transform.

    With ``n`` (the number of rows analysed) an interval is derived from Rubin's rules on the z scale with
    within-imputation variance 1 / (n - 3).
    """
    r2 = np.asarray(values, dtype=float)
    if len(r2) < 2:
        raise ValueError("at least two imputations are required to pool")
    if np.any(r2 < 0) or np.any(r2 >= 1):
        raise ValueError("R^2 values must lie in [0, 1)")
    z = np.arctanh(np.sqrt(r2))
    zbar = float(z.mean())
    ci: CI | None = None
    if n is not None and n > 3:
        pooled = pool_scalar(z, np.full(len(z), 1.0 / (n - 3)), alpha=alpha)
        low, high = math.tanh(max(pooled.ci.lower, 0.0)), math.tanh(max(pooled.ci.upper, 0.0))
        ci = CI(
            lower=low**2,
            upper=high**2,
            level=1 - alpha,
            method="Harel (2009), Fisher z of sqrt(R2)",
        )
    return PooledRSquared(
        estimate=math.tanh(zbar) ** 2, ci=ci, m=len(r2), references=[REFERENCES["harel2009"]]
    )


__all__ = [
    "LAMBDA_FLOOR",
    "MultiParameterTest",
    "PooledRSquared",
    "PooledTable",
    "PooledTerm",
    "TermMismatchError",
    "barnard_rubin_df",
    "d1_wald",
    "d2_chisq",
    "pool_r_squared",
    "pool_scalar",
    "pool_terms",
]
