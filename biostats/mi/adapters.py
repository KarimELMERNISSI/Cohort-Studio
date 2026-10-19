"""
Adapters: what to pool from each type of analysis result.

An adapter reads, from one completed-data result, the estimates on their pooling scale with their
covariance (or variances) and the complete-data df (``extract``), then turns the pooled table into the
result the page shows (``finish``): odds, rate and hazard ratios back from the log scale, D1 tests of
factors, Harel's R2, adjusted means, a pooled t-test, pooled Kaplan-Meier curves on the cloglog scale with
the median read from the pooled curve, D2 for log-rank and chi-square statistics. Analyses that cannot be
pooled raise :class:`NotPoolableError` with the reason and the alternative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from biostats._types import CI, MIInfo, TestResult
from biostats.mi.not_poolable import NotPoolable, not_poolable
from biostats.mi.pooling import (
    MultiParameterTest,
    PooledTable,
    d2_chisq,
    pool_r_squared,
    pool_scalar,
)


class NotPoolableError(ValueError):
    def __init__(self, entry: NotPoolable) -> None:
        super().__init__(f"{entry.reason} {entry.alternative}")
        self.entry = entry


@dataclass
class Extract:
    """What one completed-data analysis contributes to the pooling."""

    params: pd.Series  # estimates on the pooling scale
    vcov: pd.DataFrame | None = None
    variances: pd.Series | None = None
    dfcom: float = math.inf
    scales: dict[str, str] = field(default_factory=dict)  # term -> pooling scale
    factors: dict[str, list[str]] = field(default_factory=dict)  # factor -> its terms (D1 tests)
    chisq: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )  # name -> (statistic, df) for D2
    extra: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def mi_info(term: Any) -> MIInfo:
    """MIInfo of a pooled term."""
    return MIInfo(
        m=term.m,
        fmi=term.fmi,
        riv=term.riv,
        lam=term.lam,
        df=term.df,
        dfcom=term.dfcom,
        scale=term.scale,
    )


class Adapter:
    kind = "generic"

    def extract(self, result: Any) -> Extract:  # pragma: no cover - interface
        raise NotImplementedError

    def prepare(self, extracts: list[Extract]) -> tuple[list[Extract], list[str]]:
        """A last look at the extracts of all the imputed datasets before pooling: estimates that cannot be
        combined are dropped, with a warning that says why."""
        return extracts, []

    def tests(
        self, extracts: list[Extract], table: PooledTable | None
    ) -> dict[str, MultiParameterTest]:
        """D1 for each factor (several terms) and D2 for each chi-square statistic."""
        out: dict[str, MultiParameterTest] = {}
        first = extracts[0]
        if table is not None and table.covariances is not None:
            for factor, terms in first.factors.items():
                if len(terms) > 1 and all(t in table.names for t in terms):
                    out[factor] = table.wald(terms)
        for name in first.chisq:
            stats_, dfs = zip(*[e.chisq[name] for e in extracts], strict=True)
            out[name] = d2_chisq(list(stats_), df=int(dfs[0]))
        return out

    def finish(
        self, extracts: list[Extract], table: PooledTable | None, *, alpha: float
    ) -> dict[str, Any]:
        return {}


def _factor_terms(columns: list[str], reference_levels: dict[str, str]) -> dict[str, list[str]]:
    return {
        f: [c for c in columns if c.startswith(f"{f}_")]
        for f in reference_levels
        if any(c.startswith(f"{f}_") for c in columns)
    }


class RegressionAdapter(Adapter):
    """Logistic, Poisson and negative-binomial regression (biostats.regression.RegressionResult)."""

    kind = "regression"

    def extract(self, result: Any) -> Extract:
        if result.params is None or result.vcov is None:
            raise ValueError(
                "the model has no covariance matrix (a parameter at the boundary) and cannot be pooled"
            )
        names = list(result.params.index)
        return Extract(
            params=result.params,
            vcov=result.vcov,
            dfcom=float(result.df_resid),
            scales={n: "log" for n in names if n != "const"},
            factors=_factor_terms(names, dict(result.reference_levels)),
            warnings=list(result.warnings),
            extra={"kind": result.kind, "n": result.n, "events": result.events},
        )


class CoxAdapter(Adapter):
    """Cox proportional hazards (biostats.survival.CoxResult)."""

    kind = "cox"

    def extract(self, result: Any) -> Extract:
        if result.params is None or result.vcov is None:
            raise ValueError("the Cox model has no covariance matrix and cannot be pooled")
        names = list(result.params.index)
        return Extract(
            params=result.params,
            vcov=result.vcov,
            dfcom=float(result.df_resid),
            scales=dict.fromkeys(names, "log"),
            factors=_factor_terms(names, dict(result.reference_levels)),
            warnings=list(result.warnings),
            extra={"n": result.n, "events": result.events},
        )


class LinearAdapter(Adapter):
    """Linear regression and ANCOVA (biostats.linear)."""

    kind = "linear"

    def extract(self, result: Any) -> Extract:
        extra: dict[str, Any] = {"r2": result.r2, "r2_adj": result.r2_adj, "n": result.n}
        means = getattr(result, "adjusted_means", None)
        if means is not None:
            extra["adjusted_means"] = means.set_index("group")[["mean", "se"]]
            extra["group"] = result.group
        return Extract(
            params=result.params,
            vcov=result.vcov,
            dfcom=float(result.df_resid),
            factors={k: v for k, v in result.terms_of.items()},
            warnings=list(result.warnings),
            extra=extra,
        )

    def tests(
        self, extracts: list[Extract], table: PooledTable | None
    ) -> dict[str, MultiParameterTest]:
        out: dict[str, MultiParameterTest] = {}
        if table is not None:
            for source, terms in extracts[0].factors.items():
                out[source] = table.wald(terms)  # the Type II test of each covariate
        return out

    def finish(
        self, extracts: list[Extract], table: PooledTable | None, *, alpha: float
    ) -> dict[str, Any]:
        n = int(extracts[0].extra["n"])
        out: dict[str, Any] = {
            "r2": pool_r_squared([e.extra["r2"] for e in extracts], n=n, alpha=alpha),
        }
        if all(e.extra["r2_adj"] >= 0 for e in extracts):
            out["r2_adj"] = pool_r_squared([e.extra["r2_adj"] for e in extracts], n=n, alpha=alpha)
        if "adjusted_means" in extracts[0].extra:
            rows = []
            for level in extracts[0].extra["adjusted_means"].index:
                pooled = pool_scalar(
                    [e.extra["adjusted_means"].loc[level, "mean"] for e in extracts],
                    [e.extra["adjusted_means"].loc[level, "se"] ** 2 for e in extracts],
                    dfcom=extracts[0].dfcom,
                    alpha=alpha,
                    term=str(level),
                )
                rows.append(
                    {
                        "group": level,
                        "mean": pooled.estimate,
                        "se": pooled.se,
                        "ci_low": pooled.ci.lower,
                        "ci_high": pooled.ci.upper,
                        "df": pooled.df,
                        "fmi": pooled.fmi,
                    }
                )
            out["adjusted_means"] = pd.DataFrame(rows)
            out["group"] = extracts[0].extra["group"]
        return out


_RANK_TESTS = {
    "mann": "mann_whitney",
    "wilcoxon": "wilcoxon_signed_rank",
    "kruskal": "kruskal_wallis",
}


class TestAdapter(Adapter):
    """Tests that estimate a difference with its standard error (Welch, Student, paired, one-sample t)."""

    kind = "test"

    def extract(self, result: TestResult) -> Extract:
        if result.family == "nonparametric":
            key = next(
                (v for k, v in _RANK_TESTS.items() if k in result.name.lower()), "mann_whitney"
            )
            entry = not_poolable(key)
            assert entry is not None
            raise NotPoolableError(entry)
        if result.estimate is None or result.se is None:
            raise ValueError(
                f"{result.name} reports no estimate with a standard error and cannot be pooled"
            )
        label = result.estimate_label or "estimate"
        return Extract(
            params=pd.Series({label: result.estimate}),
            variances=pd.Series({label: result.se**2}),
            dfcom=float(result.df[0]) if result.df else math.inf,
            warnings=list(result.warnings),
            extra={"result": result},
        )

    def finish(
        self, extracts: list[Extract], table: PooledTable | None, *, alpha: float
    ) -> dict[str, Any]:
        assert table is not None
        term = table.terms[0]
        first: TestResult = extracts[0].extra["result"]
        pooled = first.model_copy(
            update={
                "name": f"{first.name} (pooled over {term.m} imputations)",
                "statistic": term.statistic,
                "df": [term.df],
                "p_value": term.p_value,
                "estimate": term.estimate,
                "estimate_ci": CI(
                    lower=term.ci.lower, upper=term.ci.upper, level=1 - alpha, method=term.ci.method
                ),
                "se": term.se,
                "mi": mi_info(term),
                "effect_sizes": [],
                "assumptions": [],
                "warnings": [],
            }
        )
        return {"result": pooled}


class ChiSquareAdapter(Adapter):
    """A chi-square test with no estimate (r x c independence): D2 combination of the statistics."""

    kind = "chisq"

    def extract(self, result: TestResult) -> Extract:
        if result.statistic is None or not result.df:
            raise ValueError(f"{result.name} reports no chi-square statistic and cannot be pooled")
        return Extract(
            params=pd.Series(dtype=float),
            chisq={"chi_square": (float(result.statistic), float(result.df[0]))},
            warnings=list(result.warnings),
            extra={"n": result.n_total},
        )

    def tests(
        self, extracts: list[Extract], table: PooledTable | None
    ) -> dict[str, MultiParameterTest]:
        if len({e.chisq["chi_square"][1] for e in extracts}) > 1:
            raise ValueError(
                "the tables have different shapes in the imputed datasets; fix the levels"
            )
        return super().tests(extracts, table)


RATIOS = ["odds_ratio", "relative_risk"]


class TwoByTwoAdapter(Adapter):
    """
    2x2 table: log odds ratio, log risk ratio and risk difference; D2 for the chi-square.

    A zero cell gets the +0.5 correction in its own imputed dataset only, so the ratios of the datasets would mix
    two estimators: they are then not pooled (the risk difference still is), unless every table was corrected
    (``analyze_twobytwo(..., haldane="always")``, a sensitivity analysis).
    """

    kind = "twobytwo"

    def extract(self, result: Any) -> Extract:
        params = pd.Series(
            {
                "odds_ratio": math.log(result.odds_ratio),
                "relative_risk": math.log(result.relative_risk),
                "risk_difference": result.risk_difference,
            }
        )
        variances = pd.Series(
            {
                "odds_ratio": result.log_or_se**2,
                "relative_risk": result.log_rr_se**2,
                "risk_difference": result.rd_se**2,
            }
        )
        return Extract(
            params=params,
            variances=variances,
            scales={"odds_ratio": "log", "relative_risk": "log"},
            chisq={"chi_square": (float(result.chi2_statistic), 1.0)},
            warnings=[g for g in result.guidance if "Haldane" in g],
            extra={
                "n": result.n_total,
                "zero_cell": min(result.a, result.b, result.c, result.d) == 0,
                "haldane": result.haldane_mode,
            },
        )

    def prepare(self, extracts: list[Extract]) -> tuple[list[Extract], list[str]]:
        zero = sum(bool(e.extra.get("zero_cell")) for e in extracts)
        if not zero or all(e.extra.get("haldane") == "always" for e in extracts):
            return extracts, []
        entry = not_poolable("twobytwo_zero_cell")
        assert entry is not None
        kept = [
            replace(
                e,
                params=e.params.drop(RATIOS),
                variances=None if e.variances is None else e.variances.drop(RATIOS),
                scales={},
                warnings=[w for w in e.warnings if "Haldane" not in w],
            )
            for e in extracts
        ]
        return kept, [
            f"Odds ratio and risk ratio not pooled: a cell is zero in {zero} of {len(extracts)} imputed datasets. "
            f"{entry.reason} {entry.alternative}"
        ]


def _step_values(
    times: np.ndarray, values: np.ndarray, grid: np.ndarray, before: float
) -> np.ndarray:
    """A right-continuous step function evaluated on ``grid`` (``before`` until the first time)."""
    idx = np.searchsorted(times, grid, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], before)
    return out.astype(float)


def pool_km_curves(curves: list[Any], *, alpha: float = 0.05) -> pd.DataFrame:
    """
    Pool m Kaplan-Meier curves of one group on the cloglog scale at every time any curve steps.

    Where all m survival probabilities lie strictly between 0 and 1 they are pooled with Rubin's rules on
    log(-log S) (Greenwood variances, delta method); elsewhere the mean is reported without an interval.
    """
    grid = np.unique(np.concatenate([np.asarray(c.times, dtype=float) for c in curves]))
    surv = np.array(
        [
            _step_values(np.asarray(c.times, float), np.asarray(c.survival, float), grid, 1.0)
            for c in curves
        ]
    )
    se = np.array(
        [
            _step_values(np.asarray(c.times, float), np.asarray(c.se, float), grid, 0.0)
            for c in curves
        ]
    )
    rows = []
    for j, t in enumerate(grid):
        s, v = surv[:, j], se[:, j] ** 2
        if np.all((s > 0) & (s < 1)) and np.all(np.isfinite(v)):
            pooled = pool_scalar(s, v, alpha=alpha, scale="cloglog", from_natural=True)
            assert pooled.natural_estimate is not None and pooled.natural_ci is not None
            rows.append(
                {
                    "time": t,
                    "survival": pooled.natural_estimate,
                    "lower": pooled.natural_ci.lower,
                    "upper": pooled.natural_ci.upper,
                    "fmi": pooled.fmi,
                }
            )
        else:
            rows.append(
                {
                    "time": t,
                    "survival": float(s.mean()),
                    "lower": np.nan,
                    "upper": np.nan,
                    "fmi": np.nan,
                }
            )
    return pd.DataFrame(rows)


def median_from_curve(curve: pd.DataFrame) -> float | None:
    below = curve.loc[curve["survival"] <= 0.5, "time"]
    return None if below.empty else float(below.iloc[0])


class SurvivalAdapter(Adapter):
    """Kaplan-Meier curves (pooled on the cloglog scale) and the log-rank test (D2)."""

    kind = "survival"

    def extract(self, result: Any) -> Extract:
        curves = result.curves if hasattr(result, "curves") else [result]
        chisq = {}
        if getattr(result, "logrank_statistic", None) is not None:
            chisq["log_rank"] = (float(result.logrank_statistic), float(result.logrank_df))
        return Extract(
            params=pd.Series(dtype=float),
            chisq=chisq,
            warnings=list(getattr(result, "warnings", [])),
            extra={"curves": {c.label: c for c in curves}},
        )

    def finish(
        self, extracts: list[Extract], table: PooledTable | None, *, alpha: float
    ) -> dict[str, Any]:
        labels = list(extracts[0].extra["curves"])
        pooled = {}
        medians = {}
        for label in labels:
            curves = [e.extra["curves"][label] for e in extracts if label in e.extra["curves"]]
            if any(c.se is None for c in curves):
                raise ValueError(
                    "the Kaplan-Meier curves carry no standard error and cannot be pooled"
                )
            curve = pool_km_curves(curves, alpha=alpha)
            pooled[label] = curve
            medians[label] = median_from_curve(curve)
        return {"curves": pooled, "medians": medians}


def adapter_for(result: Any) -> Adapter:
    """The adapter that pools results of this type."""
    from biostats.contingency import TwoByTwoResult
    from biostats.linear import LinearModelResult
    from biostats.regression import RegressionResult
    from biostats.survival import CoxResult, KMCurve, SurvivalComparison

    if isinstance(result, LinearModelResult):
        return LinearAdapter()
    if isinstance(result, RegressionResult):
        return RegressionAdapter()
    if isinstance(result, CoxResult):
        return CoxAdapter()
    if isinstance(result, TestResult):
        if result.estimate is None and result.statistic_label == "chi2":
            return ChiSquareAdapter()
        return TestAdapter()
    if isinstance(result, TwoByTwoResult):
        return TwoByTwoAdapter()
    if isinstance(result, SurvivalComparison | KMCurve):
        return SurvivalAdapter()
    raise TypeError(f"no pooling rule for results of type {type(result).__name__}")


__all__ = [
    "Adapter",
    "ChiSquareAdapter",
    "Extract",
    "NotPoolableError",
    "adapter_for",
    "median_from_curve",
    "mi_info",
    "pool_km_curves",
]
