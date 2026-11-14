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
    welch_corrected,
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
    # complete-data df of each term and of each factor's test, when they are not ``dfcom`` (Welch-type df of
    # group means with unequal variances); the smallest over the imputed datasets is used
    term_dfcom: dict[str, float] = field(default_factory=dict)
    test_dfcom: dict[str, float] = field(default_factory=dict)
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


def _smallest_cells(counts: list[dict[str, dict[str, tuple[int, int, int]]]], name: str) -> dict:
    """Of one covariate's cells over the datasets that count them: the smallest rows, events, rows with an event
    and rows without an event of each cell."""
    held = [c[name] for c in counts if name in c]
    return {
        value: (
            min(c[value][0] for c in held),
            min(c[value][1] for c in held),
            min(c[value][2] for c in held),
            min(c[value][0] - c[value][2] for c in held),
        )
        for value in held[0]
    }


def _model_sizes(extracts: list[Extract]) -> dict[str, Any]:
    """
    The sizes behind a pooled model: its rows and its events as their smallest and largest over the imputed datasets,
    its smallest residual degrees of freedom, the datasets whose fit converged, and for each factor level, and each
    value of a covariate that holds two values (an indicator), the smallest number of rows, of events, of rows with
    an event and of rows without one. A level or an outcome read from imputed values differs in size between the
    datasets; the small-cell rule of a shared result takes the smallest.
    """
    rows = [int(e.extra["n"]) for e in extracts]
    out: dict[str, Any] = {"n": (min(rows), max(rows))}
    events = [e.extra.get("events") for e in extracts]
    if all(v is not None for v in events):
        out["events"] = (int(min(events)), int(max(events)))
    # a count model: the rows that hold an event, its events being a sum of counts
    with_events = [e.extra.get("rows_with_events") for e in extracts]
    if all(v is not None for v in with_events):
        out["rows_with_events"] = (int(min(with_events)), int(max(with_events)))
    # a count model's dispersion in each dataset, as the smallest and the largest: a Poisson model above 1.5 has
    # standard errors that are too small, which a pooled model said of no dataset
    spread = [e.extra.get("dispersion") for e in extracts]
    if spread and all(isinstance(v, int | float) and math.isfinite(v) for v in spread):
        out["dispersion"] = (float(min(spread)), float(max(spread)))
    residual = [e.dfcom for e in extracts if math.isfinite(e.dfcom)]
    if residual:
        out["df_resid"] = int(min(residual))
    out["datasets"] = len(extracts)
    out["converged_in"] = sum(bool(e.extra.get("converged", True)) for e in extracts)
    if any("rows_left_out" in e.extra for e in extracts):
        # the rows a dataset's model left out for another reason than a missing value (a follow-up time below
        # zero): the most over the datasets
        out["rows_left_out"] = max(int(e.extra.get("rows_left_out") or 0) for e in extracts)
    levels = [e.extra.get("level_counts") or {} for e in extracts]
    for factor, first in levels[0].items():
        if any(list(c.get(factor, {})) != list(first) for c in levels):
            # every dataset is given the same levels: a term without its own counts would pass for a large one
            raise ValueError("the imputed datasets do not hold the same levels of a factor")
    out["level_counts"] = {factor: _smallest_cells(levels, factor) for factor in levels[0]}
    # an indicator: two values in a dataset; where an imputation gave it other values it is counted on the datasets
    # that keep the two (the observed rows are in all of them)
    indicators = [e.extra.get("indicator_counts") or {} for e in extracts]
    names = list(dict.fromkeys(name for c in indicators for name in c))
    out["indicator_counts"] = {}
    for name in names:
        held = [c for c in indicators if name in c]
        same = [c for c in held if list(c[name]) == list(held[0][name])]
        out["indicator_counts"][name] = _smallest_cells(same, name)
    return out


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
            extra={
                "kind": result.kind,
                "n": result.n,
                "events": result.events,
                "rows_with_events": getattr(result, "rows_with_events", None),
                "dispersion": getattr(result, "dispersion", None),
                "level_counts": dict(getattr(result, "level_counts", {})),
                "indicator_counts": dict(getattr(result, "indicator_counts", {})),
                "converged": getattr(result, "converged", True),
            },
        )

    def finish(
        self, extracts: list[Extract], table: PooledTable | None, *, alpha: float
    ) -> dict[str, Any]:
        return {"family": extracts[0].extra["kind"], **_model_sizes(extracts)}


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
            extra={
                "n": result.n,
                "events": result.events,
                "level_counts": dict(getattr(result, "level_counts", {})),
                "indicator_counts": dict(getattr(result, "indicator_counts", {})),
                "converged": getattr(result, "converged", True),
                "rows_left_out": int(getattr(result, "rows_left_out", 0) or 0),
            },
        )

    def finish(
        self, extracts: list[Extract], table: PooledTable | None, *, alpha: float
    ) -> dict[str, Any]:
        return _model_sizes(extracts)


class LinearAdapter(Adapter):
    """Linear regression and ANCOVA (biostats.linear)."""

    kind = "linear"

    def extract(self, result: Any) -> Extract:
        extra: dict[str, Any] = {
            "r2": result.r2,
            "r2_adj": result.r2_adj,
            "n": result.n,
            "cov_type": result.cov_type,  # robust: the pooled tests and means do not assume equal variances
            "references": list(result.references),
        }
        means = getattr(result, "adjusted_means", None)
        if means is not None:
            columns = ["mean", "se", "df"] if "df" in means else ["mean", "se"]
            extra["adjusted_means"] = means.set_index("group")[columns]
            extra["group"] = result.group
        return Extract(
            params=result.params,
            vcov=result.vcov,
            dfcom=float(result.df_resid),
            term_dfcom=dict(result.term_df),
            test_dfcom=dict(result.factor_df),
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
                # the Type II test of each covariate; a factor with its own complete-data df has Welch's (the group
                # means of welch_means), and its test then takes Welch's second-order correction too
                if all(source in e.test_dfcom for e in extracts):
                    dfcom = min(e.test_dfcom[source] for e in extracts)
                    out[source] = welch_corrected(table.wald(terms, dfcom=dfcom), dfcom)
                else:
                    out[source] = table.wald(terms)
        return out

    def finish(
        self, extracts: list[Extract], table: PooledTable | None, *, alpha: float
    ) -> dict[str, Any]:
        n = int(extracts[0].extra["n"])
        out: dict[str, Any] = {
            "r2": pool_r_squared([e.extra["r2"] for e in extracts], n=n, alpha=alpha),
            "cov_type": extracts[0].extra.get("cov_type", "nonrobust"),
            "references": extracts[0].extra.get("references", []),
        }
        if all(e.extra["r2_adj"] >= 0 for e in extracts):
            out["r2_adj"] = pool_r_squared([e.extra["r2_adj"] for e in extracts], n=n, alpha=alpha)
        if "adjusted_means" in extracts[0].extra:
            rows = []
            for level in extracts[0].extra["adjusted_means"].index:
                means = [e.extra["adjusted_means"].loc[level] for e in extracts]
                # a mean's own complete-data df when it has one (n - 1 of its group), else the model's
                dfcom = min(
                    float(row["df"]) if "df" in row else e.dfcom
                    for row, e in zip(means, extracts, strict=True)
                )
                pooled = pool_scalar(
                    [row["mean"] for row in means],
                    [row["se"] ** 2 for row in means],
                    dfcom=dfcom,
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
    # the columns named even without a row: a group with no event has no step, and still a curve
    return pd.DataFrame(rows, columns=["time", "survival", "lower", "upper", "fmi"])


def _grid_counts(curves: list[Any], times: np.ndarray, everywhere: bool) -> dict[str, Any]:
    """
    At each time of a pooled curve's grid, over the imputed datasets: the fewest events up to that time, the events
    up to that time in each dataset (``events_each``: the events between two times are counted in each dataset, since
    the difference of two smallest counts can be more than any dataset holds), and the fewest rows still at risk
    at it. The rows at risk between two of a dataset's event times are not known exactly (the censorings in between
    are counted, not timed): the number at its next event time is taken, which is never more than the truth. A
    dataset that lacks the group counts none. What a shared curve is thinned by.
    """
    events = np.full(len(times), np.iinfo(np.int64).max if everywhere else 0, dtype=np.int64)
    at_risk = events.copy()
    each: list[list[int]] = []
    for curve in curves if everywhere else []:
        own = np.asarray(curve.times, dtype=float)
        happened = getattr(curve, "events_at", None)
        if happened is None:  # a curve computed before its events were kept: nothing can be said
            return {"events": [0] * len(times), "at_risk": [0] * len(times), "events_each": []}
        so_far = np.concatenate([[0], np.cumsum(np.asarray(happened, dtype=np.int64))])
        on_grid = so_far[np.searchsorted(own, times, side="right")]
        each.append([int(v) for v in on_grid])
        events = np.minimum(events, on_grid)
        left = np.concatenate([np.asarray(curve.at_risk, dtype=np.int64), [0]])
        at_risk = np.minimum(at_risk, left[np.searchsorted(own, times, side="left")])
    return {
        "events": [int(v) for v in events],
        "at_risk": [int(v) for v in at_risk],
        "events_each": each,
    }


def _followed(curves: list[Any], everywhere: bool) -> dict[str, list]:
    """
    Over the imputed datasets, at every time a row's follow-up ends in any of them: the fewest rows still followed at
    that time. A dataset that lacks the group counts none, and a curve computed before its follow-up times were kept
    gives nothing. What a shared curve says its follow-up by.
    """
    if (
        not curves
        or not everywhere
        or any(getattr(c, "follow_times", None) is None for c in curves)
    ):
        return {"times": [], "rows": []}
    times = np.unique(np.concatenate([np.asarray(c.follow_times, dtype=float) for c in curves]))
    rows = np.full(len(times), np.iinfo(np.int64).max, dtype=np.int64)
    for curve in curves:
        own = np.asarray(curve.follow_times, dtype=float)
        left = np.concatenate([np.asarray(curve.followed, dtype=np.int64), [0]])
        rows = np.minimum(rows, left[np.searchsorted(own, times, side="left")])
    return {"times": [float(v) for v in times], "rows": [int(v) for v in rows]}


def median_from_curve(
    curve: pd.DataFrame, column: str = "survival", *, last: float | None = None
) -> float | None:
    """
    The median of a pooled curve (``column`` "survival"), or an end of its interval ("lower", "upper": the pooled
    band), by the rule of every median (biostats.survival.km_quantile); ``last`` is the last follow-up time.
    """
    from biostats.survival import km_quantile

    return km_quantile(curve["time"], curve[column], last=last)


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
            extra={
                "curves": {c.label: c for c in curves},
                # the rows in no curve: of groups too small to draw, with a follow-up time below zero
                "rows_skipped": int(getattr(result, "skipped_rows", 0) or 0),
                "rows_left_out": int(getattr(result, "rows_left_out", 0) or 0),
            },
        )

    def finish(
        self, extracts: list[Extract], table: PooledTable | None, *, alpha: float
    ) -> dict[str, Any]:
        labels = list(extracts[0].extra["curves"])
        pooled = {}
        medians = {}
        intervals = {}
        sizes = {}
        events = {}
        censored = {}
        grid = {}
        followed = {}
        for label in labels:
            curves = [e.extra["curves"][label] for e in extracts if label in e.extra["curves"]]
            # the smallest: a group read from imputed values can differ in size, and a dataset where the group was
            # too small to draw counts as none
            everywhere = len(curves) == len(extracts)
            sizes[label] = min(c.n for c in curves) if everywhere else 0
            events[label] = min(c.events for c in curves) if everywhere else 0
            censored[label] = min(c.n - c.events for c in curves) if everywhere else 0
            if any(c.se is None for c in curves):
                raise ValueError(
                    "the Kaplan-Meier curves carry no standard error and cannot be pooled"
                )
            curve = pool_km_curves(curves, alpha=alpha)
            pooled[label] = curve
            grid[label] = _grid_counts(curves, curve["time"].to_numpy(dtype=float), everywhere)
            followed[label] = _followed(curves, everywhere)
            # a stretch at exactly 0.5 runs to the last follow-up of any imputation (a curve recorded before: none)
            ends = [c.last_time for c in curves if getattr(c, "last_time", None) is not None]
            last = max(ends) if ends else None
            medians[label] = median_from_curve(curve, last=last)
            intervals[label] = (
                median_from_curve(curve, "lower", last=last),
                median_from_curve(curve, "upper", last=last),
            )
        totals = [
            (
                sum(c.n for c in e.extra["curves"].values()),
                sum(c.events for c in e.extra["curves"].values()),
            )
            for e in extracts
        ]
        return {
            "curves": pooled,
            "medians": medians,
            "median_ci": intervals,
            "sizes": sizes,
            "events": events,
            "censored": censored,  # the rows without the event, the fewest over the datasets
            # at each time of a pooled curve: the fewest events so far and the fewest rows still at risk
            "grid_counts": grid,
            # at every time a follow-up ends: the fewest rows still followed
            "followed": followed,
            # the rows in no curve, the most over the datasets: a group read from imputed values is too small to
            # draw in some of them only
            "rows_skipped": max(int(e.extra.get("rows_skipped") or 0) for e in extracts),
            "rows_left_out": max(int(e.extra.get("rows_left_out") or 0) for e in extracts),
            # the rows and the events of all the groups drawn, the smallest over the datasets (a sum of each group's
            # smallest would be fewer than any dataset holds)
            "total": {"rows": min(n for n, _ in totals), "events": min(d for _, d in totals)},
        }


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
