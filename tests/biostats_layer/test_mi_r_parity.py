"""
Parity of the pooling layer with R mice / mitml.

The reference values come from tests/reference/r/make_mi_fixtures.R, run by the manual workflow
.github/workflows/r-fixtures.yml; until they are in tests/reference/fixtures/mi these tests are skipped.
The models are fitted here on the datasets that mice completed, so the comparison covers the model fit
and the pooling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from biostats.linear import linear_regression, welch_means
from biostats.mi import MIData, analyze_mi, d2_chisq, pool_r_squared, pool_scalar, pool_terms
from biostats.regression import count_regression, logistic_regression
from biostats.survival import cox_regression

FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / "mi"
REFERENCE = FIXTURES / "mice_reference.json"
COMPLETED = FIXTURES / "mice_completed_long.csv"

pytestmark = pytest.mark.skipif(
    not (REFERENCE.exists() and COMPLETED.exists()),
    reason="R reference values not generated yet (run the 'R reference fixtures' workflow)",
)

GROUP_TERMS = ["C(group)[T.b]", "C(group)[T.c]"]
R_TERMS = {"(Intercept)": "Intercept", "groupb": "C(group)[T.b]", "groupc": "C(group)[T.c]"}
# R stops glm when the relative change in deviance is below 1e-8, so its Poisson estimates and SEs are exact
# to about 1e-6 only (ours converge to 1e-12): 1e-5 there, as for Cox
TOLERANCE = {"lm": 1e-8, "logistic": 1e-6, "poisson": 1e-5, "cox": 1e-5}


def _num(value: float | str) -> float:
    return math.inf if value == "Inf" else float(value)


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def completed() -> list[pd.DataFrame]:
    long = pd.read_csv(COMPLETED)
    return [frame.drop(columns=[".imp", ".id"]) for _, frame in long.groupby(".imp", sort=True)]


def _fit(kind: str, data: pd.DataFrame, formula: str):
    if kind == "lm":
        return smf.ols(formula.format(y="y"), data).fit()
    if kind == "logistic":
        return smf.glm(formula.format(y="bin"), data, family=sm.families.Binomial()).fit()
    if kind == "poisson":
        return smf.glm(formula.format(y="count"), data, family=sm.families.Poisson()).fit()
    rhs = formula.split("~", 1)[1]
    return smf.phreg(f"time ~ {rhs}", data, status=data["status"].to_numpy(), ties="efron").fit()


def test_pool_scalar_matches_mice(reference: dict) -> None:
    for case in reference["scalar"]:
        pooled = pool_scalar(case["Q"], case["U"], dfcom=_num(case["dfcom"]))
        for ours, theirs in [
            (pooled.estimate, case["qbar"]),
            (pooled.ubar, case["ubar"]),
            (pooled.b, case["b"]),
            (pooled.t, case["t"]),
            (pooled.df, case["df"]),
            (pooled.riv, case["riv"]),
            (pooled.fmi, case["fmi"]),
            (pooled.statistic, case["statistic"]),
            (pooled.p_value, case["p_value"]),
            (pooled.ci.lower, case["ci_low"]),
            (pooled.ci.upper, case["ci_high"]),
        ]:
            assert ours == pytest.approx(_num(theirs), rel=1e-10, abs=1e-12), case["name"]


@pytest.mark.parametrize("kind", ["lm", "logistic", "poisson", "cox"])
def test_pooled_models_match_mice(
    reference: dict, completed: list[pd.DataFrame], kind: str
) -> None:
    model = next(m for m in reference["models"] if m["model"] == kind)
    fits = [_fit(kind, data, "{y} ~ age + C(group) + x") for data in completed]
    params = [pd.Series(np.asarray(f.params), index=f.model.exog_names) for f in fits]
    covs = [
        pd.DataFrame(
            np.asarray(f.cov_params()), index=f.model.exog_names, columns=f.model.exog_names
        )
        for f in fits
    ]
    dfcom = _num(model["table"][0]["dfcom"])
    table = pool_terms(params, covariances=covs, dfcom=dfcom)
    tol = TOLERANCE[kind]
    for row in model["table"]:
        term = table.term(R_TERMS.get(row["term"], row["term"]))
        for ours, theirs in [
            (term.estimate, row["estimate"]),
            (term.se, row["std_error"]),
            (term.statistic, row["statistic"]),
            (term.df, row["df"]),
            (term.p_value, row["p_value"]),
            (term.ci.lower, row["ci_low"]),
            (term.ci.upper, row["ci_high"]),
        ]:
            assert ours == pytest.approx(theirs, rel=tol, abs=tol * 1e-2), (kind, row["term"])
        # riv, lambda and FMI depend on the variance between imputations, which magnifies the convergence
        # tolerance of R's iterative fits (glm epsilon 1e-8): ten times the tolerance of the estimates
        between_tol = tol if kind == "lm" else tol * 10
        for ours, theirs in [
            (term.riv, row["riv"]),
            (term.lam, row["lambda"]),
            (term.fmi, row["fmi"]),
        ]:
            assert ours == pytest.approx(theirs, rel=between_tol, abs=between_tol * 1e-2), (
                kind,
                row["term"],
            )

    d1 = table.wald(GROUP_TERMS)
    expected = model["d1"]
    stat = expected["F.value"]
    assert d1.statistic == pytest.approx(stat, rel=tol), kind
    assert d1.df2 == pytest.approx(expected["df2"], rel=tol), kind
    assert d1.p_value == pytest.approx(expected["P(>F)"], rel=max(tol, 1e-8), abs=1e-12), kind
    assert d1.riv == pytest.approx(expected["RIV"], rel=tol), kind

    wald = []
    for p, c in zip(params, covs, strict=True):
        q = p[GROUP_TERMS].to_numpy()
        wald.append(float(q @ np.linalg.solve(c.loc[GROUP_TERMS, GROUP_TERMS].to_numpy(), q)))
    d2 = d2_chisq(wald, df=len(GROUP_TERMS))
    expected = model["d2"]
    assert d2.statistic == pytest.approx(
        expected.get("statistic", expected.get("F.value")), rel=tol
    ), kind
    assert d2.p_value == pytest.approx(expected["P(>F)"], rel=max(tol, 1e-8), abs=1e-12), kind


def test_pooled_r_squared_matches_mice(reference: dict, completed: list[pd.DataFrame]) -> None:
    fits = [_fit("lm", data, "{y} ~ age + C(group) + x") for data in completed]
    n = int(fits[0].nobs)
    expected = reference["r_squared"]
    pooled = pool_r_squared([f.rsquared for f in fits], n=n)
    assert pooled.estimate == pytest.approx(expected["estimate"], rel=1e-10)
    assert pooled.fmi == pytest.approx(expected["fmi"], rel=1e-10)
    # mice uses the literal 1.96; the normal quantile differs in the fifth digit
    assert pooled.ci is not None
    assert pooled.ci.lower == pytest.approx(expected["low"], rel=1e-4)
    assert pooled.ci.upper == pytest.approx(expected["high"], rel=1e-4)
    adjusted = pool_r_squared([f.rsquared_adj for f in fits], n=n)
    assert adjusted.estimate == pytest.approx(expected["adjusted_estimate"], rel=1e-10)


DESIGN_TERMS = {"(Intercept)": "const", "groupb": "group_b", "groupc": "group_c"}


@pytest.mark.parametrize("kind", ["lm", "logistic", "poisson", "cox"])
def test_analyze_mi_on_the_mice_imputations_matches_mice(reference: dict, kind: str) -> None:
    """The whole chain: MIData from the mice long file, our models, the adapters and the pooling."""
    mi = MIData.from_long(pd.read_csv(COMPLETED))
    covariates = ["age", "group", "x"]
    levels = mi.levels(covariates)
    fit = {
        "lm": lambda d: linear_regression(d, "y", covariates, levels=levels),
        "logistic": lambda d: logistic_regression(
            d, "bin", covariates, event_level="1", levels=levels
        ),
        "poisson": lambda d: count_regression(d, "count", covariates, levels=levels),
        "cox": lambda d: cox_regression(d, "time", "status", covariates, levels=levels),
    }[kind]
    result = analyze_mi(fit, mi, complete_case=False)
    model = next(m for m in reference["models"] if m["model"] == kind)
    tol = TOLERANCE[kind]
    for row in model["table"]:
        term = result.term(DESIGN_TERMS.get(row["term"], row["term"]))
        assert term.estimate == pytest.approx(row["estimate"], rel=tol, abs=tol * 1e-2), (
            kind,
            row["term"],
        )
        assert term.se == pytest.approx(row["std_error"], rel=tol), (kind, row["term"])
        assert term.df == pytest.approx(row["df"], rel=tol), (kind, row["term"])
        between_tol = tol if kind == "lm" else tol * 10
        assert term.fmi == pytest.approx(row["fmi"], rel=between_tol, abs=between_tol * 1e-2), (
            kind,
            row["term"],
        )
    group = result.tests["group"]
    expected = model["d1"]
    assert group.statistic == pytest.approx(
        expected.get("statistic", expected.get("F.value")), rel=tol
    )


ROBUST_TERMS = {"Intercept": "const", "groupb": "group_b", "groupc": "group_c"}


def test_welch_type_group_means_match_sandwich_mice_and_mitml(reference: dict) -> None:
    """
    The groups compared without equal variances (biostats.linear.welch_means): HC2 variances (sandwich), Welch's
    ANOVA (oneway.test), each quantity pooled with its own complete-data df (mice::pool.scalar), and D1 (mitml).
    """
    robust = reference.get("robust")
    if robust is None or "means" not in robust:
        pytest.skip(
            "the Welch-type reference values come with the next run of the 'R reference fixtures' workflow"
        )
    mi = MIData.from_long(pd.read_csv(COMPLETED))
    levels = mi.levels(["group"])
    first = welch_means(mi.complete(1, ["y", "group"]), "y", "group", levels=levels)
    expected = robust["first"]
    terms = [ROBUST_TERMS[t] for t in expected["terms"]]
    np.testing.assert_allclose(
        first.vcov.loc[terms, terms].to_numpy(), np.asarray(expected["vcov"]), rtol=1e-10
    )
    for name, df in expected["df_terms"].items():
        assert first.term_df[ROBUST_TERMS[name]] == pytest.approx(df, rel=1e-10), name
    welch = first.type2.set_index("source").loc["group"]
    assert welch["F"] == pytest.approx(expected["welch_f"], rel=1e-10)
    assert welch["p"] == pytest.approx(expected["welch_p"], rel=1e-8, abs=1e-14)
    assert first.factor_df["group"] == pytest.approx(expected["df2"], rel=1e-10)

    result = analyze_mi(
        lambda d: welch_means(d, "y", "group", levels=levels),
        mi,
        columns=["y", "group"],
        complete_case=False,
    )
    for row in robust["terms"]:
        term = result.term(ROBUST_TERMS[row["term"]])
        for ours, theirs in [
            (term.estimate, row["estimate"]),
            (term.se, row["std_error"]),
            (term.df, row["df"]),
            (term.p_value, row["p_value"]),
            (term.riv, row["riv"]),
            (term.fmi, row["fmi"]),
            (term.dfcom, row["dfcom"]),
        ]:
            assert ours == pytest.approx(theirs, rel=1e-8, abs=1e-12), row["term"]
    means = result.extra["adjusted_means"].set_index("group")
    for row in robust["means"]:
        pooled = means.loc[row["group"]]
        for ours, theirs in [
            (pooled["mean"], row["mean"]),
            (pooled["se"], row["se"]),
            (pooled["df"], row["df"]),
            (pooled["fmi"], row["fmi"]),
        ]:
            assert ours == pytest.approx(theirs, rel=1e-8, abs=1e-12), row["group"]
    d1 = result.tests["group"]
    expected = robust["d1"]
    # mitml's D1, divided by Welch's second-order correction for these groups. mitml's testConstraints
    # differentiates the constraints numerically (numericDeriv), so its D1 carries errors of about 1e-9 even for
    # these linear constraints: 1e-7 here
    correction = 1 + 2 * (expected["df1"] + 1 - 2) / (3 * expected["dfcom"])
    statistic = expected["statistic"] / correction
    assert d1.statistic == pytest.approx(statistic, rel=1e-7)
    assert d1.df1 == expected["df1"] and d1.df2 == pytest.approx(expected["df2"], rel=1e-7)
    p_value = float(stats.f.sf(statistic, expected["df1"], expected["df2"]))
    assert d1.p_value == pytest.approx(p_value, rel=1e-7, abs=1e-12)
    assert d1.riv == pytest.approx(expected["riv"], rel=1e-7)
