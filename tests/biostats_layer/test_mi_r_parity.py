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

from biostats.mi import d2_chisq, pool_r_squared, pool_scalar, pool_terms

FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / "mi"
REFERENCE = FIXTURES / "mice_reference.json"
COMPLETED = FIXTURES / "mice_completed_long.csv"

pytestmark = pytest.mark.skipif(
    not (REFERENCE.exists() and COMPLETED.exists()),
    reason="R reference values not generated yet (run the 'R reference fixtures' workflow)",
)

GROUP_TERMS = ["C(group)[T.b]", "C(group)[T.c]"]
R_TERMS = {"(Intercept)": "Intercept", "groupb": "C(group)[T.b]", "groupc": "C(group)[T.c]"}
TOLERANCE = {"lm": 1e-8, "logistic": 1e-6, "poisson": 1e-6, "cox": 1e-5}


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
            assert ours == pytest.approx(theirs, rel=1e-10, abs=1e-12), case["name"]


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
            (term.riv, row["riv"]),
            (term.lam, row["lambda"]),
            (term.fmi, row["fmi"]),
        ]:
            assert ours == pytest.approx(theirs, rel=tol, abs=tol * 1e-2), (kind, row["term"])

    d1 = table.wald(GROUP_TERMS)
    expected = model["d1"]
    stat = expected.get("statistic", expected.get("F.value"))
    assert d1.statistic == pytest.approx(stat, rel=tol), kind
    assert d1.df2 == pytest.approx(expected["df2"], rel=tol), kind
    assert d1.p_value == pytest.approx(expected["p.value"], rel=max(tol, 1e-8)), kind

    wald = []
    for p, c in zip(params, covs, strict=True):
        q = p[GROUP_TERMS].to_numpy()
        wald.append(float(q @ np.linalg.solve(c.loc[GROUP_TERMS, GROUP_TERMS].to_numpy(), q)))
    d2 = d2_chisq(wald, df=len(GROUP_TERMS))
    expected = model["d2"]
    assert d2.statistic == pytest.approx(
        expected.get("statistic", expected.get("F.value")), rel=tol
    ), kind
    assert d2.p_value == pytest.approx(expected["p.value"], rel=max(tol, 1e-8)), kind


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
