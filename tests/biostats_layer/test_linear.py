"""Linear regression and ANCOVA (biostats.linear) against statsmodels' formula interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from biostats.design import factor_levels
from biostats.linear import ancova, linear_regression


def _data(n: int = 180, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(55, 10, n),
            "bmi": rng.normal(27, 4, n),
            "arm": rng.choice(["control", "drug_a", "drug_b"], n),
            "sex": rng.choice(["F", "M"], n),
        }
    )
    df["sbp"] = (
        110
        + 0.4 * df["age"]
        + 0.8 * df["bmi"]
        - 6 * (df["arm"] == "drug_a")
        - 9 * (df["arm"] == "drug_b")
    ) + rng.normal(0, 8, n)
    df.loc[rng.random(n) < 0.05, "bmi"] = np.nan
    return df


def test_linear_regression_matches_statsmodels_ols() -> None:
    df = _data()
    res = linear_regression(df, "sbp", ["age", "bmi", "arm"])
    ref = smf.ols("sbp ~ age + bmi + C(arm)", df.dropna()).fit()
    mapping = {
        "const": "Intercept",
        "arm_drug_a": "C(arm)[T.drug_a]",
        "arm_drug_b": "C(arm)[T.drug_b]",
    }
    for _, row in res.table.iterrows():
        name = mapping.get(row["term"], row["term"])
        assert row["coef"] == pytest.approx(ref.params[name], rel=1e-10)
        assert row["se"] == pytest.approx(ref.bse[name], rel=1e-10)
        assert row["p"] == pytest.approx(ref.pvalues[name], rel=1e-8)
        assert row["ci_low"] == pytest.approx(ref.conf_int().loc[name, 0], rel=1e-10)
    assert res.r2 == pytest.approx(ref.rsquared) and res.r2_adj == pytest.approx(ref.rsquared_adj)
    assert res.df_resid == ref.df_resid and res.n == int(ref.nobs)
    assert res.terms_of == {"age": ["age"], "bmi": ["bmi"], "arm": ["arm_drug_a", "arm_drug_b"]}
    order = [mapping.get(t, t) for t in res.params.index]  # patsy puts the factor terms first
    np.testing.assert_allclose(
        res.vcov.to_numpy(), ref.cov_params().loc[order, order].to_numpy(), rtol=1e-8
    )
    assert len(res.residuals) == len(res.fitted) == res.n


def test_type_ii_table_matches_anova_lm() -> None:
    df = _data()
    res = linear_regression(df, "sbp", ["age", "bmi", "arm", "sex"])
    ref = sm.stats.anova_lm(smf.ols("sbp ~ age + bmi + C(arm) + C(sex)", df.dropna()).fit(), typ=2)
    table = res.type2.set_index("source")
    for ours, theirs in [
        ("age", "age"),
        ("bmi", "bmi"),
        ("arm", "C(arm)"),
        ("sex", "C(sex)"),
        ("Residual", "Residual"),
    ]:
        assert table.loc[ours, "df"] == pytest.approx(ref.loc[theirs, "df"])
        assert table.loc[ours, "sum_sq"] == pytest.approx(ref.loc[theirs, "sum_sq"], rel=1e-8)
        if ours != "Residual":
            assert table.loc[ours, "F"] == pytest.approx(ref.loc[theirs, "F"], rel=1e-8)
            assert table.loc[ours, "p"] == pytest.approx(ref.loc[theirs, "PR(>F)"], rel=1e-6)
            ss, sse = ref.loc[theirs, "sum_sq"], ref.loc["Residual", "sum_sq"]
            assert table.loc[ours, "partial_eta2"] == pytest.approx(ss / (ss + sse), rel=1e-8)


def test_ancova_adjusted_means_are_predictions_at_the_covariate_means() -> None:
    df = _data()
    res = ancova(df, "sbp", "arm", ["age", "bmi"])
    assert res.group == "arm" and res.group_terms == ["arm_drug_a", "arm_drug_b"]
    used = df.dropna()
    ref = smf.ols("sbp ~ age + bmi + C(arm)", used).fit()
    grid = pd.DataFrame(
        {
            "arm": ["control", "drug_a", "drug_b"],
            "age": used["age"].mean(),
            "bmi": used["bmi"].mean(),
        }
    )
    pred = ref.get_prediction(grid).summary_frame()
    means = res.adjusted_means
    assert means is not None and list(means["group"]) == ["control", "drug_a", "drug_b"]
    np.testing.assert_allclose(means["mean"], pred["mean"], rtol=1e-10)
    np.testing.assert_allclose(means["se"], pred["mean_se"], rtol=1e-8)
    diff = means.set_index("group")["mean"]
    assert diff["drug_b"] - diff["control"] == pytest.approx(res.params["arm_drug_b"])
    assert res.type2.set_index("source").loc["arm", "p"] < 0.001  # the planted arm effect


def test_frozen_levels_and_errors() -> None:
    df = _data()
    levels = factor_levels(df, ["arm"])
    frozen = linear_regression(df, "sbp", ["age", "arm"], levels=levels)
    assert list(frozen.params.index) == ["const", "age", "arm_drug_a", "arm_drug_b"]
    with pytest.raises(ValueError, match="arm: drug_b"):
        linear_regression(df[df["arm"] != "drug_b"], "sbp", ["age", "arm"], levels=levels)
    with pytest.raises(ValueError, match="must be numeric"):
        linear_regression(df.assign(sbp="high"), "sbp", ["age"])
    with pytest.raises(ValueError, match="cannot be estimated"):
        linear_regression(df.head(3), "sbp", ["age", "bmi", "arm"])
    with pytest.raises(ValueError, match="two levels"):
        ancova(df.assign(arm="control"), "sbp", "arm", ["age"])
    small = linear_regression(df.head(30), "sbp", ["age", "bmi", "arm", "sex"])
    assert any("per parameter" in w for w in small.warnings)
