"""MIData and analyze_mi: one analysis per imputed dataset, pooled through the adapters (M2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from biostats import TestResult
from biostats.contingency import TwoByTwo, analyze_twobytwo, chi_square_independence
from biostats.inference import compare_groups
from biostats.linear import ancova, linear_regression, welch_means
from biostats.mi import (
    MIAnalysisError,
    MIData,
    NotPoolableError,
    analyze_mi,
    not_poolable,
    pool_terms,
)
from biostats.mi.pooling import d2_chisq
from biostats.regression import logistic_regression
from biostats.survival import compare_survival, cox_regression


def _truth(n: int = 300, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(60, 9, n),
            "arm": rng.choice(["control", "drug"], n),
            "site": rng.choice(["Ettelbruck", "Luxembourg", "Metz"], n),
            "smoker": rng.choice(["no", "yes"], n),
        }
    )
    df["sbp"] = 100 + 0.5 * df["age"] - 8 * (df["arm"] == "drug") + rng.normal(0, 8, n)
    lin = -4 + 0.05 * df["age"] + 0.8 * (df["smoker"] == "yes")
    df["event"] = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    df["time"] = rng.exponential(np.exp(3.5 - 0.02 * df["age"] + 0.5 * (df["arm"] == "drug")))
    return df


def _mi(m: int = 5, seed: int = 2) -> tuple[MIData, pd.DataFrame]:
    truth = _truth()
    rng = np.random.default_rng(seed)
    base = truth.copy()
    gaps = {
        "sbp": rng.random(len(base)) < 0.15,
        "age": rng.random(len(base)) < 0.10,
        "smoker": rng.random(len(base)) < 0.08,
    }
    for col, mask in gaps.items():
        base[col] = base[col].astype(object) if col == "smoker" else base[col]
        base.loc[mask, col] = np.nan
    completed = []
    for _ in range(m):
        c = base.copy()
        c.loc[gaps["sbp"], "sbp"] = truth.loc[gaps["sbp"], "sbp"] + rng.normal(
            0, 6, gaps["sbp"].sum()
        )
        c.loc[gaps["age"], "age"] = truth.loc[gaps["age"], "age"] + rng.normal(
            0, 4, gaps["age"].sum()
        )
        c.loc[gaps["smoker"], "smoker"] = rng.choice(["no", "yes"], gaps["smoker"].sum())
        completed.append(c)
    return MIData.from_completed(
        base, completed, methods={"sbp": "pmm", "age": "pmm", "smoker": "logreg"}
    ), truth


# --- MIData -----------------------------------------------------------------------------------------


def test_midata_stores_only_the_imputed_cells_and_rebuilds_each_dataset() -> None:
    mi, _ = _mi()
    assert mi.m == 5 and mi.imputed_columns == ["age", "smoker", "sbp"]
    assert mi.n_imputed() == {c: int(mi.base[c].isna().sum()) for c in mi.imputed_columns}
    observed = mi.complete(0)
    assert observed["sbp"].isna().any()
    second = mi.complete(2, ["sbp", "smoker"])
    assert list(second.columns) == ["sbp", "smoker"] and not second.isna().any().any()
    rows = mi.imputations["sbp"].index
    assert (second.loc[rows, "sbp"].to_numpy() == mi.imputations["sbp"][2].to_numpy()).all()
    assert mi.levels(["site", "smoker", "age"]) == {
        "site": ["Ettelbruck", "Luxembourg", "Metz"],
        "smoker": ["no", "yes"],
    }
    with pytest.raises(ValueError, match="does not exist"):
        mi.complete(6)


def test_long_format_round_trips_with_or_without_the_observed_rows() -> None:
    mi, _ = _mi(m=3)
    long = mi.long()
    assert len(long) == 4 * len(mi.base) and list(long.columns[:2]) == [".imp", ".id"]
    again = MIData.from_long(long)
    for i in range(4):
        pd.testing.assert_frame_equal(
            again.complete(i), mi.complete(i), check_names=False, check_dtype=False
        )
    without_observed = MIData.from_long(long[long[".imp"] > 0])
    for i in range(1, 4):
        pd.testing.assert_frame_equal(
            without_observed.complete(i), mi.complete(i), check_names=False, check_dtype=False
        )
    assert without_observed.n_imputed()["sbp"] == mi.n_imputed()["sbp"]
    with pytest.raises(ValueError, match="no imputed dataset"):
        MIData.from_long(long[long[".imp"] == 0])


def test_midata_validation_and_categorical_columns() -> None:
    base = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0], "g": pd.Categorical(["a", None, "b"], categories=["a", "b"])}
    )
    imputations = {
        "x": pd.DataFrame({1: [2.0], 2: [2.5]}, index=[1]),
        "g": pd.DataFrame({1: ["a"], 2: ["b"]}, index=[1]),
    }
    mi = MIData(base=base, imputations=imputations, m=2)
    second = mi.complete(2)
    assert isinstance(second["g"].dtype, pd.CategoricalDtype) and second.loc[1, "g"] == "b"
    with pytest.raises(ValueError, match="columns 1..2"):
        MIData(base=base, imputations={"x": pd.DataFrame({1: [2.0]}, index=[1])}, m=2)
    with pytest.raises(ValueError, match="observed"):
        MIData(base=base, imputations={"x": pd.DataFrame({1: [2.0], 2: [2.0]}, index=[0])}, m=2)
    with pytest.raises(ValueError, match="not in the data"):
        MIData(base=base, imputations={"y": pd.DataFrame({1: [2.0], 2: [2.0]}, index=[1])}, m=2)
    with pytest.raises(ValueError, match="at least 1"):
        MIData(base=base, imputations={}, m=0)
    with pytest.raises(ValueError, match="unique"):
        MIData(base=base.set_axis([0, 0, 1]), imputations={}, m=2)


# --- analyze_mi with each adapter -------------------------------------------------------------------


def test_linear_regression_is_pooled_like_pool_terms_with_type_ii_tests_and_r2() -> None:
    mi, _ = _mi()
    levels = mi.levels(["arm", "site"])
    result = analyze_mi(
        lambda d: linear_regression(d, "sbp", ["age", "arm", "site"], levels=levels), mi
    )
    fits = [
        linear_regression(d, "sbp", ["age", "arm", "site"], levels=levels)
        for _, d in mi.completed()
    ]
    expected = pool_terms(
        [f.params for f in fits], covariances=[f.vcov for f in fits], dfcom=fits[0].df_resid
    )
    assert result.kind == "linear" and result.m == 5 and not result.partial
    assert result.term("arm_drug").estimate == pytest.approx(expected.term("arm_drug").estimate)
    assert result.term("arm_drug").df == pytest.approx(expected.term("arm_drug").df)
    assert set(result.tests) == {"age", "arm", "site"} and result.tests["arm"].p_value < 0.001
    assert 0 < result.extra["r2"].estimate < 1 and result.extra["r2"].ci is not None
    assert result.complete_case is not None and result.complete_case.n < fits[0].n
    assert {"term", "estimate", "fmi"} <= set(result.to_frame())


def test_ancova_pools_the_adjusted_means() -> None:
    mi, _ = _mi()
    result = analyze_mi(lambda d: ancova(d, "sbp", "arm", ["age"], levels=mi.levels(["arm"])), mi)
    means = result.extra["adjusted_means"].set_index("group")
    assert list(means.index) == ["control", "drug"] and result.extra["group"] == "arm"
    assert means.loc["drug", "mean"] - means.loc["control", "mean"] == pytest.approx(
        result.term("arm_drug").estimate, abs=1e-9
    )
    assert (means["fmi"] >= 0).all()


def test_welch_type_group_means_are_pooled_with_their_own_df() -> None:
    # three groups or more under MI: each group keeps its own variance, and each pooled quantity its own
    # complete-data df (the smallest over the imputations): n - 1 for a group mean, Satterthwaite's for a
    # difference, Welch's for the D1 test. The residual df made the test reject too often.
    mi, _ = _mi()
    levels = {"site": ["Ettelbruck", "Luxembourg", "Metz"]}
    result = analyze_mi(lambda d: welch_means(d, "sbp", "site", levels=levels), mi)
    fits = [welch_means(d, "sbp", "site", levels=levels) for _, d in mi.completed()]
    term_dfcom = {t: min(f.term_df[t] for f in fits) for t in fits[0].term_df}
    expected = pool_terms(
        [f.params for f in fits], covariances=[f.vcov for f in fits], dfcom=term_dfcom
    )
    for term in ("const", "site_Luxembourg", "site_Metz"):
        pooled = result.term(term)
        assert pooled.dfcom == pytest.approx(term_dfcom[term]) and pooled.dfcom < fits[0].df_resid
        # the mean of the HC2 variances
        assert pooled.ubar == pytest.approx(expected.term(term).ubar)
        assert pooled.df == pytest.approx(expected.term(term).df)
    welch_df2 = min(f.factor_df["site"] for f in fits)
    test = result.tests["site"]
    reference = expected.wald(["site_Luxembourg", "site_Metz"], dfcom=welch_df2)
    correction = 1 + 2 * (3 - 2) / (3 * welch_df2)  # Welch's second-order correction, three groups
    assert test.method == "D1" and test.statistic == pytest.approx(reference.statistic / correction)
    assert test.df2 == pytest.approx(reference.df2)
    assert test.p_value == pytest.approx(stats.f.sf(test.statistic, 2, test.df2))
    means = result.extra["adjusted_means"].set_index("group")
    sizes = mi.base["site"].value_counts()
    for level in levels["site"]:  # each mean pooled with the df of its own group
        assert means.loc[level, "df"] <= sizes[level] - 1
    assert result.extra["cov_type"] == "HC2"
    assert any("Welch BL (1951)" in r for r in result.extra["references"])


def test_with_identical_imputations_the_pooled_statistic_is_welchs_f() -> None:
    # no variance between the imputed datasets (a fully observed outcome): the corrected D1 statistic is
    # Welch's F of the data, with the complete-data df of mitml's rule
    from statsmodels.stats.oneway import anova_oneway

    rng = np.random.default_rng(8)
    sizes, sds = (6, 9, 12, 20, 30), (3.0, 2.5, 2.0, 1.5, 1.0)
    names = list("abcde")
    base = pd.DataFrame(
        {
            "y": np.concatenate([rng.normal(0, s, n) for n, s in zip(sizes, sds, strict=True)]),
            "g": np.repeat(names, sizes),
            "x": rng.normal(0, 1, sum(sizes)),
        }
    )
    base.loc[[0, 7], "x"] = np.nan  # imputed identically, so nothing varies between the imputations
    completed = [base.fillna({"x": 0.0}) for _ in range(4)]
    mi = MIData.from_completed(base, completed, methods={"x": "pmm"})
    result = analyze_mi(lambda d: welch_means(d, "y", "g", levels={"g": names}), mi)
    welch = anova_oneway(
        [base.loc[base["g"] == g, "y"].to_numpy() for g in names], use_var="unequal"
    )
    test = result.tests["g"]
    assert test.statistic == pytest.approx(welch.statistic, rel=1e-10)
    df = welch.df_denom
    assert test.df2 == pytest.approx((df + 1) / (df + 3) * df, rel=1e-10)


def test_terms_with_different_complete_data_df_are_refused() -> None:
    from dataclasses import replace

    mi, _ = _mi()
    levels = {"site": ["Ettelbruck", "Luxembourg", "Metz"]}

    def uneven(data, i):
        fit = welch_means(data, "sbp", "site", levels=levels)
        return replace(fit, term_df={"const": 9.0}) if i == 2 else fit

    with pytest.raises(ValueError, match="different complete-data df"):
        analyze_mi(uneven, mi)


def test_logistic_regression_reports_odds_ratios_factor_tests_and_counted_warnings() -> None:
    mi, _ = _mi()
    levels = mi.levels(["smoker", "site"])
    result = analyze_mi(
        lambda d: logistic_regression(d, "event", ["age", "smoker", "site"], levels=levels), mi
    )
    smoker = result.term("smoker_yes")
    assert smoker.scale == "log" and smoker.natural_estimate == pytest.approx(
        math.exp(smoker.estimate)
    )
    assert (
        "site" in result.tests
        and result.tests["site"].method == "D1"
        and result.tests["site"].df1 == 2
    )
    assert all("imputed datasets)" in w for w in result.warnings)


def test_cox_regression_reports_hazard_ratios() -> None:
    mi, _ = _mi()
    levels = mi.levels(["arm", "site"])
    result = analyze_mi(
        lambda d, i: cox_regression(d, "time", "event", ["age", "arm", "site"], levels=levels),
        mi,
        columns=["time", "event", "age", "arm", "site"],
    )
    arm = result.term("arm_drug")
    assert (
        arm.natural_estimate is not None and arm.natural_estimate < 1
    )  # the planted protective effect
    assert result.tests["site"].df1 == 2


def test_t_test_is_pooled_into_a_test_result_and_rank_tests_are_refused() -> None:
    mi, _ = _mi()

    def welch(d: pd.DataFrame) -> TestResult:
        return compare_groups(
            {k: g["sbp"] for k, g in d.groupby("arm")}, kind="welch", reference="control"
        )

    result = analyze_mi(welch, mi)
    pooled = result.extra["result"]
    assert isinstance(pooled, TestResult) and pooled.mi is not None and pooled.mi.m == 5
    assert (
        "pooled over 5 imputations" in pooled.name
        and pooled.estimate is not None
        and pooled.estimate < 0
    )
    assert (
        pooled.estimate_ci is not None
        and pooled.estimate_ci.lower < pooled.estimate < pooled.estimate_ci.upper
    )

    def mann_whitney(d: pd.DataFrame) -> TestResult:
        return compare_groups(
            {k: g["sbp"] for k, g in d.groupby("arm")}, kind="mannwhitney", reference="control"
        )

    with pytest.raises(NotPoolableError) as refused:
        analyze_mi(mann_whitney, mi)
    assert refused.value.entry.analysis == "mann_whitney" and refused.value.entry.alternative


def test_two_by_two_pools_ratios_and_the_chi_square() -> None:
    mi, _ = _mi()

    def table(d: pd.DataFrame):
        a = int(((d["smoker"] == "yes") & (d["event"] == 1)).sum())
        b = int(((d["smoker"] == "yes") & (d["event"] == 0)).sum())
        c = int(((d["smoker"] == "no") & (d["event"] == 1)).sum())
        dd = int(((d["smoker"] == "no") & (d["event"] == 0)).sum())
        return analyze_twobytwo(TwoByTwo(a=a, b=b, c=c, d=dd))

    result = analyze_mi(table, mi, complete_case=False)
    assert result.term("odds_ratio").natural_estimate is not None
    assert (
        result.term("relative_risk").scale == "log"
        and result.term("risk_difference").scale == "identity"
    )
    assert result.tests["chi_square"].method == "D2" and result.complete_case is None


def zero_cell_mi() -> MIData:
    """Arm B has no observed event and two missing outcomes: a zero cell in 2 of the 5 imputed datasets."""
    base = pd.DataFrame(
        {"arm": ["A"] * 20 + ["B"] * 20, "y": [1.0] * 6 + [0.0] * 14 + [0.0] * 18 + [np.nan] * 2}
    )
    completed = []
    for fill in ([0, 0], [1, 0], [0, 1], [0, 0], [1, 1]):
        c = base.copy()
        c.loc[c["y"].isna(), "y"] = fill
        completed.append(c)
    return MIData.from_completed(base, completed, methods={"y": "logreg"})


def arm_table(d: pd.DataFrame, haldane: str = "if_zero"):
    exposed, event = d["arm"] == "B", d["y"] == 1
    counts = TwoByTwo(
        a=int((exposed & event).sum()),
        b=int((exposed & ~event).sum()),
        c=int((~exposed & event).sum()),
        d=int((~exposed & ~event).sum()),
    )
    return analyze_twobytwo(counts, haldane="always" if haldane == "always" else "if_zero")


def test_a_zero_cell_in_some_imputed_datasets_leaves_the_ratios_unpooled() -> None:
    # the +0.5 correction applied in 2 datasets only: pooling the ratios would mix two estimators (M5)
    result = analyze_mi(arm_table, zero_cell_mi(), complete_case=False)
    assert result.table is not None and result.table.names == ["risk_difference"]
    assert any(
        w.startswith(
            "Odds ratio and risk ratio not pooled: a cell is zero in 2 of 5 imputed datasets"
        )
        for w in result.warnings
    )
    assert not any("because at least one cell is zero" in w for w in result.warnings)
    assert not_poolable("twobytwo_zero_cell") is not None
    # the sensitivity analysis adds 0.5 to every cell of every dataset: one estimator, the ratios are pooled
    sensitivity = analyze_mi(lambda d: arm_table(d, "always"), zero_cell_mi(), complete_case=False)
    assert sensitivity.table is not None
    assert {"odds_ratio", "relative_risk", "risk_difference"} == set(sensitivity.table.names)
    assert any(
        "Sensitivity analysis" in w and "(in 5 of 5 imputed datasets)" in w
        for w in sensitivity.warnings
    )


def test_chi_square_independence_matches_scipy_and_names_absent_levels() -> None:
    df = pd.DataFrame({"g": ["a"] * 30 + ["b"] * 30 + ["c"] * 40, "y": ["x", "y", "z", "x"] * 25})
    res = chi_square_independence(df, "g", "y")
    statistic, p, dof, _ = stats.chi2_contingency(pd.crosstab(df["g"], df["y"]), correction=False)
    assert res.statistic == pytest.approx(statistic) and res.p_value == pytest.approx(p)
    assert res.df == [dof] and res.groups == ["a", "b", "c"] and res.n_per_group == [30, 30, 40]
    with pytest.raises(ValueError, match="absent from these rows"):
        chi_square_independence(df[df["g"] != "c"], "g", "y", levels={"g": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="outside the levels"):
        chi_square_independence(df, "g", "y", levels={"g": ["a", "b"]})


def test_a_chi_square_of_independence_is_combined_with_d2() -> None:
    mi, _ = _mi()
    levels = {"site": ["Ettelbruck", "Luxembourg", "Metz"], "smoker": ["no", "yes"]}
    result = analyze_mi(lambda d: chi_square_independence(d, "site", "smoker", levels=levels), mi)
    assert result.kind == "chisq" and result.table is None
    statistics = [
        chi_square_independence(d, "site", "smoker", levels=levels).statistic
        for _, d in mi.completed(["site", "smoker"])
    ]
    assert result.tests["chi_square"] == d2_chisq([float(s) for s in statistics], df=2)
    assert result.complete_case is not None and result.complete_case.n_total < len(mi.base)
    fisher = not_poolable("fisher_exact")
    assert fisher is not None and "odds ratio" in fisher.alternative


def test_welch_without_effect_sizes_gives_the_same_test() -> None:
    rng = np.random.default_rng(3)
    groups = {"a": rng.normal(0, 1, 80), "b": rng.normal(0.4, 1.2, 90)}
    full = compare_groups(groups, kind="welch", reference="a")
    light = compare_groups(groups, kind="welch", reference="a", effect_sizes=False)
    assert light.effect_sizes == [] and len(full.effect_sizes) == 2
    assert (light.estimate, light.se, light.p_value, light.df) == (
        full.estimate,
        full.se,
        full.p_value,
        full.df,
    )


def test_survival_curves_are_pooled_with_medians_and_log_rank() -> None:
    mi, _ = _mi()
    result = analyze_mi(lambda d: compare_survival(d, "time", "event", "arm"), mi)
    curves = result.extra["curves"]
    assert set(curves) == {"control", "drug"}
    control = curves["control"]
    assert control["survival"].is_monotonic_decreasing and (control["survival"].between(0, 1)).all()
    inside = control.dropna()
    assert (inside["lower"] <= inside["survival"]).all() and (
        inside["survival"] <= inside["upper"]
    ).all()
    assert result.extra["medians"]["control"] is not None
    assert result.tests["log_rank"].method == "D2" and result.table is None


# --- orchestration ----------------------------------------------------------------------------------


def test_a_failure_names_the_imputation_and_unsupported_results_are_refused() -> None:
    mi, _ = _mi()

    def fails_on_three(d: pd.DataFrame, i: int):
        if i == 3:
            raise RuntimeError("did not converge")
        return linear_regression(d, "sbp", ["age"])

    with pytest.raises(MIAnalysisError, match="imputed dataset 3: did not converge") as err:
        analyze_mi(fails_on_three, mi)
    assert err.value.imputation == 3
    with pytest.raises(TypeError, match="no pooling rule"):
        analyze_mi(lambda d: {"mean": 1.0}, mi)


def test_time_budget_gives_a_labelled_preview_and_fn_can_receive_the_index() -> None:
    mi, _ = _mi(m=6)
    seen: list[int] = []

    def fit(d: pd.DataFrame, i: int):
        seen.append(i)
        return linear_regression(d, "sbp", ["age"])

    result = analyze_mi(fit, mi, time_budget=0.0, min_imputations=2)
    assert result.partial and result.m == 2 and result.m_total == 6
    assert any(n.startswith("Preview: pooled over 2 of 6") for n in result.notes)
    assert seen == [1, 2, 0]  # two imputations, then the complete-case analysis (index 0)


def test_differing_complete_data_df_use_the_smallest() -> None:
    mi, _ = _mi(m=3)

    def test_with_varying_df(d: pd.DataFrame, i: int) -> TestResult:
        return TestResult(name="t", estimate=float(d["sbp"].mean()), se=1.0, df=[50.0 + i])

    result = analyze_mi(test_with_varying_df, mi, complete_case=False)
    assert result.term("estimate").dfcom == 51.0
    assert any("smallest" in n for n in result.notes)
