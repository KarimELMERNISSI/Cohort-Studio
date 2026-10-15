"""Tier C: survival, regression, rates, stratified tables, paired tests, reliability, multiple imputation (reference values)."""

import numpy as np
import pandas as pd
import pytest

from biostats.imputation import multiple_impute, pooled_statistics, rubin_pool
from biostats.paired import (
    mcnemar_test,
    one_sample_proportion,
    one_sample_t,
    paired_t,
    wilcoxon_signed_rank,
)
from biostats.rates import incidence_rate, mantel_haenszel, rate_ratio, standardised_ratio
from biostats.regression import count_regression, logistic_regression
from biostats.reliability import cohens_kappa, icc
from biostats.survival import compare_survival, cox_regression, kaplan_meier

# Freireich (1963) 6-MP trial (Gehan data): + marks censored observations
SIX_MP = [
    (6, 1),
    (6, 1),
    (6, 1),
    (6, 0),
    (7, 1),
    (9, 0),
    (10, 1),
    (10, 0),
    (11, 0),
    (13, 1),
    (16, 1),
    (17, 0),
    (19, 0),
    (20, 0),
    (22, 1),
    (23, 1),
    (25, 0),
    (32, 0),
    (32, 0),
    (34, 0),
    (35, 0),
]
PLACEBO = [(t, 1) for t in (1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23)]


def _gehan() -> pd.DataFrame:
    rows = [{"time": t, "event": e, "arm": "6-MP"} for t, e in SIX_MP] + [
        {"time": t, "event": e, "arm": "placebo"} for t, e in PLACEBO
    ]
    return pd.DataFrame(rows)


def test_kaplan_meier_medians_and_logrank_gehan() -> None:
    res = compare_survival(_gehan(), "time", "event", "arm")
    medians = {c.label: c.median for c in res.curves}
    assert medians["6-MP"] == 23 and medians["placebo"] == 8
    assert res.logrank_statistic == pytest.approx(16.79, abs=0.02)
    assert res.logrank_p < 1e-4 and res.logrank_df == 1
    curve = kaplan_meier([t for t, _ in PLACEBO], [e for _, e in PLACEBO], "placebo")
    assert curve.n == 21 and curve.events == 21 and curve.survival[-1] == pytest.approx(0.0)
    assert (curve.lower <= curve.survival + 1e-9).all() and (
        curve.upper >= curve.survival - 1e-9
    ).all()


def test_cox_hazard_ratio_gehan_efron() -> None:
    res = cox_regression(_gehan(), "time", "event", ["arm"])
    row = res.table.iloc[0]
    assert row["term"] == "arm_placebo"
    assert row["log_hr"] == pytest.approx(
        1.5721, abs=1e-3
    )  # placebo vs 6-MP: HR 4.82 (6-MP vs placebo 0.2076)
    assert row["se"] == pytest.approx(0.4124, abs=1e-3)
    assert 1 / row["hr"] == pytest.approx(0.2076, abs=1e-3)
    assert row["ci_low"] < row["hr"] < row["ci_high"] and row["p"] < 0.001
    assert res.n == 42 and res.events == 30 and res.ties == "efron"


def test_a_negative_follow_up_time_is_left_out_with_groups_and_covariates() -> None:
    clean = _gehan()
    dirty = pd.concat(
        [clean, pd.DataFrame([{"time": -3, "event": 0, "arm": "placebo"}])], ignore_index=True
    )
    km, km_clean = (
        compare_survival(dirty, "time", "event", "arm"),
        compare_survival(clean, "time", "event", "arm"),
    )
    assert km.logrank_statistic == pytest.approx(km_clean.logrank_statistic)
    assert km.warnings == ["1 row(s) with a negative follow-up time were left out"]
    cox, cox_clean = (
        cox_regression(dirty, "time", "event", ["arm"]),
        cox_regression(clean, "time", "event", ["arm"]),
    )
    assert cox.n == 42 and cox.table["log_hr"].iloc[0] == pytest.approx(
        cox_clean.table["log_hr"].iloc[0]
    )
    assert "1 row(s) with a negative follow-up time were left out" in cox.warnings


def test_logistic_regression_spector() -> None:
    from statsmodels.datasets import spector

    data = spector.load_pandas().data
    res = logistic_regression(data, "GRADE", ["GPA", "TUCE", "PSI"], event_level="1.0")
    ors = dict(zip(res.table["term"], res.table["odds_ratio"], strict=True))
    assert ors["GPA"] == pytest.approx(16.88, abs=0.02)
    assert ors["TUCE"] == pytest.approx(1.0998, abs=1e-3)
    assert ors["PSI"] == pytest.approx(10.79, abs=0.02)
    assert res.auc is not None and 0.8 < res.auc < 1.0 and res.auc_ci is not None
    assert res.n == 32 and res.events == 11
    assert any("events per parameter" in w for w in res.warnings)


def test_count_regression_recovers_rate_ratio() -> None:
    rng = np.random.default_rng(3)
    n = 400
    exposed = rng.integers(0, 2, n)
    time = rng.uniform(0.5, 3.0, n)
    counts = rng.poisson(0.2 * time * np.exp(np.log(2.0) * exposed))
    df = pd.DataFrame({"events": counts, "exposed": exposed, "person_years": time})
    res = count_regression(df, "events", ["exposed"], exposure_col="person_years")
    assert res.kind == "poisson"
    irr = float(res.table.loc[res.table["term"] == "exposed", "rate_ratio"].iloc[0])
    assert 1.5 < irr < 2.7
    nb = count_regression(
        df, "events", ["exposed"], exposure_col="person_years", family="negative_binomial"
    )
    assert nb.kind == "negative_binomial" and len(nb.table) == 1


def test_incidence_rate_and_ratio_reference_values() -> None:
    rate = incidence_rate(15, 1200, per=1000)
    assert rate.rate * 1000 == pytest.approx(12.5)
    assert rate.ci.lower * 1000 == pytest.approx(
        7.00, abs=0.01
    ) and rate.ci.upper * 1000 == pytest.approx(20.62, abs=0.01)
    rr = rate_ratio(15, 1200, 4, 1000)
    assert rr.ratio == pytest.approx(3.125)
    assert rr.ci.lower < 3.125 < rr.ci.upper and rr.p_value < 0.05
    with pytest.raises(ValueError):
        incidence_rate(3, 0)


def test_mantel_haenszel_matches_statsmodels_and_flags_confounding() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for stratum, (p_exp, base) in {"young": (0.7, 0.1), "old": (0.3, 0.3)}.items():
        for _ in range(300):
            exposed = rng.random() < p_exp
            logit = np.log(base / (1 - base)) + (
                np.log(2.5) if exposed else 0.0
            )  # common odds ratio 2.5
            risk = 1 / (1 + np.exp(-logit))
            rows.append(
                {
                    "exposed": "yes" if exposed else "no",
                    "event": "yes" if rng.random() < risk else "no",
                    "age": stratum,
                }
            )
    df = pd.DataFrame(rows)
    res = mantel_haenszel(df, "exposed", "event", "age", exposure_level="yes", event_level="yes")
    assert res.n_strata == 2
    assert res.or_mh_ci.lower < res.or_mh < res.or_mh_ci.upper
    assert res.rr_mh_ci.lower < res.rr_mh < res.rr_mh_ci.upper
    assert 1.5 < res.or_mh < 4.0 and 1.2 < res.rr_mh < res.or_mh
    assert res.breslow_day_p > 0.01
    assert any(
        "confounder" in w for w in res.warnings
    )  # exposure is associated with age, age with the outcome
    assert len(res.per_stratum) == 2 and "odds_ratio" in res.per_stratum.columns


def test_standardised_ratio_reference_value() -> None:
    smr = standardised_ratio(25, 18.4)
    assert smr.ratio == pytest.approx(1.3587, abs=1e-3)
    assert smr.ci.lower == pytest.approx(0.8793, abs=1e-3) and smr.ci.upper == pytest.approx(
        2.0057, abs=1e-3
    )
    assert 0 < smr.p_value <= 1


def test_paired_tests() -> None:
    before = [10, 12, 9, 14, 11, 13, 10, 12, 15, 11]
    after = [12, 13, 11, 15, 12, 15, 11, 14, 16, 13]
    t = paired_t(after, before)
    assert t.estimate == pytest.approx(1.5) and t.p_value < 0.001
    assert t.estimate_ci is not None and t.estimate_ci.lower < 1.5 < t.estimate_ci.upper
    assert t.effect_sizes[0].name == "Cohen's d_z" and t.effect_sizes[0].ci is not None
    w = wilcoxon_signed_rank(after, before, n_boot=200)
    assert w.p_value < 0.01 and w.estimate > 0 and w.estimate_ci is not None
    one = one_sample_t(after, 12)
    assert one.estimate == pytest.approx(np.mean(after) - 12)
    mc = mcnemar_test([[59, 6], [16, 80]])
    assert mc.p_value == pytest.approx(0.0525, abs=1e-3)
    assert mc.estimate == pytest.approx((6 - 16) / 161)
    prop = one_sample_proportion(30, 100, 0.2)
    assert prop.p_value < 0.05 and prop.estimate_ci.lower < 0.3 < prop.estimate_ci.upper


def test_kappa_and_icc_reference_values() -> None:
    k = cohens_kappa([[20, 5], [10, 15]])
    assert k.kappa == pytest.approx(0.400, abs=1e-6)
    assert k.ci.lower == pytest.approx(0.1511, abs=1e-3) and k.ci.upper == pytest.approx(
        0.6489, abs=1e-3
    )
    assert k.interpretation.startswith("fair") and k.n == 50
    weighted = cohens_kappa([[10, 2, 0], [3, 12, 2], [0, 1, 9]], weights="quadratic")
    assert weighted.weighted and 0.7 < weighted.kappa < 1.0
    # Shrout & Fleiss (1979) worked example: 6 targets, 4 judges
    ratings = np.array(
        [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    )
    two_one = icc(ratings, model="ICC(2,1)")
    three_one = icc(ratings, model="ICC(3,1)")
    assert two_one.icc == pytest.approx(0.29, abs=0.01)
    assert three_one.icc == pytest.approx(0.71, abs=0.01)
    assert two_one.ci.lower < two_one.icc < two_one.ci.upper
    assert three_one.ci.lower < three_one.icc < three_one.ci.upper


def test_rubin_pooling_reference_values_and_multiple_imputation() -> None:
    pooled = rubin_pool([1.0, 1.2, 0.9, 1.1, 1.05], [0.04, 0.05, 0.045, 0.04, 0.05])
    assert pooled.estimate == pytest.approx(1.05)
    assert pooled.total_variance == pytest.approx(0.0600, abs=1e-6)
    assert pooled.df == pytest.approx(64.0, abs=1e-6)
    assert pooled.fmi == pytest.approx(0.272, abs=1e-3)
    assert pooled.ci.lower < 1.05 < pooled.ci.upper
    with pytest.raises(ValueError):
        rubin_pool([1.0], [0.1])

    rng = np.random.default_rng(8)
    x = rng.normal(50, 10, 300)
    y = 0.5 * x + rng.normal(0, 5, 300)
    df = pd.DataFrame({"x": x, "y": y, "label": ["a", "b"] * 150})
    df.loc[rng.random(300) < 0.2, "y"] = np.nan
    completed = multiple_impute(df, m=4, exclude=["label"])
    assert len(completed) == 4 and all(not c["y"].isna().any() for c in completed)
    assert (completed[0]["label"] == df["label"]).all()
    table = pooled_statistics(completed, ["y"])
    assert table.loc[0, "m"] == 4 and table.loc[0, "between_variance"] > 0
    assert abs(table.loc[0, "estimate"] - y.mean()) < 2.0
