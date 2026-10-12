"""Welch-first inference, post-hoc tables with every pair, and multiplicity adjustment (reference: R PlantGrowth)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from biostats.inference import compare_groups, groups_from_frame, recommend
from biostats.multiplicity import adjust, normalize_method
from biostats.posthoc import default_method, pairwise_comparisons

CTRL = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
TRT1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
TRT2 = [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]
PLANT = {"ctrl": CTRL, "trt1": TRT1, "trt2": TRT2}


def test_welch_anova_matches_r_oneway_test() -> None:
    res = compare_groups(PLANT, kind="welch_anova")
    assert res.name == "Welch ANOVA"
    assert res.statistic == pytest.approx(5.181, abs=1e-3)
    assert res.p_value == pytest.approx(0.01739, abs=1e-5)
    assert res.df[0] == 2 and res.df[1] == pytest.approx(17.128, abs=1e-3)
    assert [e.name for e in res.effect_sizes] == ["eta squared", "omega squared"]
    assert any(c.name.startswith("equal variances") for c in res.assumptions)
    assert any(c.name == "sample size" and c.status == "caution" for c in res.assumptions)


def test_classic_anova_and_kruskal_reference_values() -> None:
    anova = compare_groups(PLANT, kind="anova")
    assert anova.statistic == pytest.approx(4.846, abs=1e-3) and anova.p_value == pytest.approx(0.01591, abs=1e-5)
    kw = compare_groups(PLANT, kind="kruskal")
    assert kw.name == "Kruskal-Wallis" and kw.family == "nonparametric"
    assert kw.statistic == pytest.approx(7.9882, abs=1e-4) and kw.p_value == pytest.approx(0.01842, abs=1e-5)
    assert kw.effect_sizes[0].estimate == pytest.approx((7.9882 - 2) / 27, abs=1e-4)


def test_auto_policy_is_welch_first_and_never_gates_on_assumptions() -> None:
    assert compare_groups(PLANT).name == "Welch ANOVA"
    two = compare_groups({"ctrl": CTRL, "trt2": TRT2})
    assert two.name == "Welch's t-test"
    assert compare_groups({"ctrl": CTRL, "trt2": TRT2}, nonparametric=True).name == "Mann-Whitney U"
    assert recommend({"a": np.array(CTRL), "b": np.array(TRT2)})[0] == "welch"
    skewed = {"a": np.array([1, 1, 1, 1, 50.0]), "b": np.array([2, 2, 2, 2, 90.0])}
    assert recommend(skewed)[0] == "mannwhitney"


def test_two_group_estimate_is_comparison_minus_reference() -> None:
    res = compare_groups({"ctrl": CTRL, "trt2": TRT2}, kind="welch", reference="ctrl")
    assert res.reference_group == "ctrl" and res.comparison_group == "trt2"
    assert res.estimate == pytest.approx(np.mean(TRT2) - np.mean(CTRL))
    assert res.estimate_ci is not None and res.estimate_ci.lower < res.estimate < res.estimate_ci.upper
    scipy_p = stats.ttest_ind(TRT2, CTRL, equal_var=False).pvalue
    assert res.p_value == pytest.approx(scipy_p)
    flipped = compare_groups({"ctrl": CTRL, "trt2": TRT2}, kind="welch", reference="trt2")
    assert flipped.estimate == pytest.approx(-res.estimate)
    assert flipped.primary_effect.estimate == pytest.approx(-res.primary_effect.estimate)
    mw = compare_groups({"ctrl": CTRL, "trt2": TRT2}, kind="mannwhitney", reference="ctrl", n_boot=200)
    assert mw.estimate_label.startswith("Hodges-Lehmann") and mw.estimate_ci is not None
    assert mw.effect_sizes[0].name == "rank-biserial r"


def test_groups_from_frame_and_degenerate_input() -> None:
    df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10 + ["c"] * 10, "v": CTRL + TRT1 + TRT2})
    groups = groups_from_frame(df, "g", "v")
    assert list(groups) == ["a", "b", "c"] and len(groups["a"]) == 10
    empty = compare_groups({"a": [1.0], "b": [2.0, 3.0]})
    assert empty.p_value is None and empty.warnings


def test_tukey_reports_all_pairs_with_r_reference_values() -> None:
    table = pairwise_comparisons(PLANT, method="tukey")
    assert len(table) == 3
    assert table["p_adj"].round(3).tolist() == pytest.approx([0.391, 0.198, 0.012], abs=1e-3)
    assert table["estimate"].tolist() == pytest.approx([-0.371, 0.494, 0.865], abs=1e-3)
    assert (table["ci_low"] < table["estimate"]).all() and (table["estimate"] < table["ci_high"]).all()
    assert table.attrs["references"]


def test_games_howell_reduces_to_welch_t_for_two_groups_and_orders_plantgrowth() -> None:
    two = pairwise_comparisons({"ctrl": CTRL, "trt2": TRT2}, method="games-howell")
    welch_p = stats.ttest_ind(TRT2, CTRL, equal_var=False).pvalue
    assert two["p_adj"].iloc[0] == pytest.approx(welch_p, abs=1e-6)
    assert two["estimate"].iloc[0] == pytest.approx(np.mean(TRT2) - np.mean(CTRL))
    gh = pairwise_comparisons(PLANT, method="games-howell")
    assert len(gh) == 3 and gh["adjustment"].iloc[0].startswith("intrinsic")
    by_pair = dict(zip(zip(gh["group_1"], gh["group_2"], strict=True), gh["p_adj"], strict=True))
    assert by_pair[("trt1", "trt2")] < 0.05 < by_pair[("ctrl", "trt1")]
    assert 0.01 < by_pair[("trt1", "trt2")] < 0.03


def test_dunn_matches_hand_calculation_and_holm() -> None:
    table = pairwise_comparisons({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, method="dunn", adjust_method="holm")
    z_ab = (5 - 2) / np.sqrt(7.5 * (2 / 3))
    assert table.loc[0, "statistic"] == pytest.approx(z_ab)
    p_ab = 2 * stats.norm.sf(z_ab)
    assert table.loc[0, "p_raw"] == pytest.approx(p_ab)
    p_ac = 2 * stats.norm.sf((8 - 2) / np.sqrt(7.5 * (2 / 3)))
    assert table["p_adj"].tolist() == pytest.approx([max(p_ab * 2, p_ac * 3), p_ac * 3, max(p_ab * 2, p_ac * 3)], abs=1e-9)
    assert default_method("Kruskal-Wallis") == "dunn" and default_method("Welch ANOVA") == "games-howell"
    assert default_method("One-way ANOVA") == "tukey"


def test_pairwise_t_and_mannwhitney_adjust_holm_over_all_pairs() -> None:
    welch = pairwise_comparisons(PLANT, method="welch")
    assert (welch["p_adj"] >= welch["p_raw"]).all() and welch["adjustment"].iloc[0] == "Holm"
    mw = pairwise_comparisons(PLANT, method="mannwhitney", adjust_method="Benjamini-Hochberg (FDR)")
    assert mw["adjustment"].iloc[0] == "Benjamini-Hochberg"
    with pytest.raises(ValueError):
        pairwise_comparisons(PLANT, method="scheffe")


def test_adjust_is_nan_safe_and_label_tolerant() -> None:
    assert adjust([0.01, 0.04, 0.03], "holm").tolist() == pytest.approx([0.03, 0.06, 0.06])
    out = adjust([0.01, None, 0.03], "Holm (recommended)")
    assert np.isnan(out[1]) and out[0] == pytest.approx(0.02)
    assert adjust([0.2, 0.5], "None").tolist() == [0.2, 0.5]
    assert normalize_method("Benjamini-Hochberg (FDR)") == "fdr_bh"
    with pytest.raises(ValueError):
        normalize_method("sidak")
