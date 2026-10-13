"""Guidance of the hypothesis-tests page: plan before a run, ANCOVA checks, planning effect sizes, sentences."""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import utils.analysis_utils as au
from ui.guidance.analysis import (
    HELP,
    TAB_GUIDE,
    TEST_NOTES,
    ancova_check,
    cohens_d_from_means,
    cohens_f_from_means,
    cohens_w_from_proportions,
    p_superiority_from_d,
    parse_numbers,
    plan_group_comparison,
    power_sentence,
    recommend_correction,
    sample_size_sentence,
)

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)


def _frame() -> pd.DataFrame:
    n = 60
    df = pd.DataFrame(
        {
            "arm": ["A"] * 30 + ["B"] * 30,
            "x": np.r_[rng.normal(10, 2, 30), rng.normal(11, 2, 30)],
            "event": ["yes", "no"] * 30,
            "constant": [1.0] * n,
            "note": ["text"] * n,
            "lab": np.r_[rng.normal(5, 1, 30), rng.normal(5, 1, 30)],
        }
    )
    df.loc[:19, "lab"] = np.nan  # a third missing
    small = pd.DataFrame({"arm": ["C"] * 3, "x": [9.0, 10.0, 11.0], "event": ["yes", "no", "no"],
                          "constant": [1.0] * 3, "note": ["t"] * 3, "lab": [5.0, 5.0, 5.0]})
    return pd.concat([df, small], ignore_index=True)


NUM = ["x", "constant", "lab"]
CAT = ["event"]


def _plan(df, groups, targets, **kw):
    return plan_group_comparison(df, "arm", groups, targets, numeric_cols=NUM, categorical_cols=CAT, **kw)


def test_two_groups_plan_tests_and_advice() -> None:
    plan = _plan(_frame(), ["A", "B"], ["x", "event", "constant", "note", "lab"])
    tests = {p.variable: p for p in plan.tests}
    assert plan.group_sizes == {"A": 30, "B": 30}
    assert tests["x"].test == "Welch's t-test" and "Welch" in tests["x"].reason
    assert tests["event"].test == "Chi-square" and "risk ratio" in tests["event"].reason
    assert tests["constant"].kind == "skipped" and tests["note"].kind == "skipped"
    assert tests["lab"].missing_share == pytest.approx(20 / 60)
    texts = " ".join(a.text for a in plan.advice)
    assert "lab: 33% missing" in texts
    assert "3 tests without correction" in texts  # x, event, lab
    assert plan.correction.option == "Holm (recommended)" and plan.pairs == 0


def test_small_and_many_groups() -> None:
    plan = _plan(_frame(), ["A", "B", "C"], ["x"], correction="Holm (recommended)")
    assert plan.group_sizes["C"] == 3
    assert any(a.level == "caution" and "C: fewer than 5" in a.text for a in plan.advice)
    assert plan.tests[0].test == "Welch ANOVA" and plan.pairs == 3
    assert "3 pairs" in plan.notes[0]
    assert not any("without correction" in a.text for a in plan.advice)


def test_skewed_small_groups_get_a_rank_test_and_strategies_are_respected() -> None:
    df = pd.DataFrame({"arm": ["A"] * 8 + ["B"] * 8, "y": [1, 1, 1, 1, 1, 2, 2, 30] * 2})
    auto = plan_group_comparison(df, "arm", ["A", "B"], ["y"], numeric_cols=["y"], categorical_cols=[])
    assert auto.tests[0].test == "Mann-Whitney U"
    forced = plan_group_comparison(df, "arm", ["A", "B"], ["y"], numeric_cols=["y"], categorical_cols=[],
                                   strategy="Force Parametric (T-test/ANOVA)")
    assert forced.tests[0].test == "Welch's t-test"


def test_the_plan_predicts_what_the_engine_runs() -> None:
    df = _frame()
    sub = df[df["arm"].isin(["A", "B", "C"])]
    plan = _plan(df, ["A", "B", "C"], ["x"])
    engine = au.analyze_variable(sub, "arm", "x", NUM, CAT, [])
    assert engine["Test Used"] == plan.tests[0].test


def test_sparse_2x2_uses_fisher() -> None:
    df = pd.DataFrame({"arm": ["A"] * 10 + ["B"] * 10, "event": ["yes"] * 2 + ["no"] * 8 + ["yes"] + ["no"] * 9})
    plan = plan_group_comparison(df, "arm", ["A", "B"], ["event"], numeric_cols=[], categorical_cols=["event"])
    assert plan.tests[0].test == "Fisher's exact test"


def test_correction_recommendation() -> None:
    assert recommend_correction(1).option == "None"
    assert recommend_correction(4).option == "Holm (recommended)"
    assert recommend_correction(25).option == "Benjamini-Hochberg (FDR)"


def test_ancova_capacity_and_covariate_warnings() -> None:
    n = 60
    group = np.repeat(["T", "C"], n // 2)
    mediator = (group == "T") * 10 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "g": group, "age": rng.normal(50, 10, n), "med": mediator})
    ok = ancova_check(df, "y", "g", ["age"])
    assert ok.parameters == 3 and ok.per_parameter == pytest.approx(20) and ok.advice[0].level == "ok"
    flagged = ancova_check(df, "y", "g", ["age", "med"])
    assert any("med is largely explained by g" in a.text for a in flagged.advice)
    thin_df = df.copy()
    thin_df.loc[::2, "age"] = np.nan  # half the rows, both groups
    thin = ancova_check(thin_df, "y", "g", ["age", "med"])
    assert thin.n_complete == 30 and thin.n_excluded == 30
    assert thin.advice[0].level == "violated"  # 30 / 4 = 7.5 per parameter
    one_group = df.copy()
    one_group.loc[:29, "age"] = np.nan  # every T row
    assert "Only one group" in ancova_check(one_group, "y", "g", ["age"]).advice[0].text


def test_planning_effect_sizes() -> None:
    assert cohens_d_from_means(10, 12, 4) == pytest.approx(0.5)
    assert cohens_f_from_means([0, 0.5, 1.0], 1.0) == pytest.approx(np.sqrt(1 / 6))
    assert cohens_w_from_proportions([0.20, 0.35]) == pytest.approx(0.16797, abs=1e-4)
    assert p_superiority_from_d(0) == pytest.approx(0.5)
    assert p_superiority_from_d(0.5) == pytest.approx(0.6382, abs=1e-4)
    assert parse_numbers("120, 125; 132") == [120.0, 125.0, 132.0]
    with pytest.raises(ValueError):
        cohens_w_from_proportions([0.2, 1.0])
    with pytest.raises(ValueError):
        cohens_d_from_means(1, 2, 0)


def test_result_sentences() -> None:
    text = sample_size_sentence("a difference of d = 0.50", power=0.8, alpha=0.05, n_groups=[64, 64])
    assert "64 participants per group (128 in total)" in text and "80% power" in text and "dropouts" in text
    assert "20 in group 1 and 40 in group 2 (60 in total)" in sample_size_sentence(
        "d = 0.8", power=0.9, alpha=0.05, n_groups=[19.2, 39.1]
    )
    assert "88 participants in total" in sample_size_sentence("w = 0.3", power=0.8, alpha=0.05, n_total=87.2)
    assert "would say little" in power_sentence("d = 0.3", 0.32, sample="30 per group")
    assert "below the usual 80%" in power_sentence("d = 0.5", 0.7, sample="50 per group")
    assert "meets the usual" in power_sentence("d = 0.5", 0.85, sample="70 per group")


def test_orientation_matches_the_page_tabs_and_help_texts_are_complete() -> None:
    source = (ROOT / "app_pages" / "epidemiology.py").read_text(encoding="utf-8")
    tabs_call = re.search(r"tabs = st\.tabs\((\[.*?\])", source, re.S)
    assert tabs_call
    assert [g.tab for g in TAB_GUIDE] == ast.literal_eval(tabs_call.group(1))
    assert all(text.strip() for text in HELP.values()) and len(HELP) >= 45
    assert set(TEST_NOTES) >= {"Auto-Detect", "Student's t-test", "Welch's t-test", "Mann-Whitney U",
                               "ANOVA", "Kruskal-Wallis", "Chi-Square", "Fisher's Exact"}
