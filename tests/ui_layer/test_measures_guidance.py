"""
The guidance of two tabs of Analyze > Hypothesis tests & epidemiology (ui/guidance/measures.py: what is offered, and
the plan of each analysis, checked on the data) and what the tabs share on the screen (app_pages/analysis_layout.py:
how a number, an interval and a p-value are written; a field that depends on the column chosen above it).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app_pages import analysis_layout as layout
from biostats._types import CI
from ui.guidance import measures as m

N = 240


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    ill = rng.random(N) < 0.3
    marker = np.where(ill, rng.normal(60, 10, N), rng.normal(30, 10, N)).round(1)
    falling = np.where(ill, rng.normal(30, 6, N), rng.normal(45, 6, N)).round(0)
    first = rng.normal(130, 15, N).round(0)
    second = (first - 5 + rng.normal(0, 6, N)).round(0)
    grade = rng.choice(["mild", "moderate", "severe"], N, p=[0.5, 0.3, 0.2])
    again = np.where(rng.random(N) < 0.8, grade, rng.choice(["mild", "moderate", "severe"], N))
    frame = pd.DataFrame(
        {
            "ill": np.where(ill, "Yes", "No"),
            "arm": rng.choice(["Control", "Intervention"], N),
            "marker": marker,
            "falling": falling,
            "sbp_first": first,
            "sbp_second": second,
            "height": rng.normal(170, 9, N).round(0),
            "hba1c": rng.normal(6.55, 0.8, N).round(1),
            "grade_local": grade,
            "grade_central": again,
            "smoker_first": rng.choice(["Yes", "No"], N, p=[0.3, 0.7]),
            "site": rng.choice(
                ["Metz", "Metz ", "Luxembourg", "Arlon"], N, p=[0.4, 0.02, 0.38, 0.2]
            ),
            "id": [f"P{i:04d}" for i in range(N)],
        }
    )
    frame["smoker_second"] = np.where(
        (frame["smoker_first"] == "Yes") & (rng.random(N) < 0.3), "No", frame["smoker_first"]
    )
    frame.loc[:29, "sbp_second"] = np.nan
    return frame


def _levels(plan) -> list[str]:
    return [item.level for item in plan.advice]


def _text(plan) -> str:
    return " ".join([*(step.text for step in plan.steps), *(item.text for item in plan.advice)])


# ----------------------------------------------------------------------------- what is offered


def test_columns_are_offered_by_what_an_analysis_can_read(df: pd.DataFrame) -> None:
    assert {"marker", "sbp_first", "height"} <= set(
        m.measurements(df)
    ) and "id" not in m.measurements(df)
    assert {"ill", "arm", "smoker_first"} <= set(m.two_valued(df))
    assert "site" not in m.two_valued(df) and "grade_local" not in m.two_valued(
        df
    )  # more than two values
    assert {"grade_local", "grade_central", "ill"} <= set(m.rated(df)) and "id" not in m.rated(df)
    assert {"site", "arm", "grade_local"} <= set(m.strata(df)) and "id" not in m.strata(df)


def test_a_second_column_is_offered_by_its_name_first(df: pd.DataFrame) -> None:
    """The first columns of the file were preselected (age against height); a second column named like the first is
    the likely one."""
    assert m.look_alikes("sbp_first", m.measurements(df))[0] == "sbp_second"
    assert "sbp_first" not in m.look_alikes("sbp_first", m.measurements(df))
    assert m.look_alikes("grade_local", m.rated(df))[0] == "grade_central"
    assert m.look_alikes(None, ["b", "a"]) == ["b", "a"]


def test_the_value_that_says_present_is_offered_only_when_a_word_says_it(df: pd.DataFrame) -> None:
    assert m.present_value(m.levels_of(df["ill"])) == "Yes"
    assert (
        m.present_value(m.levels_of(df["arm"])) is None
    )  # neither arm is "the condition": a choice to make
    assert m.present_value([0, 1]) == 1 and m.present_value([]) is None


# ----------------------------------------------------------------------------- accuracy of a test


def test_the_accuracy_plan_asks_which_way_the_test_points(df: pd.DataFrame) -> None:
    nothing = m.plan_accuracy(df, None, None, None, None, m.NO_CUTOFF, None, None)
    assert nothing.status == "idle" and not nothing.steps
    asked = m.plan_accuracy(df, "ill", "Yes", "marker", None, m.NO_CUTOFF, None, None)
    assert (
        asked.status == "idle" and not asked.steps and "which way the test points" in _text(asked)
    )
    ready = m.plan_accuracy(df, "ill", "Yes", "marker", m.HIGHER, m.NO_CUTOFF, None, None)
    assert ready.status == "ok" and len(ready.steps) == 2
    assert (
        not ready.advice
    )  # the data agree with the direction: the medians stand beside the question already
    assert ready.steps[0].used == N


def test_a_direction_the_data_contradict_is_said(df: pd.DataFrame) -> None:
    """A marker that falls with the condition, read as rising: the cut-off found called everyone positive."""
    wrong = m.plan_accuracy(df, "ill", "Yes", "falling", m.HIGHER, m.NO_CUTOFF, None, None)
    assert wrong.status == "caution" and "Check the direction" in _text(wrong)
    right = m.plan_accuracy(df, "ill", "Yes", "falling", m.LOWER, m.FIXED, 38.0, None)
    assert right.status == "ok" and "at most 38" in _text(right)


def test_a_cut_off_is_checked_against_the_test(df: pd.DataFrame) -> None:
    none = m.plan_accuracy(df, "ill", "Yes", "marker", m.HIGHER, m.FIXED, None, None)
    assert none.status == "idle" and not none.steps  # nothing to compute until it is typed
    beyond = m.plan_accuracy(df, "ill", "Yes", "marker", m.HIGHER, m.FIXED, 1000.0, None)
    assert beyond.status == "violated" and "no one would be called positive" in _text(beyond)
    found = m.plan_accuracy(df, "ill", "Yes", "marker", m.HIGHER, m.YOUDEN, None, None)
    assert found.status == "caution" and "exploratory" in _text(found)
    typed = m.plan_accuracy(df, "ill", "Yes", "marker", m.HIGHER, m.FIXED, 45.0, 0.02)
    assert "the prevalence typed (2.0%)" in _text(typed)


def test_a_reference_standard_holds_two_values(df: pd.DataFrame) -> None:
    plan = m.plan_accuracy(df, "site", None, "marker", None, m.NO_CUTOFF, None, None)
    assert plan.status == "violated" and not plan.steps
    assert "a reference standard holds two" in _text(plan)


# ----------------------------------------------------------------------------- agreement


def test_two_measurements_are_checked_to_be_of_one_quantity(df: pd.DataFrame) -> None:
    one = m.plan_measurements(df, ["sbp_first"], [m.BLAND_ALTMAN, m.ICC], m.ABSOLUTE)
    assert one.status == "idle" and not one.steps
    two = m.plan_measurements(df, ["sbp_first", "sbp_second"], [m.BLAND_ALTMAN, m.ICC], m.ABSOLUTE)
    assert two.status == "ok" and [step.used for step in two.steps] == [N - 30, N - 30]
    assert "30 lack `sbp_second`" in _text(two)
    other = m.plan_measurements(df, ["sbp_first", "height"], [m.BLAND_ALTMAN], m.ABSOLUTE)
    assert other.status == "caution" and "hardly related" in _text(other)
    three = m.plan_measurements(
        df, ["sbp_first", "sbp_second", "height"], [m.BLAND_ALTMAN, m.ICC], m.ABSOLUTE
    )
    assert len(three.steps) == 1 and "the ICC alone is computed" in _text(three)


def test_two_raters_need_the_same_categories_and_a_stated_order(df: pd.DataFrame) -> None:
    categories, *_ = m.rater_categories(df, "grade_local", "grade_central")
    assert categories == ["mild", "moderate", "severe"]
    names = m.plan_raters(df, "grade_local", "grade_central", m.NAMES, None, m.LINEAR)
    assert names.status == "ok" and "categories without an order" in _text(names)
    graded = m.plan_raters(df, "grade_local", "grade_central", m.GRADES, categories, m.LINEAR)
    assert graded.status == "ok" and "Order used: `mild` < `moderate` < `severe`" in _text(graded)
    short = m.plan_raters(df, "grade_local", "grade_central", m.GRADES, categories[:2], m.LINEAR)
    assert (
        short.status == "violated"
        and not short.steps
        and "the order lacks `severe`" in _text(short)
    )
    apart = m.plan_raters(df, "arm", "ill", m.NAMES, None, m.LINEAR)
    assert apart.status == "violated" and "share no category" in _text(apart)


# ----------------------------------------------------------------------------- paired, one group


def test_the_paired_plan_says_who_is_left_out_and_leaves_what_never_changes_to_the_reading(
    df: pd.DataFrame,
) -> None:
    plan = m.plan_paired(df, "sbp_first", "sbp_second", m.T_TEST)
    assert plan.status == "ok" and plan.steps[0].used == N - 30
    assert "30 lack `sbp_second`" in _text(plan)
    # what is true of every change within one group is no check of these data: it closed every plan, and is said
    # once where the result is read
    assert "not the effect of a treatment" not in _text(plan)
    assert "not the effect of a treatment" in m.WITHIN_ONE_GROUP
    assert m.plan_paired(df, "sbp_first", None, m.T_TEST).status == "idle"


def test_a_change_in_a_yes_or_no_characteristic_needs_the_same_two_values(df: pd.DataFrame) -> None:
    plan = m.plan_paired_yes_no(df, "smoker_first", "smoker_second", "Yes")
    assert plan.status == "ok" and "participants changed" in _text(plan)
    assert m.plan_paired_yes_no(df, "smoker_first", "smoker_second", None).status == "idle"
    other = m.plan_paired_yes_no(df, "arm", "ill", None)
    assert other.status == "violated" and "the same two values in both" in _text(other)


def test_a_reference_value_is_checked_against_the_measurement(df: pd.DataFrame) -> None:
    asked = m.plan_one_mean(df, "hba1c", None, m.ONE_T)
    assert (
        asked.status == "idle"
        and not asked.steps
        and "fixed before looking at these data" in _text(asked)
    )
    far = m.plan_one_mean(df, "hba1c", 70.0, m.ONE_T)
    assert far.status == "caution" and "the right variable, the right unit?" in _text(far)
    assert m.plan_one_mean(df, "hba1c", 7.0, m.ONE_T).status == "ok"


def test_a_reference_that_is_the_datas_own_summary_is_asked_about(df: pd.DataFrame) -> None:
    """The mean for the mean, the median for the typical value. Compared at the reference's own precision, 7 "was"
    the mean of values averaging 6.55; at the precision of the values, 50 "was" the mean of ages averaging 50.43,
    which a t-test tells apart from 50."""
    mean = float(df["hba1c"].mean())
    assert round(mean) == 7 and round(mean, 1) != 7.0
    assert "also the mean" not in _text(m.plan_one_mean(df, "hba1c", 7.0, m.ONE_T))
    own = m.plan_one_mean(df, "hba1c", round(mean, 2), m.ONE_T)
    assert (
        own.status == "caution"
        and "is also the mean of these values: was it read from them?" in _text(own)
    )
    assert "also the mean" in _text(m.plan_one_mean(df, "hba1c", round(mean, 3), m.ONE_T))
    # values recorded to 0.1: their mean rounded to 0.1 is not "the mean" (it is known one decimal finer)
    assert "also the mean" not in _text(m.plan_one_mean(df, "hba1c", round(mean, 1), m.ONE_T))
    ages = pd.DataFrame(
        {"age": np.r_[np.full(57, 50.0), np.full(43, 51.0)]}
    )  # mean 50.43, median 50
    assert "also the" not in _text(m.plan_one_mean(ages, "age", 50.0, m.ONE_T))
    typical = m.plan_one_mean(ages, "age", 50.0, m.SIGNED_RANK)
    assert "50 is also the median of these values" in _text(
        typical
    )  # a question: a reference may well be round


def test_a_share_is_counted_and_its_reference_checked(df: pd.DataFrame) -> None:
    with_it, counted = m.share_counts(df, "ill", "Yes")
    assert (with_it, counted) == (int((df["ill"] == "Yes").sum()), N)
    assert m.plan_share(df, "ill", "Yes", 0.3).status == "ok"
    assert m.plan_share(df, "ill", "Yes", 0.0).status == "violated"
    assert m.plan_share(df, "ill", None, 0.3).status == "idle"
    assert m.check_share(30, 100, 0.2) == (True, [])
    ready, advice = m.check_share(30, 20, 0.2)
    assert not ready and "cannot exceed" in advice[0].text
    assert m.check_share(None, 100, 0.2) == (
        False,
        [],
    )  # a number is missing: nothing is said, nothing computed


# ----------------------------------------------------------------------------- strata, rates


def test_the_strata_plan_names_what_cannot_be_combined(df: pd.DataFrame) -> None:
    plan = m.plan_strata(df, "ill", "Yes", "arm", "Control", "grade_local")
    assert plan.status == "ok" and "`Intervention` against `Control`" in _text(plan)
    assert m.plan_strata(df, None, None, None, None, None).status == "idle"
    spelled = m.plan_strata(df, "ill", "Yes", "arm", "Control", "site")
    assert spelled.status == "caution" and "written several ways" in _text(spelled)
    assert [item.column for item in spelled.advice if item.column] == [
        "site"
    ]  # cleaned from the plan itself


def test_the_calculators_check_the_numbers_typed() -> None:
    assert m.check_rates(m.TWO_RATES, 15, 1200.0, None, 1000.0) == (
        False,
        [],
    )  # a number is missing
    ready, advice = m.check_rates(m.TWO_RATES, 15, 1200.0, 4, 1000.0)
    # nothing to say of these numbers: "count first events" stood above every result, and is in its reading
    assert ready and not advice
    assert "first event" in m.RATE_READING and m.RATE_READING in m.RATIO_READING
    assert "taken as known" in m.EXPECTED_READING
    ready, advice = m.check_rates(m.TWO_RATES, 15, 1200.0, 0, 1000.0)
    assert (
        ready and "no upper bound" in advice[0].text
    )  # 0 events meant "one rate" and gave an infinite ratio
    assert m.check_rates(m.ONE_RATE, 15, 1200.0, None, None)[0]
    assert not m.check_rates(m.ONE_RATE, 15, 0.0, None, None)[0]  # no person-time, no rate
    assert m.check_expected(25, 18.4)[0] and not m.check_expected(3, 0.0)[0]
    assert "almost nothing" in m.check_expected(0, 0.5)[1][0].text


# ----------------------------------------------------------------------------- what the two reads of the tabs found


def test_data_that_show_no_direction_are_not_said_to_fit_one(df: pd.DataFrame) -> None:
    """The same data read "the direction fits" for higher and for lower: the plan read the interval of the AUC only
    to say that the data contradict the direction."""
    flat = df.assign(noise=np.random.default_rng(3).normal(50, 10, N).round(1))
    for direction in (m.HIGHER, m.LOWER):
        plan = m.plan_accuracy(flat, "ill", "Yes", "noise", direction, m.NO_CUTOFF, None, None)
        assert "the direction fits" not in _text(plan) and "Check the direction" not in _text(plan)
        assert "do not show which way the test points (AUC 0." in _text(plan)
        assert plan.status == "ok"  # said, never a block
    none = df.assign(empty=np.nan)
    refused = m.plan_accuracy(none, "ill", "Yes", "empty", m.HIGHER, m.NO_CUTOFF, None, None)
    assert (
        refused.status == "violated"
        and "No row holds both `ill` and a number in `empty`" in _text(refused)
    )


def test_the_order_of_grades_is_offered_only_when_the_grades_say_it() -> None:
    """ "high, low, medium, none" is the alphabet's order: offered under a plan that read "ready", it gave a weighted
    kappa of 0.63 for 0.76."""
    assert m.grade_order(["10", "2", "1"]) == ["1", "2", "10"]
    assert m.grade_order(["V", "IX", "I", "X", "iv"]) == [
        "I",
        "iv",
        "V",
        "IX",
        "X",
    ]  # by value: IX after V
    assert m.grade_order(["low", "medium", "high"]) == []
    assert m.grade_order(["I", "II", "severe"]) == [] and m.grade_order(["mild", "mix"]) == []
    assert m.grade_order([]) == []


def test_grades_written_in_words_are_placed_by_their_user(df: pd.DataFrame) -> None:
    categories, *_ = m.rater_categories(df, "grade_local", "grade_central")
    offered = m.grade_order(categories)
    assert offered == []  # mild, moderate, severe: words
    plan = m.plan_raters(df, "grade_local", "grade_central", m.GRADES, offered, m.LINEAR)
    assert plan.status == "violated" and not plan.steps  # no Run until every grade is placed
    assert "Grades written in words carry no order of their own" in _text(plan)
    placed = m.plan_raters(df, "grade_local", "grade_central", m.GRADES, categories, m.LINEAR)
    assert placed.status == "ok" and len(placed.steps) == 1


def test_one_reads_as_present_beside_zero_only() -> None:
    """With the codes 1 and 2 the fields opened on 1: 1 = censored and 2 = event is a coding too (R's survival)."""
    assert m.present_value([0, 1]) == 1 and m.present_value([False, True]) is True
    assert m.present_value([1, 2]) is None and m.present_value(["1", "2"]) is None
    assert m.present_value(["No", "Yes"]) == "Yes"


def test_values_that_differ_by_a_space_are_written_apart(df: pd.DataFrame) -> None:
    """Written as code alone, "`after` holds `No` and `Yes`" refused a column for not holding No and Yes."""
    spaced = df.assign(smoker_second=df["smoker_second"].replace({"Yes": "Yes "}))
    plan = m.plan_paired_yes_no(spaced, "smoker_first", "smoker_second", None)
    assert plan.status == "violated" and "`smoker_second` holds `No` and `Yes␣`" in _text(plan)
    assert [item.column for item in plan.advice if item.column] == [
        "smoker_second"
    ]  # cleaned from the plan
    raters = df.assign(grade_central=df["grade_central"].replace({"mild": "mild "}))
    said = m.plan_raters(raters, "grade_local", "grade_central", m.NAMES, None, m.LINEAR)
    assert "written alike but not identical (`mild`, `mild␣`)" in _text(said)
    assert "`mild␣` is given by `grade_central` only" in _text(said)
    assert [item.column for item in said.advice if item.column] == ["grade_central"]


def test_codes_that_read_as_one_number_are_kept_apart_and_said() -> None:
    """Two columns of texts: "250.0" and "250.00" are two codes, and the plan says so with the column to clean if
    they are one. A column of numbers beside a column of texts: the texts are read as numbers, and the plan names
    the writings that were read as one."""
    codes = pd.DataFrame(
        {"a": ["250.0"] * 12 + ["401.9"] * 18, "b": ["250.00"] * 6 + ["250.0"] * 6 + ["401.9"] * 18}
    )
    plan = m.plan_raters(codes, "a", "b", m.NAMES, None, m.LINEAR)
    text = _text(plan)
    assert plan.steps and plan.status == "caution"
    assert (
        "`250.0` and `250.00` read as the same number and count as categories of their own" in text
    )
    assert "3 categories" in text  # as written
    assert [item.column for item in plan.advice if getattr(item, "column", None)] == ["a"]
    grades = pd.DataFrame({"a": [2.0, 3.0, 2.0, 4.0] * 6, "b": ["2.0", "3", "2", "04"] * 6})
    plan = m.plan_raters(grades, "a", "b", m.NAMES, None, m.LINEAR)
    text = _text(plan)
    assert plan.steps and "3 categories" in text
    assert "`b` holds texts beside a column of numbers: its values are read as numbers" in text
    assert "`2.0` as `2`" in text and "`04` as `4`" in text
    # two columns of numbers, whole against decimal: one grade, and nothing to say
    whole = pd.DataFrame({"a": [2, 3, 2, 4] * 6, "b": [2.0, 3.0, 2.0, 4.0] * 6})
    assert "read as" not in _text(m.plan_raters(whole, "a", "b", m.NAMES, None, m.LINEAR))
    # a grade written "nan" gives no order to offer
    assert m.grade_order(["2", "nan", "1"]) == []


def test_kappa_at_its_edges_is_said_before_the_run() -> None:
    one = pd.DataFrame({"a": ["X"] * 30, "b": ["X"] * 30})
    nothing = m.plan_raters(one, "a", "b", m.NAMES, None, m.LINEAR)
    assert nothing.status == "violated" and "there is nothing kappa can compare" in _text(nothing)
    fixed = pd.DataFrame({"a": ["X"] * 40, "b": ["X"] * 30 + ["Y"] * 10})
    zero = m.plan_raters(fixed, "a", "b", m.NAMES, None, m.LINEAR)
    assert (
        zero.status == "caution"
        and "`a` puts every participant in `X`: kappa is 0 by construction" in _text(zero)
    )
    # 95 of 100 in X for one rater and in Y for the other: no agreement, and no word on "the same category"
    apart = pd.DataFrame({"a": ["X"] * 95 + ["Y"] * 5, "b": ["Y"] * 95 + ["X"] * 5})
    assert "the same category" not in _text(m.plan_raters(apart, "a", "b", m.NAMES, None, m.LINEAR))
    alike = pd.DataFrame({"a": ["X"] * 95 + ["Y"] * 5, "b": ["X"] * 93 + ["Y"] * 7})
    assert "90% or more of the participants in the same category" in _text(
        m.plan_raters(alike, "a", "b", m.NAMES, None, m.LINEAR)
    )


def test_agreement_of_columns_that_do_not_vary_is_refused() -> None:
    """Two constant columns: the plan read "ready", the headline "ICC nan" and the benchmark "excellent"."""
    constant = pd.DataFrame({"a": [5.0] * 20, "b": [6.0] * 20})
    plan = m.plan_measurements(constant, ["a", "b"], [m.BLAND_ALTMAN, m.ICC], m.ABSOLUTE)
    assert plan.status == "violated" and not plan.steps and "one value only" in _text(plan)


def test_the_roughness_of_the_limits_is_the_one_of_their_interval() -> None:
    """ "About 0.65 SD either way" was printed for any number of pairs below 30: it is 0.65 at 30, 1.24 at 10."""
    rng = np.random.default_rng(5)

    def said(pairs: int) -> str:
        first = rng.normal(100, 10, pairs).round(0)
        frame = pd.DataFrame({"a": first, "b": first + rng.normal(0, 3, pairs).round(0)})
        return _text(m.plan_measurements(frame, ["a", "b"], [m.BLAND_ALTMAN], m.ABSOLUTE))

    assert "10 pairs: each limit of agreement is known to about 1.24 SD" in said(10)
    assert "the limits are rough" in said(10) and "the limits are rough" not in said(30)
    assert "30 pairs: each limit of agreement is known to about 0.65 SD" in said(30)
    assert "limit of agreement is known" not in said(60)


def _imputed(df: pd.DataFrame, columns: list[str]):
    """Three imputed datasets of ``df`` in which ``columns`` are filled (with their mean, 1 to 3 added)."""
    from biostats.mi import MIData

    completed = [df.fillna({c: float(df[c].mean()) + i for c in columns}) for i in range(1, 4)]
    return MIData.from_completed(df, completed, methods=dict.fromkeys(columns, "pmm"))


def test_the_plan_of_a_pooled_test_counts_what_the_imputed_datasets_give(df: pd.DataFrame) -> None:
    """With imputed data the plan counted the complete pairs and said the others were left out, over a test that
    reads every row."""
    numbers = df[["sbp_first", "sbp_second", "hba1c"]]
    filled = m.pooled_rows(_imputed(numbers, ["sbp_second"]), ["sbp_first", "sbp_second"])
    assert (filled.rows, filled.used, filled.imputed["sbp_second"], filled.datasets) == (
        N,
        N,
        30,
        3,
    )
    plan = m.plan_paired(df, "sbp_first", "sbp_second", m.T_TEST, filled)
    assert plan.rows == N and plan.steps[0].used == N and plan.status == "ok"
    assert "pooled over the 3 imputed datasets" in plan.steps[0].text
    assert "30 values of `sbp_second` are imputed" in _text(plan) and "left out" not in _text(plan)
    # a column left out of the imputation: each dataset reads the pairs that have it, and the plan says why
    left = m.pooled_rows(_imputed(numbers, ["hba1c"]), ["sbp_first", "sbp_second"])
    assert (left.used, left.imputed) == (N - 30, {"sbp_first": 0, "sbp_second": 0})
    partial = m.plan_paired(df, "sbp_first", "sbp_second", m.T_TEST, left)
    assert partial.steps[0].used == N - 30
    assert (
        "`sbp_second` is not imputed: each imputed dataset reads the 210 rows that have it"
        in _text(partial)
    )
    one = m.plan_one_mean(
        df.assign(hba1c=numbers["hba1c"].mask(df.index < 20)), "hba1c", 7.0, m.ONE_T
    )
    assert one.steps[0].used == N - 20  # observed: the values recorded
    gaps = numbers.assign(hba1c=numbers["hba1c"].mask(numbers.index < 20))
    pooled = m.plan_one_mean(
        gaps, "hba1c", 7.0, m.ONE_T, m.pooled_rows(_imputed(gaps, ["hba1c"]), ["hba1c"])
    )
    assert pooled.steps[0].used == N and "20 values of `hba1c` are imputed" in _text(pooled)


def test_a_pooled_paired_test_is_stamped_with_the_rows_its_datasets_read(df: pd.DataFrame) -> None:
    """Every row of the population was stamped as analysed, where a column left out of the imputation lacks 30."""
    from app_pages.analysis_extras import _rows_read
    from biostats.mi import analyze_mi
    from biostats.paired import paired_t

    numbers = df[["sbp_first", "sbp_second", "hba1c"]]
    mi = _imputed(numbers, ["hba1c"])  # sbp_second keeps its 30 missing values
    found = analyze_mi(
        lambda d: paired_t(d["sbp_second"], d["sbp_first"]), mi, columns=["sbp_first", "sbp_second"]
    )
    assert _rows_read(found, mi) == N - 30 == m.pooled_rows(mi, ["sbp_first", "sbp_second"]).used
    whole = _imputed(numbers, ["sbp_second"])
    found = analyze_mi(
        lambda d: paired_t(d["sbp_second"], d["sbp_first"]),
        whole,
        columns=["sbp_first", "sbp_second"],
    )
    assert _rows_read(found, whole) == N


def test_the_strata_plan_counts_the_strata_the_tests_read(df: pd.DataFrame) -> None:
    lone = df.assign(group=np.where(df.index < 6, "small", df["grade_local"]))
    lone.loc[lone.index < 6, "arm"] = "Control"  # a stratum of one group only
    plan = m.plan_strata(lone, "ill", "Yes", "arm", "Control", "group")
    assert "1 of 4 strata holds one group only or one outcome only" in _text(plan)
    assert "it adds nothing to the odds ratio, and the two tests read the other 3" in _text(plan)
    assert "cannot be computed" not in _text(plan)


# ----------------------------------------------------------------------------- how a result is written


def test_numbers_intervals_and_p_values_are_written_for_reading() -> None:
    assert (
        layout.p_text(1.5e-43) == "< 0.001" and layout.p_text(0.0) == "< 0.001"
    )  # never "1.5e-43", never "0"
    assert (
        layout.p_text(0.0171) == "= 0.017"
        and layout.p_text(None) == "n/a"
        and layout.p_text(float("nan")) == "n/a"
    )
    assert layout.num(5.98707) == "5.99" and layout.num(None) == "n/a"
    # the whole part is kept and grouped at any size (60,962.5 read "61,000", and two means 1,200 apart read
    # "243,700 to 244,900" beside a change of "1,230"); no exponent ("+9.37e+04", "3.65e+06", "0" for 0.00001)
    assert layout.num(1234.5) == "1,234" and layout.num(60962.5) == "60,962"
    assert layout.num(93656) == "93,656" and layout.num(3650000) == "3,650,000"
    assert layout.num(1234.5, 6) == "1,234.5"
    assert layout.num(0.00001) == "0.00001" and layout.num(0.0004) == "0.0004"
    assert layout.num(-0.0) == "0"  # a range of cut-offs read "between -0 and 0"
    # next to 0.05, a fourth decimal: 0.0499 and 0.0501 both read "= 0.050"
    assert layout.p_text(0.0499) == "= 0.0499" and layout.p_text(0.0501) == "= 0.0501"
    assert layout.p_text(0.051) == "= 0.051" and layout.p_text(0.2) == "= 0.200"
    assert layout.percent(0.8614) == "86.1%" and layout.percent(None) == "n/a"
    # two bounds with the same decimals: each rounded on its own read "-9 to -7.38" and "66% to 82.6%"
    assert layout.span(-8.998, -7.384) == "-9.00 to -7.38"
    assert layout.span(19.358, 20.973) == "19.4 to 21.0"
    assert layout.span(0.996, 12.935) == "0.996 to 12.935"  # "1.00 to 12.94" beside p = 0.051
    assert layout.span(66.0, 82.6, unit="%") == "66.0% to 82.6%"
    assert layout.span(None, 1.0) == "n/a to 1"
    ci = CI(lower=0.9019, upper=0.9527, level=0.95, method="DeLong")
    assert layout.interval(ci) == "95% CI 0.902 to 0.953" and layout.interval(None) == ""
    assert layout.interval(ci, scale=100, unit="%") == "95% CI 90.2% to 95.3%"
    # without bounds, no interval: "95% CI n/a% to n/a%" stood beside a predictive value that has no value
    nothing = CI(lower=float("nan"), upper=float("nan"), level=0.95, method="")
    assert layout.interval(nothing) == "" and layout.interval(nothing, scale=100, unit="%") == ""


def test_a_plan_writes_a_number_as_a_result_does(df: pd.DataFrame) -> None:
    """One rule for the two: a plan said that a score "runs from 174 to 2.802e+04", where the page beside it wrote
    28,020."""
    assert m.number(28017.0, 4) == "28,017" == layout.num(28017.0, 4)
    assert m._num(123456.7) == "123,457" and m._num(1589.0) == "1,589" and m._num(67.6) == "67.6"
    assert m._num(0.0000123) == "0.0000123" and m._num(float("nan")) == "n/a"
    large = df.assign(marker=df["marker"] * 1000)
    plan = m.plan_accuracy(large, "ill", "Yes", "marker", m.HIGHER, m.FIXED, 5.0, None)
    text = _text(plan)
    assert "everyone is called positive" in text and "e+0" not in text
    assert re.search(r"runs from [\d,]+ to [\d,]+\)", text), text


def test_an_estimate_and_its_interval_are_written_with_the_same_decimals() -> None:
    """Each rounded on its own, a rate ratio read "3.12, 95% CI 0.996 to 12.935" and a typical change "-5.5, 95% CI
    -6.00 to -4.50": the three numbers take the decimals the one nearest 0 needs."""

    def ci(lower: float, upper: float) -> CI:
        return CI(lower=lower, upper=upper, level=0.95, method="")

    assert layout.with_interval(3.125, ci(0.996, 12.935)) == ("3.125", "95% CI 0.996 to 12.935")
    assert layout.with_interval(-5.5, ci(-6.0, -4.5)) == ("-5.50", "95% CI -6.00 to -4.50")
    assert layout.with_interval(1.359, ci(0.879, 2.006)) == ("1.359", "95% CI 0.879 to 2.006")
    assert layout.with_interval(5.98707, ci(5.52, 6.45)) == ("5.99", "95% CI 5.52 to 6.45")
    # an estimate without a value, or without an upper end: each said for itself
    assert layout.with_interval(None, ci(0.1, 0.2)) == ("n/a", "95% CI 0.100 to 0.200")
    assert layout.with_interval(float("inf"), ci(2.99, float("inf"))) == ("n/a", "")
    # two limits of agreement and the bounds of their intervals: "20.17" stood over "95% CI 19.4 to 21.0"
    assert layout.alike(-8.1912, 20.1653, -8.998, -7.384, 19.358, 20.973) == [
        "-8.19",
        "20.17",
        "-9.00",
        "-7.38",
        "19.36",
        "20.97",
    ]
    assert layout.alike(None, 1.5) == ["n/a", "1.50"]


def _runs() -> None:
    from app_pages import analysis_layout as layout
    from ui.guidance.outcomes import Advice, ModelPlan

    idle = ModelPlan(0, (), (Advice("info", "Choose the first and the second measurement."),))
    layout.run_button("waiting", ready=False, what="the paired test", plan=idle)
    layout.run_button("bare", ready=False, what="the paired test")
    layout.run_button("ready", ready=True, what="the paired test", plan=idle)


def test_a_disabled_run_says_what_is_missing() -> None:
    """With Guidance off the plan is not drawn: Run was grey and nothing on the screen said why. Its help says what
    the plan says; a Run that can be clicked has no help (it stayed open over the plan after the click)."""
    at = AppTest.from_function(_runs)
    at.run()
    assert not at.exception
    waiting, bare, ready = (at.button(key=key) for key in ("waiting", "bare", "ready"))
    assert waiting.disabled and waiting.proto.help == "Choose the first and the second measurement."
    assert bare.disabled and "complete the choices above to run the paired test" in bare.proto.help
    assert not ready.disabled and not ready.proto.help


def test_a_plan_that_runs_nothing_says_what_to_choose(df: pd.DataFrame) -> None:
    """What a disabled Run reads: the first line of a plan without a step names the choice to make."""
    for plan in (
        m.plan_paired(df, "sbp_first", None, m.T_TEST),
        m.plan_one_mean(df, "hba1c", None, m.ONE_T),
        m.plan_share(df, None, None, None),
        m.plan_strata(df, None, None, None, None, None),
    ):
        assert not plan.steps and plan.advice and plan.advice[0].text[:6] in ("Choose", "Type t")


def test_a_share_whose_test_and_interval_part_says_so() -> None:
    """An exact binomial test beside a Wilson interval: 3 of 10 against 10% gives p = 0.070 and an interval from
    10.8% to 60.3%, which leaves the reference out. Two methods; the result says so, on the page and in the dossier
    (it is one of the result's warnings)."""
    from biostats.paired import one_sample_proportion

    parted = one_sample_proportion(3, 10, 0.10)
    assert parted.p_value == pytest.approx(0.0702, abs=5e-5) and parted.estimate_ci.lower > 0.10
    assert len(parted.warnings) == 1 and "does not reject the reference" in parted.warnings[0]
    assert "report both" in parted.warnings[0]
    for with_it, counted, reference in ((30, 100, 0.20), (302, 928, 0.30), (9, 10, 0.5)):
        assert not one_sample_proportion(with_it, counted, reference).warnings
    assert parted.statistic_label == "number counted"  # the dossier read "successes = 3"


def _dependent_fields() -> None:
    import streamlit as st

    from app_pages import analysis_layout as layout

    column = st.selectbox("column", ["ill", "arm"], index=None, key="column")
    layout.depends("level", column, {"ill": "Yes"}.get(column))
    st.selectbox(
        "level",
        {"ill": ["No", "Yes"], "arm": ["Control", "Intervention"]}.get(column, []),
        index=None,
        key="level",
    )
    layout.depends("reference", column, None)
    st.number_input("reference", value=None, key="reference")


def test_a_field_follows_the_column_chosen_above_it_and_keeps_what_is_typed() -> None:
    """A level offered for one column, a number typed for it: the next column starts from its own offer, and empty.
    (The value goes through the widget's state: a default reached the script but not the screen, which went on
    showing "Choose the value" over a result computed for Yes; the browser check reads the screen.)"""
    at = AppTest.from_function(_dependent_fields)
    at.run()
    assert at.selectbox(key="level").value is None
    at.selectbox(key="column").select("ill").run()
    assert at.selectbox(key="level").value == "Yes"  # offered
    at.selectbox(key="level").select("No").run()
    at.number_input(key="reference").set_value(4.0).run()
    at.run()
    assert (
        at.selectbox(key="level").value == "No" and at.number_input(key="reference").value == 4.0
    )  # kept
    at.selectbox(key="column").select("arm").run()
    assert not at.exception
    assert at.selectbox(key="level").value is None  # neither arm says "present": a choice to make
    assert (
        at.number_input(key="reference").value is None
    )  # a reference is in the unit of its measurement


# ----------------------------------------------------------------------------- a table an analysis cannot read


def _tabs_on_a_poor_table() -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.analysis_diagnostic import run_diagnostic_accuracy
    from app_pages.analysis_extras import run_paired_and_rates

    poor = pd.DataFrame({"id": [f"P{i:02d}" for i in range(30)], "note": ["seen"] * 30})
    if st.session_state.get("tab") == "paired":
        run_paired_and_rates(poor, None)
    else:
        run_diagnostic_accuracy(poor, None)


@pytest.mark.parametrize(
    ("tab", "choices", "said"),
    [
        ("diagnostic", {"diag_question": m.ACCURACY}, "No column with two values"),
        ("diagnostic", {"diag_question": m.MEASUREMENTS}, "Fewer than two numeric columns"),
        ("diagnostic", {"diag_question": m.RATERS}, "Fewer than two columns of categories"),
        (
            "paired",
            {"paired_question": m.PAIRED, "pair_measure": m.NUMBER},
            "Fewer than two numeric columns",
        ),
        (
            "paired",
            {"paired_question": m.PAIRED, "pair_measure": m.YES_NO},
            "Fewer than two yes/no columns",
        ),
        ("paired", {"paired_question": m.ONE_GROUP, "one_compare": m.MEAN}, "No numeric column"),
        ("paired", {"paired_question": m.ONE_GROUP, "one_compare": m.SHARE}, "No yes/no column"),
        ("paired", {"paired_question": m.STRATA}, "Not enough columns for a stratified analysis"),
    ],
)
def test_an_analysis_the_table_cannot_feed_says_what_it_needs(
    tab: str, choices: dict, said: str
) -> None:
    """A table of identifiers and one constant: no form of empty lists, no Run on nothing, but what is missing."""
    at = AppTest.from_function(_tabs_on_a_poor_table, default_timeout=60)
    at.session_state["tab"] = tab
    for key, value in choices.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert any(said in block.value for block in at.markdown), [
        block.value for block in at.markdown
    ][:6]
    assert not [button for button in at.button if button.label == "Run analysis"]


def test_a_link_of_an_empty_state_opens_the_tab_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Create new variables" opened Enrich on its first tab, Import External Data: a link may name the tab of its
    page, and the two tabs name it."""
    import ui.components as components
    from app_pages import analysis_diagnostic, analysis_extras
    from ui import nav

    opened: list[tuple] = []
    monkeypatch.setattr(
        components,
        "page_link",
        lambda key, label=None, *, tab=None, **_: opened.append((key, label, tab)),
    )
    components.empty_state(
        "Nothing to read",
        "What is missing.",
        links=(("clean", "Clean categories"), ("enrich", "Create new variables", "variables")),
    )
    assert opened == [
        ("clean", "Clean categories", None),
        ("enrich", "Create new variables", "variables"),
    ]
    assert nav.tab_query("enrich", "variables")  # a tab the page declares
    for module in (analysis_diagnostic, analysis_extras):
        source = __import__("inspect").getsource(module)
        assert '("enrich", "Create new variables")' not in source, module.__name__


def test_what_the_2x2_panel_holds_is_given_as_the_stratified_analysis_takes_it(
    df: pd.DataFrame,
) -> None:
    """The values a button of another tab sets before the stratified analysis is drawn: each choice with the mark
    the dependent field keeps (so that the field shows the choice carried, not its own offer), and only what
    the analysis offers."""
    from app_pages.analysis_extras import strata_choices

    carried = strata_choices(
        df, outcome="ill", event="Yes", compare="arm", reference="Intervention"
    )
    assert carried == {
        "mh_outcome": "ill",
        "mh_ev_level": "Yes",
        "mh_ev_level::for": ("ill", ("No", "Yes")),
        "mh_exposure": "arm",
        "mh_reference": "Intervention",
        "mh_reference::for": ("arm", ("Control", "Intervention")),
    }
    assert layout.given("level", ("ill", ("No", "Yes")), "Yes") == {
        "level": "Yes",
        "level::for": ("ill", ("No", "Yes")),
    }
    # a variable of more than two values is not offered here: it is left empty, with its reference
    partly = strata_choices(df, outcome="ill", event="Yes", compare="site", reference="Metz")
    assert partly["mh_outcome"] == "ill" and partly["mh_exposure"] is None
    assert "mh_reference" not in partly
    # a level the outcome does not hold is not carried: the field then makes its own offer
    assert "mh_ev_level" not in strata_choices(
        df, outcome="ill", event="1", compare="arm", reference="Control"
    )
    # the plan then names what is still to choose, and nothing that is chosen
    said = [
        m.plan_strata(df, outcome, "Yes", compare, "Control", stratum).advice[0].text
        for outcome, compare, stratum in (
            ("ill", "arm", None),
            ("ill", None, None),
            (None, None, None),
        )
    ]
    assert said == [
        "Choose the variable to adjust for.",
        "Choose what is compared and the variable to adjust for.",
        "Choose the outcome, what is compared and the variable to adjust for.",
    ]
    # the variable to adjust for kept from an earlier use is let go when it is now the outcome or the groups
    kept = strata_choices(
        df, outcome="ill", event="Yes", compare="arm", reference="Control", stratum="arm"
    )
    assert kept["mh_stratum"] is None
    other = strata_choices(
        df, outcome="ill", event="Yes", compare="arm", reference="Control", stratum="site"
    )
    assert "mh_stratum" not in other


def test_the_calculators_need_no_table() -> None:
    at = AppTest.from_function(_tabs_on_a_poor_table, default_timeout=60)
    at.session_state["tab"] = "paired"
    at.session_state["paired_question"] = m.RATES
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    at.number_input(key="rate_e1").set_value(15).run()
    at.number_input(key="rate_t1").set_value(1200.0).run()
    at.number_input(key="rate_e0").set_value(4).run()
    at.number_input(key="rate_t0").set_value(1000.0).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    # with the decimals of its interval (0.996 to 12.935): "3.12" beside them read as another precision
    assert {metric.label: metric.value for metric in at.metric}["Rate ratio"] == "3.125"
