"""
The two measure tabs of Analyze > Hypothesis tests & epidemiology on results at their edges, through the code that
draws them (AppTest on made-up tables, where the tutorial cohort holds no such case): a stratum without an upper
bound, strata that cannot be compared, opposite strata, two columns that hold the same values or stand a constant
apart, a rater who uses one category, a test and an interval that disagree, grades written in words.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from biostats.reliability import cohens_kappa, rater_table
from ui.guidance import measures as m


def _drawn(script: Callable[[], None], case: str = "") -> AppTest:
    at = AppTest.from_function(script, default_timeout=120)
    at.session_state["case"] = case
    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    return at


def _texts(at: AppTest) -> str:
    """What the screen says, as it is read: a warning is written with Markdown's escapes."""
    said = [
        *(c.value for c in at.caption),
        *(md.value for md in at.markdown),
        *(w.value for w in at.warning),
    ]
    return "\n".join(said).replace("\\", "")


def _metrics(at: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in at.metric}


# ----------------------------------------------------------------------------- strata


def _strata() -> None:
    import pandas as pd
    import streamlit as st

    from app_pages import analysis_extras as extras
    from biostats.rates import mantel_haenszel

    cases = {
        "unbounded": {"A": (12, 0, 5, 9), "B": (20, 20, 25, 15), "C": (18, 22, 24, 16)},
        "one": {"A": (10, 10, 12, 8), "B": (5, 0, 7, 0)},
        "opposite": {"A": (30, 10, 10, 30), "B": (10, 30, 30, 10)},
    }
    rows: list[tuple[str, str, str]] = []
    for name, (a, b, c, d) in cases[st.session_state["case"]].items():
        rows += [(name, "Treated", "Yes")] * a + [(name, "Treated", "No")] * b
        rows += [(name, "Control", "Yes")] * c + [(name, "Control", "No")] * d
    frame = pd.DataFrame(rows, columns=["stratum", "group", "event"])
    result = mantel_haenszel(
        frame, "group", "event", "stratum", exposure_level="Treated", event_level="Yes"
    )
    extras._show_strata(
        result, "event", "Yes", "group", "Treated", "Control", "stratum", len(frame)
    )


def test_a_stratum_without_an_upper_bound_is_named_not_left_out() -> None:
    """Twelve events of 12 against 5 of 14 in one stratum: its odds ratio read "not estimable", and the line said
    that the strata "run from 0.545 to 0.600: they differ", leaving out the one that differed."""
    at = _drawn(_strata, "unbounded")
    text = _texts(at)
    assert (
        "The odds ratios of the strata run from 0.545 to 0.600 in 2 of the 3 strata tested; in the other the ratio "
        "has no upper bound (a cell of the table is empty): they differ (Breslow-Day p < 0.001)."
        in text
    )
    table = at.dataframe[0].value
    assert list(table["Odds ratio"]) == ["no upper bound", "0.600", "0.545"]
    assert list(table["Risk ratio"]) == ["2.800", "0.800", "0.750"]
    assert "the strata differ: read each stratum" in text  # beside the combined ratio
    # each ratio with the decimals of its interval
    assert _metrics(at)["Odds ratio, adjusted for stratum"] == "0.895"
    assert "95% CI 0.510 to 1.571 · p = 0.803" in text


def test_strata_that_were_not_compared_say_so_and_write_no_empty_number() -> None:
    """One stratum holds both groups and both outcomes: the test of the association reads it, and the details
    read "Breslow-Day: chi2 = n/a, p n/a"."""
    at = _drawn(_strata, "one")
    text = _texts(at)
    assert "strata not tested" in text
    assert (
        "Whether the strata differ was not tested: fewer than two strata hold both groups and both outcomes."
        in text
    )
    assert (
        "The test that the odds ratios of the strata are alike (Breslow-Day) was not run." in text
    )
    assert "n/a" not in text
    assert (
        "the test of the association reads the other one; whether the strata differ needs two"
        in text
    )
    assert list(at.dataframe[0].value["Odds ratio"]) == ["0.667", "not estimable"]


def test_opposite_strata_are_compared_at_a_combined_ratio_of_1() -> None:
    """Odds ratios of 9.00 and 0.11 combine to exactly 1, where the test had no value: the page read "strata not
    compared" over the two strata that differ most."""
    at = _drawn(_strata, "opposite")
    text = _texts(at)
    assert _metrics(at)["Odds ratio, adjusted for stratum"] == "1.000"
    assert (
        "The odds ratios of the strata run from 0.111 to 9.000: they differ (Breslow-Day p < 0.001)."
        in text
    )
    assert "chi2 = 40, p < 0.001" in text and "not tested" not in text
    assert "the strata differ: read each stratum" in text


# ----------------------------------------------------------------------------- two measurements


def _measurements() -> None:
    import numpy as np
    import pandas as pd
    import streamlit as st

    from app_pages import analysis_diagnostic as diag
    from biostats.agreement import bland_altman
    from biostats.reliability import icc

    first = np.random.default_rng(2).normal(100, 10, 40).round(0)
    frame = pd.DataFrame(
        {"a": first, "b": first + {"same": 0, "shifted": 10}[st.session_state["case"]]}
    )
    diag._show_bland_altman(bland_altman(frame["a"], frame["b"]), "a", "b", None)
    for model in ("ICC(2,1)", "ICC(3,1)"):
        diag._show_icc(icc(frame.to_numpy(), model=model), len(frame))


def test_two_columns_that_hold_the_same_values_are_read_as_such() -> None:
    """A column chosen under two names: "`a` reads 0 higher than `b` on average. For about 95 of 100 participants the
    two differ by between 0 and 0", an ICC of "1.000" over "(Shrout-Fleiss)" alone, and "p n/a" twice."""
    at = _drawn(_measurements, "same")
    text = _texts(at)
    assert "`a` and `b` hold the same value in every pair." in text and "reads 0" not in text
    shown = _metrics(at)
    assert shown["ICC(2,1)"] == "1.000" == shown["ICC(3,1)"]
    assert "no interval: the columns hold the same values in these rows" in text
    assert (
        "no interval: apart from a constant shift, the columns hold the same values in these rows"
        in text
    )
    assert text.count("No F test: apart from a constant shift between them") == 2
    assert "every pair differs by the same amount" in text
    assert "n/a" not in text and "Benchmark" not in text and "close to 95%" not in text
    # the plan says it before the run: a copy of a column is the likely cause
    frame = pd.DataFrame(
        {"a": [101.0, 95.5, 110.2, 99.9, 104.1], "b": [101.0, 95.5, 110.2, 99.9, 104.1]}
    )
    plan = m.plan_measurements(frame, ["a", "b"], [m.BLAND_ALTMAN, m.ICC], m.ABSOLUTE)
    assert plan.steps and plan.status == "caution"
    assert any(
        item.text == "`a` and `b` hold the same value in every row used: check that they are two "
        "measurements, not one column twice."
        for item in plan.advice
    )


def test_two_columns_a_constant_apart_are_read_as_such() -> None:
    """Agreement in ranking and spacing is complete (consistency, 1), agreement in value is not: the caption of
    the first said "the columns hold the same values", which they do not."""
    at = _drawn(_measurements, "shifted")
    text = _texts(at)
    assert "`a` reads 10.0 lower than `b` in every pair." in text
    shown = _metrics(at)
    assert shown["ICC(3,1)"] == "1.000" and shown["ICC(2,1)"] == "0.640"
    assert (
        "no interval: apart from a constant shift, the columns hold the same values in these rows"
        in text
    )
    assert "no interval: the columns hold the same values in these rows" not in text
    frame = pd.DataFrame({"a": [101.0, 95.5, 110.2, 99.9, 104.1]})
    frame["b"] = frame["a"] + 0.3  # in floating point the five differences are not one number
    plan = m.plan_measurements(frame, ["a", "b"], [m.BLAND_ALTMAN, m.ICC], m.ABSOLUTE)
    assert any(
        "differ by the same amount (0.3) in every row used: one looks computed from the other"
        in item.text
        for item in plan.advice
    )
    usual = frame.assign(b=[100.0, 97.0, 108.5, 101.2, 103.0])
    said = " ".join(
        item.text
        for item in m.plan_measurements(usual, ["a", "b"], [m.BLAND_ALTMAN], m.ABSOLUTE).advice
    )
    assert "in every row used" not in said


# ----------------------------------------------------------------------------- two raters


def _one_category() -> None:
    import pandas as pd

    from app_pages import analysis_diagnostic as diag

    frame = pd.DataFrame({"a": ["X"] * 40, "b": ["X"] * 30 + ["Y"] * 10})
    diag._show_kappa(diag.kappa_of(frame, "a", "b", order=None, weights=None), "a", "b")


def test_a_rater_who_uses_one_category_gives_a_kappa_without_an_interval() -> None:
    at = _drawn(_one_category)
    text = _texts(at)
    assert _metrics(at)["Cohen's kappa"] == "0.000"
    assert "0 by construction: one rater uses one category only" in text
    assert "No standard error, and no interval: one rater uses one category only." in text
    assert "95% CI" not in text and "Benchmark" not in text and "nan" not in text


def _raters_in_words() -> None:
    import numpy as np
    import pandas as pd
    import streamlit as st

    from app_pages import analysis_diagnostic as diag

    scale = ["none", "low", "medium", "high"]
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 4, 300)
    other = np.clip(truth + rng.choice([-1, 0, 0, 0, 1], 300), 0, 3)
    frame = pd.DataFrame(
        {"investigator": [scale[i] for i in truth], "committee": [scale[i] for i in other]}
    )
    if st.session_state.get("case") == "result":
        # as Run computes and draws it (a result is kept with the dataset of the session: none is loaded here)
        order = st.session_state["order"]
        found = diag.kappa_of(frame, "investigator", "committee", order=order, weights="quadratic")
        diag._show_kappa(found, "investigator", "committee")
    else:
        diag.run_diagnostic_accuracy(frame, None)


def test_grades_in_words_are_placed_by_hand_and_weighted_in_that_order() -> None:
    """Sorted as text, "high, low, medium, none" was the scale of a weighted kappa (0.53 for 0.82). Words carry no
    order of their own: Run waits until every grade is placed, and the weights read the order given."""
    at = AppTest.from_function(_raters_in_words, default_timeout=120)
    at.session_state["diag_question"] = m.RATERS
    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    at.selectbox(key="kappa_a").select("investigator").run()
    at.selectbox(key="kappa_b").select("committee").run()
    at.radio(key="kappa_scale").set_value(m.GRADES).run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    assert at.multiselect(key="kappa_order").value == []  # no order is offered for words
    assert at.button(key="kappa_run").disabled
    assert "Grades written in words carry no order of their own" in _texts(at)
    scale = ["none", "low", "medium", "high"]
    at.multiselect(key="kappa_order").set_value(scale).run()
    at.radio(key="kappa_w").set_value(m.QUADRATIC).run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    assert not at.button(key="kappa_run").disabled  # every grade is placed
    assert "Order used: `none` < `low` < `medium` < `high`" in _texts(at)
    # the result, in the order placed: the same ratings held as numbers give the same weighted kappa
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 4, 300)
    other = np.clip(truth + rng.choice([-1, 0, 0, 0, 1], 300), 0, 3)
    numbers = rater_table(pd.Series(truth), pd.Series(other)).to_numpy()
    expected = cohens_kappa(numbers, weights="quadratic").kappa
    at = AppTest.from_function(_raters_in_words, default_timeout=120)
    at.session_state["case"] = "result"
    at.session_state["order"] = scale
    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    shown = _metrics(at)
    assert shown["Weighted kappa (quadratic)"] == f"{expected:.3f}"
    assert "grades in the order none < low < medium < high" in _texts(at)
    by_name = cohens_kappa(
        rater_table(
            pd.Series([scale[i] for i in truth]), pd.Series([scale[i] for i in other])
        ).to_numpy(),
        weights="quadratic",
    ).kappa
    assert expected - by_name > 0.05  # what the alphabet's order gave


# ----------------------------------------------------------------------------- a change, a share


def _changes() -> None:
    import numpy as np
    import pandas as pd
    import streamlit as st

    from app_pages import analysis_extras as extras
    from biostats.paired import mcnemar_test, one_sample_proportion, one_sample_t, paired_t
    from ui.guidance import measures as m

    case = st.session_state["case"]
    rng = np.random.default_rng(5)
    before = rng.normal(140, 12, 60).round(0)
    after = (before - 5.5 + rng.normal(0, 4, 60)).round(0)
    if case == "large":
        frame = pd.DataFrame({"before": before * 1000, "after": after * 1000 + 2473})
        extras._show_change(
            paired_t(frame["after"], frame["before"]),
            frame,
            "before",
            "after",
            m.T_TEST,
            pooled=False,
        )
    elif case == "mean":
        frame = pd.DataFrame({"after": after})
        extras._show_mean(
            one_sample_t(frame["after"], 140.0), "after", 140.0, m.ONE_T, len(frame), 0
        )
    elif case == "mcnemar":
        table = pd.DataFrame([[35, 0], [5, 0]], index=["No", "Yes"], columns=["No", "Yes"])
        extras._show_yes_no(mcnemar_test(table.to_numpy()), table, "t0", "t1")
    else:
        extras._share_metrics(one_sample_proportion(2, 4, 0.15), 2, 4, 0.15, "the characteristic")


@pytest.mark.parametrize(
    ("case", "figure", "sentence"),
    [
        ("large", "Mean change", "`after` is {} lower than `before` on average (paired t-test)."),
        (
            "mean",
            "Difference from 140",
            "The mean of `after` is {} below the reference of 140 (one-sample t-test).",
        ),
    ],
)
def test_the_sentence_under_a_result_repeats_the_figure_of_the_headline(
    case: str, figure: str, sentence: str
) -> None:
    """ "5.5 lower" stood under "-5.50", and a change of 3,660 read "3.66e+03" in the reading of the test."""
    at = _drawn(_changes, case)
    value = _metrics(at)[figure]
    assert value.startswith("-") and "e" not in value
    assert sentence.format(value.lstrip("-")) in _texts(at)


def test_mcnemar_s_test_and_interval_that_disagree_are_said_under_the_result() -> None:
    """Five changed one way and none the other among 40. The exact p is 0.0625: rounded to the even digit it read
    "0.062", where every other number on the screen rounds a half away from zero."""
    at = _drawn(_changes, "mcnemar")
    text = _texts(at)
    assert _metrics(at)["Change in the share with Yes"] == "-12.5 points"
    assert "95% CI -23.6 to -1.4 · p = 0.063" in text
    assert (
        "The test and the interval disagree: the test does not reject no change, the interval leaves 0 out."
        in text
    )
    assert "2 warnings below" in text


def test_a_share_s_note_gives_the_end_of_the_interval_that_the_headline_rounds() -> None:
    """2 of 4 against 15%: the headline's interval reads "15.0% to 85.0%" beside "the interval leaves it out"."""
    at = _drawn(_changes, "share")
    text = _texts(at)
    assert "2 of 4 · 95% CI 15.0% to 85.0% (Wilson)" in text
    assert "the interval leaves it out (its lower end is 15.004%)" in text
    assert "The share is 35.0 points above the reference." in text
