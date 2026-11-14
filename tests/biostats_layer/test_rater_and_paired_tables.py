"""
The tables two analyses of agreement and of change are computed on, built from two columns: the ratings of two raters
compared as values and in a stated order (Cohen's kappa), and a yes/no characteristic measured twice, which must hold
the same two values (McNemar's test). And a cut-off that calls no one, or everyone, positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biostats.diagnostic import diagnostic_accuracy
from biostats.paired import mcnemar_test, one_sample_proportion, paired_table
from biostats.reliability import (
    by_value,
    category,
    category_order,
    cohens_kappa,
    koo_li,
    landis_koch,
    number_of,
    rater_table,
)


def test_a_rating_is_the_category_of_its_value() -> None:
    assert [category(v) for v in (2, 2.0, np.int64(2), np.float64(2.0), "2", "2.0", " 2 ")] == [
        "2"
    ] * 7
    assert category(2.5) == "2.5" and category("II") == "II" and category("low ") == "low "
    assert category(True) == "True" and category(np.bool_(False)) == "False"
    assert category("nan") == "nan" and category("inf") == "inf"  # texts, not numbers
    # numbers by value, anything else by name
    assert category_order(["10", "2", "1"]) == ["1", "2", "10"]
    assert category_order(["low", "High", "medium"]) == ["High", "low", "medium"]


def test_a_text_is_a_number_only_when_written_as_one() -> None:
    """ "1e3", "1_000", "nan" and "inf" all parse as numbers: as ratings they are names, as codes are. Spaces around
    a number are not part of it."""
    names = ["1e3", "1_000", "nan", "inf", "Infinity", "0x10", "1,5"]
    assert [category(name) for name in names] == names
    assert [category(v) for v in ("2.0", "02", " 2 ", "+2", "2.", ".5", "-1.50")] == [
        "2",
        "2",
        "2",
        "2",
        "2",
        "0.5",
        "-1.5",
    ]
    # a number itself, whatever its size: written without an exponent, so that it reads back as the number it is
    assert category(2.5e-7) == "0.00000025" and category(1e16) == "10000000000000000"
    assert number_of(category(2.5e-7)) == 2.5e-7
    # one name among the categories: all by name
    assert category_order(["10", "9", "1e3", "2"]) == ["10", "1e3", "2", "9"]
    assert number_of(" 12 ") == 12.0 and number_of(".5") == 0.5 and number_of("-1.5") == -1.5
    for name in ("1e3", "nan", "inf", "1_000", ""):
        with pytest.raises(ValueError):
            number_of(name)


def test_a_kappa_without_a_standard_error_has_no_interval() -> None:
    """One rater uses one category only: kappa is 0 by construction and its standard error has no value. The
    interval was cut to "-1 to 1", which reads as an interval that says nothing, where there is none."""
    one_category = cohens_kappa(np.array([[19, 1], [0, 0]]))
    assert one_category.kappa == 0.0
    for value in (one_category.se, one_category.ci.lower, one_category.ci.upper):
        assert np.isnan(value)
    # an interval that passes an end of the scale is still cut there
    near_one = cohens_kappa(np.array([[10, 0], [1, 9]]))
    assert near_one.kappa == pytest.approx(0.9) and near_one.ci.upper == 1.0
    assert -1 <= near_one.ci.lower < near_one.kappa


def test_two_raters_codes_are_compared_as_values() -> None:
    """One rater's grades read as whole numbers, the other's as decimals (a missing grade makes a column of
    decimals): compared as text they shared no category, and kappa was 0.000 with "slight agreement"."""
    first = pd.Series([1, 2, 3, 4, 2, 3, 1, 4, 2, 3] * 6, name="reader_1")
    second = first.astype(float).rename("reader_2")
    second.iloc[[0, 7, 13]] = [2.0, 3.0, np.nan]
    table = rater_table(first, second)
    assert list(table.index) == list(table.columns) == ["1", "2", "3", "4"]
    assert (table.index.name, table.columns.name) == ("reader_1", "reader_2")
    counts = table.to_numpy()
    assert int(counts.sum()) == 59 and int(np.trace(counts)) == 57  # the participants rated by both
    assert cohens_kappa(counts).kappa > 0.9
    # as text, the way the page built it: no category in common
    as_text = pd.crosstab(first.astype(str), second.astype(str))
    assert not set(as_text.index) & set(as_text.columns)


def test_two_codes_are_one_category_only_beside_a_column_of_numbers() -> None:
    """Texts that read as the same number were one category whatever the columns: "250.0" and "250.00", "01" and
    "1" are codes, and two raters who never wrote the same code had a kappa of 1. Two columns of texts are compared
    as written; texts beside a column of numbers are the same grades written two ways, compared by value."""
    first = pd.Series(["250.0", "250.0", "401.9", "01", "250.0", "401.9"], name="a")
    second = pd.Series(["250.00", "250.0", "401.9", "1", "250.00", "401.9"], name="b")
    assert not by_value(first, second)
    table = rater_table(first, second)
    # two writings of one number are two categories, in an order that no run changes
    assert list(table.index) == ["01", "1", "250.0", "250.00", "401.9"] == list(table.columns)
    assert int(np.trace(table.to_numpy())) == 3 and int(table.to_numpy().sum()) == 6
    assert cohens_kappa(table).kappa < 1
    # a column of numbers beside a column of texts: by value
    grades = pd.Series([2, 3, 2, 4, 3, 4], name="a")
    typed = pd.Series(["2.0", "3", "2", "04", "3.0", "4"], name="b")
    assert by_value(grades, typed) and by_value(typed, grades)
    table = rater_table(grades, typed)
    assert list(table.index) == ["2", "3", "4"] and int(np.trace(table.to_numpy())) == 6
    # a column of True and False is no column of numbers
    assert not by_value(pd.Series([True, False]), pd.Series(["1", "0"]))
    # a category "nan" is a text: sorted as a number it stood between 2 and 1
    assert category_order(["2", "nan", "1"]) == ["1", "2", "nan"]
    assert category_order(["10", "9", "09"]) == ["09", "9", "10"]


def test_a_weighted_kappa_reads_the_grades_in_their_stated_order() -> None:
    """Sorted as text, "high, low, medium, none" was the scale of a weighted kappa (0.53, "moderate", for ratings
    whose weighted kappa is 0.82), and grades from 1 to 10 read 1, 10, 2."""
    scale = ["none", "low", "medium", "high"]
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 4, 300)
    other = np.clip(truth + rng.choice([-1, 0, 0, 0, 1], 300), 0, 3)
    first = pd.Series([scale[i] for i in truth], name="investigator")
    second = pd.Series([scale[i] for i in other], name="committee")
    by_name = rater_table(first, second)
    assert list(by_name.index) == ["high", "low", "medium", "none"]
    stated = rater_table(first, second, order=scale)
    assert list(stated.index) == list(stated.columns) == scale
    # the same ratings held as numbers: the scale's own order
    numbers = rater_table(pd.Series(truth), pd.Series(other))
    for weights in ("linear", "quadratic"):
        right = cohens_kappa(numbers.to_numpy(), weights=weights).kappa
        assert cohens_kappa(stated.to_numpy(), weights=weights).kappa == pytest.approx(right)
        assert cohens_kappa(by_name.to_numpy(), weights=weights).kappa < right - 0.05
    # without weights the order does not matter
    assert cohens_kappa(by_name.to_numpy()).kappa == pytest.approx(
        cohens_kappa(stated.to_numpy()).kappa
    )
    grades = pd.Series(list(range(1, 11)) * 3)
    assert list(rater_table(grades, grades).index) == [str(grade) for grade in range(1, 11)]
    with pytest.raises(ValueError, match="every category once"):
        rater_table(first, second, order=["none", "low", "medium"])
    with pytest.raises(ValueError, match="every category once"):
        rater_table(first, second, order=[*scale, "low"])


def test_the_benchmark_words_can_be_read_on_an_interval() -> None:
    assert [landis_koch(k) for k in (-0.1, 0.1, 0.3, 0.5, 0.71, 0.9)] == [
        "poor (less than chance)",
        "slight",
        "fair",
        "moderate",
        "substantial",
        "almost perfect",
    ]
    assert [koo_li(v) for v in (0.4, 0.55, 0.87, 0.94)] == ["poor", "moderate", "good", "excellent"]


def test_a_paired_yes_no_table_holds_the_same_two_values() -> None:
    """ "Negative / Positive" before and "Detected / Not detected" after were paired by the alphabetical order of
    their labels: a difference of 0.666 with p = 2e-107, where the share had fallen by 6 points."""
    before = pd.Series(["Positive"] * 30 + ["Negative"] * 70, name="t0")
    after = pd.Series(["Detected"] * 20 + ["Not detected"] * 80, name="t1")
    with pytest.raises(ValueError, match="the same two values"):
        paired_table(before, after)
    same = pd.Series(["Positive"] * 20 + ["Negative"] * 80, name="t1")
    table = paired_table(before, same)
    assert list(table.index) == list(table.columns) == ["Negative", "Positive"]
    assert (table.index.name, table.columns.name) == ("t0", "t1")
    assert table.to_numpy().tolist() == [[70, 0], [10, 20]]
    result = mcnemar_test(table.to_numpy())
    assert result.estimate == pytest.approx(-0.10)  # the share with Positive: 30% before, 20% after
    # the value counted comes second: the same test, the change in the share of the other value
    counted = paired_table(before, same, counted="Negative")
    assert list(counted.index) == ["Positive", "Negative"]
    other = mcnemar_test(counted.to_numpy())
    assert other.estimate == pytest.approx(0.10) and other.p_value == pytest.approx(result.p_value)
    with pytest.raises(ValueError, match="not one of the two values"):
        paired_table(before, same, counted="Maybe")
    # the rows where both are recorded; 1 and 1.0 are one value
    first = pd.Series([1, 0, 1, 0, 1, None], name="a")
    second = pd.Series([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], name="b")
    assert int(paired_table(first, second).to_numpy().sum()) == 5
    with pytest.raises(ValueError, match="the same two values"):
        paired_table(pd.Series(["Yes"] * 5), pd.Series(["Yes"] * 5))  # one value only


def test_mcnemar_s_test_and_its_interval_say_when_they_disagree() -> None:
    """Five participants changed one way and none the other among 40: the exact test gives p = 0.0625 beside an
    interval of the change from -23.6 to -1.4 points, which leaves 0 out. The two are two methods, and they part
    more often than those of a share."""
    result = mcnemar_test(np.array([[35, 0], [5, 0]]))
    assert result.p_value == pytest.approx(0.0625)
    assert result.estimate_ci is not None
    assert (result.estimate_ci.lower, result.estimate_ci.upper) == pytest.approx(
        (-0.236, -0.014), abs=1e-3
    )
    assert any(
        "The test and the interval disagree: the test does not reject no change, the interval leaves 0 out"
        in said
        and "McNemar's exact test, a Wald interval" in said
        for said in result.warnings
    )
    # with pairs enough for no other warning: 6 against 16 changed, p = 0.0525 beside -11.9 to -0.5 points
    textbook = mcnemar_test([[59, 6], [16, 80]])
    assert [said[:35] for said in textbook.warnings] == ["The test and the interval disagree:"]
    # they agree: nothing is added
    assert not mcnemar_test(np.array([[744, 0], [58, 126]])).warnings
    assert not mcnemar_test(
        np.array([[40, 12], [12, 40]])
    ).warnings  # no change, and an interval about 0


def test_a_share_s_test_and_interval_that_disagree_give_the_end_of_the_interval() -> None:
    """2 of 4 against 15%: the Wilson interval starts at 15.004%. At one decimal the screen read "15.0% to 85.0%"
    beside "the interval leaves 15% out"; the note gives the end with the decimals that tell it from the
    reference."""
    close = one_sample_proportion(2, 4, 0.15)
    assert close.p_value > 0.05
    assert any(
        "the test does not reject the reference, the interval leaves it out (its lower end is 15.004%)"
        in said
        for said in close.warnings
    )
    wide = one_sample_proportion(3, 10, 0.10)
    assert any("(its lower end is 10.8%)" in said for said in wide.warnings)
    # the two agree: no note
    assert not one_sample_proportion(30, 100, 0.2).warnings
    assert not one_sample_proportion(0, 39, 0.09).warnings


def test_a_cut_off_that_calls_no_one_or_everyone_positive_is_no_division_by_zero() -> None:
    """A marker that points the other way, or a cut-off above every score, leaves a row of the 2x2 table empty: the
    predictive value at a prevalence was 0 / 0, the page said "could not be computed", and the caution written for
    that case was lost with the result."""
    everyone = diagnostic_accuracy(tp=302, fp=626, fn=0, tn=0, prevalence=0.325)
    assert everyone.npv_at_prevalence is None
    assert everyone.ppv_at_prevalence == pytest.approx(0.325)
    assert any("everyone is called positive" in said for said in everyone.warnings)
    no_one = diagnostic_accuracy(tp=0, fp=0, fn=101, tn=827, prevalence=0.5)
    assert no_one.ppv_at_prevalence is None and no_one.npv_at_prevalence == pytest.approx(0.5)
    assert any("no one is called positive" in said for said in no_one.warnings)
    # a cut-off inside the scores: the same numbers as before, and no such warning
    usual = diagnostic_accuracy(tp=87, fp=126, fn=14, tn=701, prevalence=0.02)
    sens, spec = 87 / 101, 701 / 827
    assert usual.ppv_at_prevalence == sens * 0.02 / (sens * 0.02 + (1 - spec) * 0.98)
    assert usual.npv_at_prevalence == spec * 0.98 / (spec * 0.98 + (1 - sens) * 0.02)
    assert not any("called positive" in said for said in usual.warnings)
