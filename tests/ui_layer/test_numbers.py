"""How a number is written in a plan, a sentence and a result (ui.guidance.numbers): one rule for the three."""

from __future__ import annotations

import numpy as np
import pytest

from biostats._types import CI, EffectSize
from biostats._types import TestResult as StatResult  # alias: pytest must not try to collect it
from ui.guidance import how_to_read
from ui.guidance import numbers as n


def _ci(lower: float, upper: float) -> CI:
    return CI(lower=lower, upper=upper, level=0.95, method="")


def test_a_half_is_rounded_away_from_zero_and_a_zero_has_no_sign() -> None:
    """The built-in rounding goes to the even digit: on one screen 60,962.5 read "60,962" and 1,235.5 "1,236"."""
    assert [n.fixed(value, 0) for value in (60962.5, 1234.5, 1235.5, -2.5)] == [
        "60,963",
        "1,235",
        "1,236",
        "-3",
    ]
    # the number as it is written, not as it is stored (2.675 is held a little below)
    assert n.fixed(2.675, 2) == "2.68" and n.fixed(0.125, 2) == "0.13"
    assert n.fixed(np.float64(1.005), 2) == "1.01" and n.fixed(3, 2) == "3.00"
    assert (
        n.fixed(-0.001, 2) == "0.00" and n.fixed(-0.0, 1) == "0.0" and n.fixed(-0.006, 2) == "-0.01"
    )
    assert n.fixed(1e15, 0) == "1,000,000,000,000,000"  # nothing is left to round
    for nothing in (None, float("nan"), float("inf"), "a text"):
        assert n.fixed(nothing, 2) == "n/a" and n.number(nothing) == "n/a"


def test_a_number_alone_keeps_its_whole_part_and_writes_no_exponent() -> None:
    assert n.number(1234.5) == "1,235" and n.number(60962.5) == "60,963"
    assert n.number(93656) == "93,656" and n.number(2.0) == "2" and n.number(0.5) == "0.5"
    assert n.number(5.98707) == "5.99" and n.number(5.98707, 4) == "5.987"
    assert n.number(-0.0004) == "-0.0004" and n.number(0.0000123) == "0.0000123"
    # below what twelve decimals hold: 0, without a sign
    assert n.number(1e-13) == "0" and n.number(-1e-13) == "0" and n.number(-0.0) == "0"
    assert "e" not in n.number(2.5e-9) and "e" not in n.number(3.65e12)


def test_the_decimals_of_an_estimate_and_its_interval() -> None:
    """Those the bounds need, never more than one beyond the estimate's. With the decimals of the number nearest 0
    for all three, a result got more digits the nearer it was to "no difference": "-0.0711, 95% CI -0.6941 to
    0.5518"."""
    assert n.together((3.125,), (0.996, 12.935)) == 3
    assert n.together((-0.0711,), (-0.6941, 0.5518)) == 3
    assert n.together((0.5,), (0.00847, 9.60453)) == 4  # one beyond the estimate's three
    assert n.together((0.0,), (-0.25, 0.25)) == 3  # an estimate of 0 leaves the bounds to decide
    assert n.together((None,), (0.1, 0.2)) == 3
    assert n.together((3.125,), ()) == 2 and n.together((), ()) == 0
    assert n.together((1e-9,), (1e-10, 2e-9)) == n.MOST
    assert n.with_interval(-0.0711, _ci(-0.6941, 0.5518)) == ("-0.071", "95% CI -0.694 to 0.552")
    assert n.with_interval(0.5, _ci(0.00847, 9.60453)) == ("0.5000", "95% CI 0.0085 to 9.6045")
    assert n.with_interval(3.125, _ci(0.996, 12.935)) == ("3.125", "95% CI 0.996 to 12.935")
    assert n.with_interval(2473.4, _ci(1890.2, 3056.6)) == ("2,473", "95% CI 1,890 to 3,057")
    assert n.with_interval(2.5, None) == ("2.5", "")
    # a kind of number with decimals of its own: an AUC, a kappa, an ICC
    assert n.with_interval(0.0731, _ci(0.0473, 0.0981), places=3) == (
        "0.073",
        "95% CI 0.047 to 0.098",
    )
    assert n.with_interval(0.0, _ci(float("nan"), float("nan")), places=3) == ("0.000", "")
    assert n.with_interval(-0.00004, _ci(-0.1, 0.1), places=3) == (
        "0.000",
        "95% CI -0.100 to 0.100",
    )
    assert n.with_interval(None, _ci(0.1, 0.2), places=3) == ("n/a", "95% CI 0.100 to 0.200")
    # several estimates read together: the first of them are estimates, the others bounds
    assert n.alike(12.0, 0.5, 11.25, 12.75, estimates=2) == ["12.0", "0.5", "11.3", "12.8"]
    assert n.alike(0.01234, 0.5, 0.4, 0.6) == ["0.012", "0.500", "0.400", "0.600"]
    assert n.percent(0.12345) == "12.3%" and n.percent(None) == "n/a"


@pytest.mark.parametrize(
    ("p", "text"),
    [
        (0.0499, "= 0.0499"),
        (0.0501, "= 0.0501"),
        (0.04996, "= 0.04996"),  # four decimals read "0.0500"
        (0.050004, "= 0.050004"),
        (0.05, "= 0.0500"),
        (0.0495, "= 0.0495"),
        (0.0494, "= 0.049"),
        (0.051, "= 0.051"),
        (0.001, "= 0.001"),
        (0.00099, "< 0.001"),
        (1.0, "= 1.000"),
        (0.0, "< 0.001"),
        (None, "n/a"),
        (float("nan"), "n/a"),
    ],
)
def test_a_p_value_next_to_the_threshold_takes_the_decimals_that_tell_it_from_it(p, text) -> None:
    assert n.p_text(p) == text


def test_the_reading_of_a_test_writes_its_numbers_as_the_result_does() -> None:
    """ "How to read this" had a rule of its own: a mean change of 2,473 read "2.47e+03", and "-5.01 (95% CI
    -5.677 to -4.343)" stood under a headline "-5.01, 95% CI -5.68 to -4.34"."""
    large = StatResult(
        name="Paired t-test",
        p_value=0.0499,
        estimate=2473.4,
        estimate_ci=_ci(1890.2, 3056.6),
        estimate_label="mean of the differences",
        effect_sizes=[
            EffectSize(name="matched-pairs r", kind="r", estimate=-0.529, ci=_ci(-0.6, -0.45))
        ],
        groups=["pairs"],
    )
    text = how_to_read(large)
    assert "The mean of the differences is 2,473 (95% CI 1,890 to 3,057)." in text
    assert "e+0" not in text
    assert (
        "Matched-pairs r = -0.529 (95% CI -0.600 to -0.450)" in text
    )  # a sentence starts with a capital
    assert "The p-value (0.0499)" in text
    small = StatResult(
        name="Paired t-test", p_value=0.2, estimate=-5.01, estimate_ci=_ci(-5.677, -4.343)
    )
    assert "is -5.01 (95% CI -5.68 to -4.34)." in how_to_read(small)
    # one group, or pairs: a rank test has no second group to "tend to have larger values"
    paired = StatResult(
        name="Wilcoxon signed-rank test", p_value=0.03, family="nonparametric", groups=["pairs"]
    )
    assert "about the typical difference (the pseudo-median), not about the mean" in how_to_read(
        paired
    )
    two = StatResult(
        name="Mann-Whitney U test", p_value=0.03, family="nonparametric", groups=["a", "b"]
    )
    assert "which group tends to have larger values" in how_to_read(two)
