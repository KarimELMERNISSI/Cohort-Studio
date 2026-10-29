"""
The figures and numbers of the Survival & regression page (app_pages/analysis_survival_results.py): round time ticks
with the numbers at risk under them, round ratios on a log axis, times without an exponent, and a level's text drawn
as text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app_pages.analysis_survival_results import (
    LABEL_GAP,
    Curve,
    _median,
    _median_ci,
    _ratio_ticks,
    _ticks,
    forest_figure,
    km_figure,
)


@pytest.mark.parametrize(
    ("top", "expected"),
    [
        (1450.0, [0, 250, 500, 750, 1000, 1250]),  # every 500 left three ticks under 1,450 days
        (10.0, [0, 2, 4, 6, 8, 10]),
        (3.7, [0, 1, 2, 3]),
        (0.0, [0]),
    ],
)
def test_time_ticks_are_round_and_about_five(top, expected) -> None:
    assert list(_ticks(top)) == expected


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0.48, 4.1, [0.5, 1.0, 2.0, 4.0]),
        (0.1, 12.0, [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
        (0.75, 1.0, [0.8, 0.9, 1.0]),  # a narrow interval below 1
    ],
)
def test_ratio_ticks_read_as_ratios(lo, hi, expected) -> None:
    assert _ratio_ticks(lo, hi) == expected


def test_a_wide_range_keeps_at_most_eight_ratio_ticks_and_one() -> None:
    ticks = _ratio_ticks(0.001, 1000.0)
    assert len(ticks) <= 8 and 1.0 in ticks


def test_times_read_without_an_exponent() -> None:
    assert _median(1119.0) == "1,119"  # ".3g" gave 1.12e+03
    assert _median(854.5) == "854.5" and _median(2.37) == "2.37" and _median(None) == "not reached"
    assert _median_ci(1213.0, None) == "1,213 to not reached"
    assert _median_ci(None, None) == "not reached"  # not "not reached to not reached"


def _curve(label: str) -> Curve:
    times = np.array([1.0, 4.0, 9.0])
    survival = np.array([0.9, 0.7, 0.5])
    return Curve(label, label, times, survival, survival - 0.1, survival + 0.05)


def test_the_numbers_at_risk_sit_under_the_ticks_and_the_labels_clear_of_them() -> None:
    hostile = "<a href='https://example.org'>B</a>"
    fig = km_figure(
        [_curve("A"), _curve(hostile)],
        time_label="fu_days",
        at_risk={"A": [5, 4, 2, 1, 1], hostile: [6, 5, 3, 2, 1]},
    )
    ticks = list(fig.layout.xaxis.tickvals)
    assert ticks == [0, 2, 4, 6, 8]
    labels = [a for a in fig.layout.annotations if a.xref == "paper"]
    assert {a.xshift for a in labels} == {
        -LABEL_GAP
    }  # left of the axis: the count at 0 is centred on it
    counts = [a for a in fig.layout.annotations if a.xref == "x"]
    assert sorted({a.x for a in counts}) == ticks and len(counts) == 10
    texts = " ".join(str(a.text) for a in fig.layout.annotations) + " ".join(
        t.name or "" for t in fig.data
    )
    assert "<a href" not in texts and "&lt;a href" in texts  # a level's text stays text


def test_the_forest_puts_the_comparison_first_and_leaves_out_an_infinite_interval() -> None:
    table = pd.DataFrame(
        {
            "term": ["age", "arm_B", "sex_M", "site_X"],
            "odds_ratio": [1.02, 0.63, 1.4, 3.0],
            "ci_low": [1.01, 0.48, 0.9, 0.0],
            "ci_high": [1.03, 0.84, 2.2, np.inf],
            "p": [0.001, 0.002, 0.13, 0.99],
        }
    )
    fig = forest_figure(
        table, ratio="odds_ratio", highlight={"arm_B"}, axis_title="Odds ratio (log scale)"
    )
    order = list(fig.layout.yaxis.categoryarray)
    assert (
        order[-1] == "arm_B" and "site_X" not in order
    )  # the top of the axis is the end of the list
    assert list(fig.layout.xaxis.tickvals) == _ratio_ticks(0.48, 2.2)
    assert [t.name for t in fig.data] == ["Comparison", "Adjustment terms"]
