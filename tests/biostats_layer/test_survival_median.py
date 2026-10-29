"""
One rule for every median of a survival curve (M6): R's quantile.survfit (Brookmeyer & Crowley 1982), the first time
at which S(t) <= 0.5, and the midpoint of a stretch where the curve stays at exactly 0.5. Three rules lived in the code:
statsmodels' strict one for a Kaplan-Meier curve (times 1 to 10, every one an event: 6.0), the first time at or below
0.5 for a pooled curve (5.0), and R's (5.5). The median's interval is the same rule on the band's curves.
"""

from __future__ import annotations

import pandas as pd
import pytest

from biostats.mi.adapters import median_from_curve
from biostats.references import REFERENCES
from biostats.survival import compare_survival, kaplan_meier, km_quantile

DECREASING = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


@pytest.mark.parametrize(
    ("survival", "p", "last", "expected"),
    [
        (DECREASING, 0.5, None, 5.5),  # at exactly 0.5 from 5 to 6
        (DECREASING, 0.25, None, 3.0),  # 0.75 is never exactly held: the first time below it
        (DECREASING, 0.75, None, 8.0),
        ([0.9, 0.8, 0.5, 0.5, 0.5], 0.5, 9.0, 6.0),  # at 0.5 from 3 to the last follow-up, 9
        ([0.9, 0.8, 0.5, 0.5, 0.5], 0.5, None, 4.0),  # without it, to the last time of the curve
        ([0.9, 0.8, 0.7], 0.5, None, None),  # never reached
        (
            [0.9, 0.5 + 1e-12, 0.2],
            0.5,
            None,
            2.5,
        ),  # a product of fractions that is 0.5 up to rounding
    ],
)
def test_the_quantile_of_a_curve(survival, p, last, expected) -> None:
    times = list(range(1, len(survival) + 1))
    assert km_quantile(times, survival, p, last=last) == expected


def test_a_kaplan_meier_median_and_its_interval() -> None:
    curve = kaplan_meier(list(range(1, 11)), [1] * 10)
    assert curve.median == 5.5  # 6.0 by statsmodels' strict rule
    assert curve.median_lower is not None and curve.median_upper is not None
    assert curve.median_lower <= curve.median <= curve.median_upper and curve.level == 0.95
    flat = kaplan_meier([1, 2, 3, 4, 5, 5, 5, 5], [1, 1, 1, 1, 0, 0, 0, 0])
    assert flat.median == 4.5  # at 0.5 from 4 to the last follow-up, 5
    falls = kaplan_meier([1, 2, 3, 4, 6, 7, 8, 9], [1, 1, 1, 1, 0, 1, 1, 1])
    assert falls.median == 5.5  # at 0.5 from 4 until it falls at 7
    assert kaplan_meier([1, 2, 3, 4], [1, 0, 0, 0]).median is None


def test_a_median_the_curve_falls_through_does_not_move() -> None:
    """The 6-MP and placebo arms of the Gehan leukaemia data: no stretch at exactly 0.5."""
    six_mp = [6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35]
    six_mp_event = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    placebo = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]
    frame = pd.DataFrame(
        {
            "time": six_mp + placebo,
            "event": six_mp_event + [1] * len(placebo),
            "arm": ["6-MP"] * len(six_mp) + ["placebo"] * len(placebo),
        }
    )
    result = compare_survival(frame, "time", "event", "arm")
    assert {c.label: c.median for c in result.curves} == {"6-MP": 23, "placebo": 8}
    assert REFERENCES["brookmeyer1982"] in result.references


def test_a_pooled_curve_has_the_same_rule() -> None:
    curve = pd.DataFrame({"time": range(1, 11), "survival": DECREASING})
    assert median_from_curve(curve) == 5.5  # the first time at or below 0.5 gave 5
