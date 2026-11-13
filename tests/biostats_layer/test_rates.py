"""Rate ratios: the interval and the test agree, and the exact interval is the one of R's poisson.test (M5)."""

from __future__ import annotations

import pandas as pd
import pytest
from scipy import stats

from biostats.rates import mantel_haenszel, rate_ratio

CASES = [
    (15, 1200, 4, 1000),
    (11, 800, 21, 3011),
    (0, 500, 7, 900),
    (9, 250, 0, 400),
    (120, 10500.5, 95, 11200),
    (3, 100, 3, 100),
]


def test_the_exact_interval_is_the_one_of_r_poisson_test() -> None:
    # the example of R's help page ?poisson.test: poisson.test(c(11, 6 + 8 + 7), c(800, 1083 + 1050 + 878))
    # gives the rate ratio 1.971488 with the 95 percent confidence interval 0.8584264 to 4.2772659
    rr = rate_ratio(11, 800, 6 + 8 + 7, 1083 + 1050 + 878)
    assert rr.ratio == pytest.approx(1.971488, abs=1e-6)
    assert rr.ci.lower == pytest.approx(0.8584264, abs=1e-7)
    assert rr.ci.upper == pytest.approx(4.2772659, abs=1e-7)
    assert rr.ci.method == "exact conditional (Clopper-Pearson)" and rr.test == "exact conditional"


@pytest.mark.parametrize("counts", CASES)
def test_the_exact_interval_and_p_value_agree(counts: tuple[int, float, int, float]) -> None:
    rr = rate_ratio(*counts)
    assert (rr.p_value < 0.05) == (rr.ci.lower > 1 or rr.ci.upper < 1)
    # twice the smaller tail of the exposed events, binomial given the total under IRR = 1
    d1, t1, d0, t0 = counts
    null = t1 / (t1 + t0)
    tails = stats.binom.cdf(d1, d1 + d0, null), stats.binom.sf(d1 - 1, d1 + d0, null)
    assert rr.p_value == pytest.approx(min(1.0, 2 * min(tails)))


def test_the_score_option_pairs_the_score_interval_with_the_score_test() -> None:
    for counts in [c for c in CASES if min(c[0], c[2]) > 0]:
        rr = rate_ratio(*counts, ci_method="score")
        assert rr.ci.method == "score" and rr.test == "score"
        assert (rr.p_value < 0.05) == (rr.ci.lower > 1 or rr.ci.upper < 1)


def test_a_group_without_events() -> None:
    # the score interval collapsed to a single value here (0.988 to 0.988); the exact interval is open-ended
    none_exposed = rate_ratio(0, 500, 7, 900)
    assert none_exposed.ratio == 0 and none_exposed.ci.lower == 0 and 1 < none_exposed.ci.upper < 2
    none_reference = rate_ratio(9, 250, 0, 400)
    assert none_reference.ratio == float("inf") and none_reference.ci.upper == float("inf")
    assert none_reference.ci.lower > 1 and none_reference.p_value < 0.001
    with pytest.raises(ValueError, match="no events in either group"):
        rate_ratio(0, 500, 0, 900)
    with pytest.raises(ValueError, match="events in both groups"):
        rate_ratio(0, 500, 7, 900, ci_method="score")
    with pytest.raises(ValueError, match="unknown interval method"):
        rate_ratio(5, 500, 7, 900, ci_method="wald")  # type: ignore[arg-type]


def _strata(cells: dict[str, tuple[int, int, int, int]]) -> pd.DataFrame:
    """Rows from the 2x2 counts of each stratum: exposed events, exposed non-events, reference events, reference
    non-events."""
    rows = []
    for stratum, (a, b, c, d) in cells.items():
        for exposed, event, n in (
            ("yes", "yes", a),
            ("yes", "no", b),
            ("no", "yes", c),
            ("no", "no", d),
        ):
            rows += [{"exposed": exposed, "event": event, "stratum": stratum}] * n
    return pd.DataFrame(rows)


def test_odds_ratios_that_differ_without_confounding_are_not_called_confounding() -> None:
    # exposure independent of the stratum (half exposed in each), the same odds ratio of 9 in both strata: the
    # crude odds ratio (5.4) differs from the adjusted one only because odds ratios are non-collapsible, and the
    # crude and adjusted risk ratios agree (2.33)
    df = _strata({"low risk": (50, 50, 10, 90), "high risk": (90, 10, 50, 50)})
    res = mantel_haenszel(
        df, "exposed", "event", "stratum", exposure_level="yes", event_level="yes"
    )
    assert res.or_mh == pytest.approx(9.0) and res.crude_or == pytest.approx(49 / 9)
    assert res.crude_rr == pytest.approx(res.rr_mh) == pytest.approx(7 / 3)
    assert not any("may be a confounder" in w for w in res.warnings)
    assert any("non-collapsible" in w and "need not mean confounding" in w for w in res.warnings)


def test_confounding_shows_on_the_risk_ratios() -> None:
    # exposure concentrated in the high-risk stratum, no effect within the strata: the crude ratios are confounded
    df = _strata({"low risk": (2, 18, 16, 144), "high risk": (90, 90, 10, 10)})
    res = mantel_haenszel(
        df, "exposed", "event", "stratum", exposure_level="yes", event_level="yes"
    )
    assert res.rr_mh == pytest.approx(1.0) and res.crude_rr > 2
    assert any("risk ratios differ" in w and "may be a confounder" in w for w in res.warnings)


# ----------------------------------------------------------------------------- strata that carry no information

SITES = {
    "A": (35, 85, 46, 74),
    "B": (25, 71, 26, 62),
    "C": (39, 119, 68, 94),
    "D": (29, 60, 32, 58),
}


def _mh(cells: dict[str, tuple[int, int, int, int]]):
    return mantel_haenszel(
        _strata(cells), "exposed", "event", "stratum", exposure_level="yes", event_level="yes"
    )


def test_a_stratum_where_everyone_had_the_event_does_not_count_in_the_tests() -> None:
    """It has no odds ratio of its own. Counted among the strata compared, it gave the Breslow-Day test a degree of
    freedom too many: the same statistic read p = 0.48 where the four informative strata give 0.33."""
    four = _mh(SITES)
    five = _mh({**SITES, "E": (10, 0, 10, 0)})
    assert (four.n_strata, four.n_tested, five.n_strata, five.n_tested) == (4, 4, 5, 4)
    assert five.breslow_day_statistic == pytest.approx(four.breslow_day_statistic)
    assert five.breslow_day_p == pytest.approx(four.breslow_day_p)
    assert five.breslow_day_p == pytest.approx(
        stats.chi2.sf(four.breslow_day_statistic, 3)
    )  # 3 df, not 4
    assert five.mh_test_p == pytest.approx(four.mh_test_p)
    assert five.or_mh == pytest.approx(four.or_mh)
    assert (five.or_mh_ci.lower, five.or_mh_ci.upper) == pytest.approx(
        (four.or_mh_ci.lower, four.or_mh_ci.upper)
    )
    # the risk ratio reads it: both risks are 1 there, which pulls the combined ratio towards 1
    assert abs(five.rr_mh - 1) < abs(four.rr_mh - 1)
    assert any("1 of 5 strata holds one group only or one outcome only" in w for w in five.warnings)
    assert not any("one group only" in w for w in four.warnings)


@pytest.mark.parametrize(
    "added",
    [
        {"E": (0, 10, 0, 10)},  # no event
        {"E": (2, 4, 0, 0)},  # one group only
        {"E": (1, 0, 0, 0)},  # a single row: it left both tests without a value
        {"E": (10, 0, 10, 0), "F": (0, 3, 0, 0), "G": (0, 1, 0, 0)},
    ],
)
def test_the_two_tests_read_the_strata_that_hold_both_groups_and_both_outcomes(added: dict) -> None:
    four, more = _mh(SITES), _mh({**SITES, **added})
    assert more.n_strata == 4 + len(added) and more.n_tested == 4
    assert more.mh_test_statistic == pytest.approx(four.mh_test_statistic)
    assert more.mh_test_p == pytest.approx(four.mh_test_p)
    assert more.breslow_day_statistic == pytest.approx(four.breslow_day_statistic)
    assert more.breslow_day_p == pytest.approx(four.breslow_day_p)
    assert more.or_mh == pytest.approx(four.or_mh)
    assert more.n == four.n + sum(sum(cells) for cells in added.values())
    assert len(more.per_stratum) == more.n_strata  # every stratum is still shown


def test_one_informative_stratum_has_no_other_to_differ_from() -> None:
    res = _mh({"A": SITES["A"], "E": (10, 0, 10, 0)})
    assert res.n_tested == 1 and res.breslow_day_p != res.breslow_day_p  # not a number: not tested
    assert res.mh_test_p == pytest.approx(_mh_alone(SITES["A"]))
    assert res.or_mh == pytest.approx((35 * 74) / (85 * 46))
    none = _mh({"E": (10, 0, 10, 0), "F": (0, 5, 0, 5)})
    assert none.n_tested == 0 and none.mh_test_p != none.mh_test_p and none.or_mh != none.or_mh


def _mh_alone(cells: tuple[int, int, int, int]) -> float:
    """The Mantel-Haenszel test of one 2x2 table, continuity-corrected."""
    a, b, c, d = cells
    n1, n0, m1, m0, n = a + b, c + d, a + c, b + d, a + b + c + d
    expected, variance = n1 * m1 / n, n1 * n0 * m1 * m0 / (n**2 * (n - 1))
    return float(stats.chi2.sf((abs(a - expected) - 0.5) ** 2 / variance, 1))


def test_the_result_names_what_it_compares_and_the_rows_it_read() -> None:
    res = _mh(SITES)
    assert (res.compared, res.reference, res.event) == ("yes", "no", "yes")
    assert res.n == sum(sum(cells) for cells in SITES.values())
