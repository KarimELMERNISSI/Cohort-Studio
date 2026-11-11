"""
The rank-based estimates and their intervals, computed without random draws (biostats.rank_intervals): the
Hodges-Lehmann shift of two groups, the pseudo-median of paired differences, and the rank-biserial r. Until M8 they
were percentile bootstraps: an end moved with the seed, and a comparison of two large groups took minutes.
"""

from __future__ import annotations

import math
from math import comb

import numpy as np
import pytest
from scipy import stats

import biostats.rank_intervals as ri
from biostats.effect_sizes import rank_biserial
from biostats.inference import compare_groups
from biostats.paired import one_sample_wilcoxon, wilcoxon_signed_rank
from biostats.rank_intervals import (
    APPROXIMATE,
    EXACT,
    disagreement,
    hodges_lehmann,
    pseudo_median,
)

# Hollander & Wolfe's examples on the help page of R's wilcox.test, with the values R reports for them
PERMEABILITY_X = [0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46]
PERMEABILITY_Y = [1.15, 0.88, 0.90, 0.74, 1.21]
DEPRESSION_BEFORE = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]
DEPRESSION_AFTER = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29]


def test_the_shift_of_two_small_groups_is_the_one_r_reports() -> None:
    estimate, ci = hodges_lehmann(np.array(PERMEABILITY_X), np.array(PERMEABILITY_Y))
    assert estimate == pytest.approx(0.305, abs=1e-12)
    assert ci is not None and ci.method == EXACT and ci.level == 0.95
    assert (ci.lower, ci.upper) == pytest.approx((-0.15, 0.76), abs=1e-12)


def test_the_pseudo_median_of_paired_differences_is_the_one_r_reports() -> None:
    d = np.array(DEPRESSION_BEFORE) - np.array(DEPRESSION_AFTER)
    estimate, ci = pseudo_median(d)
    assert estimate == pytest.approx(0.46, abs=1e-12)
    assert ci is not None and ci.method == EXACT
    assert (ci.lower, ci.upper) == pytest.approx((0.010, 0.786), abs=1e-12)


def test_the_null_distributions_are_counted_exactly() -> None:
    for m, n in ((3, 3), (5, 7), (10, 5), (12, 12), (49, 49)):
        counts = ri._mann_whitney_counts(m, n)
        assert len(counts) == m * n + 1 and counts[0] == 1
        assert counts.sum() == pytest.approx(comb(m + n, m), rel=1e-12)
        assert np.array_equal(counts, counts[::-1])  # U and mn - U are equally likely
    assert list(ri._mann_whitney_counts(2, 3)) == [1, 1, 2, 2, 2, 1, 1]
    for n in (5, 9, 20, 49):
        counts = ri._signed_rank_counts(n)
        assert counts.sum() == 2**n and np.array_equal(counts, counts[::-1])
    assert list(ri._signed_rank_counts(3)) == [1, 1, 1, 2, 1, 1, 1]
    # the quantile of R's qwilcox and qsignrank: the smallest value whose cumulative probability reaches p
    assert ri._lower_quantile(ri._mann_whitney_counts(10, 5), 0.025) == 9
    assert ri._lower_quantile(ri._signed_rank_counts(9), 0.025) == 6


def test_groups_too_small_for_the_level_get_no_interval() -> None:
    estimate, ci = hodges_lehmann(np.array([1.0, 2.5, 4.0]), np.array([0.5, 3.0, 6.0]))
    assert (
        ci is None and estimate == -0.5
    )  # two groups of 3 allow 90% at most: no interval labelled 95%
    assert hodges_lehmann(np.array([1.0, 2.0]), np.arange(20.0))[1] is None  # a group below 3
    assert pseudo_median(np.array([1.0, -2.0, 3.0, 4.5]))[1] is None  # fewer than 5 differences
    ninety = hodges_lehmann(np.array([1.0, 2.5, 4.0]), np.array([0.5, 3.0, 6.0]), alpha=0.10)[1]
    assert ninety is not None and (ninety.lower, ninety.upper) == (
        -5.0,
        3.5,
    )  # the whole range, at 90%


def test_ties_do_not_make_a_sample_large_enough_for_the_level() -> None:
    """5 differences hold 93.75% at most and groups of 3 and 4 hold 94.3%, whatever their ties: the normal
    approximation, which ties call for, gave "2.000 to 3.000" labelled 95%."""
    assert pseudo_median(np.array([2.0, 2.0, 2.0, 2.0, 3.0])) == (2.0, None)
    assert pseudo_median(np.ones(5))[1] is None
    assert hodges_lehmann(np.array([1.0, 1.0, 2.0]), np.array([3.0, 3.0, 3.0, 4.0])) == (-2.0, None)
    assert hodges_lehmann(np.array([1.0, 1.0, 2.0]), np.array([3.0, 3.0, 4.0]))[1] is None
    # one more value, and an interval of 95% exists: 6 differences, groups of 3 and 5
    assert pseudo_median(np.array([2.0, 2.0, 2.0, 2.0, 3.0, 5.0]))[1] is not None
    assert (
        hodges_lehmann(np.array([1.0, 1.0, 2.0]), np.array([3.0, 3.0, 3.0, 4.0, 6.0]))[1]
        is not None
    )
    assert (
        pseudo_median(np.array([2.0, 2.0, 2.0, 2.0, 3.0]), alpha=0.10)[1] is not None
    )  # 90% exists for 5


def test_ties_or_a_group_of_fifty_use_the_normal_approximation() -> None:
    rng = np.random.default_rng(3)
    tied = hodges_lehmann(rng.normal(120, 15, 25).round(), rng.normal(128, 15, 30).round())[1]
    large = hodges_lehmann(rng.lognormal(0, 1, 60), rng.lognormal(0.3, 1, 20))[1]
    untied = hodges_lehmann(rng.lognormal(0, 1, 49), rng.lognormal(0.3, 1, 20))[1]
    assert tied is not None and large is not None and untied is not None
    assert (tied.method, large.method, untied.method) == (APPROXIMATE, APPROXIMATE, EXACT)
    d = rng.normal(-4, 8, 30).round()
    assert (d == 0).any() or len(np.unique(np.abs(d))) < len(d)
    assert pseudo_median(d[d != 0], zeros=True)[1].method == APPROXIMATE
    clean = rng.normal(-4, 8, 30)
    assert pseudo_median(clean)[1].method == EXACT
    assert pseudo_median(clean, zeros=True)[1].method == APPROXIMATE  # a zero was left out: as R


def test_the_interval_is_the_set_of_shifts_the_test_does_not_reject() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.lognormal(0, 1, 60), rng.lognormal(0.5, 1, 80)
    _, ci = hodges_lehmann(x, y)
    assert ci is not None

    def p(shift: float) -> float:
        return float(
            stats.mannwhitneyu(x - shift, y, method="asymptotic", use_continuity=True).pvalue
        )

    step = 1e-9
    assert p(ci.lower - step) <= 0.05 < p(ci.lower + step)
    assert p(ci.upper + step) <= 0.05 < p(ci.upper - step)

    d = rng.standard_t(3, 70) + 0.5
    _, one = pseudo_median(d)
    assert one is not None

    def p1(shift: float) -> float:
        return float(stats.wilcoxon(d - shift, method="approx", correction=True).pvalue)

    assert p1(one.lower - step) <= 0.05 < p1(one.lower + step)
    assert p1(one.upper + step) <= 0.05 < p1(one.upper - step)

    # the exact branch, against the exact tests: small untied samples
    a, b = rng.normal(0, 1, 9), rng.normal(0.8, 1, 12)
    _, small = hodges_lehmann(a, b)
    assert small is not None and small.method == EXACT

    def p2(shift: float) -> float:
        return float(stats.mannwhitneyu(a - shift, b, method="exact").pvalue)

    assert p2(small.lower - step) <= 0.05 < p2(small.lower + step)
    assert p2(small.upper + step) <= 0.05 < p2(small.upper - step)
    few = rng.normal(0.4, 1, 14)
    _, exact = pseudo_median(few)
    assert exact is not None and exact.method == EXACT

    def p3(shift: float) -> float:
        return float(stats.wilcoxon(few - shift, method="exact").pvalue)

    assert p3(exact.lower - step) <= 0.05 < p3(exact.lower + step)
    assert p3(exact.upper + step) <= 0.05 < p3(exact.upper - step)


def test_a_reversed_comparison_is_the_mirror_image() -> None:
    rng = np.random.default_rng(5)
    for x, y in (
        (rng.lognormal(0, 1, 14), rng.lognormal(0.4, 1, 22)),
        (rng.normal(0, 1, 80).round(1), rng.normal(0.3, 1, 65).round(1)),
    ):
        forward, back = hodges_lehmann(x, y), hodges_lehmann(y, x)
        assert forward[0] == -back[0]
        assert (forward[1].lower, forward[1].upper) == (-back[1].upper, -back[1].lower)
        r, s = rank_biserial(x, y), rank_biserial(y, x)
        assert r.estimate == pytest.approx(-s.estimate, abs=1e-15)
        assert (r.ci.lower, r.ci.upper) == pytest.approx((-s.ci.upper, -s.ci.lower), abs=1e-12)


def test_an_order_statistic_found_by_counting_is_the_one_of_the_sorted_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(11)
    cases = []
    for trial in range(60):
        m, n = int(rng.integers(3, 70)), int(rng.integers(3, 70))
        if trial % 2:
            cases.append(
                (rng.normal(0, 1, m).round(1), rng.normal(0.3, 1, n).round(1))
            )  # many ties
        else:
            cases.append((rng.lognormal(0, 1, m), rng.lognormal(0.2, 1, n)))
    written = [(hodges_lehmann(x, y), pseudo_median(x)) for x, y in cases]
    monkeypatch.setattr(ri, "WRITTEN_OUT", 0)  # never write the pairs out: count them
    counted = [(hodges_lehmann(x, y), pseudo_median(x)) for x, y in cases]
    assert counted == written
    for (x, y), ((shift, _), (centre, _)) in zip(cases, written, strict=True):
        assert shift == float(np.median(np.subtract.outer(x, y)))
        walsh = (x[:, None] + x[None, :]) / 2
        assert centre == float(np.median(walsh[np.triu_indices(len(x))]))


@pytest.mark.parametrize(
    "values",
    [
        [0.3, -1.2, 2.5, 0.7, 4.1, -0.4, 1.9],  # no two equal
        [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 5.0, 5.0],  # ties
        [4.0] * 6,  # every value the same
        [-0.0, 0.0, -0.0, 1.0, -1.0, 0.0],  # zeros of both signs
        [
            1e300,
            -1e300,
            1.0,
            5e-324,
            -5e-324,
            0.1,
        ],  # twenty orders of magnitude, and the smallest numbers
        [1.0, 1.0 + 2**-52, 1.0 + 2**-51, 1.0 - 2**-53, 3.0],  # neighbouring floats
    ],
)
def test_every_order_statistic_found_by_counting(values: list[float]) -> None:
    """Every place from the first to the last, of the differences and of the Walsh averages."""
    x = np.array(values)
    y = np.array(values[::-1][:-1]) / 2 + 0.25
    written, rows = ri._differences(x, y)
    expected = np.sort(written())
    counted = rows()
    assert [counted.kth(k) for k in range(1, len(expected) + 1)] == expected.tolist()
    written, rows = ri._walsh(x)
    expected = np.sort(written())
    counted = rows()
    assert [counted.kth(k) for k in range(1, len(expected) + 1)] == expected.tolist()
    with pytest.raises(ValueError, match="no such order statistic"):
        counted.kth(len(expected) + 1)


def test_a_zero_is_given_without_a_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    """-0.04 rounded to one decimal is -0.0: an end printed "-0.000"."""
    x, y = np.array([-0.0, -0.0, -0.0, 1.0, -1.0]), np.array([0.0, 0.0, 0.0, 0.5, -0.5])
    for limit in (ri.WRITTEN_OUT, 0):
        monkeypatch.setattr(ri, "WRITTEN_OUT", limit)
        estimate, _ = hodges_lehmann(x, y)
        assert estimate == 0 and math.copysign(1, estimate) == 1
        centre, _ = pseudo_median(np.array([-0.0, -0.0, -0.0, 2.0, -2.0, -0.0]))
        assert centre == 0 and math.copysign(1, centre) == 1


def test_above_the_real_threshold_the_counted_places_are_those_of_a_full_sort() -> None:
    rng = np.random.default_rng(21)
    x, y = (
        rng.normal(0, 1, 1500).round(2),
        rng.normal(0.1, 1, 1450).round(2),
    )  # 2,175,000 pairs, many tied
    assert len(x) * len(y) > ri.WRITTEN_OUT
    estimate, ci = hodges_lehmann(x, y)
    ordered = np.sort(np.subtract.outer(x, y).ravel())
    pairs = len(ordered)
    assert estimate == (ordered[(pairs + 1) // 2 - 1] + ordered[pairs // 2]) / 2
    total = len(x) + len(y)
    ties = sum(
        float(np.sum(c.astype(float) ** 3 - c))
        for c in (np.unique(v, return_counts=True)[1] for v in (x, y))
    )
    sigma = math.sqrt(pairs / 12 * ((total + 1) - ties / (total * (total - 1))))
    k = math.floor(pairs / 2 + 0.5 - stats.norm.ppf(0.975) * sigma)
    assert (ci.lower, ci.upper) == (ordered[k - 1], ordered[pairs - k])


def test_two_groups_of_twenty_thousand_need_no_table_of_their_differences() -> None:
    rng = np.random.default_rng(2)
    x, y = (
        rng.normal(0, 1, 20_000),
        rng.normal(0.05, 1, 20_000),
    )  # 400 million differences: 3.2 GB written out
    estimate, ci = hodges_lehmann(x, y)
    assert ci is not None and ci.lower < estimate < ci.upper
    assert estimate == pytest.approx(-0.05, abs=0.03) and ci.upper - ci.lower < 0.1


# ----------------------------------------------------------------------------- the rank-biserial r


def test_the_rank_biserial_interval_is_delong_s_on_the_logit_scale() -> None:
    rng = np.random.default_rng(7)
    x, y = rng.normal(0.6, 1, 40), rng.normal(0, 1, 55)
    r = rank_biserial(x, y)
    assert r.ci is not None and r.ci.method == "DeLong, logit scale"
    assert -1 < r.ci.lower < r.estimate < r.ci.upper < 1
    # by hand: A, DeLong's variance from the placements, the logit interval of A, then r = 2A - 1
    v10 = np.array([(np.sum(y < v) + 0.5 * np.sum(y == v)) / len(y) for v in x])
    v01 = np.array([(np.sum(x > v) + 0.5 * np.sum(x == v)) / len(x) for v in y])
    a = v10.mean()
    se = math.sqrt(v10.var(ddof=1) / len(x) + v01.var(ddof=1) / len(y))
    half = stats.norm.ppf(0.975) * se / (a * (1 - a))
    logit = math.log(a / (1 - a))
    expected = [2 / (1 + math.exp(-(logit - half))) - 1, 2 / (1 + math.exp(-(logit + half))) - 1]
    assert r.estimate == pytest.approx(2 * a - 1, abs=1e-12)
    assert (r.ci.lower, r.ci.upper) == pytest.approx(expected, abs=1e-12)
    assert rank_biserial(x, y, interval=False).ci is None
    assert rank_biserial(x, y) == rank_biserial(x, y)  # nothing is drawn


def test_no_interval_when_one_group_is_wholly_above_the_other() -> None:
    apart = rank_biserial([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0])
    assert (
        apart.estimate == 1.0 and apart.ci is None
    )  # A = 1: no standard error to build an interval from
    assert rank_biserial([1.0, 2.0], [3.0, 4.0, 5.0]).ci is None  # a group below 3


# ----------------------------------------------------------------------------- through the tests that report them


def test_the_mann_whitney_result_carries_the_computed_intervals() -> None:
    rng = np.random.default_rng(9)
    groups = {"control": rng.lognormal(1, 1, 45), "drug": rng.lognormal(1.5, 1, 52)}
    result = compare_groups(groups, kind="mannwhitney", reference="control")
    again = compare_groups(groups, kind="mannwhitney", reference="control")
    assert result.estimate_ci == again.estimate_ci and result.estimate_ci.method == APPROXIMATE
    assert result.estimate_label == "Hodges-Lehmann shift (drug - control)"
    assert result.estimate_ci.lower < result.estimate < result.estimate_ci.upper
    assert result.primary_effect.ci.method == "DeLong, logit scale"
    # untied, and the test computed as the interval is (a normal approximation corrected for continuity): the
    # interval leaves 0 out when the test rejects, and the result has nothing to add
    excludes_zero = result.estimate_ci.lower > 0 or result.estimate_ci.upper < 0
    assert excludes_zero == (result.p_value < 0.05)
    assert result.warnings == ["Mann-Whitney U selected manually"]


def test_a_test_that_rejects_beside_an_interval_that_reaches_zero_says_so() -> None:
    """With tied values an end of the interval can be a shift that the test rejects: on the tutorial, hospital
    visits by arm read p = 0.00003 beside "-1.000 to 0.000"."""
    reference = [2, 3, 5, 5, 5, 5, 5, 5, 6, 6, 6, 8]
    comparison = [4, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7]
    result = compare_groups(
        {"reference": reference, "comparison": comparison},
        kind="mannwhitney",
        reference="reference",
    )
    assert result.statistic == 109.5 and result.p_value == pytest.approx(0.027, abs=5e-4)
    assert result.estimate == 1.0
    assert (result.estimate_ci.lower, result.estimate_ci.upper) == (0.0, 2.0)
    said = [w for w in result.warnings if "ends at 0" in w]
    assert len(said) == 1 and "tied values" in said[0] and "rank-biserial r" in said[0]
    # the three cases of the sentence, and the two where the test and the interval agree
    from biostats._types import CI

    def ci(lower: float, upper: float) -> CI:
        return CI(lower=lower, upper=upper, level=0.95, method="")

    assert disagreement(0.20, ci(-1.0, 2.0)) is None and disagreement(0.01, ci(0.5, 2.0)) is None
    assert "ends at 0" in disagreement(0.01, ci(0.0, 2.0))
    assert "holds 0" in disagreement(0.049, ci(-0.01, 2.0))
    assert "leaves 0 out" in disagreement(0.051, ci(0.01, 2.0))
    assert disagreement(None, ci(0.0, 1.0)) is None and disagreement(0.01, None) is None
    # untied groups of 9 and 12: SciPy's test is a normal approximation above 8 values, the interval is exact
    low = [-1.83, -1.63, -0.32, -0.24, -0.23, -0.03, 0.12, 0.65, 1.14]
    high = [-0.63, -0.36, 0.08, 0.16, 0.24, 0.25, 0.28, 0.72, 1.2, 1.79, 2.3, 2.44]
    border = compare_groups({"low": low, "high": high}, kind="mannwhitney", reference="low")
    assert border.p_value == pytest.approx(0.0507, abs=5e-5) and border.estimate_ci.lower > 0
    assert any("part at the border" in w and "leaves 0 out" in w for w in border.warnings)


def test_an_infinite_value_is_ranked_by_the_test_and_left_out_of_the_shift() -> None:
    """A ratio with a zero denominator: above 2,000,000 pairs, where the pairs are counted and not written out, the
    search for an order statistic did not end."""
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 1500), rng.normal(0.1, 1, 1500)
    plain = compare_groups({"a": a[1:], "b": b[1:]}, kind="mannwhitney", reference="b")
    a[0], b[0] = np.inf, -np.inf
    result = compare_groups({"a": a, "b": b}, kind="mannwhitney", reference="b")
    assert (result.estimate, result.estimate_ci) == (plain.estimate, plain.estimate_ci)
    assert result.statistic != plain.statistic  # the test ranks them
    assert any(w.startswith("2 infinite values: the test ranks them") for w in result.warnings)
    assert hodges_lehmann(np.array([np.inf, np.inf]), np.array([1.0, 2.0, 3.0])) == (
        pytest.approx(math.nan, nan_ok=True),
        None,
    )
    paired = wilcoxon_signed_rank([np.inf, 2, 3, 4, 5, 6.5, 7.2], [0, 0.5, 0.2, 0.1, 0.3, 0.9, 0.4])
    assert paired.estimate == pseudo_median(np.array([1.5, 2.8, 3.9, 4.7, 5.6, 6.8]))[0]
    assert paired.warnings[0].startswith("1 infinite difference: the test ranks it as the largest")
    # the counted route refuses a number that is not finite: it can never look for one without end
    rows = ri._Rows(lambda i, j: np.array([1.0, 2.0, np.inf])[j] + 0 * i, np.zeros(2), 3)
    with pytest.raises(ValueError, match="finite"):
        rows.kth(2)


def test_spreads_that_differ_between_groups_of_unequal_size_are_named() -> None:
    """The interval, like the test, assumes two distributions of the same shape: it holds 87% for a nominal 95%
    when the smaller group is the more spread, 98% the other way round."""
    rng = np.random.default_rng(12)
    wide, tight = rng.normal(0, 3, 12), rng.normal(0, 1, 36)
    narrow = compare_groups({"wide": wide, "tight": tight}, kind="mannwhitney")
    said = [w for w in narrow.warnings if w.startswith("The spreads differ")]
    assert (
        len(said) == 1 and "smaller group is the more spread" in said[0] and "too narrow" in said[0]
    )
    assert "rank-biserial r and its interval do not assume it" in said[0]
    other = compare_groups(
        {"wide": rng.normal(0, 3, 36), "tight": rng.normal(0, 1, 12)}, kind="mannwhitney"
    )
    assert any("larger group is the more spread" in w and "too wide" in w for w in other.warnings)
    same_size = compare_groups(
        {"wide": rng.normal(0, 3, 30), "tight": rng.normal(0, 1, 30)}, kind="mannwhitney"
    )
    same_spread = compare_groups(
        {"a": rng.normal(0, 1, 12), "b": rng.normal(0, 1, 36)}, kind="mannwhitney"
    )
    for result in (same_size, same_spread):
        assert not any(w.startswith("The spreads differ") for w in result.warnings)


def test_one_group_wholly_above_the_other_says_why_r_has_no_interval() -> None:
    apart = compare_groups(
        {"low": [1.0, 2.0, 3.0, 4.0], "high": [5.0, 6.0, 7.0, 8.0]}, kind="mannwhitney"
    )
    assert abs(apart.primary_effect.estimate) == 1 and apart.primary_effect.ci is None
    assert any("wholly above the other" in w for w in apart.warnings)


def test_the_signed_rank_tests_carry_the_computed_interval() -> None:
    paired = wilcoxon_signed_rank(DEPRESSION_BEFORE, DEPRESSION_AFTER)
    assert paired.estimate == pytest.approx(0.46, abs=1e-12)
    assert (paired.estimate_ci.lower, paired.estimate_ci.upper) == pytest.approx(
        (0.010, 0.786), abs=1e-12
    )
    assert paired.estimate_ci.method == EXACT
    whole = wilcoxon_signed_rank(
        [140, 150, 135, 160, 150, 145, 155], [130, 150, 130, 150, 140, 150, 140]
    )
    assert (
        whole.estimate_ci.method == APPROXIMATE
    )  # a pair without a difference was left out, and ranks are tied
    assert whole.warnings == [
        "1 pair without a difference was left out, by the signed-rank test's rule: the test, the pseudo-median "
        "and its interval are those of the 6 pairs that differ."
    ]
    assert (
        paired.warnings == []
        and paired.estimate_label == "pseudo-median difference (after - before)"
    )
    # 12 pairs of which 6 do not change: 1.5 is the pseudo-median of the 6 that do (0.5 on every pair)
    half = wilcoxon_signed_rank([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, -1], [0] * 12)
    assert half.estimate == 1.5 and half.n_per_group == [6]
    assert half.warnings[0].startswith("6 pairs without a difference were left out")
    centred = one_sample_wilcoxon([5, 5, 6, 7, 8, 9, 10, 4.5, 3, 12], 5)
    assert centred.estimate_label == "pseudo-median minus reference (5)"
    assert centred.warnings[0].startswith("2 values equal to the reference were left out")
    assert "8 values that differ from it" in centred.warnings[0]
    one = one_sample_wilcoxon(np.array(DEPRESSION_BEFORE) - np.array(DEPRESSION_AFTER) + 1.0, 1.0)
    assert one.estimate == pytest.approx(0.46, abs=1e-12)
    assert (one.estimate_ci.lower, one.estimate_ci.upper) == pytest.approx(
        (0.010, 0.786), abs=1e-12
    )
