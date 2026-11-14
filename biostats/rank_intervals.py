"""
The rank-based estimates and their confidence intervals, computed without random draws.

- The Hodges-Lehmann shift between two groups is the median of the pairwise differences. Its interval is two order
  statistics of those differences, D(k) and D(mn + 1 - k), with k from the null distribution of the Mann-Whitney
  statistic (Lehmann 1963; Bauer 1972; Hollander, Wolfe & Chicken 2014, section 4.3). It is the interval that
  belongs to the Mann-Whitney test, under the same assumption (two distributions of the same shape, one shifted):
  it holds the shifts that the test does not reject, and its two ends. With tied values an end can be a shift that
  the test itself rejects, so the interval can touch 0 beside a small p-value (:func:`disagreement`).
- The pseudo-median of paired differences or of one sample is the median of the Walsh averages (the averages of
  every pair of differences, each with itself included). Its interval is the same from the ordered Walsh averages
  and the null distribution of the signed-rank statistic (Hollander, Wolfe & Chicken, section 3.3).

k is exact when both groups are below 50 and no value is tied (one sample: fewer than 50 differences, none tied and
none zero); otherwise it comes from the normal approximation with a continuity correction and the variance corrected
for ties. This is the rule of R's ``wilcox.test(conf.int = TRUE)``, whose large-sample limits are these order
statistics up to the tolerance of its root search. Three things differ from R. The estimate is always the median of
the pairs: in its large-sample branch R searches a root, and reports any point between the two middle pairs when
their number is even. A sample too small for the level gets no interval, with or without ties (5 differences hold
93.75% at most, groups of 3 and 4 hold 94.3%): R gives the whole range, or an interval at a lower level. And the
values that are not finite are left out of the estimate and of its interval (the tests rank them).

Until M8 these intervals were percentile bootstraps of 300 or 1,000 resamples: an end then moved with the seed by
0.15 standard error at 300 resamples, enough for an interval to exclude zero beside a test that did not reject, and
a comparison of two groups of 2,000 took half a minute per variable.

The differences of two groups of 50,000 are 2.5 thousand million numbers: an order statistic is found among them
without writing them out (:class:`_Rows`), by counting how many lie below a value.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from scipy import stats

from biostats._types import CI
from biostats.references import REFERENCES

REFERENCES.setdefault(
    "lehmann1963",
    "Lehmann EL (1963). Nonparametric confidence intervals for a shift parameter. Ann Math Stat 34:1507-1512.",
)
REFERENCES.setdefault(
    "bauer1972",
    "Bauer DF (1972). Constructing confidence sets using rank statistics. JASA 67:687-690.",
)
REFERENCES.setdefault(
    "hollander2014",
    "Hollander M, Wolfe DA, Chicken E (2014). Nonparametric Statistical Methods, 3rd ed. Wiley.",
)

EXACT_BELOW = 50  # group sizes below which the null distribution is enumerated (as R)
EXACT = "exact, from the null distribution of the rank statistic"
APPROXIMATE = "normal approximation of the rank statistic, corrected for continuity and ties"
WRITTEN_OUT = 2_000_000  # pairs up to which the differences are written out and partitioned


# ----------------------------------------------------------------------------- null distributions


@lru_cache(maxsize=128)
def _mann_whitney_counts(m: int, n: int) -> np.ndarray:
    """
    The number of arrangements of two groups of m and n untied values that give each U from 0 to mn, by the
    recurrence f(u; m, n) = f(u - n; m - 1, n) + f(u; m, n - 1): additions only, so a count is exact up to 2^53
    and good to 13 digits above (two groups of 49 reach 2.5e28).
    """
    previous = [np.ones(1) for _ in range(n + 1)]  # i = 0, for each j
    for i in range(1, m + 1):
        row = [np.ones(1)]  # j = 0
        for j in range(1, n + 1):
            out = np.zeros(i * j + 1)
            shifted = previous[j]  # f(.; i - 1, j), moved up by j
            out[j : j + len(shifted)] += shifted
            same = row[j - 1]  # f(.; i, j - 1)
            out[: len(same)] += same
            row.append(out)
        previous = row
    return previous[n]


@lru_cache(maxsize=128)
def _signed_rank_counts(n: int) -> np.ndarray:
    """The number of sets of ranks among 1..n that add up to each value from 0 to n(n + 1)/2."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        counts[rank:] = counts[rank:] + counts[:-rank]
    return counts


def _lower_quantile(counts: np.ndarray, p: float) -> int:
    """The smallest value whose cumulative probability reaches p (R's qwilcox and qsignrank)."""
    cdf = np.cumsum(counts) / counts.sum()
    return int(np.searchsorted(cdf, p * (1 - 64 * np.finfo(float).eps), side="left"))


def _exact_place(counts: np.ndarray, alpha: float) -> int:
    """
    The place k of the interval's ends among the ordered pairs: the interval from the k-th smallest to the k-th
    largest holds its level. 0 when no interval of this level exists (the whole range of the pairs holds less): R
    then reports the whole range with a warning; no interval is given here under a level it does not hold.
    """
    k = _lower_quantile(counts, alpha / 2)
    if k == 0 and 2 * counts[0] / counts.sum() <= alpha * (1 + 1e-12):
        return 1  # the whole range holds the level exactly (two groups of 3 at 90%)
    return k


def _tie_sum(*groups: np.ndarray) -> float:
    """The sum of t^3 - t over the groups of tied values, each array counted on its own."""
    total = 0.0
    for values in groups:
        _, sizes = np.unique(values, return_counts=True)
        total += float(np.sum(sizes.astype(float) ** 3 - sizes))
    return total


# ----------------------------------------------------------------------------- order statistics of pairs


class _Rows:
    """
    Numbers v(i, j) for j from ``first[i]`` to ``last`` (not included), each row increasing with j, never written
    out: the k-th smallest of them all is found by counting, row by row, how many lie at or below a value (a
    bisection on j for every row at once, with the numbers as they are computed, so that the result is the one a
    full sort would give).
    """

    def __init__(
        self,
        value: Callable[[np.ndarray, np.ndarray], np.ndarray],
        first: np.ndarray,
        last: int,
    ) -> None:
        self.value = value
        self.first = first.astype(np.int64)
        self.last = int(last)
        self.rows = np.arange(len(first), dtype=np.int64)
        self.total = int(np.sum(self.last - self.first))

    def _above(self, t: float) -> np.ndarray:
        """For each row, the place of its first number above ``t`` (``last`` when none is)."""
        low = self.first.copy()
        high = np.full(len(low), self.last, dtype=np.int64)
        while True:
            active = low < high
            if not active.any():
                return low
            middle = (low + high) // 2
            values = self.value(self.rows, np.minimum(middle, self.last - 1))
            right = active & (values <= t)
            left = active & ~right
            low = np.where(right, middle + 1, low)
            high = np.where(left, middle, high)

    def kth(self, k: int) -> float:
        """The k-th smallest number (k from 1). The numbers are finite: an infinite one would be looked for
        without end."""
        if not 1 <= k <= self.total:
            raise ValueError("no such order statistic")
        held = self.first < self.last
        top = float(np.max(self.value(self.rows[held], np.full(int(held.sum()), self.last - 1))))
        if not math.isfinite(top):
            raise ValueError("an order statistic is counted among finite numbers")
        bottom, above = -math.inf, self.first.copy()  # no number is at or below ``bottom`` so far
        while True:
            has = above < self.last
            found = float(
                np.min(self.value(self.rows[has], above[has]))
            )  # the smallest above ``bottom``
            if not math.isfinite(found):
                raise ValueError("an order statistic is counted among finite numbers")
            place = self._above(found)
            if int(np.sum(place - self.first)) >= k:
                return found
            bottom, above = found, place
            middle = bottom + (top - bottom) / 2
            if not bottom < middle < top:
                continue  # no number can lie between them: the next one found is ``top``
            place = self._above(middle)
            if int(np.sum(place - self.first)) < k:
                bottom, above = middle, place
            else:
                top = middle


def _picked(
    written: Callable[[], np.ndarray], rows: Callable[[], _Rows], total: int, ks: list[int]
):
    """The order statistics ``ks`` (from 1) of ``total`` numbers: written out and partitioned when they are few,
    counted otherwise. A zero is given without a sign (-0.04 rounded to one decimal is -0.0)."""
    if total <= WRITTEN_OUT:
        values = written()
        part = np.partition(values, sorted({k - 1 for k in ks}))
        return [float(part[k - 1]) + 0.0 for k in ks]
    counted = rows()
    return [counted.kth(k) + 0.0 for k in ks]


def _too_small(arrangements: int, alpha: float) -> bool:
    """
    Whether no interval of level 1 - alpha exists for a sample: its two most extreme arrangements (every value of
    one group above every value of the other; every difference of one sign), 2 of ``arrangements``, are together
    more likely than alpha. Ties do not change that.
    """
    return 2 / arrangements > alpha * (1 + 1e-12)


def _middle(total: int) -> list[int]:
    """The places of the one or two middle numbers, whose mean is the median."""
    return [(total + 1) // 2, total // 2 + 1]


def _differences(
    x: np.ndarray, y: np.ndarray
) -> tuple[Callable[[], np.ndarray], Callable[[], _Rows]]:
    def written() -> np.ndarray:
        return np.subtract.outer(x, y).ravel()

    def rows() -> _Rows:
        down = np.sort(y)[::-1]  # x[i] - down[j] increases with j
        return _Rows(lambda i, j: x[i] - down[j], np.zeros(len(x)), len(y))

    return written, rows


def _walsh(d: np.ndarray) -> tuple[Callable[[], np.ndarray], Callable[[], _Rows]]:
    def written() -> np.ndarray:
        i, j = np.triu_indices(len(d))
        return (d[i] + d[j]) / 2

    def rows() -> _Rows:
        up = np.sort(d)  # (up[i] + up[j]) / 2 increases with j, taken from j = i
        return _Rows(lambda i, j: (up[i] + up[j]) / 2, np.arange(len(d)), len(d))

    return written, rows


# ----------------------------------------------------------------------------- the two estimates


def hodges_lehmann(
    comparison: np.ndarray, reference: np.ndarray, *, alpha: float = 0.05
) -> tuple[float, CI | None]:
    """
    The Hodges-Lehmann shift (comparison - reference) and its interval, of the finite values. No interval when a
    group has fewer than 3 values, or when the groups are too small for an interval of this level, with or without
    ties (two groups of 3 allow 90% at most).
    """
    x, y = np.asarray(comparison, dtype=float), np.asarray(reference, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return math.nan, None
    pairs = m * n
    written, rows = _differences(x, y)
    k = 0
    method = EXACT
    small = m < EXACT_BELOW and n < EXACT_BELOW
    if m >= 3 and n >= 3 and not (small and _too_small(math.comb(m + n, m), alpha)):
        untied = len(np.unique(np.concatenate([x, y]))) == m + n
        if small and untied:
            k = _exact_place(_mann_whitney_counts(m, n), alpha)
        else:
            total = m + n
            sigma = math.sqrt(pairs / 12 * ((total + 1) - _tie_sum(x, y) / (total * (total - 1))))
            k = math.floor(pairs / 2 + 0.5 - stats.norm.ppf(1 - alpha / 2) * sigma)
            method = APPROXIMATE
    places = _middle(pairs) + ([k, pairs + 1 - k] if k >= 1 else [])
    found = _picked(written, rows, pairs, places)
    estimate = (found[0] + found[1]) / 2
    if k < 1:
        return estimate, None
    return estimate, CI(lower=found[2], upper=found[3], level=1 - alpha, method=method)


def pseudo_median(
    differences: np.ndarray, *, alpha: float = 0.05, zeros: bool = False
) -> tuple[float, CI | None]:
    """
    The pseudo-median of ``differences`` (the non-zero ones; ``zeros`` says some were left out, which rules the exact
    distribution out, as in R) and its interval, of the finite ones. No interval below 5 differences, or when they
    are too few for this level, with or without ties (5 differences allow 93.75% at most).
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return math.nan, None
    pairs = n * (n + 1) // 2
    written, rows = _walsh(d)
    k = 0
    method = EXACT
    if n >= 5 and not (n < EXACT_BELOW and _too_small(2**n, alpha)):
        untied = len(np.unique(np.abs(d))) == n
        if n < EXACT_BELOW and untied and not zeros:
            k = _exact_place(_signed_rank_counts(n), alpha)
        else:
            sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - _tie_sum(d) / 48)
            k = math.floor(pairs / 2 + 0.5 - stats.norm.ppf(1 - alpha / 2) * sigma)
            method = APPROXIMATE
    places = _middle(pairs) + ([k, pairs + 1 - k] if k >= 1 else [])
    found = _picked(written, rows, pairs, places)
    estimate = (found[0] + found[1]) / 2
    if k < 1:
        return estimate, None
    return estimate, CI(lower=found[2], upper=found[3], level=1 - alpha, method=method)


def disagreement(p_value: float | None, interval: CI | None, alpha: float = 0.05) -> str | None:
    """
    What a result says when its test and the interval beside it do not say the same of a shift of 0; None when they
    agree. The interval holds the shifts that the test does not reject and its two ends: with tied values an end can
    be a shift that the test itself rejects (on whole numbers, in 5% to 59% of datasets), and without ties the two
    part at the border when one is exact and the other approximate (SciPy's Mann-Whitney test is exact up to 8
    values in a group, the interval below 50).
    """
    if interval is None or p_value is None or not math.isfinite(p_value):
        return None
    rejects, inside = p_value < alpha, interval.lower <= 0 <= interval.upper
    if rejects != inside:
        return None
    read = "The rank-biserial r gives the size of the difference."
    if rejects and 0 in (interval.lower, interval.upper):
        return (
            "The test rejects a shift of 0 and the interval ends at 0: with tied values, an end of the interval can "
            f"be a shift that the test itself rejects. {read}"
        )
    said = (
        "the test rejects a shift of 0 and the interval holds 0"
        if rejects
        else "the test does not reject a shift of 0 and the interval leaves 0 out"
    )
    return (
        f"The test and the interval disagree at the border: {said}. One of the two is exact and the other a "
        "normal "
        f"approximation, or values are tied. {read}"
    )


__all__ = ["APPROXIMATE", "EXACT", "disagreement", "hodges_lehmann", "pseudo_median"]
