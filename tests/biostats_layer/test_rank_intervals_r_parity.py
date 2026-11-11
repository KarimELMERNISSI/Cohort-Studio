"""
The rank-based intervals against R's stats::wilcox.test(conf.int = TRUE): the Hodges-Lehmann shift of two groups and
the pseudo-median of one sample or of paired values.

The reference values come from tests/reference/r/make_rank_fixtures.R, run by the manual workflow
.github/workflows/r-fixtures.yml; until they are in tests/reference/fixtures/rank these tests are skipped. Each case
holds what R decided (its method, the level its interval holds, its warnings) and the two middle pairs from R's own
sort. What is compared, and where the two are not expected to give the same number:

- The estimate here is always the median of the pairs: it is compared with R's two middle pairs. R's own estimate
  is that median in its exact branch; in its normal branch it is the root of a step function, found within 1e-4,
  and any point between the two middle pairs when their number is even.
- The ends: the same order statistics in R's exact branch, R's roots (within its tolerance) in its normal branch.
- A sample too small for the level (5 differences, groups of 3 and 4) gets no interval here; R gives the whole
  range, at a lower level or under the level asked for.
- Where R lowered its level (a few tied differences), its interval lies inside the one given here at the level
  asked for.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from biostats._types import CI
from biostats.rank_intervals import APPROXIMATE, EXACT, hodges_lehmann, pseudo_median

REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "fixtures"
    / "rank"
    / "wilcox_reference.json"
)

pytestmark = pytest.mark.skipif(
    not REFERENCE.is_file(),
    reason="R reference values not generated yet (run the 'R reference fixtures' workflow)",
)
ROOT = 1e-3  # of R's root search (tol 1e-4 on the argument), with some room
SAME = {"rel": 1e-12, "abs": 1e-12}  # the same order statistic, computed on both sides


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _numbers(values: list[str]) -> np.ndarray:
    return np.array([float(v) for v in values], dtype=float)


def _compare(
    case: dict[str, Any], estimate: float, ci: CI | None, *, too_small: bool, shift: float = 0.0
) -> None:
    name = case["name"]
    low, high = (float(v) for v in case["middle"])
    r_estimate, r_lower, r_upper = (float(case[key]) for key in ("estimate", "lower", "upper"))
    exact = "exact" in case["method"]
    near = SAME if exact else {"abs": ROOT}

    assert estimate + shift == pytest.approx((low + high) / 2, **SAME), name
    if exact or case["pairs"] % 2:
        assert estimate + shift == pytest.approx(r_estimate, **near), name
    else:
        assert low - ROOT <= r_estimate <= high + ROOT, (
            name
        )  # R stops anywhere between the two middle pairs

    if too_small:
        assert ci is None, name  # R: the whole range, which does not hold the level asked for
        return
    assert ci is not None, name
    assert ci.level == pytest.approx(case["conf_level"]), name
    lower, upper = ci.lower + shift, ci.upper + shift
    if case["achieved_level"] < case["conf_level"] - 1e-9:
        # R lowered its level: its narrower interval lies inside the one of the level asked for
        assert lower <= r_lower + ROOT and upper >= r_upper - ROOT, (name, case["warnings"])
        return
    assert ci.method == (EXACT if exact else APPROXIMATE), name
    assert (lower, upper) == pytest.approx((r_lower, r_upper), **near), name


def test_the_shift_of_two_groups_matches_r(reference: dict[str, Any]) -> None:
    assert len(reference["two_sample"]) >= 25
    for case in reference["two_sample"]:
        x, y = _numbers(case["x"]), _numbers(case["y"])
        alpha = 1 - case["conf_level"]
        m, n = len(x), len(y)
        assert case["pairs"] == m * n
        too_small = min(m, n) < 3 or (m < 50 and n < 50 and 2 / comb(m + n, m) > alpha)
        estimate, ci = hodges_lehmann(x, y, alpha=alpha)
        _compare(case, estimate, ci, too_small=too_small)


def test_the_pseudo_median_matches_r(reference: dict[str, Any]) -> None:
    assert len(reference["one_sample"]) >= 24
    for case in reference["one_sample"]:
        x, mu = _numbers(case["x"]), float(case["mu"])
        d = x - _numbers(case["y"]) if "y" in case else x - mu
        nonzero = d[d != 0]
        alpha = 1 - case["conf_level"]
        n = len(nonzero)
        assert (case["differences"], case["nonzero"]) == (len(d), n)
        assert case["pairs"] == n * (n + 1) // 2
        too_small = n < 5 or (n < 50 and 2 / 2**n > alpha)
        estimate, ci = pseudo_median(nonzero, alpha=alpha, zeros=n < len(d))
        _compare(case, estimate, ci, too_small=too_small, shift=mu)
