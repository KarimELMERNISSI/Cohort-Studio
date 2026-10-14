"""The sorted-search DeLong placements equal the pairwise definition (ties count one half)."""

from __future__ import annotations

import numpy as np
import pytest

from biostats.diagnostic import delong_auc


def _pairwise(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    pos = labels == 1
    x, y = scores[pos], scores[~pos]
    m, n = len(x), len(y)
    v10 = np.array([(np.sum(y < xi) + 0.5 * np.sum(y == xi)) / n for xi in x])
    v01 = np.array([(np.sum(x > yj) + 0.5 * np.sum(x == yj)) / m for yj in y])
    return float(v10.mean()), float(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n)


@pytest.mark.parametrize("n", [30, 500])
def test_placements_match_the_pairwise_definition_with_ties(n: int) -> None:
    rng = np.random.default_rng(n)
    labels = rng.integers(0, 2, n)
    scores = np.round(rng.normal(labels * 0.7, 1, n), 1)  # rounding creates ties
    result = delong_auc(labels, scores, pos_label=1)
    auc, variance = _pairwise(labels, scores)
    assert result["auc"] == pytest.approx(auc, abs=1e-12)
    assert result["variance"] == pytest.approx(variance, abs=1e-12)


def test_perfect_separation_and_identical_scores() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert delong_auc(labels, np.array([1, 2, 3, 4, 5, 6]), pos_label=1)["auc"] == 1.0
    assert delong_auc(labels, np.zeros(6), pos_label=1)["auc"] == 0.5
