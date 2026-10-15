"""
Statistical properties of the whole chain (engine, analysis, pooling) under MAR: bias and coverage.

Slow (a few minutes): run with ``pytest -m slow``. A regression of y on x and z, x missing at random
given y and z; the pooled 95% interval of the coefficient of x must cover the truth in about 95% of
the replications and the estimate must be unbiased, while the complete-case estimate is biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biostats.linear import linear_regression
from biostats.mi import ImputationSpec, analyze_mi, impute

pytestmark = pytest.mark.slow

BETA_X = 0.5
REPLICATIONS = 200


def _replicate(seed: int) -> tuple[float, bool, float]:
    rng = np.random.default_rng(seed)
    n = 300
    z = rng.normal(size=n)
    x = 0.6 * z + rng.normal(scale=0.8, size=n)
    y = 1.0 + BETA_X * x + 0.4 * z + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    # MAR on (y, z): x missing more often when y is high (about 35% missing)
    df.loc[rng.random(n) < 1 / (1 + np.exp(-(-1.0 + 1.2 * (y - y.mean())))), "x"] = np.nan
    run = impute(df, ImputationSpec(columns=["x", "y", "z"], m=10, maxit=5, seed=seed))
    pooled = analyze_mi(lambda d: linear_regression(d, "y", ["x", "z"]), run.mi)
    term = pooled.term("x")
    complete_case = float(pooled.complete_case.params["x"])
    return term.estimate, term.ci.contains(BETA_X), complete_case


def test_pmm_is_unbiased_with_nominal_coverage_under_mar() -> None:
    results = [_replicate(seed) for seed in range(REPLICATIONS)]
    estimates = np.array([r[0] for r in results])
    coverage = float(np.mean([r[1] for r in results]))
    bias = float(estimates.mean() - BETA_X)
    mc_se = float(estimates.std(ddof=1) / np.sqrt(REPLICATIONS))
    assert abs(bias) < 2 * mc_se + 0.01, (bias, mc_se)
    # Monte Carlo SE of a 95% coverage over 200 replications is 1.5 points
    assert 0.91 <= coverage <= 0.99, coverage
    complete_case_bias = float(np.mean([r[2] for r in results]) - BETA_X)
    assert abs(complete_case_bias) > abs(
        bias
    )  # outcome-dependent missingness biases the complete cases
