"""
Rates against R's stats::poisson.test: the exact interval of one rate, and the exact conditional interval of a
rate ratio with its central p-value.

The reference values come from tests/reference/r/make_rates_fixtures.R, run by the manual workflow
.github/workflows/r-fixtures.yml; until they are in tests/reference/fixtures/rates these tests are skipped.
poisson.test reports a minimum-likelihood p-value, which can disagree with its own interval, so the script also
computes the central one (twice the smaller tail), which is the p-value of biostats.rates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from biostats.rates import incidence_rate, rate_ratio

REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "fixtures"
    / "rates"
    / "poisson_reference.json"
)

pytestmark = pytest.mark.skipif(
    not REFERENCE.is_file(),
    reason="R reference values not generated yet (run the 'R reference fixtures' workflow)",
)


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _number(value: Any) -> float:
    return float(value)  # jsonlite writes the infinities as "Inf"


def test_the_rate_ratio_matches_r(reference: dict[str, Any]) -> None:
    for case in reference["rate_ratio"]:
        rr = rate_ratio(case["d1"], case["t1"], case["d0"], case["t0"])
        assert rr.ratio == pytest.approx(_number(case["ratio"]), rel=1e-9), case
        assert rr.ci.lower == pytest.approx(_number(case["lower"]), rel=1e-7, abs=1e-12), case
        assert rr.ci.upper == pytest.approx(_number(case["upper"]), rel=1e-7), case
        assert rr.p_value == pytest.approx(_number(case["p_central"]), rel=1e-7), case


def test_the_rate_interval_matches_r(reference: dict[str, Any]) -> None:
    for case in reference["rate"]:
        rate = incidence_rate(case["events"], case["time"], per=1)
        assert rate.ci.lower == pytest.approx(_number(case["lower"]), rel=1e-7, abs=1e-12), case
        assert rate.ci.upper == pytest.approx(_number(case["upper"]), rel=1e-7), case
