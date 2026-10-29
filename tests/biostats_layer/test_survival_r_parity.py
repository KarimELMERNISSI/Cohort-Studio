"""
biostats.survival's quantiles of a Kaplan-Meier curve and their limits against R's survfit (log-log band) and
quantile.survfit.

The reference values come from tests/reference/r/make_survival_fixtures.R, run by the manual workflow
.github/workflows/r-fixtures.yml; until they are in tests/reference/fixtures/survival these tests are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biostats.survival import kaplan_meier, km_quantile

REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "fixtures"
    / "survival"
    / "km_quantiles.json"
)

pytestmark = pytest.mark.skipif(
    not REFERENCE.is_file(),
    reason="R reference values not generated yet (run the 'R reference fixtures' workflow)",
)


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _cases() -> list[str]:
    if not REFERENCE.is_file():
        return []
    return list(json.loads(REFERENCE.read_text(encoding="utf-8"))["cases"])


@pytest.mark.parametrize("case", _cases())
def test_the_quantiles_and_their_limits_are_rs(reference: dict, case: str) -> None:
    r = reference["cases"][case]
    curve = kaplan_meier(r["time"], r["event"])
    last = max(r["time"])
    for i, p in enumerate(r["probs"]):
        ours = (
            km_quantile(curve.times, curve.survival, p, last=last),
            km_quantile(curve.times, curve.lower, p, last=last),
            km_quantile(curve.times, curve.upper, p, last=last),
        )
        for name, value, expected in zip(
            ("quantile", "lower", "upper"),
            ours,
            (r["quantile"][i], r["lower"][i], r["upper"][i]),
            strict=True,
        ):
            if expected is None:
                assert value is None, (case, p, name)
            else:
                assert value == pytest.approx(expected, abs=1e-9), (case, p, name)
