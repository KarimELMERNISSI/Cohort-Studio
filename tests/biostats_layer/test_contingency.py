"""2x2 orientation and measures of association (audit S1). Reference values computed with scipy 1.18 /
statsmodels 0.15 for Placebo 40/100 events vs Treatment 15/100 events."""

import numpy as np
import pandas as pd
import pytest

from biostats.contingency import (
    TwoByTwo,
    analyze_twobytwo,
    detect_event_level,
    detect_reference_level,
)


@pytest.fixture
def trial_df() -> pd.DataFrame:
    rows = (
        [("Placebo", "Yes")] * 40
        + [("Placebo", "No")] * 60
        + [("Treatment", "Yes")] * 15
        + [("Treatment", "No")] * 85
    )
    return pd.DataFrame(rows, columns=["arm", "event"])


class TestDetection:
    def test_event_and_reference_tokens(self) -> None:
        assert detect_event_level(["No", "Yes"]) == "Yes"
        assert detect_event_level([0, 1]) == 1
        assert detect_event_level([False, True]) is True
        assert detect_event_level(["A", "B"]) is None
        assert detect_reference_level(["Treatment", "Placebo"]) == "Placebo"
        assert detect_reference_level([0, 1]) == 0
        assert detect_reference_level(["A", "B"]) is None


class TestOrientation:
    def test_explicit_exposure_placebo(self, trial_df: pd.DataFrame) -> None:
        t = TwoByTwo.from_crosstab(trial_df, "arm", "event", exposure_level="Placebo")
        assert (t.a, t.b, t.c, t.d) == (40, 60, 15, 85)
        assert t.exposure_level == "Placebo" and t.reference_level == "Treatment"
        assert t.event_level == "Yes" and t.nonevent_level == "No"
        assert t.as_frame().iloc[2, 2] == 200

    def test_auto_detection_uses_placebo_as_reference(self, trial_df: pd.DataFrame) -> None:
        t = TwoByTwo.from_crosstab(trial_df, "arm", "event")
        assert t.reference_level == "Placebo" and t.exposure_level == "Treatment"
        assert (t.a, t.b, t.c, t.d) == (15, 85, 40, 60)
        res = analyze_twobytwo(t)
        assert res.relative_risk == pytest.approx(0.375, rel=1e-6)

    def test_sorted_01_outcome_is_not_inverted(self) -> None:
        df = pd.DataFrame(
            {"g": ["A"] * 100 + ["B"] * 100, "y": [1] * 40 + [0] * 60 + [1] * 15 + [0] * 85}
        )
        t = TwoByTwo.from_crosstab(df, "g", "y")
        assert t.event_level == "1" and t.reference_level == "A" and t.exposure_level == "B"
        assert (t.a, t.b, t.c, t.d) == (15, 85, 40, 60)
        res = analyze_twobytwo(t)
        assert res.prevalence_event == pytest.approx(0.275)  # was reported as 72.5 % before the fix
        assert res.risk_reference == pytest.approx(0.40) and res.risk_exposed == pytest.approx(0.15)

    def test_level_overrides_accept_strings(self) -> None:
        df = pd.DataFrame({"g": [0, 0, 1, 1, 0, 1], "y": [1, 0, 1, 1, 0, 0]})
        t = TwoByTwo.from_crosstab(df, "g", "y", exposure_ref="1", event_level="0")
        assert t.reference_level == "1" and t.exposure_level == "0" and t.event_level == "0"

    def test_requires_two_levels(self) -> None:
        df = pd.DataFrame({"g": ["A", "B", "C"], "y": [1, 0, 1]})
        with pytest.raises(ValueError, match="exactly two levels"):
            TwoByTwo.from_crosstab(df, "g", "y")


class TestMeasures:
    def test_reference_values(self, trial_df: pd.DataFrame) -> None:
        res = analyze_twobytwo(
            TwoByTwo.from_crosstab(trial_df, "arm", "event", exposure_level="Placebo")
        )
        assert res.relative_risk == pytest.approx(2.6667, abs=1e-4)
        assert (res.relative_risk_ci.lower, res.relative_risk_ci.upper) == pytest.approx(
            (1.5780, 4.5065), abs=1e-3
        )
        assert res.odds_ratio == pytest.approx(3.7778, abs=1e-4)
        assert (res.odds_ratio_ci.lower, res.odds_ratio_ci.upper) == pytest.approx(
            (1.9154, 7.4511), abs=1e-3
        )
        assert res.risk_difference == pytest.approx(0.25)
        assert res.risk_difference_ci.method.startswith("Newcombe")
        assert (res.risk_difference_ci.lower, res.risk_difference_ci.upper) == pytest.approx(
            (0.1272, 0.3633), abs=1e-3
        )
        assert res.nnt == pytest.approx(4.0)
        assert res.nnt_ci is not None
        assert (res.nnt_ci.lower, res.nnt_ci.upper) == pytest.approx((2.75, 7.86), abs=0.01)
        assert res.nnt_label.startswith("NNH")
        assert res.prevalence_event == pytest.approx(0.275)
        assert res.chi2_statistic == pytest.approx(15.674, abs=1e-3)
        assert res.chi2_pvalue == pytest.approx(7.53e-5, rel=1e-2)
        assert res.fisher_exact_pvalue == pytest.approx(1.197e-4, rel=1e-2)
        assert res.primary_test.startswith("Pearson") and res.primary_pvalue == res.chi2_pvalue
        assert res.attributable_fraction_exposed == pytest.approx(0.625, abs=1e-4)
        assert not res.haldane_applied

    def test_wald_option_and_legacy_dict(self, trial_df: pd.DataFrame) -> None:
        res = analyze_twobytwo(
            TwoByTwo.from_crosstab(trial_df, "arm", "event", exposure_level="Placebo"), rd_ci="wald"
        )
        assert res.risk_difference_ci.method == "Wald"
        assert (res.risk_difference_ci.lower, res.risk_difference_ci.upper) == pytest.approx(
            (0.1312, 0.3688), abs=1e-3
        )
        payload = res.legacy_dict()
        for key in (
            "a",
            "b",
            "c",
            "d",
            "n_total",
            "odds_ratio",
            "odds_ratio_ci",
            "relative_risk",
            "risk_difference",
            "nnt",
            "guidance",
        ):
            assert key in payload
        assert set(payload["odds_ratio_ci"]) == {"lower", "upper", "level", "method"}

    def test_zero_cell_haldane(self) -> None:
        res = analyze_twobytwo(TwoByTwo(a=15, b=0, c=5, d=20))
        assert res.haldane_applied and np.isfinite(res.odds_ratio) and res.odds_ratio > 0
        assert any("Haldane-Anscombe" in g for g in res.guidance)
        # expected counts are 7.5/7.5/12.5/12.5 here, so Pearson remains the primary test
        assert res.expected_min == pytest.approx(7.5) and res.primary_test.startswith("Pearson")

    def test_small_expected_counts_make_fisher_primary(self) -> None:
        res = analyze_twobytwo(TwoByTwo(a=3, b=1, c=0, d=4))
        assert res.expected_min < 5
        assert (
            res.primary_test.startswith("Fisher") and res.primary_pvalue == res.fisher_exact_pvalue
        )
        assert any("expected count" in g for g in res.guidance)

    def test_unbounded_nnt_when_rd_interval_includes_zero(self) -> None:
        res = analyze_twobytwo(TwoByTwo(a=6, b=14, c=5, d=15))
        assert res.nnt is not None and res.nnt_ci is None
        assert any("not bounded" in g for g in res.guidance)

    def test_case_control_guidance(self) -> None:
        res = analyze_twobytwo(TwoByTwo(a=30, b=70, c=10, d=90), design="case-control")
        assert any("Case-control design" in g for g in res.guidance)

    def test_empty_group_rejected(self) -> None:
        with pytest.raises(ValueError):
            analyze_twobytwo(TwoByTwo(a=0, b=0, c=5, d=5))
