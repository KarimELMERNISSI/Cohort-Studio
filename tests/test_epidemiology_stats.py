"""
Unit Tests for Epidemiological Statistics and Biostatistical Engines.

Validates:
- 2x2 contingency analysis (OR, RR, RD, NNT, Woolf/Katz CIs, Haldane-Anscombe correction)
- Effect sizes with confidence intervals (Cohen's d, Hedges' g, η², ω²)
- ANCOVA assumption diagnostics (homogeneity of regression slopes, normality, homoscedasticity)
- Clinical diagnostic test accuracy metrics (Sensitivity, Specificity, PPV, NPV, Wilson CIs)
"""

import pytest
import numpy as np
import pandas as pd
from utils.epidemiology_utils import (
    calculate_2x2_epidemiology_metrics,
    compute_cohens_d_with_ci,
    compute_eta_and_omega_squared,
    check_ancova_assumptions,
    calculate_diagnostic_accuracy,
)


class TestContingency2x2Epidemiology:
    """Tests for 2x2 table metrics and epidemiological measures."""

    def test_standard_2x2_cohort_trial(self):
        # Benchmark clinical data: Treated vs Placebo on Event
        # a=20 (Treated Event), b=80 (Treated No Event) -> Risk = 20/100 = 0.20
        # c=40 (Placebo Event), d=60 (Placebo No Event) -> Risk = 40/100 = 0.40
        res = calculate_2x2_epidemiology_metrics((20, 80, 40, 60))

        assert res.n_total == 200
        assert not res.haldane_applied
        
        # RR should be 0.20 / 0.40 = 0.50
        assert pytest.approx(res.relative_risk, rel=1e-2) == 0.50
        assert res.relative_risk_ci.lower < res.relative_risk < res.relative_risk_ci.upper

        # OR should be (20*60) / (80*40) = 1200 / 3200 = 0.375
        assert pytest.approx(res.odds_ratio, rel=1e-2) == 0.375
        assert res.odds_ratio_ci.lower < res.odds_ratio < res.odds_ratio_ci.upper

        # RD = 0.20 - 0.40 = -0.20
        assert pytest.approx(res.risk_difference, abs=1e-2) == -0.20
        
        # NNT = 1 / |-0.20| = 5.0
        assert pytest.approx(res.nnt, abs=1e-2) == 5.0

        # Significance
        assert res.chi2_pvalue < 0.01
        assert res.fisher_exact_pvalue < 0.01

    def test_zero_cell_haldane_anscombe_correction(self):
        # Cell with 0: a=15, b=0, c=5, d=20
        res = calculate_2x2_epidemiology_metrics((15, 0, 5, 20))
        assert res.haldane_applied is True
        # Check that OR and RR are finite numbers and do not raise ZeroDivisionError
        assert np.isfinite(res.odds_ratio)
        assert np.isfinite(res.relative_risk)
        assert res.odds_ratio > 0
        assert any("Haldane-Anscombe" in g for g in res.guidance)

    def test_rare_disease_guidance(self):
        # Rare disease: a=2, b=998, c=1, d=999 (prevalence ~ 0.15% < 10%)
        res = calculate_2x2_epidemiology_metrics((2, 998, 1, 999))
        assert any("Rare disease assumption holds" in g for g in res.guidance)


class TestEffectSizeCalculations:
    """Tests for standardized effect sizes with confidence intervals."""

    def test_cohens_d_and_hedges_g(self):
        np.random.seed(42)
        group1 = np.random.normal(loc=110, scale=15, size=50)
        group2 = np.random.normal(loc=100, scale=15, size=50)

        res = compute_cohens_d_with_ci(group1, group2)
        assert res.metric == "Cohen's d"
        assert res.estimate > 0.4  # Expected d around 0.67
        assert res.ci is not None
        assert res.ci.lower < res.estimate < res.ci.upper
        # Hedges' g should be slightly smaller due to small-sample correction factor J
        assert res.hedges_g <= res.estimate

    def test_identical_groups_zero_effect(self):
        group = [10.0, 10.0, 10.0, 10.0]
        res = compute_cohens_d_with_ci(group, group)
        assert res.estimate == 0.0

    def test_eta_and_omega_squared(self):
        g1 = [10, 12, 11, 13, 12]
        g2 = [15, 16, 14, 17, 15]
        g3 = [20, 22, 21, 23, 22]

        res = compute_eta_and_omega_squared([g1, g2, g3])
        assert 0.0 < res["eta_squared"] <= 1.0
        assert 0.0 < res["omega_squared"] <= 1.0
        # Omega squared is less biased than eta squared
        assert res["omega_squared"] <= res["eta_squared"]


class TestANCOVADiagnostics:
    """Tests for ANCOVA epidemiological assumption checks."""

    def test_homogeneity_of_slopes_holds_when_parallel(self):
        np.random.seed(123)
        n = 100
        covariate = np.random.normal(50, 10, n)
        group = np.random.choice(["A", "B"], n)
        
        # Parallel model: identical slope of 0.8 for both groups
        noise = np.random.normal(0, 2, n)
        target = 10.0 + (5.0 * (group == "B")) + 0.8 * covariate + noise

        df = pd.DataFrame({"target": target, "group": group, "cov": covariate})
        diag = check_ancova_assumptions(df, target="target", group="group", covariates=["cov"])

        assert diag.slopes_homogeneous is True
        assert diag.homogeneity_of_slopes_pvalue is not None
        assert diag.homogeneity_of_slopes_pvalue >= 0.05

    def test_homogeneity_of_slopes_violation_detected(self):
        np.random.seed(456)
        n = 150
        covariate = np.random.uniform(20, 80, n)
        group = np.random.choice(["A", "B"], n)
        
        # Strong interaction: slope is 0.2 for A, but 2.5 for B
        slope = np.where(group == "B", 2.5, 0.2)
        noise = np.random.normal(0, 3, n)
        target = 10.0 + slope * covariate + noise

        df = pd.DataFrame({"target": target, "group": group, "cov": covariate})
        diag = check_ancova_assumptions(df, target="target", group="group", covariates=["cov"])

        assert diag.slopes_homogeneous is False
        assert diag.homogeneity_of_slopes_pvalue < 0.05
        assert any("Violation of Parallel Slopes" in w for w in diag.warnings)


class TestDiagnosticTestAccuracy:
    """Tests for diagnostic testing accuracy and Wilson score intervals."""

    def test_diagnostic_accuracy_metrics(self):
        # 100 diseased: 90 TP, 10 FN -> Sensitivity = 90%
        # 100 healthy:  85 TN, 15 FP -> Specificity = 85%
        res = calculate_diagnostic_accuracy(tp=90, fp=15, fn=10, tn=85)

        assert pytest.approx(res.sensitivity, abs=1e-3) == 0.90
        assert res.sensitivity_ci.lower < 0.90 < res.sensitivity_ci.upper

        assert pytest.approx(res.specificity, abs=1e-3) == 0.85
        assert res.specificity_ci.lower < 0.85 < res.specificity_ci.upper

        # PPV = 90 / (90 + 15) = 90 / 105 ~ 0.857
        assert pytest.approx(res.ppv, abs=1e-2) == 0.857

        # NPV = 85 / (85 + 10) = 85 / 95 ~ 0.895
        assert pytest.approx(res.npv, abs=1e-2) == 0.895

        # Youden index J = 0.90 + 0.85 - 1 = 0.75
        assert pytest.approx(res.youden_index, abs=1e-2) == 0.75
        assert res.lr_positive > 1.0
        assert res.lr_negative < 1.0
