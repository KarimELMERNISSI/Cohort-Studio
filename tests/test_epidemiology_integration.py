"""
Integration Tests for Epidemiology and Biostatistics Pipelines.

Tests end-to-end analytical pipelines:
- `analyze_variable` with automatic test recommendation and manual overrides
- Integration of 2x2 epidemiological contingency metrics and Cohen's d CIs
- Post-hoc pairwise analysis and multiple testing corrections
- Missingness pattern analysis (Little's MCAR test)
"""

import pytest
import numpy as np
import pandas as pd
from utils.analysis_utils import (
    analyze_variable,
    recommend_statistical_test,
    perform_post_hoc,
    littles_mcar_test,
)
import statsmodels.stats.multitest as smt


@pytest.fixture
def clinical_cohort_df():
    """Generates a representative clinical cohort DataFrame."""
    np.random.seed(101)
    n = 120

    # Group / Exposure: Treatment (T) vs Placebo (P)
    group = np.random.choice(["Treatment", "Placebo"], size=n, p=[0.5, 0.5])
    
    # Biomarker (continuous, higher in treatment)
    biomarker = np.where(group == "Treatment", 
                         np.random.normal(loc=12.5, scale=2.0, size=n), 
                         np.random.normal(loc=10.0, scale=2.0, size=n))
    
    # Multiclass Stage (3 groups)
    stage = np.random.choice(["Stage_I", "Stage_II", "Stage_III"], size=n, p=[0.4, 0.35, 0.25])
    
    # Blood Pressure across stages
    sbp = np.where(stage == "Stage_I", np.random.normal(120, 10, n),
          np.where(stage == "Stage_II", np.random.normal(135, 12, n),
                                       np.random.normal(150, 15, n)))
    
    # Binary clinical event (hypertension: Yes/No, more frequent in Placebo)
    event_prob = np.where(group == "Placebo", 0.40, 0.15)
    adverse_event = np.random.binomial(1, event_prob, n)
    event_label = np.where(adverse_event == 1, "Yes", "No")

    return pd.DataFrame({
        "group": group,
        "biomarker": biomarker,
        "stage": stage,
        "sbp": sbp,
        "adverse_event": event_label,
    })


class TestEpidemiologyAnalysisPipeline:
    """Tests analytical pipelines executed during univariate analysis."""

    def test_two_group_continuous_pipeline(self, clinical_cohort_df):
        df = clinical_cohort_df
        res = analyze_variable(
            df=df,
            group_col="group",
            target="biomarker",
            numeric_cols=["biomarker", "sbp"],
            categorical_cols=["stage"],
            binary_cols=["adverse_event"],
            manual_test="Student's t-test",
        )

        assert res["Variable"] == "biomarker"
        assert res["Test Used"] == "Student's t-test"
        assert res["P-Value"] < 0.001
        assert res["Effect Type"] == "Cohen's d"
        assert abs(res["Effect Size"]) > 0.8  # Strong effect
        assert "N1" in res["Sample Info"]
        assert "N2" in res["Sample Info"]

    def test_multi_group_anova_and_posthoc(self, clinical_cohort_df):
        df = clinical_cohort_df
        res = analyze_variable(
            df=df,
            group_col="stage",
            target="sbp",
            numeric_cols=["biomarker", "sbp"],
            categorical_cols=["group"],
            binary_cols=["adverse_event"],
            manual_test="ANOVA",
        )

        assert res["Test Used"] == "ANOVA"
        assert res["P-Value"] < 0.001
        assert res["Effect Type"] == "Eta Squared"
        assert res["Effect Size"] > 0.14  # Large effect

        # Post-hoc pairwise comparison
        ph = perform_post_hoc(df, group_col="stage", target_col="sbp", test_type="ANOVA")
        assert ph is not None
        assert not ph.empty
        assert "group1" in ph.columns and "group2" in ph.columns
        assert "p-adj" in ph.columns

    def test_categorical_2x2_epidemiology_pipeline(self, clinical_cohort_df):
        df = clinical_cohort_df
        res = analyze_variable(
            df=df,
            group_col="group",
            target="adverse_event",
            numeric_cols=["biomarker", "sbp"],
            categorical_cols=["stage"],
            binary_cols=["group", "adverse_event"],
            manual_test="Auto-Detect",
        )

        assert res["Variable"] == "adverse_event"
        assert res["Test Used"] in ["Chi-Square", "Fisher's Exact"]
        assert "epi_2x2" in res
        
        epi = res["epi_2x2"]
        assert "odds_ratio" in epi
        assert "relative_risk" in epi
        assert "risk_difference" in epi
        assert "odds_ratio_ci" in epi
        assert "relative_risk_ci" in epi
        assert epi["n_total"] == len(df)

    def test_test_recommendation_engine(self, clinical_cohort_df):
        df = clinical_cohort_df
        # 2 groups + continuous -> T-test or Mann-Whitney
        rec_test, reasoning = recommend_statistical_test(
            df, group_col="group", target_col="biomarker",
            numeric_cols=["biomarker", "sbp"],
            categorical_cols=["stage"],
            binary_cols=["adverse_event"],
        )
        assert rec_test in ["Student's t-test", "Mann-Whitney U", "Welch's t-test"]
        assert len(reasoning) > 0

    def test_multiple_testing_correction(self):
        # 10 hypothetical p-values with several false-positive candidates
        raw_pvalues = [0.001, 0.008, 0.024, 0.045, 0.055, 0.12, 0.35, 0.50, 0.80, 0.95]
        
        # Bonferroni
        _, p_bonf, _, _ = smt.multipletests(raw_pvalues, method="bonferroni")
        assert p_bonf[0] == 0.010  # 0.001 * 10
        assert p_bonf[1] == 0.080  # 0.008 * 10

        # Benjamini-Hochberg (FDR)
        _, p_fdr, _, _ = smt.multipletests(raw_pvalues, method="fdr_bh")
        assert (p_fdr <= p_bonf).all()

    def test_littles_mcar_test(self):
        np.random.seed(99)
        data = np.random.normal(size=(50, 4))
        df_mcar = pd.DataFrame(data, columns=["A", "B", "C", "D"])
        
        # Inject completely random missingness
        mask = np.random.binomial(1, 0.1, size=df_mcar.shape).astype(bool)
        df_mcar[mask] = np.nan

        res = littles_mcar_test(df_mcar)
        assert "p_value" in res
        assert "statistic" in res
        assert "result" in res
