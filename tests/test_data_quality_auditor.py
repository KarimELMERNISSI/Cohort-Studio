"""
Unit tests for DataQualityAuditor (Multi-dimensional Data Quality & Anomaly Profiling).
"""

import numpy as np
import pandas as pd
import pytest

from explore.data_quality_auditor import DataQualityAuditor


@pytest.fixture
def clean_cohort_df() -> pd.DataFrame:
    """Fixture providing a perfectly clean clinical dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "patient_id": [f"ID_{i:04d}" for i in range(n)],
        "age": np.random.randint(30, 75, size=n).astype(float),
        "sbp": np.random.normal(120, 10, size=n),
        "gender": np.random.choice(["Male", "Female"], size=n),
    })


@pytest.fixture
def dirty_cohort_df() -> pd.DataFrame:
    """Fixture providing a dataset with missing values, duplicates, outliers, and formatting issues."""
    data = {
        "id": ["P1", "P2", "P3", "P4", "P4"],  # 1 duplicate row
        "age": [25.0, np.nan, 30.0, 45.0, 45.0],  # 1 missing value
        "glucose": [90.0, 95.0, 92.0, 999.0, 999.0],  # extreme outlier
        "country": ["France", " France", "italy", "Italy", "Italy"],  # whitespace & case inconsistency
    }
    return pd.DataFrame(data)


class TestDataQualityAuditor:
    """Test suite for DataQualityAuditor quality dimensions and audit engine."""

    def test_completeness_perfect_and_missing(self, clean_cohort_df: pd.DataFrame, dirty_cohort_df: pd.DataFrame) -> None:
        """Completeness score reflects ratio of non-null cells."""
        auditor_clean = DataQualityAuditor(clean_cohort_df)
        assert auditor_clean.compute_completeness() == 100.0

        auditor_dirty = DataQualityAuditor(dirty_cohort_df)
        # 20 total cells, 1 NaN -> (1 - 1/20) * 100 = 95.0%
        assert auditor_dirty.compute_completeness() == 95.0

        empty_df = pd.DataFrame()
        assert DataQualityAuditor(empty_df).compute_completeness() == 0.0

    def test_uniqueness_calculation(self, clean_cohort_df: pd.DataFrame, dirty_cohort_df: pd.DataFrame) -> None:
        """Uniqueness score calculates percentage of distinct rows."""
        auditor_clean = DataQualityAuditor(clean_cohort_df)
        assert auditor_clean.compute_uniqueness() == 100.0

        auditor_dirty = DataQualityAuditor(dirty_cohort_df)
        # 5 rows total, 4 unique -> 4/5 * 100 = 80.0%
        assert auditor_dirty.compute_uniqueness() == 80.0

    def test_validity_outlier_detection_methods(self) -> None:
        """Evaluates statistical outlier detection algorithms (IQR, Z-score, Quantile)."""
        # Create a sample with 30 normal observations and 1 extreme outlier
        np.random.seed(42)
        values = list(np.random.normal(100.0, 5.0, size=29)) + [9999.0]
        outlier_df = pd.DataFrame({"marker": values})
        auditor = DataQualityAuditor(outlier_df)

        # IQR method detects 9999.0 outlier
        iqr_score = auditor.compute_validity(method="iqr", params={"multiplier": 1.5})
        assert iqr_score < 100.0

        # Z-score method
        z_score = auditor.compute_validity(method="zscore", params={"threshold": 2.0})
        assert z_score < 100.0

        # Quantile method
        q_score = auditor.compute_validity(method="quantile", params={"lower": 0.05, "upper": 0.95})
        assert q_score < 100.0

    def test_validity_constant_data_edge_case(self) -> None:
        """Constant data has IQR=0 and std=0; should not falsely flag values as outliers."""
        constant_df = pd.DataFrame({"col_a": [10.0] * 50, "col_b": [25.0] * 50})
        auditor = DataQualityAuditor(constant_df)

        assert auditor.compute_validity(method="iqr") == 100.0
        assert auditor.compute_validity(method="zscore") == 100.0

    def test_validity_multivariate_methods(self, clean_cohort_df: pd.DataFrame) -> None:
        """Evaluates Isolation Forest and Local Outlier Factor without crashes."""
        auditor = DataQualityAuditor(clean_cohort_df)
        if_score = auditor.compute_validity(method="Isolation Forest")
        assert 0.0 <= if_score <= 100.0

        lof_score = auditor.compute_validity(method="Local Outlier Factor")
        assert 0.0 <= lof_score <= 100.0

    def test_consistency_check(self) -> None:
        """Consistency identifies text/object columns that could be numeric or datetime."""
        clean_df = pd.DataFrame({"names": ["Alice", "Bob", "Charlie"]})
        assert DataQualityAuditor(clean_df).compute_consistency() == 100.0

        inconsistent_df = pd.DataFrame({
            "numeric_as_text": ["10", "20", "30"],
            "legit_text": ["A", "B", "C"]
        })
        auditor = DataQualityAuditor(inconsistent_df)
        # 1 of 2 text columns is mis-typed -> 50.0%
        assert auditor.compute_consistency() == 50.0

    def test_uniformity_whitespace_and_capitalization(self, dirty_cohort_df: pd.DataFrame) -> None:
        """Uniformity flags trailing whitespace and casing mismatches."""
        auditor = DataQualityAuditor(dirty_cohort_df)
        uniformity = auditor.compute_uniformity()
        assert uniformity < 100.0
        assert len(auditor.uniformity_details) > 0
        issues = [d["Issues"] for d in auditor.uniformity_details]
        assert any("Whitespace" in i or "Capitalization" in i for i in issues)

    def test_clinical_validity_with_and_without_rules(self) -> None:
        """Clinical validity evaluates custom anomaly rules when provided in config."""
        df = pd.DataFrame({
            "sbp": [120, 130, 240, 115],  # 240 is clinically abnormal (> 200)
            "dbp": [80, 85, 130, 78],
        })

        # No config -> returns None (N/A)
        auditor_no_config = DataQualityAuditor(df)
        assert auditor_no_config.compute_clinical_validity() is None

        # Config with clinical anomaly rule formatted as expression
        config = {
            "mask_families": {
                "clinical_anomalies": {
                    "hypertensive_crisis": {
                        "expression": "df['sbp'] > 200"
                    }
                }
            }
        }
        auditor_with_rules = DataQualityAuditor(df, config=config)
        score = auditor_with_rules.compute_clinical_validity()
        # 1 out of 4 rows triggers anomaly -> 75.0%
        assert score == 75.0
        assert auditor_with_rules.clinical_anomalies_df is not None
        assert len(auditor_with_rules.clinical_anomalies_df) == 1

    def test_run_audit_and_generate_advice(self, dirty_cohort_df: pd.DataFrame) -> None:
        """Full audit run compiles all dimension scores and generates actionable advice."""
        auditor = DataQualityAuditor(dirty_cohort_df)
        metrics = auditor.run_audit()

        assert "Completeness" in metrics
        assert "Uniqueness" in metrics
        assert "Statistical Validity" in metrics
        assert "Consistency" in metrics
        assert "Uniformity" in metrics

        assert len(auditor.advice) > 0
        categories = [a["category"] for a in auditor.advice]
        assert "Completeness" in categories or "Uniqueness" in categories

    def test_mcar_missingness_test(self) -> None:
        """Evaluates heuristic Little's MCAR missingness analysis."""
        df_no_missing = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        res = DataQualityAuditor(df_no_missing).perform_mcar_test()
        assert res["is_mcar"] is True
        assert "No missing data" in res["interpretation"]
