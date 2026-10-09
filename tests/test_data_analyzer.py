"""
Unit tests for DataAnalyzer and DatasetProfile.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from utils.data_analyzer import DataAnalyzer, DatasetProfile


@pytest.fixture
def sample_clinical_df():
    """Provides a synthetic DataFrame with mixed data types."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "patient_id": [f"P_{i:03d}" for i in range(n)],
        "age": np.random.randint(20, 80, size=n),
        "bmi": np.random.normal(25.0, 4.0, size=n),
        "gender": np.random.choice(["Male", "Female"], size=n),
        "hypertension": np.random.choice([0, 1], size=n),
        "stage": np.random.choice(["Stage_I", "Stage_II", "Stage_III"], size=n),
        "admission_date": [
            (datetime(2023, 1, 1) + timedelta(days=int(i))).strftime("%Y-%m-%d")
            for i in range(n)
        ],
        "low_card_numeric": np.random.choice([1, 2, 3], size=n),
    })


class TestDataAnalyzer:
    """Test suite for DataAnalyzer column classification and profiling."""

    def test_analyzer_column_categorization(self, sample_clinical_df):
        analyzer = DataAnalyzer(sample_clinical_df)

        assert "age" in analyzer.numeric_cols or "bmi" in analyzer.numeric_cols
        assert "gender" in analyzer.binary_cols
        assert "hypertension" in analyzer.binary_cols
        assert "stage" in analyzer.categorical_cols
        assert "admission_date" in analyzer.date_cols
        assert "low_card_numeric" in analyzer.low_cardinality_numeric_cols

    def test_dataset_profile_immutability(self, sample_clinical_df):
        analyzer = DataAnalyzer(sample_clinical_df)
        profile = analyzer.profile

        assert isinstance(profile, DatasetProfile)
        assert profile.total_rows == 50
        assert profile.total_columns == 8
        assert "gender" in profile.binary_cols

        # Test that dataclass is frozen (immutable)
        with pytest.raises(Exception):
            profile.total_rows = 100

    def test_analyzer_refresh_after_mutation(self, sample_clinical_df):
        analyzer = DataAnalyzer(sample_clinical_df)
        assert "new_metric" not in analyzer.numeric_cols

        # Add new column
        updated_df = sample_clinical_df.copy()
        updated_df["new_metric"] = np.random.normal(100, 15, size=len(updated_df))

        analyzer.refresh(updated_df)
        assert "new_metric" in analyzer.numeric_cols
        assert analyzer.profile.total_columns == 9

    def test_get_suitable_columns(self, sample_clinical_df):
        analyzer = DataAnalyzer(sample_clinical_df)

        hist_cols = analyzer.get_suitable_columns("Histogram")
        assert "x" in hist_cols and "color" in hist_cols
        assert len(hist_cols["x"]) > 0

        scatter_cols = analyzer.get_suitable_columns("Scatter Plot")
        assert "x" in scatter_cols and "y" in scatter_cols

        roc_cols = analyzer.get_suitable_columns("ROC Curve")
        assert "true_class" in roc_cols and "score" in roc_cols
        assert "gender" in roc_cols["true_class"] or "hypertension" in roc_cols["true_class"]

    def test_edge_cases_constant_and_empty(self):
        empty_df = pd.DataFrame(columns=["col_a", "col_b"])
        analyzer = DataAnalyzer(empty_df)
        assert analyzer.profile.total_rows == 0
        assert analyzer.profile.total_columns == 2

        constant_df = pd.DataFrame({
            "constant_num": [5.0] * 10,
            "constant_str": ["Same"] * 10,
        })
        analyzer_const = DataAnalyzer(constant_df)
        assert "constant_num" in analyzer_const.low_cardinality_numeric_cols
        assert analyzer_const.profile.total_rows == 10
