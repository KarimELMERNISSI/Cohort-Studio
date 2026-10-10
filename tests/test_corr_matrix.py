"""
Unit Tests for Correlation Matrix and Graph Utilities (explore/corr_matrix.py).

Tests correlation computation, Excel workbook generation with conditional formatting,
sheet name sanitization, openpyxl chart export, and yFiles network graph generation.
"""

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

import explore.corr_matrix as ecm


@pytest.fixture
def clinical_df() -> pd.DataFrame:
    """Fixture providing clinical cohort DataFrame with numeric and categorical features."""
    return pd.DataFrame({
        "age": [45, 52, 60, 39, 65, 58],
        "bmi": [24.5, 28.2, 31.0, 22.1, 29.4, 27.0],
        "systolic_bp": [120, 135, 145, 115, 150, 130],
        "treatment_arm": ["Control", "Arm A", "Arm B", "Control", "Arm A", "Arm B"],
        "special_group": ["Site:1/A", "Site:2/B", "Site:3[C]", "Site:1/A", "Site:2/B", "Site:3[C]"],
    })


class TestCorrelationComputation:
    """Tests for _compute_corr_and_counts function."""

    def test_compute_corr_and_counts_symmetric(self, clinical_df: pd.DataFrame) -> None:
        """Symmetric columns compute diagonal self-correlations cleanly without duplicate column errors."""
        cols = ["age", "bmi", "systolic_bp"]
        corrs, counts = ecm._compute_corr_and_counts(clinical_df, targets=cols, predictors=cols, method="pearson")

        assert corrs.shape == (3, 3)
        assert counts.shape == (3, 3)

        # Diagonals must equal 1.0
        for col in cols:
            assert corrs.loc[col, col] == 1.0
            assert counts.loc[col, col] == len(clinical_df)

        # Off-diagonal values must be finite floats between -1 and 1
        assert -1.0 <= corrs.loc["age", "bmi"] <= 1.0

    def test_compute_corr_and_counts_asymmetric(self, clinical_df: pd.DataFrame) -> None:
        """Asymmetric target and predictor sets compute correlation and count matrices."""
        targets = ["systolic_bp"]
        predictors = ["age", "bmi"]
        corrs, counts = ecm._compute_corr_and_counts(clinical_df, targets=targets, predictors=predictors, method="pearson")

        assert corrs.shape == (2, 1)
        assert counts.shape == (2, 1)
        assert -1.0 <= corrs.loc["age", "systolic_bp"] <= 1.0
        assert counts.loc["age", "systolic_bp"] == len(clinical_df)

    def test_compute_corr_and_counts_missing_columns(self, clinical_df: pd.DataFrame) -> None:
        """Handles missing column gracefully with None values rather than crashing."""
        corrs, counts = ecm._compute_corr_and_counts(
            clinical_df, targets=["non_existent"], predictors=["age"], method="pearson"
        )
        assert corrs.loc["age", "non_existent"] is None
        assert counts.loc["age", "non_existent"] == 0

    def test_compute_corr_and_counts_constant_column(self) -> None:
        """Constant column has zero variance and results in None correlation without crashing."""
        df = pd.DataFrame({"constant_col": [10, 10, 10, 10], "varying_col": [1, 2, 3, 4]})
        corrs, _ = ecm._compute_corr_and_counts(
            df, targets=["varying_col"], predictors=["constant_col"], method="pearson"
        )
        assert corrs.loc["constant_col", "varying_col"] is None


class TestExcelCorrelationExport:
    """Tests for custom_corr_mat_to_excel and add_correlation_chart_openpyxl."""

    def test_custom_corr_mat_to_excel_symmetric(self, clinical_df: pd.DataFrame) -> None:
        """Symmetric columns export to Excel without 'At least one sheet must be visible' IndexError."""
        cols = ["age", "bmi", "systolic_bp"]
        buf = BytesIO()
        ecm.custom_corr_mat_to_excel(
            data=clinical_df,
            targets=cols,
            predictors=cols,
            file_name=buf,
            method="pearson"
        )
        excel_bytes = buf.getvalue()
        assert len(excel_bytes) > 0

        xls = pd.ExcelFile(BytesIO(excel_bytes))
        assert "Global(pearson)" in xls.sheet_names
        assert "Correlation Summary" in xls.sheet_names
        assert "Correlation Strength Info" in xls.sheet_names

    def test_custom_corr_mat_to_excel_grouped_with_special_characters(self, clinical_df: pd.DataFrame) -> None:
        """Grouping column containing characters like ':', '/', '[' sanitizes sheet names."""
        cols = ["age", "bmi"]
        buf = BytesIO()
        ecm.custom_corr_mat_to_excel(
            data=clinical_df,
            targets=cols,
            predictors=cols,
            group_column="special_group",
            file_name=buf,
            method="spearman"
        )
        excel_bytes = buf.getvalue()
        assert len(excel_bytes) > 0

        xls = pd.ExcelFile(BytesIO(excel_bytes))
        for sheet in xls.sheet_names:
            assert len(sheet) <= 31
            for forbidden in [":", "/", "\\", "?", "*", "[", "]"]:
                assert forbidden not in sheet

    def test_custom_corr_mat_to_excel_empty_selection(self, clinical_df: pd.DataFrame) -> None:
        """Empty targets and predictors export safely with fallback sheet instead of crashing."""
        buf = BytesIO()
        ecm.custom_corr_mat_to_excel(
            data=clinical_df,
            targets=[],
            predictors=[],
            file_name=buf,
            method="kendall"
        )
        excel_bytes = buf.getvalue()
        assert len(excel_bytes) > 0

        xls = pd.ExcelFile(BytesIO(excel_bytes))
        assert "Correlation Summary" in xls.sheet_names

    def test_add_correlation_chart_openpyxl_empty_matrix(self) -> None:
        """Empty correlation DataFrame does not crash openpyxl chart sheet writer."""
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            ecm.add_correlation_chart_openpyxl(writer, pd.DataFrame(), method="pearson")
        assert len(buf.getvalue()) > 0


class TestYFilesNetworkData:
    """Tests for get_yfiles_network_data function."""

    def test_get_yfiles_network_data_empty(self) -> None:
        """Empty dataframe or empty precalculated matrix returns empty nodes and edges."""
        nodes, edges = ecm.get_yfiles_network_data(
            data=pd.DataFrame(),
            targets=[],
            predictors=[],
            method="pearson",
            threshold_category="Moderate",
            correlation_matrix=pd.DataFrame()
        )
        assert nodes == []
        assert edges == []

    def test_get_yfiles_network_data_valid(self, clinical_df: pd.DataFrame) -> None:
        """Generates valid node and edge structures for non-empty clinical data."""
        cols = ["age", "bmi", "systolic_bp"]
        nodes, edges = ecm.get_yfiles_network_data(
            data=clinical_df,
            targets=cols,
            predictors=cols,
            method="pearson",
            threshold_category="Negligible"
        )
        assert len(nodes) == 3
        assert isinstance(edges, list)
