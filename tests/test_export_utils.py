"""
Unit Tests for Export Utilities (utils/export_utils.py).

Tests DataFrame Excel and CSV conversions, Plotly figure HTML/JSON exports,
and publication modebar configuration.
"""

from io import BytesIO
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import pytest

from utils.export_utils import (
    fig_to_html_str,
    fig_to_json_str,
    get_publication_plot_config,
    to_csv_bytes,
    to_excel,
    to_excel_sheets,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture providing a sample clinical research DataFrame."""
    return pd.DataFrame({
        "patient_id": [101, 102, 103],
        "systolic_bp": [120, 135, 140],
        "treatment_group": ["Control", "Arm A", "Arm B"],
    })


@pytest.fixture
def sample_figure() -> go.Figure:
    """Fixture providing a sample Plotly figure."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 20, 30], mode="lines+markers"))
    fig.update_layout(title="Clinical Biomarker Trend")
    return fig


class TestExportUtils:
    """Test suite for data and figure export utilities."""

    def test_to_excel_single_sheet(self, sample_df: pd.DataFrame) -> None:
        """Converts DataFrame into valid Excel bytes."""
        excel_bytes = to_excel(sample_df)
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 0

        # Verify readability via openpyxl / pandas
        loaded_df = pd.read_excel(BytesIO(excel_bytes), sheet_name="Sheet1", index_col=0)
        assert loaded_df.shape == sample_df.shape
        assert list(loaded_df.columns) == list(sample_df.columns)

    def test_to_excel_multi_sheets(self, sample_df: pd.DataFrame) -> None:
        """Converts multiple DataFrames into an Excel workbook with separate sheets."""
        sheets = {
            "Baseline": sample_df,
            "FollowUp": sample_df.copy(),
        }
        excel_bytes = to_excel_sheets(sheets)
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 0

        # Read back both sheets
        xls = pd.ExcelFile(BytesIO(excel_bytes))
        assert "Baseline" in xls.sheet_names
        assert "FollowUp" in xls.sheet_names

    def test_to_excel_empty_or_none(self) -> None:
        """Empty DataFrame or None exports safely with fallback sheet instead of crashing."""
        bytes_empty = to_excel(pd.DataFrame())
        assert isinstance(bytes_empty, bytes)
        assert len(bytes_empty) > 0

        bytes_none = to_excel(None)
        assert isinstance(bytes_none, bytes)
        assert len(bytes_none) > 0

    def test_to_excel_sheets_empty_dict(self) -> None:
        """Empty dictionary exports safely with a default sheet instead of raising IndexError."""
        excel_bytes = to_excel_sheets({})
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 0
        xls = pd.ExcelFile(BytesIO(excel_bytes))
        assert len(xls.sheet_names) >= 1
        assert "Summary" in xls.sheet_names

    def test_to_excel_sheets_sanitizes_sheet_names(self, sample_df: pd.DataFrame) -> None:
        """Special characters in sheet names are sanitized to prevent openpyxl exceptions."""
        sheets = {
            "Group:2024/01/01[Arm*A]?": sample_df,
            "A" * 50: sample_df,
        }
        excel_bytes = to_excel_sheets(sheets)
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 0
        xls = pd.ExcelFile(BytesIO(excel_bytes))
        assert len(xls.sheet_names) == 2
        for name in xls.sheet_names:
            assert len(name) <= 31
            for char in [":", "/", "\\", "?", "*", "[", "]"]:
                assert char not in name

    def test_to_csv_bytes(self, sample_df: pd.DataFrame) -> None:
        """Converts DataFrame to UTF-8 CSV bytes."""
        csv_bytes = to_csv_bytes(sample_df, index=False)
        assert isinstance(csv_bytes, bytes)
        csv_str = csv_bytes.decode("utf-8")
        assert "patient_id,systolic_bp,treatment_group" in csv_str
        assert "101,120,Control" in csv_str

    def test_fig_to_html_str(self, sample_figure: go.Figure) -> None:
        """Exports Plotly figure into standalone HTML with CDN script."""
        html_str = fig_to_html_str(sample_figure, include_plotlyjs="cdn")
        assert isinstance(html_str, str)
        assert "<html>" in html_str
        assert "cdn.plot.ly" in html_str or "plotly.js" in html_str
        assert "Clinical Biomarker Trend" in html_str

    def test_fig_to_html_str_handles_invalid_object(self) -> None:
        """Invalid figure object fails gracefully returning comment string."""
        html_str = fig_to_html_str("not_a_figure")
        assert "Failed to export figure to HTML" in html_str

    def test_fig_to_json_str(self, sample_figure: go.Figure) -> None:
        """Exports Plotly figure into serialized JSON specification."""
        json_str = fig_to_json_str(sample_figure)
        assert isinstance(json_str, str)
        assert "data" in json_str
        assert "layout" in json_str
        assert "Clinical Biomarker Trend" in json_str

    def test_fig_to_json_str_handles_invalid_object(self) -> None:
        """Invalid figure object fails gracefully returning empty JSON dict."""
        json_str = fig_to_json_str(None)
        assert json_str == "{}"

    def test_get_publication_plot_config(self) -> None:
        """Generates Plotly modebar configuration for publication vector exports."""
        config = get_publication_plot_config(
            filename="oncology_survival_curve",
            format_type="svg",
            width=1400,
            height=900,
            scale=3,
        )
        assert config["displaylogo"] is False
        assert "toImageButtonOptions" in config
        image_opts = config["toImageButtonOptions"]
        assert image_opts["format"] == "svg"
        assert image_opts["filename"] == "oncology_survival_curve"
        assert image_opts["width"] == 1400
        assert image_opts["height"] == 900
        assert image_opts["scale"] == 3
