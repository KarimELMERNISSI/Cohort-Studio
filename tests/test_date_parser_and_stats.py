"""
Unit Tests for Date Parser (utils/date_parser.py) and Statistics Utilities (utils/statistics_utils.py).

Tests automated clinical datetime format discovery, Excel serial conversion,
and normality screening methods (Shapiro-Wilk, D'Agostino, KS, Anderson-Darling).
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from app_pages.home import get_statistics_dataframe as home_get_stats
from explore.statistics import get_statistics_dataframe as explore_get_stats
from utils.data_analyzer import DataAnalyzer
from utils.date_parser import smart_parse_dates
from utils.statistics_utils import normality_test, sanitize_dataframe_for_arrow


class TestSmartDateParser:
    """Test suite for utils/date_parser.py."""

    def test_empty_series_handling(self) -> None:
        """Empty series returns with explanation note."""
        s = pd.Series([], dtype=object)
        res, notes = smart_parse_dates(s)
        assert len(res) == 0
        assert "Empty Column" in notes

    def test_excel_serial_number_detection(self) -> None:
        """Numeric serial dates (e.g. days since 1899-12-30) are correctly converted."""
        # 44927 corresponds to ~2023-01-01
        serials = pd.Series([44927, 44928, 44929, 44930])
        parsed, notes = smart_parse_dates(serials)
        assert "Excel Serial Detected" in notes
        assert parsed.dt.year.iloc[0] == 2023
        assert parsed.dt.month.iloc[0] == 1

    def test_iso_date_strings_parsing(self) -> None:
        """Parses YYYY-MM-DD standard clinical dates."""
        dates = pd.Series(["2026-01-15", "2026-02-20", "2026-03-25", "2026-04-30"])
        parsed, _ = smart_parse_dates(dates)
        assert parsed.dt.year.tolist() == [2026, 2026, 2026, 2026]
        assert parsed.dt.month.tolist() == [1, 2, 3, 4]
        assert parsed.dt.day.tolist() == [15, 20, 25, 30]

    def test_european_format_day_first(self) -> None:
        """Deduces Day/Month/Year when first position exceeds 12."""
        dates = pd.Series(["25/01/2026", "28/02/2026", "15/03/2026", "18/04/2026"])
        parsed, _ = smart_parse_dates(dates)
        assert parsed.dt.day.iloc[0] == 25
        assert parsed.dt.month.iloc[0] == 1
        assert parsed.dt.year.iloc[0] == 2026


class TestNormalityUtilities:
    """Test suite for utils/statistics_utils.py."""

    @pytest.fixture
    def normal_sample(self) -> pd.Series:
        """Generate sample from standard normal distribution."""
        np.random.seed(42)
        return pd.Series(np.random.normal(loc=100.0, scale=15.0, size=200))

    @pytest.fixture
    def skewed_sample(self) -> pd.Series:
        """Generate sample from heavily skewed exponential distribution."""
        np.random.seed(42)
        return pd.Series(np.random.exponential(scale=2.0, size=200))

    def test_shapiro_normality(self, normal_sample: pd.Series, skewed_sample: pd.Series) -> None:
        """Shapiro-Wilk accepts normal and rejects skewed distributions."""
        p_normal = normality_test(normal_sample, method="shapiro")
        assert p_normal > 0.05

        p_skewed = normality_test(skewed_sample, method="shapiro")
        assert p_skewed < 0.05

    def test_dagostino_normality(self, normal_sample: pd.Series, skewed_sample: pd.Series) -> None:
        """D'Agostino Omnibus test screens normality."""
        p_normal = normality_test(normal_sample, method="dagostino")
        assert p_normal > 0.05

        p_skewed = normality_test(skewed_sample, method="dagostino")
        assert p_skewed < 0.05

    def test_ks_test_normality(self, normal_sample: pd.Series) -> None:
        """Kolmogorov-Smirnov test returns valid p-value."""
        p_val = normality_test(normal_sample, method="ks")
        assert 0.0 <= p_val <= 1.0

    def test_anderson_darling(self, normal_sample: pd.Series) -> None:
        """Anderson-Darling test returns significance level."""
        sig_level = normality_test(normal_sample, method="anderson")
        assert sig_level is not None
        assert not np.isnan(sig_level)

    def test_all_null_column_handling(self) -> None:
        """All-null column returns np.nan without raising exception."""
        null_col = pd.Series([np.nan, np.nan, np.nan])
        res = normality_test(null_col, method="shapiro")
        assert np.isnan(res)


class TestPyArrowStatisticsCompatibility:
    """Test suite ensuring descriptive statistics DataFrames serialize cleanly to PyArrow."""

    def test_sanitize_dataframe_for_arrow_mixed_timestamps(self) -> None:
        """Mixed float and Timestamp objects (ArrowInvalid trigger) are sanitized safely."""
        df_mixed = pd.DataFrame({
            "mean": [25.5, pd.Timestamp("2017-11-16 16:20:00")],
            "feature": ["age", "event_date"],
        })
        sanitized = sanitize_dataframe_for_arrow(df_mixed)
        table = pa.Table.from_pandas(sanitized)
        assert table.num_rows == 2

    def test_sanitize_dataframe_for_arrow_mixed_timedeltas(self) -> None:
        """Mixed float and Timedelta objects serialize to PyArrow without exception."""
        df_timedeltas = pd.DataFrame({
            "mean": [12.0, pd.Timedelta(days=5, hours=2)],
            "feature": ["metric", "followup_duration"],
        })
        sanitized = sanitize_dataframe_for_arrow(df_timedeltas)
        table = pa.Table.from_pandas(sanitized)
        assert table.num_rows == 2

    def test_sanitize_dataframe_preserves_numeric_dtypes(self) -> None:
        """Purely numeric columns retain native float and integer dtypes."""
        df_numeric = pd.DataFrame({
            "mean": [10.5, 20.3],
            "count": [100, 200],
        })
        sanitized = sanitize_dataframe_for_arrow(df_numeric)
        assert sanitized["mean"].dtype == np.float64
        assert sanitized["count"].dtype == np.int64
        table = pa.Table.from_pandas(sanitized)
        assert table.schema.field("mean").type == pa.float64()

    def test_home_get_statistics_dataframe_arrow_compatibility(self) -> None:
        """Descriptive statistics in home.py with datetime columns serialize to PyArrow without ArrowInvalid."""
        df_clinical = pd.DataFrame({
            "patient_id": [f"PT_{i:03d}" for i in range(30)],
            "age": [float(20 + i) for i in range(30)],
            "sbp": [120.0 + float(i % 10) for i in range(30)],
            "admission_date": pd.date_range("2017-01-01", periods=30, freq="W"),
            "discharge_date": pd.date_range("2017-01-15", periods=30, freq="W"),
            "category": ["Group_A" if i % 2 == 0 else "Group_B" for i in range(30)],
        })
        analyzer = DataAnalyzer(df_clinical)
        num_stats, cat_stats, date_stats = home_get_stats(df_clinical, analyzer)

        # PyArrow conversion must succeed for all three statistical tables
        t_num = pa.Table.from_pandas(num_stats)
        assert t_num.num_rows > 0
        assert "admission_date" not in num_stats.index  # Dates separated into date_stats

        t_cat = pa.Table.from_pandas(cat_stats)
        assert t_cat.num_rows > 0

        t_date = pa.Table.from_pandas(date_stats)
        assert t_date.num_rows == 2
        assert "admission_date" in date_stats.index
        assert "discharge_date" in date_stats.index

    def test_explore_get_statistics_dataframe_arrow_compatibility(self) -> None:
        """Descriptive statistics in explore/statistics.py serializes cleanly with mixed column types."""
        df_mixed = pd.DataFrame({
            "score": [1.0, 2.5, 3.8, 4.2],
            "event_time": pd.to_datetime(["2017-01-01", "2017-02-01", "2017-03-01", "2017-04-01"]),
            "status": ["Active", "Paused", "Active", "Completed"],
        })
        stats_df = explore_get_stats(df_mixed)
        table = pa.Table.from_pandas(stats_df)
        assert table.num_rows == 3
