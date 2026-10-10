"""
Unit Tests for Date Parser (utils/date_parser.py) and Statistics Utilities (utils/statistics_utils.py).

Tests automated clinical datetime format discovery, Excel serial conversion,
and normality screening methods (Shapiro-Wilk, D'Agostino, KS, Anderson-Darling).
"""

import numpy as np
import pandas as pd
import pytest

from utils.date_parser import smart_parse_dates
from utils.statistics_utils import normality_test


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
