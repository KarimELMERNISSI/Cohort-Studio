"""The grouping variables offered on Analyze > Univariate group comparison: each column once."""

from __future__ import annotations

import pandas as pd

from app_pages.analysis_config import grouping_options
from ingest.analyzer import DataAnalyzer


def test_a_two_level_text_column_is_offered_once() -> None:
    # "arm" is both categorical and binary: it was listed twice ("arm", "arm")
    df = pd.DataFrame(
        {
            "arm": ["control", "drug"] * 15,
            "nyha": ["I", "II", "III"] * 10,
            "death": ["No", "Yes", "No"] * 10,
            "age": list(range(30, 60)),
            "visits": [1, 2, 3] * 10,
        }
    )
    analyzer = DataAnalyzer(df)
    options = grouping_options(df, analyzer.numeric_cols, analyzer.categorical_cols, analyzer.binary_cols)
    assert options[0] == "None" and len(options) == len(set(options))
    assert {"arm", "nyha", "death", "visits"} <= set(options) and "age" not in options  # 30 values: not a factor
    assert options[1:] == sorted(options[1:])
