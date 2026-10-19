"""
Outliers flag the same rows on every screen, and the clip previewed on the Validate page is the clip recorded and
replayed (M5). Before, the three screens had their own detectors (quantile 370 vs 400 rows, Isolation Forest 140
vs 133, LOF 30 vs 27 on the tutorial cohort), Validate skipped nullable integer columns, and its clip preview used
the last column's outliers for every column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app_pages.data_monitoring import OutlierHandler
from biostats.outliers import detect_outliers
from explore.data_quality_auditor import DataQualityAuditor
from manage.reproduction_manager import _outlier_clipping

TUTORIAL = Path(__file__).resolve().parents[1] / "assets" / "tutorial" / "tutorial_cohort.csv"
LABS = ["nt_probnp_pg_ml", "creatinine_umol_l", "hba1c_pct", "sbp_baseline"]


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return pd.read_csv(TUTORIAL)


@pytest.mark.parametrize(
    ("method", "params"),
    [("zscore", {"threshold": 3.0}), ("iqr", {"multiplier": 1.5}), ("quantile", {"lower": 0.01, "upper": 0.99})],
)
def test_validate_and_the_health_score_flag_the_same_values(cohort: pd.DataFrame, method: str, params: dict) -> None:
    handler = OutlierHandler(cohort, {})
    _, flags = handler.handle_outliers(LABS, method, "none", **params)
    detection = detect_outliers(cohort, LABS, method, params)
    assert flags.equals(detection.flags)
    numeric = list(cohort.select_dtypes("number").columns)
    everywhere = detect_outliers(cohort, numeric, method, params)
    observed = sum(int(cohort[c].notna().sum()) for c in everywhere.columns)
    expected = round((1 - int(everywhere.flags.to_numpy().sum()) / observed) * 100, 2)
    assert DataQualityAuditor(cohort).compute_validity(method, params) == expected


@pytest.mark.parametrize("method", ["Isolation Forest", "Local Outlier Factor"])
def test_a_multivariate_method_flags_the_same_rows(cohort: pd.DataFrame, method: str) -> None:
    handler = OutlierHandler(cohort, {})
    _, flags = handler.handle_outliers(LABS, method, "none")
    assert flags["overall"].equals(detect_outliers(cohort, LABS, method).flags["overall"])


@pytest.mark.parametrize("method", ["zscore", "iqr", "quantile"])
def test_the_clip_preview_is_the_clip_replayed(cohort: pd.DataFrame, method: str) -> None:
    # the z-score preview used the last column's outliers for every column: on the tutorial it showed a maximum
    # NT-proBNP of 28,017 while the recorded clip gave 9,055
    handler = OutlierHandler(cohort, {})
    preview, _ = handler.handle_outliers(LABS, method, "clip")
    assert handler.detection is not None
    step = {"function": "outlier_clipping", "params": {"bounds": handler.detection.bounds}}
    replayed, _ = _outlier_clipping(cohort, step, 0, None, [])
    pd.testing.assert_frame_equal(preview, replayed)
    low, high = handler.detection.bounds["nt_probnp_pg_ml"]
    assert preview["nt_probnp_pg_ml"].max() <= high and preview["nt_probnp_pg_ml"].min() >= low


def test_a_binary_column_is_never_clipped() -> None:
    # mostly zeros: the IQR is 0 and its bounds [0, 0] would have turned every 1 into 0
    df = pd.DataFrame({"smoker": [0] * 18 + [1, 1], "age": list(range(40, 60))})
    handler = OutlierHandler(df, {})
    preview, _ = handler.handle_outliers(["smoker", "age"], "iqr", "clip")
    assert preview["smoker"].tolist() == df["smoker"].tolist() and handler.detection is not None
    assert handler.detection.skipped == ["smoker"]


def test_nullable_integer_columns_are_analysed() -> None:
    df = pd.DataFrame({"visits": pd.array([3, 4, 5, 4, 3, 4, 5, 4, 40], dtype="Int64")})
    _, flags = OutlierHandler(df, {}).handle_outliers(["visits"], "iqr", "none")
    assert flags["visits"].tolist() == [False] * 8 + [True]
