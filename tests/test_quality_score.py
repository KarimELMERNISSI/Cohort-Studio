"""Global data-quality score: weighted decomposition, audit details behind lost points, and the charts."""

import numpy as np
import pandas as pd
import pytest

from core.progress import revision_labels
from explore.data_quality_auditor import DETAIL_COLUMNS, DataQualityAuditor
from explore.quality_score import DEFAULT_WEIGHTS, band, interpretation, weighted_score
from ui.quality_charts import LOST_COLOR, history_figure, loss_figure, ring_figure

METRICS = {"Completeness": 98.0, "Uniqueness": 100.0, "Statistical Validity": 90.0, "Consistency": 100.0}


def test_global_score_is_100_minus_the_points_lost() -> None:
    q = weighted_score(METRICS, {"Completeness": 1, "Uniqueness": 1, "Statistical Validity": 2, "Consistency": 1})
    shares = {d.name: d.share for d in q.dimensions}
    assert shares["Statistical Validity"] == pytest.approx(0.4) and shares["Completeness"] == pytest.approx(0.2)
    assert q.total == pytest.approx((98 + 100 + 2 * 90 + 100) / 5)
    assert q.total == pytest.approx(100 - q.lost)
    assert [d.name for d in q.by_loss()][:2] == ["Statistical Validity", "Completeness"]
    assert q.by_loss()[0].lost == pytest.approx(4.0)


def test_missing_dimensions_and_zero_weights() -> None:
    q = weighted_score({**METRICS, "Domain Validity": None}, DEFAULT_WEIGHTS)
    assert "Domain Validity" not in {d.name for d in q.dimensions}  # no rules: left out, not counted as 0
    zero = weighted_score(METRICS, dict.fromkeys(METRICS, 0.0))
    assert zero.total == 0.0 and all(d.share == 0 for d in zero.dimensions)
    assert [d.name for d in q.dimensions] == ["Completeness", "Statistical Validity", "Consistency", "Uniqueness"]


def test_bands_and_interpretation() -> None:
    assert [band(s) for s in (95, 85, 60, 10)] == ["excellent", "good", "fair", "poor"]
    assert interpretation(99.6).startswith("Excellent")


def test_auditor_records_where_points_are_lost() -> None:
    df = pd.DataFrame(
        {
            "age": [30, 31, 32, 33, 34, 35, 36, 37, 38, 400.0],
            "sbp": [120, np.nan, 130, np.nan, 125, 128, 122, 127, 124, 126],
            "code": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "sex": ["M", "m", "F", "F ", "M", "F", "M", "F", "M", "F"],
        }
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one exact duplicate row
    auditor = DataQualityAuditor(df)
    auditor.run_audit()
    details = auditor.details
    for table in details.values():
        assert list(table.columns) == list(DETAIL_COLUMNS)
    completeness = details["Completeness"]
    assert completeness.iloc[0].tolist()[:3] == ["sbp", "empty cells", 2]
    assert details["Uniqueness"].iloc[0]["Count"] == 1
    validity = details["Statistical Validity"]
    assert "age" in set(validity["Where"]) and validity.set_index("Where").loc["age", "Count"] >= 1
    assert details["Consistency"].iloc[0][["Where", "Issue"]].tolist() == ["code", "numbers stored as text"]
    assert details["Uniformity"].iloc[0]["Where"] == "sex"


def test_revision_labels_name_each_version_of_the_data() -> None:
    steps = [
        {"function": "initial_load"},
        {"function": "rename_columns", "description": "Renamed 2 columns"},
        {"function": "imputation", "description": "Imputation (KNN)"},
        {"function": "apply_filter", "description": "Kept 480 of 500 rows"},
    ]
    assert revision_labels(steps) == ["as loaded", "Imputation (KNN)", "Kept 480 of 500 rows"]


def test_ring_is_built_from_the_dimensions() -> None:
    q = weighted_score(METRICS, DEFAULT_WEIGHTS)
    fig = ring_figure(q, previous=q.total - 1.0)
    pie = fig.data[0]
    assert sum(pie.values) == pytest.approx(100.0)  # earned + lost of every dimension
    assert pie.labels[0] == "Completeness" and pie.labels[1] == "Completeness: points lost"
    assert pie.marker.colors[1] == LOST_COLOR
    assert "weight 1" in pie.customdata[0] and "loses 0.50" in pie.customdata[0]
    texts = [a.text for a in fig.layout.annotations]
    assert texts[0] == f"<b>{q.total:.1f}</b>" and any("▲ 1.0 since the last check" in t for t in texts)
    assert len(ring_figure(q).layout.annotations) == 2  # no previous check, no delta


def test_loss_bars_and_history() -> None:
    q = weighted_score(METRICS, DEFAULT_WEIGHTS)
    bars = loss_figure(q)
    assert bars is not None and list(bars.data[0].y) == ["Statistical Validity", "Completeness"]
    assert loss_figure(weighted_score(dict.fromkeys(METRICS, 100.0))) is None

    checks = [
        {"revision": 0, "label": "as loaded", "metrics": METRICS, "at": "2026-10-13T10:00:00"},
        {
            "revision": 1,
            "label": "Imputation (KNN)",
            "metrics": {**METRICS, "Completeness": 100.0},
            "at": "2026-10-13T10:05:00",
        },
    ]
    fig = history_figure(checks, DEFAULT_WEIGHTS)
    totals = list(fig.data[0].y)
    assert totals[1] - totals[0] == pytest.approx(2.0 / 4)  # completeness +2 points with a 1/4 share
    assert len(fig.data) == 1 + len(METRICS) and fig.data[1].visible == "legendonly"
    assert fig.layout.xaxis.ticktext[1] == "1. Imputation (KNN)"
