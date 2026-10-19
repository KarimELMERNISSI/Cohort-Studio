"""
Enrich's FAMD preview (M5, AppTest): the columns offered, and the explained variance in % over every dimension. On
the tutorial cohort the plot showed "0.79" for 3 components: 0.79% of 810 dimensions, the date column making one
category per day.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

pytest.importorskip("prince", reason="FAMD runs with prince (the 'ml' extra)")


def _script() -> None:
    from app_pages.data_enrichment import compute_transformations
    from ingest.loaders import read_path
    from ui.state import TUTORIAL_COHORT_PATH

    compute_transformations(read_path(str(TUTORIAL_COHORT_PATH)).df)


def _by_label(widgets, label: str):
    return next(w for w in widgets if w.label == label)


def test_the_famd_preview_leaves_the_date_out_and_reaches_100_percent() -> None:
    at = AppTest.from_function(_script, default_timeout=180)
    at.run()
    _by_label(at.selectbox, "Choose a transformation type:").set_value(
        "Dimensionality Reduction"
    ).run()
    _by_label(at.selectbox, "Choose a dimensional reduction method:").set_value("FAMD").run()
    columns = _by_label(at.multiselect, "Choose columns to apply the selected transformation:")
    assert "enrol_date" not in columns.options and "age" in columns.options
    assert any("enrol_date" in c.value and "a date" in c.value for c in at.caption)
    columns.set_value(
        [
            "site",
            "consent",
            "arm",
            "age",
            "sex",
            "height_cm",
            "weight_kg",
            "smoking_status",
            "diabetes",
            "hypertension",
        ]
    ).run()
    _by_label(at.checkbox, "Show Explained Variance").check().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    summary = next(c.value for c in at.caption if "of the variance" in c.value)
    assert summary.startswith(
        "3 components explain 16.4% of the variance; the data has 25 dimensions"
    )
