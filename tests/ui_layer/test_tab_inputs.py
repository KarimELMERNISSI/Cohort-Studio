"""Each tab reads its own inputs (M5b): a tab no longer depends on what another tab of the page did in the run."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "tab_inputs_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    yield at
    dp.DATA_ROOT = previous


def test_the_outlier_tab_does_not_see_the_flags_of_the_criteria_tabs(app: AppTest) -> None:
    """The anomaly check added its flag columns to the frame the outlier tab analysed, in the same run."""
    app.switch_page(nav.page_path("validate")).run()
    _ok(app)
    app.session_state["validate_tab"] = "Handle Clinical Anomalies"
    app.run()
    next(b for b in app.button if b.label == "Run Anomaly Detection").click().run()
    _ok(app)
    app.session_state["validate_tab"] = "Handle Outliers"
    app.run()
    method = next(s for s in app.selectbox if s.label == "Outlier Detection Method")
    method.set_value("Isolation Forest").run()
    _ok(app)
    columns = next(m for m in app.multiselect if m.label == "Select columns for outlier handling")
    dataset = set(app.session_state["_data_session"].base.columns)
    assert columns.value and set(columns.value) <= dataset  # no clinical_anomalies_any or _list


def test_the_analysis_tab_transforms_the_variables_as_before() -> None:
    from app_pages.epidemiology import _transformed

    df = pd.DataFrame(
        {"a": [1.0, 2.0, 4.0], "b": [-1.0, 0.0, 3.0], "c": [5.0, 5.0, 5.0], "s": ["x", "y", "z"]}
    )
    numeric = ["a", "b", "c"]
    raw = _transformed(df, numeric, "None (Raw Data)")
    assert raw.equals(df) and raw is not df
    log = _transformed(df, numeric, "Log Transformation (Log1p)")
    assert np.allclose(log["a"], np.log1p([1.0, 2.0, 4.0])) and log["b"].equals(df["b"])  # b < 0
    z = _transformed(df, numeric, "Z-Score Standardization")
    assert np.allclose(z["a"], (df["a"] - df["a"].mean()) / df["a"].std(ddof=1))
    assert z["c"].equals(df["c"])  # constant: left as it is
    scaled = _transformed(df, numeric, "Min-Max Normalization")
    assert np.allclose(scaled["a"], [0.0, 1 / 3, 1.0]) and scaled["c"].equals(df["c"])
    assert scaled["s"].equals(df["s"])
    assert df["a"].tolist() == [1.0, 2.0, 4.0]  # the input is unchanged


def test_the_statistics_tab_lists_a_column_renamed_in_the_renaming_tab(app: AppTest) -> None:
    """The Statistics tab used the frame the Renaming tab returned; it now reads the analysis rows itself."""
    app.switch_page(nav.page_path("explore")).run()
    _ok(app)
    app.text_input(key="rename_age").set_value("age_years")
    next(b for b in app.button if b.label == "Apply Renaming").click().run()
    _ok(app)
    listed = " ".join(m.value for m in app.markdown if "column(s)" in m.value)
    assert "age_years" in listed and "age," not in listed.replace("age_years", "")
