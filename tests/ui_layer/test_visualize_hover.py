"""
Visualize: the columns chosen for the tooltip reach every point (M5, AppTest). The helper that added them crashed
with Plotly 7 (a trace has no .get any more), and before that gave every trace the values of the whole table: with a
colour group, a point showed another row's values.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")


@pytest.fixture
def visualize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "hover_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_sample").click().run()
    at.switch_page(nav.page_path("visualize")).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def _select(widgets, label: str, value: str) -> None:
    next(w for w in widgets if w.label == label).set_value(value)


def _figure(at: AppTest) -> dict:
    charts = at.get("plotly_chart")
    assert charts, "no chart drawn"
    return json.loads(charts[-1].proto.spec)


def test_the_tooltip_columns_follow_the_points_of_each_group(visualize: AppTest) -> None:
    at = visualize
    _select(at.sidebar.selectbox, "Plot Category", "Relationships")
    at.run()
    _select(at.sidebar.selectbox, "Plot Type", "Scatter Plot")
    at.run()
    _select(at.selectbox, "X Variable", "age")
    _select(at.selectbox, "Y Variable", "sbp")
    _select(at.selectbox, "Group by (optional)", "sex")
    at.run()
    at.multiselect(key="viz_hover_cols").set_value(["patient_id"]).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    data = at.session_state["data"]
    traces = [t for t in _figure(at)["data"] if t.get("type") == "scatter"]
    assert len(traces) >= 2  # one per sex
    for trace in traces:
        assert "patient_id=%{customdata[0]}" in trace["hovertemplate"]
        rows = data[data["sex"].astype(str) == trace["name"]]
        assert [c[0] for c in trace["customdata"]] == rows["patient_id"].tolist()


def test_each_file_of_a_figure_says_what_it_holds(visualize: AppTest) -> None:
    """A researcher attached the interactive file of a histogram to a paper as a supplement: every participant's
    value went with it, and nothing on the page said so (the data-protection review, PS-M1). The file of a scatter
    plot with an identifier in its tooltip holds every participant's identifier beside their values, and the cohort
    files hold every column of the rows, not only the two drawn."""
    from app_pages.visualization import (
        EXPORT_DATA_CAPTION,
        EXPORT_IMAGE_CAPTION,
        HTML_HELP,
        JSON_HELP,
        ROWS_HELP,
    )

    at = visualize
    _select(at.sidebar.selectbox, "Plot Category", "Relationships")
    at.run()
    _select(at.sidebar.selectbox, "Plot Type", "Scatter Plot")
    at.run()
    _select(at.selectbox, "X Variable", "age")
    _select(at.selectbox, "Y Variable", "sbp")
    at.run()
    at.multiselect(key="viz_hover_cols").set_value(["patient_id"]).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    captions = [c.value for c in at.caption]
    # the image first, which answers the expander's title; then the data
    assert captions.index(EXPORT_IMAGE_CAPTION) + 1 == captions.index(EXPORT_DATA_CAPTION)
    buttons = {b.proto.label: b.proto.help for b in at.get("download_button")}
    assert buttons["Interactive HTML (.html)"] == HTML_HELP
    assert buttons["Plotly Figure JSON (.json)"] == JSON_HELP
    assert (
        buttons["Filtered Cohort CSV (.csv)"]
        == buttons["Filtered Cohort Excel (.xlsx)"]
        == ROWS_HELP
    )
    # what the sentences say: the figure's own specification holds each participant's value and identifier
    data = at.session_state["data"]
    (trace,) = [t for t in _figure(at)["data"] if t.get("type") == "scatter"]
    assert [c[0] for c in trace["customdata"]] == data["patient_id"].tolist()
    for said in (EXPORT_DATA_CAPTION, HTML_HELP, JSON_HELP):
        assert "the data behind the figure" in said and "where the dataset itself may go" in said
    # the cohort files are the rows, every column, as their help says
    from utils.export_utils import to_csv_bytes

    written = to_csv_bytes(data).decode("utf-8").splitlines()[0].split(",")
    assert len(written) == data.shape[1] > 2 and "patient_id" in written
    assert "Every column of the rows behind this figure" in ROWS_HELP


def test_no_tooltip_columns_where_a_mark_is_not_a_row(visualize: AppTest) -> None:
    at = visualize
    _select(at.sidebar.selectbox, "Plot Category", "Distribution")
    at.run()
    _select(at.sidebar.selectbox, "Plot Type", "Histogram")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert not [m for m in at.multiselect if m.key == "viz_hover_cols"]


def test_the_smooth_density_draws_a_curve_and_every_figure_keeps_its_own_template(
    visualize: AppTest,
) -> None:
    """ "Smooth Density (KDE)" called figure_factory's distplot, which Plotly 7.1 removed: the mode raised an error.
    And Streamlit's theme painted the ticks and axis titles light in the dark theme, on the figure's white paper."""
    at = visualize
    _select(at.sidebar.selectbox, "Plot Category", "Distribution")
    at.run()
    _select(at.sidebar.selectbox, "Plot Type", "Histogram")
    at.run()
    _select(at.selectbox, "Representation", "Smooth Density (KDE)")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    figure = _figure(at)
    curves = [trace for trace in figure["data"] if trace.get("mode") == "lines"]
    assert curves
    # an array travels as typed bytes (8 for each number)
    held = curves[0]["x"]
    points = len(base64.b64decode(held["bdata"])) // 8 if isinstance(held, dict) else len(held)
    assert points == 500
    # the page's own template, not Streamlit's theme
    assert at.get("plotly_chart")[-1].proto.theme == ""
