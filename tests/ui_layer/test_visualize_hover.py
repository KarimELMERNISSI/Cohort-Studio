"""
Visualize: the columns chosen for the tooltip reach every point (M5, AppTest). The helper that added them crashed
with Plotly 7 (a trace has no .get any more), and before that gave every trace the values of the whole table: with a
colour group, a point showed another row's values.
"""

from __future__ import annotations

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


def test_no_tooltip_columns_where_a_mark_is_not_a_row(visualize: AppTest) -> None:
    at = visualize
    _select(at.sidebar.selectbox, "Plot Category", "Distribution")
    at.run()
    _select(at.sidebar.selectbox, "Plot Type", "Histogram")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert not [m for m in at.multiselect if m.key == "viz_hover_cols"]
