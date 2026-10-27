"""
Enrich: the column choice of the transformation form survives a transformation (M5 browser check, AppTest).

A transformation adds a column to the draft, so the list of columns changes. The unkeyed column picker was then a
new widget: the next choice vanished on the following rerun, and the Apply button stayed disabled. The same
transformation applied twice now gives a second column with a suffix, and the trace replays both.
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
COLUMNS = "Choose columns to apply the selected transformation:"


def _by_label(widgets, label: str):
    return next(w for w in widgets if w.label == label)


@pytest.fixture
def enrich(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "repeat_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    at.switch_page(nav.page_path("enrich")).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def test_the_choice_survives_a_transformation_and_twice_replays(enrich: AppTest) -> None:
    at = enrich
    _by_label(at.selectbox, "Choose a transformation type:").set_value("Scaling").run()
    _by_label(at.selectbox, "Choose a scaling method:").set_value("Z-Score Standardization").run()
    _by_label(at.multiselect, COLUMNS).set_value(["age"]).run()
    _by_label(at.button, "Apply Transformation").click().run()
    at.run()  # the draft has one more column: the choice is kept
    assert _by_label(at.multiselect, COLUMNS).value == ["age"]
    _by_label(at.button, "Apply Transformation").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    at.button(key="draft_apply").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    data = at.session_state["data"]
    assert {"age_Z-Score_Standardization", "age_Z-Score_Standardization_1"} <= set(data.columns)

    from core.workspace import UserScope
    from manage.paths import PathResolver
    from manage.reproduction_manager import run_trace

    trace = at.session_state["transformation_manager"].get_trace()
    functions = [s["function"] for s in trace["steps"]]
    assert functions.count("variable_transformation") == 2
    # the trace holds the source's path inside the workspace, where the user's resolver finds it
    replay = run_trace(
        json.loads(json.dumps(trace)),  # as written to and read from the trace file
        paths=PathResolver(scope=UserScope.for_user("repeat_user")),
    )
    assert replay.error is None and replay.reproducible, replay.warnings
