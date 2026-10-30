"""Stored results carry what they were computed on, and are not shown as current after a change (AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def _script(change: str) -> None:
    import pandas as pd
    import streamlit as st

    from core.population import criterion_id
    from core.state import DatasetRef
    from ui.results import provenance, recall, remember
    from ui.state import get_state

    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        st.session_state["config"] = {
            "mask_families": {
                "study_inclusion": {"operator": "all", "Adult": {"expression": "df['age'] >= 18"}}
            }
        }
        state.set_dataset(
            pd.DataFrame({"age": [10, 30, 50, None]}),
            DatasetRef(name="cohort", origin="upload", file_hash="h"),
        )
        remember("mean_age", 30.0, config=("age",), n_analysed=3)
        if change == "data":
            state.commit_change(
                state.base.rename(columns={"age": "age_years"}), "rename_columns", {}, "Renamed"
            )
        elif change == "population":
            state.set_ticked([criterion_id("study_inclusion", "Adult")])
        elif change == "settings":
            pass
    config = ("age", "other") if change == "settings" else ("age",)
    value = recall("mean_age", config=config)
    st.session_state["value"] = value
    if value is not None:
        provenance("mean_age")


def _run(change: str) -> AppTest:
    at = AppTest.from_function(_script, kwargs={"change": change}, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_a_current_result_is_shown_with_what_it_was_computed_on(data_root: Path) -> None:
    at = _run("none")
    assert at.session_state["value"] == 30.0
    caption = " ".join(c.value for c in at.caption)
    assert "Computed on cohort" in caption and "population: all 4 rows" in caption
    assert "N analysed 3 (1 not used: missing values)" in caption


def test_after_a_data_change_the_result_is_announced_stale(data_root: Path) -> None:
    at = _run("data")
    assert at.session_state["value"] is None
    assert any("data changed" in w.value for w in at.warning)


def test_after_a_population_change_the_result_is_announced_stale(data_root: Path) -> None:
    at = _run("population")
    assert at.session_state["value"] is None
    assert any("another analysis population" in w.value for w in at.warning)


def test_other_settings_simply_hide_the_result(data_root: Path) -> None:
    at = _run("settings")
    assert at.session_state["value"] is None and not at.warning


def _reloaded() -> None:
    import sys
    import types
    from dataclasses import dataclass

    import pandas as pd
    import streamlit as st

    from core.state import DatasetRef
    from ui.results import add_to_report, recall, remember, report_items, report_status
    from ui.state import get_state

    module = sys.modules.setdefault("cmp_reload_probe", types.ModuleType("cmp_reload_probe"))

    @dataclass(frozen=True)
    class Curve:  # defined anew on each run, as a reloaded module defines its classes
        median: float

    Curve.__module__, Curve.__qualname__ = module.__name__, "Curve"
    module.Curve = Curve
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        get_state().set_dataset(
            pd.DataFrame({"age": [10, 30]}),
            DatasetRef(name="cohort", origin="upload", file_hash="h"),
        )
        remember("km", Curve(30.0), config=())
        add_to_report("km")
    st.session_state["value"] = recall("km", config=())
    st.session_state["report"] = [report_status(item) for item in report_items()]


def test_a_result_stored_before_the_code_was_updated_is_not_shown(data_root: Path) -> None:
    """A result kept across a code reload is an object of the old class: the page and the dossier read fields it
    lacks (a Kaplan-Meier curve's level) and stopped with an AttributeError."""
    at = AppTest.from_function(_reloaded, default_timeout=120).run()
    assert not at.exception and at.session_state["value"] is not None
    assert at.session_state["report"] == ["current"]
    at.run()  # the module now defines another class of that name
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["value"] is None
    assert any("updated since this result was computed" in w.value for w in at.warning)
    assert at.session_state["report"] == ["stale_code"]
