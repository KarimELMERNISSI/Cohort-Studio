"""The Enrich draft banner: apply a current draft, and refuse one built on data that changed since (AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def _banner_script(stale: bool) -> None:
    import pandas as pd
    import streamlit as st

    from core.state import DatasetRef
    from ui.components.staging import render_draft_banner
    from ui.state import get_state

    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        state.set_dataset(
            pd.DataFrame({"a": [1, 2, 3]}), DatasetRef(name="d", origin="upload", file_hash="h")
        )
        state.stage(
            state.working_frame().assign(b=1),
            "variable_computation",
            {"variable_name": "b"},
            "Computed new variable: b",
        )
        if stale:  # the dataset changes after the draft was started (a rename on Explore)
            state.commit_change(
                state.base.rename(columns={"a": "A"}), "rename_columns", {}, "Renamed 1 column"
            )
    render_draft_banner()
    st.session_state["probe"] = {
        "columns": list(state.base.columns),
        "status": state.staging_status().state,
        "steps": [s["function"] for s in state.transformation_manager.history],
    }


def _labels(at: AppTest) -> list[str]:
    return [b.label for b in at.button]


def test_a_current_draft_is_applied_as_traced_steps(data_root: Path) -> None:
    at = AppTest.from_function(_banner_script, kwargs={"stale": False}, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"]["status"] == "ok"
    assert "Computed new variable: b" in " ".join(m.value for m in at.markdown)
    assert _labels(at) == ["Apply to the dataset", "Discard the draft"]
    at.button(key="draft_apply").click().run()
    probe = at.session_state["probe"]
    assert probe["columns"] == ["a", "b"] and probe["status"] == "empty"
    assert probe["steps"] == ["initial_load", "variable_computation"]


def test_a_draft_built_on_older_data_can_only_be_discarded(data_root: Path) -> None:
    at = AppTest.from_function(_banner_script, kwargs={"stale": True}, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"]["status"] == "base_changed"
    assert any("cannot be applied" in w.value for w in at.warning)
    assert _labels(at) == ["Discard the draft and restart from the current data"]
    at.button(key="draft_discard_stale").click().run()
    probe = at.session_state["probe"]
    # the rename is kept and the draft is gone
    assert probe["columns"] == ["A"] and probe["status"] == "empty"
    assert probe["steps"] == ["initial_load", "rename_columns"]
