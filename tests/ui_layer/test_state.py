"""AppState.set_dataset is the single invalidation point and starts a trace in the user's workspace (audit E8)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


def _script(username: str) -> None:
    import os

    import pandas as pd
    import streamlit as st

    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state["username"] = username
    st.session_state["user_role"] = "researcher"
    st.session_state["epidemiology_results"] = {"stale": True}
    st.session_state["rename_dict"] = {"a": "b"}
    state = get_state()
    df = pd.DataFrame({"a": [1, 2, 3]})
    ref = state.set_dataset(
        df,
        DatasetRef(name="demo", origin="upload", file_hash="abc"),
        source_bytes=b"a\n1\n2\n3\n",
        source_name="demo.csv",
    )
    tm = state.transformation_manager
    ok, msg, saved = state.save_version("demo")
    st.session_state["probe"] = {
        "name": ref.name,
        "rows": ref.n_rows,
        "cols": ref.n_cols,
        "source_path": ref.source_path,
        "source_exists": os.path.exists(ref.source_path),
        "stale_cleared": "epidemiology_results" not in st.session_state
        and "rename_dict" not in st.session_state,
        "current": st.session_state["current_dataset_name"],
        "working_rows": len(st.session_state["working_df"]),
        "trace_path": tm.get_trace_path(),
        "steps": [s.get("function_name") or s.get("function") for s in tm.history],
        "saved": saved,
        "save_ok": ok,
        "label": state.dataset.label,
    }
    sample = state.load_sample_cohort()
    st.session_state["probe2"] = {
        "origin": sample.origin,
        "rows": sample.n_rows,
        "data_rows": len(st.session_state["data"]),
        "dayfirst": sample.parse.dayfirst if sample.parse else None,
    }


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def test_set_dataset_resets_derived_state_and_writes_trace(data_root: Path) -> None:
    at = AppTest.from_function(_script, kwargs={"username": "alice"}, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    probe = at.session_state["probe"]
    assert probe["name"] == "demo" and probe["rows"] == 3 and probe["cols"] == 1
    assert probe["stale_cleared"]
    assert probe["working_rows"] == 3
    assert probe["source_exists"]
    assert (
        Path(probe["source_path"])
        .resolve()
        .is_relative_to((data_root / "users" / "alice").resolve())
    )
    assert (
        Path(probe["trace_path"])
        .resolve()
        .is_relative_to((data_root / "users" / "alice" / "traces").resolve())
    )
    assert probe["save_ok"] and probe["saved"] == "demo_v1" and probe["label"] == "demo (v1)"
    assert probe["current"] == "demo_v1" or probe["current"] == "demo"
    assert (data_root / "users" / "alice" / "datasets" / "demo_v1.parquet").exists()
    assert os.path.basename(probe["source_path"]) == "source.csv"

    probe2 = at.session_state["probe2"]
    assert probe2["origin"] == "sample" and probe2["rows"] == 500 and probe2["data_rows"] == 500
    assert probe2["dayfirst"] is True


def _results_script() -> None:
    import pandas as pd
    import streamlit as st

    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state["username"] = "bob"
    st.session_state["user_role"] = "researcher"
    stored = ("cox_result", "surv_result", "mh_result", "mi_result", "clustering_results")
    for key in stored:
        st.session_state[key] = {"from": "the previous dataset"}
    get_state().set_dataset(
        pd.DataFrame({"a": [1, 2]}), DatasetRef(name="next", origin="upload", file_hash="def")
    )
    st.session_state["left"] = [key for key in stored if key in st.session_state]


def test_analysis_results_do_not_survive_a_dataset_swap(data_root: Path) -> None:
    at = AppTest.from_function(_results_script, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["left"] == []


def _filter_script() -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.data_monitoring import commit_filter_to_session
    from core.state import DatasetRef
    from ui.state import get_state

    if "probe" not in st.session_state:
        state = get_state()
        full = pd.DataFrame({"x": range(10), "flag": [False] * 6 + [True] * 4})
        state.set_dataset(full, DatasetRef(name="filtered", origin="upload", file_hash="f"))
        # a change prepared on Enrich before the filter: it must not be applied over the filtered data
        state.stage(state.working_frame().assign(y=1), "variable_computation", {}, "y")
        commit_filter_to_session(full[~full["flag"]], "anomaly_data", 10, 6)
        st.session_state["probe"] = {
            "data": len(st.session_state["data"]),
            "base": len(state.base),
            "draft": state.staging_status().state,
            "steps": [s["function"] for s in state.transformation_manager.history],
        }


def test_an_applied_filter_reaches_the_dataset_and_outdates_the_enrich_draft(
    data_root: Path,
) -> None:
    """Imputation on Enrich must not be fitted on, or restore, rows excluded on Validate (audit E8)."""
    at = AppTest.from_function(_filter_script, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"] == {
        "data": 6,
        "base": 6,
        "draft": "base_changed",
        "steps": ["initial_load", "apply_filter"],
    }
