"""The multiple imputation in the session: status against the population, versions (M2, AppTest)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


def _script() -> None:
    import streamlit as st

    from biostats.mi import ImputationSpec, impute
    from ui.state import get_state

    state = get_state()
    ss = st.session_state
    action = ss.get("action", "impute")
    if action == "impute":
        state.load_tutorial_cohort()
        frame = state.analysis_frame()
        spec = ImputationSpec(
            columns=["hba1c_pct", "ldl_mmol_l", "age", "sex"], m=2, maxit=2, seed=1
        )
        state.set_mi(impute(frame, spec))
    elif action == "tick":
        state.activate_family("study_inclusion", True)
    elif action == "untick":
        state.activate_family("study_inclusion", False)
    elif action == "save":
        ok, msg, saved = state.save_version("mi_check")
        ss["saved"] = saved
        ss["save_msg"] = msg
    elif action == "reload":
        ss["load"] = state.load_saved_version(ss["saved"])
    elif action == "swap":
        state.load_sample_cohort()
    current = state.mi_status()
    ss["status"] = None if current is None else current.kind
    usable = state.analysis_mi()
    ss["mi_rows"] = None if usable is None else len(usable.base)


@pytest.fixture
def data_root(tmp_path: Path):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path)
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    yield tmp_path
    dp.DATA_ROOT = previous


def _run(at: AppTest, action: str) -> AppTest:
    at.session_state["action"] = action
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def _app() -> AppTest:
    return AppTest.from_function(_script, default_timeout=240)


def test_imputation_follows_the_population_and_travels_with_versions(data_root: Path) -> None:
    at = _run(_app(), "impute")
    assert at.session_state["status"] == "fresh" and at.session_state["mi_rows"] == 1206
    _run(at, "tick")  # a narrower population: a strict subset of the imputed rows
    assert at.session_state["status"] == "subset" and at.session_state["mi_rows"] is None
    _run(at, "untick")
    assert at.session_state["status"] == "fresh"
    _run(at, "save")
    assert "Saved with its multiple imputation (m = 2)" in at.session_state["save_msg"]
    assert (data_root / "datasets").exists() or any(data_root.rglob("mi_check*.parquet"))
    _run(at, "reload")
    ok, msg = at.session_state["load"]
    assert ok and "Multiple imputation restored (m = 2)" in msg
    assert at.session_state["status"] == "fresh" and at.session_state["mi_rows"] == 1206
    _run(at, "swap")  # another dataset: the imputation is dropped
    assert at.session_state["status"] is None
