"""Values filled by a single imputation are known and flagged under the results that use them (M2, C10)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


def _script() -> None:
    import streamlit as st

    from enrich.imputation_service import impute_rows
    from ui.results import provenance, remember
    from ui.state import get_state

    state = get_state()
    ss = st.session_state
    action = ss.get("action", "impute")
    if action == "impute":
        state.load_tutorial_cohort()
        outcome = impute_rows(
            state.working_frame(), None, numerical="Median", categorical="Most_Frequent"
        )
        state.stage(outcome.frame, "imputation", outcome.params, "Median imputation")
        ss["commit"] = state.commit_staging()
        ss["expected"] = outcome.params["imputed_values"]
    elif action == "save":
        ok, msg, saved = state.save_version("single_check")
        ss["saved"] = saved
    elif action == "reload":
        ss["load"] = state.load_saved_version(ss["saved"])
    elif action == "swap":
        state.load_sample_cohort()
    ss["counts"] = state.single_imputed_counts()
    remember("check", 1, config="c", columns=["hba1c_pct", "age"])
    provenance("check")


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


def test_single_imputed_values_are_counted_flagged_and_kept_with_versions(data_root: Path) -> None:
    at = AppTest.from_function(_script, default_timeout=300)
    at.session_state["username"] = "single_user"
    _run(at, "impute")
    assert at.session_state["commit"][0], at.session_state["commit"]
    expected = at.session_state["expected"]
    assert expected.get("hba1c_pct") and at.session_state["counts"] == expected
    warning = " ".join(w.value for w in at.warning)
    assert "filled by single imputation" in warning and "hba1c_pct" in warning
    assert "standard errors are too small" in warning

    _run(at, "save")
    assert at.session_state["saved"]
    _run(at, "swap")
    assert at.session_state["counts"] == {} and not at.warning
    _run(at, "reload")
    assert at.session_state["counts"] == expected  # restored from the lineage sidecar
