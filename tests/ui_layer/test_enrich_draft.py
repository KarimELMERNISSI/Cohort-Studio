"""The Enrich draft: apply a current draft, redo or discard one built on older data or another population (AppTest)."""

import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def _banner_script(change: str) -> None:
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
        frame = state.working_frame()
        state.stage(
            frame.assign(b=frame["a"] * 2),
            "variable_computation",
            {"variable_name": "b", "formula": "a * 2"},
            "Computed new variable: b",
        )
        if change == "rename":  # the formula's column is renamed after the draft was started
            state.commit_change(
                state.base.rename(columns={"a": "A"}), "rename_columns", {}, "Renamed 1 column"
            )
        elif change == "added":  # the dataset gains a column after the draft was started
            state.commit_change(state.base.assign(c=0), "external_write", {}, "Added c")
    render_draft_banner()
    draft = state.working_frame()
    st.session_state["probe"] = {
        "columns": list(state.base.columns),
        "draft_columns": list(draft.columns),
        "status": state.staging_status().state,
        "steps": [s["function"] for s in state.transformation_manager.history],
    }


def _labels(at: AppTest) -> list[str]:
    return [b.label for b in at.button]


def _run(change: str) -> AppTest:
    at = AppTest.from_function(_banner_script, kwargs={"change": change}, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_a_current_draft_is_applied_as_traced_steps(data_root: Path) -> None:
    at = _run("none")
    assert at.session_state["probe"]["status"] == "ok"
    assert "Computed new variable: b" in " ".join(m.value for m in at.markdown)
    assert _labels(at) == ["Apply to the dataset", "Discard the draft"]
    at.button(key="draft_apply").click().run()
    probe = at.session_state["probe"]
    assert probe["columns"] == ["a", "b"] and probe["status"] == "empty"
    assert probe["steps"] == ["initial_load", "variable_computation"]


def test_a_draft_built_on_older_data_is_redone_on_the_current_data(data_root: Path) -> None:
    at = _run("added")
    assert at.session_state["probe"]["status"] == "base_changed"
    assert any("cannot be applied as it is" in w.value for w in at.warning)
    assert _labels(at) == [
        "Redo my 1 change on the current data",
        "Discard the draft and restart from the current data",
    ]
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    assert probe["status"] == "ok" and probe["draft_columns"] == ["a", "c", "b"]
    at.button(key="draft_apply").click().run()
    assert at.session_state["probe"]["columns"] == ["a", "c", "b"]


def test_a_change_that_no_longer_fits_the_data_can_only_be_discarded(data_root: Path) -> None:
    at = _run("rename")
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    assert probe["status"] == "base_changed"  # the formula refers to a column that was renamed
    assert any("could not" in w.value or "failed" in w.value for w in at.warning)
    at.button(key="draft_discard_stale").click().run()
    probe = at.session_state["probe"]
    # the rename is kept and the draft is gone
    assert probe["columns"] == ["A"] and probe["status"] == "empty"
    assert probe["steps"] == ["initial_load", "rename_columns"]


def _population_script() -> None:
    import pandas as pd
    import streamlit as st

    from core.population import criterion_id
    from core.state import DatasetRef
    from enrich.imputation_service import impute_rows
    from ui.components.population import population_banner
    from ui.components.staging import render_draft_banner
    from ui.state import get_state

    state = get_state()
    adult = criterion_id("study_inclusion", "Adult")
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        st.session_state["config"] = {
            "mask_families": {
                "study_inclusion": {"operator": "all", "Adult": {"expression": "df['age'] >= 18"}}
            }
        }
        state.set_dataset(
            pd.DataFrame({"age": [10, 30, 50, 70], "bmi": [None, 20.0, None, 30.0]}),
            DatasetRef(name="d", origin="upload", file_hash="h"),
        )
        state.set_ticked([adult])
        population = state.population
        outcome = impute_rows(state.working_frame(), population.included, numerical="Mean")
        state.stage(
            outcome.frame, "imputation", outcome.params, "Mean imputation", population=population
        )
        state.set_ticked([])  # the population changes after the imputation was fitted
    if st.session_state.get("apply_now"):
        st.session_state["apply_now"] = False
        state.commit_staging()
    render_draft_banner()
    population_banner()
    st.session_state["probe"] = {
        "status": state.staging_status().state,
        "draft_bmi": state.working_frame()["bmi"].tolist(),
    }


def test_an_imputation_fitted_on_another_population_is_refitted(data_root: Path) -> None:
    at = AppTest.from_function(_population_script, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"]["status"] == "population_changed"
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    assert probe["status"] == "ok"
    bmi = probe["draft_bmi"]
    # refitted on every row: the child (excluded before) is imputed too, like the other missing value
    assert not any(math.isnan(v) for v in bmi) and bmi[0] == bmi[2] and bmi[1::2] == [20.0, 30.0]


def _applied_script(untick: bool) -> None:
    import pandas as pd
    import streamlit as st

    from core.population import criterion_id
    from core.state import DatasetRef
    from enrich.imputation_service import impute_rows
    from ui.components.population import population_banner
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
            pd.DataFrame({"age": [10, 30, 50, 70], "bmi": [None, 20.0, None, 30.0]}),
            DatasetRef(name="d", origin="upload", file_hash="h"),
        )
        state.set_ticked([criterion_id("study_inclusion", "Adult")])
        population = state.population
        outcome = impute_rows(state.working_frame(), population.included, numerical="Mean")
        state.stage(
            outcome.frame, "imputation", outcome.params, "Mean imputation", population=population
        )
        state.commit_staging()
        if untick:  # the analyses now include the child, whose missing value was not imputed
            state.set_ticked([])
    population_banner()


@pytest.mark.parametrize("untick", [False, True])
def test_imputed_values_from_another_population_are_flagged_on_analysis_pages(
    data_root: Path, untick: bool
) -> None:
    at = AppTest.from_function(
        _applied_script, kwargs={"untick": untick}, default_timeout=120
    ).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    flagged = any("fitted on other rows" in w.value for w in at.warning)
    assert flagged is untick
