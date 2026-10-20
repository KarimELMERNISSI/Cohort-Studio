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
    stored = ("cox_result", "surv_result", "mh_result", "clustering_results")
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


def _population_script() -> None:
    import pandas as pd
    import streamlit as st

    from core.state import DatasetRef
    from ui.components.population import activate_family
    from ui.state import get_state

    if "probe" not in st.session_state:
        state = get_state()
        st.session_state["config"] = {
            "mask_families": {
                "clinical_anomalies": {"operator": "any", "Flagged": {"expression": "df['flag']"}}
            }
        }
        full = pd.DataFrame({"x": range(10), "flag": [False] * 6 + [True] * 4})
        state.set_dataset(full, DatasetRef(name="filtered", origin="upload", file_hash="f"))
        # a change prepared on Enrich before the criteria are ticked
        state.stage(state.working_frame().assign(y=1), "variable_computation", {}, "y")
        activate_family("clinical_anomalies")  # Validate's "Exclude these anomalies..." button
        st.session_state["probe"] = {
            "analysed": len(st.session_state["data"]),
            "base": len(state.base),
            "draft": state.staging_status().state,
            "steps": [s["function"] for s in state.transformation_manager.history],
        }


def test_ticking_criteria_filters_the_analyses_without_deleting_rows(data_root: Path) -> None:
    """Criteria are filters: the analyses see the population, the dataset keeps every row (audit E8)."""
    at = AppTest.from_function(_population_script, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"] == {
        "analysed": 6,
        "base": 10,
        "draft": "ok",  # the data did not change: the Enrich draft can still be applied
        "steps": ["initial_load"],  # ticking a criterion is not a change of the data
    }


def _lineage_script(tamper: bool) -> None:
    import pandas as pd
    import streamlit as st

    from core.population import criterion_id
    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state["username"] = "lin"
    st.session_state["user_role"] = "researcher"
    if "probe" in st.session_state:
        return
    state = get_state()
    st.session_state["config"] = {
        "mask_families": {
            "study_inclusion": {"operator": "all", "Adult": {"expression": "df['age'] >= 18"}}
        }
    }
    state.set_dataset(
        pd.DataFrame({"age": [10, 30, 50]}),
        DatasetRef(name="cohort", origin="upload", file_hash="h"),
    )
    state.set_ticked([criterion_id("study_inclusion", "Adult")])
    ok, _, saved = state.save_version("cohort")
    lineage = state.db_manager.load_lineage(saved)
    recorded = [s["function"] for s in state.transformation_manager.history]
    head = state.transformation_manager.head_hash
    if tamper:  # the file is replaced outside CohortStudio
        pd.DataFrame({"age": [99]}).to_parquet(
            state.db_manager.lineage_path(saved).replace(".lineage.json", ".parquet")
        )
    state.set_ticked([])
    _, message = state.load_saved_version(saved)
    st.session_state["probe"] = {
        "saved": ok and saved == "cohort_v1",
        "recorded": recorded,
        "same_fp": lineage["fingerprint"] == state.session.data_fp,
        "head": lineage["trace"]["head_hash"] == head,
        "parent": state.transformation_manager.parent,
        "ticked": list(state.ticked),
        "population": (lineage["population"]["block"] or {}).get("n_included"),
        "message": message,
    }


def test_a_saved_version_links_to_its_trace_and_restores_its_population(data_root: Path) -> None:
    at = AppTest.from_function(_lineage_script, kwargs={"tamper": False}, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    probe = at.session_state["probe"]
    assert probe["saved"] and probe["same_fp"] and probe["head"]
    assert probe["recorded"] == ["initial_load", "population_definition", "save_version"]
    assert probe["parent"]["dataset"] == "cohort_v1" and probe["parent"]["trace_head"]
    assert probe["ticked"] == ["study_inclusion::Adult"] and probe["population"] == 2
    assert "modified outside" not in probe["message"]


def test_a_version_changed_outside_the_app_is_reported(data_root: Path) -> None:
    at = AppTest.from_function(_lineage_script, kwargs={"tamper": True}, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "modified outside" in at.session_state["probe"]["message"]


def _analyzer_script() -> None:
    import pandas as pd
    import streamlit as st

    from core.population import criterion_id
    from core.state import DatasetRef
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
            pd.DataFrame({"age": range(10, 130, 10), "site": ["a", "b"] * 6}),
            DatasetRef(name="d", origin="upload", file_hash="h"),
        )
    action = st.session_state.pop("action", None)
    if action == "tick":
        state.set_ticked([criterion_id("study_inclusion", "Adult")])
    elif action == "commit":
        state.commit_change(
            state.base.assign(bmi=state.base["age"] / 2), "external_write", {}, "bmi"
        )
    analysis, every_row = state.analyzer(), state.analyzer("base")
    analysis.numeric_cols.append("changed by a page")  # a page may change its own analyzer
    st.session_state["probe"] = {
        "rows": (len(analysis.df), len(every_row.df)),
        "numeric": state.analyzer().numeric_cols,
    }


def test_the_column_profile_is_computed_once_per_data_and_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every page analysed the columns on every run (dates tried in a dozen formats); now once per data."""
    import utils.data_analyzer as da

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    analysed: list[int] = []
    analyze = da.DataAnalyzer.analyze_columns

    def counted(self) -> None:
        analysed.append(len(self.df))
        analyze(self)

    monkeypatch.setattr(da.DataAnalyzer, "analyze_columns", counted)
    at = AppTest.from_function(_analyzer_script, default_timeout=120).run()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert analysed == [12]  # nothing ticked: the analysis rows are every row, profiled once
    assert at.session_state["probe"] == {"rows": (12, 12), "numeric": ["age"]}
    at.session_state["action"] = "tick"
    at.run()
    at.run()
    assert analysed == [12, 11] and at.session_state["probe"]["rows"] == (11, 12)
    at.session_state["action"] = "commit"
    at.run()
    assert analysed == [12, 11, 11, 12]  # the data changed: both profiles again
    assert at.session_state["probe"]["numeric"] == ["age", "bmi"]


def _dashboard_script() -> None:
    import streamlit as st

    from app_pages.data_quality_dashboard import render_dashboard
    from ui.state import get_state

    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        state.load_tutorial_cohort()
    if st.session_state.pop("action", None) == "commit":
        base = state.base
        state.commit_change(base.drop(columns=base.columns[-1]), "external_write", {}, "Dropped")
    ss = st.session_state
    settings = (  # what the dashboard audits with in this run
        ss.get("validity_outlier_method", "iqr"),
        ss.get("validity_outlier_params", {"multiplier": 1.5}),
    )
    render_dashboard(state.base, ss["config"])
    ss["probe"] = {"metrics": state.quality_audit(*settings)[1], "base": state.base}  # kept


def test_the_quality_audit_runs_once_per_data_and_validity_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every click on Validate audited the whole dataset again, a weight slider included."""
    import explore.data_quality_auditor as dqa

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    audits: list[str | None] = []
    run_audit = dqa.DataQualityAuditor.run_audit

    def counted(self, *args, **kwargs):
        audits.append(kwargs.get("validity_method"))
        return run_audit(self, *args, **kwargs)

    monkeypatch.setattr(dqa.DataQualityAuditor, "run_audit", counted)
    at = AppTest.from_function(_dashboard_script, default_timeout=180).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert audits == ["iqr"]
    # the tutorial cohort's audit, kept, equals a fresh one
    probe = at.session_state["probe"]
    fresh = dqa.DataQualityAuditor(probe["base"], config=at.session_state["config"])
    assert probe["metrics"] == run_audit(
        fresh, validity_method="iqr", validity_params={"multiplier": 1.5}
    )
    weight = next(s for s in at.slider if s.label.endswith("Weight"))
    weight.set_value(3.0).run()  # the score changes, not the audit
    assert not at.exception and audits == ["iqr"]
    at.selectbox(key="validity_detection_method").set_value("zscore").run()
    at.run()  # the scorecard at the top of the page uses the method chosen below it in the run before
    assert audits == ["iqr", "zscore"]
    at.session_state["action"] = "commit"
    at.run()
    assert not at.exception and len(audits) == 3  # the data changed: audited again
