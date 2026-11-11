"""
Analyze > Hypothesis tests & epidemiology, the Analysis tab's transformation (AppTest, the tutorial cohort): a result
is never shown or stored as current for values it was not computed on. Switched to Log1p after a run, the ANCOVA on
the raw values was still shown under the banner "Log1p applied"; a Log1p run of the group comparison was shown while
the store, and so the study dossier, kept the raw run, labelled current.
"""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")
RAW, LOG = "None (Raw Data)", "Log Transformation (Log1p)"


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "transform_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    at.switch_page(nav.page_path("analyze")).run()
    _ok(at)
    yield at
    dp.DATA_ROOT = previous


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _shown(at: AppTest) -> str:
    return " | ".join([s.value for s in at.subheader] + [m.value for m in at.markdown])


def test_an_ancova_on_the_raw_values_is_not_shown_once_log1p_is_chosen(app: AppTest) -> None:
    app.session_state["epi_analysis_mode"] = "Multivariate (ANCOVA/Regression)"
    app.run()
    app.selectbox(key="ancova_target").select("sbp_6m").run()
    app.selectbox(key="ancova_group").select("arm").run()
    app.multiselect(key="ancova_covs").set_value(["sbp_baseline"]).run()
    app.button(key="ancova_run").click().run()
    _ok(app)
    assert "ANCOVA Results" in _shown(app)
    app.selectbox(key="epi_transformation").select(LOG).run()
    _ok(app)
    assert "ANCOVA Results" not in _shown(app)  # computed on other values: run again
    app.selectbox(key="epi_transformation").select(RAW).run()
    _ok(app)
    assert "ANCOVA Results" in _shown(app)  # the raw run, still kept for the raw values


def test_the_group_comparison_stored_is_the_one_shown_after_a_log1p_run(app: AppTest) -> None:
    app.selectbox(key="epi_transformation").select(RAW).run()
    app.session_state["epi_analysis_mode"] = "Univariate (Group Comparison)"
    app.run()
    app.selectbox(key="group_col").select("arm").run()
    app.multiselect(key="target_cols").set_value(["sbp_6m"]).run()
    app.button(key="univariate_run").click().run()
    _ok(app)
    raw = app.session_state["_results"]["univariate"].value["results"][0]["Estimate"]
    app.selectbox(key="epi_transformation").select(LOG).run()
    app.button(key="univariate_run").click().run()
    _ok(app)
    shown = app.session_state["univariate_cache"]["results"][0]["Estimate"]
    stored = app.session_state["_results"]["univariate"]
    assert stored.value["results"][0]["Estimate"] == shown != raw
    assert LOG in str(stored.config)  # its settings name the transformation and its columns


def test_the_group_comparison_counts_the_rows_of_each_variable(app: AppTest) -> None:
    """`sbp_6m` is recorded for 778 of the tutorial's 1,206 rows: the comparison said 1,206 analysed."""
    app.selectbox(key="epi_transformation").select(RAW).run()
    app.multiselect(key="target_cols").set_value(["sbp_6m", "age"]).run()
    app.button(key="univariate_run").click().run()
    _ok(app)
    stored = app.session_state["_results"]["univariate"]
    frame = app.session_state["working_df"]
    by_variable = {row["Variable"]: row["N"] for row in stored.value["results"]}
    assert by_variable == {
        "sbp_6m": int(frame[["arm", "sbp_6m"]].dropna().shape[0]),
        "age": int(frame[["arm", "age"]].dropna().shape[0]),
    }
    assert by_variable["sbp_6m"] < by_variable["age"]
    assert (
        stored.stamp.n_analysed == by_variable["sbp_6m"]
        and stored.stamp.n_most == by_variable["age"]
    )
    assert (
        f"N analysed {by_variable['sbp_6m']:,} to {by_variable['age']:,} by variable"
        in stored.stamp.caption
    )
