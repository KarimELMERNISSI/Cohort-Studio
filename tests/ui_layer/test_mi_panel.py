"""The guided multiple-imputation panel of the Enrich page (M2, AppTest)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


def _panel() -> None:
    from app_pages.enrichment_mi import render_multiple_imputation
    from ui.state import get_state

    state = get_state()
    if not state.has_data:
        state.load_tutorial_cohort()
    render_multiple_imputation()


@pytest.fixture
def app(tmp_path: Path):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path)
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_function(_panel, default_timeout=240)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    yield at
    dp.DATA_ROOT = previous


def test_panel_guides_from_missingness_to_a_methods_paragraph(app: AppTest) -> None:
    metrics = {m.label: m.value for m in app.metric}
    assert (
        metrics["Analysis population"] == "1,206 rows" and metrics["Suggested imputations"] == "56"
    )
    assert app.number_input(key="mi_m").value == 56
    app.number_input(key="mi_m").set_value(2)
    app.number_input(key="mi_maxit").set_value(2)
    app.multiselect(key="mi_outcomes").set_value(["sbp_6m"])
    app.button(key="mi_run").click().run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    stored = app.session_state["_mi"]
    assert stored.mi.m == 2 and set(stored.mi.imputations) == {
        "hba1c_pct",
        "nt_probnp_pg_ml",
        "sbp_6m",
        "ldl_mmol_l",
        "height_cm",
        "bmi_kg_m2",
        "egfr_ml_min_1_73m2",
        "hf_phenotype",
    }
    assert app.selectbox(key="mi_diag_variable").options == list(stored.mi.imputations)
    paragraph = app.code[-1].value
    assert (
        "multiple imputation by chained equations" in paragraph
        and "We created 2 imputed datasets" in paragraph
    )
    app.button(key="mi_discard").click().run()
    assert not app.exception
    assert app.session_state["_mi"] is None if "_mi" in app.session_state else True


def test_a_planned_cox_model_adds_the_cumulative_hazard(app: AppTest) -> None:
    app.number_input(key="mi_m").set_value(2)
    app.number_input(key="mi_maxit").set_value(1)
    app.toggle(key="mi_cox").set_value(True).run()
    app.selectbox(key="mi_time").set_value("fu_days")
    app.selectbox(key="mi_event").set_value("death")
    app.button(key="mi_run").click().run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    stored = app.session_state["_mi"]
    assert tuple(stored.binding.spec["survival"]) == ("fu_days", "death")
    assert "Nelson-Aalen" in app.code[-1].value


def test_parallel_run_from_the_app_matches_the_sequential_one(app: AppTest) -> None:
    """Worker processes must not re-run the page script (it is __main__ inside Streamlit and AppTest)."""
    app.number_input(key="mi_m").set_value(2)
    app.number_input(key="mi_maxit").set_value(1)
    app.toggle(key="mi_parallel").set_value(True)
    app.button(key="mi_run").click().run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    assert "_mi" in app.session_state, [
        e.value for e in app.error
    ]  # the panel shows why the run failed
    parallel = app.session_state["_mi"].mi.fingerprint()
    app.toggle(key="mi_parallel").set_value(False)
    app.button(key="mi_run").click().run()
    assert not app.exception
    assert app.session_state["_mi"].mi.fingerprint() == parallel
