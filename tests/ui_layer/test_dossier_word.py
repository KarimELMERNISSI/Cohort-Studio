"""The Word study dossier: sections, flow figure, results with their provenance, stale results (M3, AppTest)."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pytest
from docx import Document
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


def _script() -> None:
    import streamlit as st

    from app_pages.study_dossier import build_dossier_bytes, session_parts
    from biostats.inference import compare_groups, groups_from_frame
    from biostats.survival import compare_survival, cox_regression
    from ui.results import add_to_report, remember, report_items
    from ui.state import get_state

    state = get_state()
    ss = st.session_state
    action = ss.get("action", "setup")
    if action == "setup":
        state.load_tutorial_cohort()
        state.activate_family("clinical_anomalies", True)
        state.activate_family("study_inclusion", True)
        df = state.analysis_frame()
        cox = cox_regression(df, "fu_days", "death", ["arm", "hba1c_pct"])
        remember(
            "cox",
            cox,
            config=("fu_days", "death", ("arm", "hba1c_pct")),
            n_analysed=cox.n,
            columns=["fu_days", "death", "arm", "hba1c_pct"],
        )
        add_to_report("cox")
        km = compare_survival(df, "fu_days", "death", "arm")
        remember(
            "survival",
            km,
            config=("fu_days", "death", "arm"),
            n_analysed=sum(c.n for c in km.curves),
        )
        add_to_report("survival")
        welch = compare_groups(groups_from_frame(df, "arm", "sbp_6m"), kind="welch")
        remember("paired", welch, config=("arm", "sbp_6m"), n_analysed=welch.n_total)
        add_to_report("paired")
        rows = [  # the observed group comparison of Analyze keeps a list of rows, not pooled results
            {
                "Variable": "age",
                "Test Used": "Welch's t-test",
                "P-Value": 0.59,
                "Estimate": 0.34,
                "Estimate CI": "-0.91 to 1.59",
            },
            {
                "Variable": "weight_kg",
                "Test Used": "Welch's t-test",
                "P-Value": 0.46,
                "Estimate": 0.84,
                "Estimate CI": "-1.37 to 3.04",
            },
        ]
        remember(
            "univariate",
            {"results": rows, "group": "arm", "correction": "Holm (recommended)"},
            config=("arm", ("age", "weight_kg")),
            n_analysed=len(df),
            columns=["arm", "age", "weight_kg"],
        )
        add_to_report("univariate")
    elif action == "narrow":
        state.activate_family("study_inclusion", False)
    elif action == "mi":
        from biostats.mi import ImputationSpec, analyze_mi, impute

        frame = state.analysis_frame()
        spec = ImputationSpec(
            columns=["fu_days", "death", "arm", "age", "hba1c_pct", "diabetes"],
            m=3,
            maxit=2,
            seed=1,
        )
        state.set_mi(impute(frame, spec))
        mi = state.analysis_mi()
        levels = mi.levels(["arm", "hba1c_pct"])
        pooled = analyze_mi(
            lambda d: cox_regression(
                d, "fu_days", "death", ["arm", "hba1c_pct"], levels=levels, diagnostics=False
            ),
            mi,
            columns=["fu_days", "death", "arm", "hba1c_pct"],
        )
        remember(
            "cox_mi",
            pooled,
            config=("fu_days", "death", ("arm", "hba1c_pct"), "token"),
            n_analysed=len(mi.base),
        )
        add_to_report("cox_mi")
    items = report_items()
    ss["parts"] = session_parts(state, items, include_stale=False)
    ss["dossier"] = build_dossier_bytes(
        state, items, include_stale=False, title="Tutorial dossier", author="tester"
    )
    ss["dossier_stale"] = build_dossier_bytes(
        state, items, include_stale=True, title="Tutorial dossier", author="tester"
    )


@pytest.fixture
def data_root(tmp_path: Path):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path)
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    yield tmp_path
    dp.DATA_ROOT = previous


def _run(at: AppTest, action: str) -> None:
    at.session_state["action"] = action
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _text(data: bytes) -> tuple[str, list[str], int, int]:
    doc = Document(BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    return text, headings, len(doc.tables), len(doc.inline_shapes)


def test_the_word_dossier_documents_the_flow_and_the_results(data_root: Path) -> None:
    at = AppTest.from_function(_script, default_timeout=300)
    at.session_state["username"] = "dossier_word"
    _run(at, "setup")
    text, headings, n_tables, n_pictures = _text(at.session_state["dossier"])
    for heading in (
        "1. Provenance",
        "2. Analysis population and participant flow",
        "3. Data handling",
        "4. Missing data",
        "5. Results",
        "6. References",
        "Cox proportional-hazards model",
        "Kaplan-Meier survival and log-rank test",
    ):
        assert heading in headings, heading
    assert "Analysis population: 928 of 1,206 rows (12 criteria)." in text
    assert n_pictures == 2  # the participant flow and the Kaplan-Meier curves
    assert (
        n_tables >= 6
    )  # provenance, exclusions, criteria, missing values, Cox estimates, Kaplan-Meier
    assert "Settings: fu_days, death, arm, hba1c_pct" in text
    assert "Computed on tutorial_cohort" in text and "population: 928 of 1,206 rows" in text
    assert "Log-rank test: chi2 = " in text
    assert "von Elm E" in text  # STROBE, for the participant flow
    assert "No multiple imputation was used" in text
    assert "does not certify compliance" in text
    assert (
        "Group comparison" in headings
    )  # the observed comparison (a list of rows) is documented too


def test_only_pooled_results_count_imputations() -> None:
    from app_pages.study_dossier import imputations_of

    assert (
        imputations_of({"results": [{"Variable": "age", "P-Value": 0.5}], "group": "arm"}) is None
    )
    assert imputations_of({"results": {}, "group": "arm"}) is None
    assert imputations_of({"group": "arm"}) is None
    assert imputations_of(None) is None


def test_results_of_another_population_are_left_out_unless_kept_marked(data_root: Path) -> None:
    at = AppTest.from_function(_script, default_timeout=300)
    at.session_state["username"] = "dossier_stale"
    _run(at, "setup")
    _run(at, "narrow")  # the population changes: the three results are stale
    text, headings, _, _ = _text(at.session_state["dossier"])
    assert "Cox proportional-hazards model" not in headings
    assert "Left out because they were computed on other data or another population" in text
    assert "No result was added to the dossier." in text
    marked, headings, _, _ = _text(at.session_state["dossier_stale"])
    assert "Cox proportional-hazards model" in headings
    assert "Not current: computed on another analysis population" in marked


def test_a_multiple_imputation_and_a_pooled_result_are_documented(data_root: Path) -> None:
    at = AppTest.from_function(_script, default_timeout=300)
    at.session_state["username"] = "dossier_mi"
    _run(at, "setup")
    _run(at, "mi")
    text, headings, _, _ = _text(at.session_state["dossier"])
    assert (
        "Multiple imputation" in headings and "Cox proportional-hazards model, pooled" in headings
    )
    assert "Missing data were handled by multiple imputation by chained equations" in text
    assert "Pooled over 3 imputed datasets with Rubin's rules" in text
    assert "Status against the current population: fresh" in text
    assert "Rubin DB (1987)" in text and "Sterne JAC" in text


def test_the_download_builds_the_dossier_without_the_session(data_root: Path) -> None:
    """The download button builds the document after the script run, where the session is not available."""
    from app_pages.study_dossier import _word_bytes

    at = AppTest.from_function(_script, default_timeout=300)
    at.session_state["username"] = "dossier_download"
    _run(at, "setup")
    data = _word_bytes(at.session_state["parts"], title="Built later", author="")  # outside the run
    text, headings, _, n_pictures = _text(data)
    assert text.startswith("Built later") and "Cox proportional-hazards model" in headings
    assert n_pictures == 2


def test_values_imputed_by_an_exact_formula_are_documented_apart() -> None:
    from core.attrition import FlowBox, ParticipantFlow
    from manage.dossier import DossierInput, build_dossier

    inp = DossierInput(
        title="Check",
        flow=ParticipantFlow(main=(FlowBox("a", "Records read", 10, "loaded"),)),
        population_label="all 10 rows",
        exact_imputed={"weight_kg": 3},
        single_imputed={"ldl_mmol_l": 2},
    )
    text = "\n".join(p.text for p in Document(build_dossier(inp)).paragraphs)
    assert "3 values were imputed by an exact formula" in text and "(weight_kg: 3)" in text
    assert "2 values were filled by single imputation (ldl_mmol_l: 2)" in text
