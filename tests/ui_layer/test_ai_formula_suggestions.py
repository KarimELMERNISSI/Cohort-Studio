"""The AI engine's formula suggestions on Enrich (owner request): each is read as the application computes it and tried
on the data before it can be used; its diagram comes from the formula, never from the model's description."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav
from ui.components.population import _family_key

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")


class _Engine:
    """An AI engine that is running: the pages read its suggestions from the session."""

    initialized = True


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture
def enrich(tmp_path: Path):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path)
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "suggestion_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    at.session_state["rag_manager"] = _Engine()
    at.session_state["ai_variable_suggestions"] = [
        # what models write: a backtick-quoted column, ^ for a power, and no list where a list was asked
        {
            "name": "bmi_check",
            "title": "BMI",
            "formula": "`weight_kg` / (height_cm / 100) ^ 2",
            "missing_variables": None,
            "source_type": "Model Knowledge",
        },
        {
            "name": "pulse_pressure",
            "title": "Pulse pressure",
            "formula": "sbp_baseline - dbp_reading",
            "source_type": "Model Knowledge",
        },
        {
            # a model's text with a link, an image and a colour in it, as a hostile document could make it write
            "name": "hba1c_high",
            "title": "[Read more](http://e.x)",
            "description": "![beacon](http://e.x/p.png) :red[alarm]",
            "formula": "np.where(hba1c_pct >= 6.5, 1, 0)",
            "source_type": "Model Knowledge",
        },
    ]
    at.session_state["target_impute_var"] = "sbp_6m"
    at.session_state["imputation_suggestions_for"] = "sbp_6m"
    at.session_state["imputation_suggestions"] = [
        {"name": "Imputed_sbp_6m", "formula": "sbp_baseline + np.where(nyha_class > 2, 5.0, -2.0)"},
        {
            "name": "Imputed_sbp_6m",
            "formula": "sbp_baseline + np.where(nyha_class == 'III', 5.0, -2.0)",
        },
    ]
    at.switch_page(nav.page_path("enrich")).run()
    _ok(at)
    yield at
    dp.DATA_ROOT = previous


def test_a_variable_suggestion_is_drawn_and_tried_as_it_will_be_computed(enrich: AppTest) -> None:
    diagrams = [chart.proto.spec for chart in enrich.get("graphviz_chart")]
    bmi = next(d for d in diagrams if "bmi_check" in d)
    assert '"input:weight_kg"' in bmi and '"input:height_cm"' in bmi and "Error" not in bmi
    pulse = next(d for d in diagrams if "pulse_pressure" in d)
    # a variable the data do not have, from the formula itself
    assert "dbp_reading (missing)" in pulse
    captions = " ".join(c.value for c in enrich.caption)
    assert "On your data: a value for" in captions
    assert "Cannot be computed on your data: `not in the data: dbp_reading`" in captions
    enrich.button(key="apply_var_0_bmi_check").click().run()
    _ok(enrich)
    # what the card computed: in the application's notation
    assert enrich.session_state["formula_input"] == '""weight_kg"" / (height_cm / 100) ** 2'


def test_an_imputation_suggestion_that_the_data_refuse_cannot_be_used(enrich: AppTest) -> None:
    captions = " ".join(c.value for c in enrich.caption)
    assert "Cannot be computed on your data" in captions and "nyha_class holds text" in captions
    assert "On your data: fills 183 of the 183 missing values of `sbp_6m`" in captions
    assert enrich.button(key="use_imp_sugg_0").disabled
    enrich.button(key="use_imp_sugg_1").click().run()
    _ok(enrich)
    assert enrich.session_state["imputation_formula_input"] == (
        "sbp_baseline + np.where(nyha_class == 'III', 5.0, -2.0)"
    )
    # applied with no criterion ticked: every row, and the population is recorded all the same, so the draft
    # knows when a criterion is ticked before it is applied
    next(b for b in enrich.button if b.label == "Apply Targeted Imputation").click().run()
    _ok(enrich)
    step = enrich.session_state["_data_session"].draft.steps[-1]
    assert step.params["filled_count"] == 183 and step.params["row_wise"] is True
    assert step.description == "Imputed 183 values in sbp_6m using a formula"
    assert step.population is not None and step.population["criteria"] == []
    assert step.population["n_included"] == 1206


def test_a_targeted_imputation_is_tried_on_and_fills_the_analysis_population(
    enrich: AppTest,
) -> None:
    # biostatistics review (2026-10-21): as the global imputation, the excluded patients neither inform nor receive
    for family in ("clinical_anomalies", "study_inclusion"):
        enrich.checkbox(key=_family_key(family)).check().run()
        _ok(enrich)
    assert any("fills, the 928 rows of the analysis population" in i.value for i in enrich.info)
    captions = " ".join(c.value for c in enrich.caption)
    assert "On your data: fills 150 of the 150 missing values of `sbp_6m`" in captions

    enrich.button(key="use_imp_sugg_1").click().run()
    next(b for b in enrich.button if b.label == "Apply Targeted Imputation").click().run()
    _ok(enrich)
    session = enrich.session_state["_data_session"]
    step = session.draft.steps[-1]
    assert step.function == "targeted_imputation" and step.params["filled_count"] == 150
    assert step.population and step.population["n_included"] == 928
    included = session.population(enrich.session_state["config"]).included
    excluded = session.base.index.difference(included)
    # the excluded rows keep their missing values
    kept = session.draft.frame.loc[excluded, "sbp_6m"].isna()
    assert kept.equals(session.base.loc[excluded, "sbp_6m"].isna())


def test_an_exact_formula_is_one_of_each_rows_own_values(enrich: AppTest) -> None:
    # biostatistics review (2026-10-21): an aggregate ticked "Exact" was kept as observed values
    enrich.text_area(key="imputation_formula_input").input("sbp_baseline.mean() + 0 * age").run()
    enrich.checkbox(key="imputation_exact").check().run()
    next(b for b in enrich.button if b.label == "Apply Targeted Imputation").click().run()
    _ok(enrich)
    assert any("This one also uses other rows" in e.value for e in enrich.error)
    draft = enrich.session_state["_data_session"].draft
    assert draft is None or not draft.steps


def test_a_new_formula_with_a_caret_is_refused(enrich: AppTest) -> None:
    enrich.text_input(key="variable_name_input").input("bmi_caret").run()
    enrich.text_area(key="formula_input").input("weight_kg / (height_cm / 100) ^ 2").run()
    _ok(enrich)
    assert any("^ is not a power in a formula" in e.value for e in enrich.error)


def test_a_models_text_stays_text_and_a_value_from_a_missing_input_is_named(
    enrich: AppTest,
) -> None:
    shown = " ".join(m.value for m in enrich.markdown)
    assert r"\[Read more\]\(http\://e\.x\)" in shown and "[Read more](http" not in shown
    assert r"\!\[beacon\]" in shown and r"\:red\[alarm\]" in shown
    captions = " ".join(c.value for c in enrich.caption)
    # np.where answers "false" for a missing HbA1c: those rows get a value all the same, and the card says how many
    assert "a value for 1,206 of 1,206 rows (266 from a missing `hba1c_pct`)" in captions
