"""The shared analysis population: sidebar checkboxes, banner of the analysis pages, freeze (AppTest)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from core.population import criterion_id
from ui import nav
from ui.components.population import _family_key, _widget_key
from ui.components.text import plain

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")


@pytest.fixture
def app(tmp_path: Path):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path)
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "population_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    yield at
    dp.DATA_ROOT = previous


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _rows(at: AppTest) -> int:
    return len(at.session_state["data"])


def test_the_sidebar_selector_filters_every_page_without_deleting_rows(app: AppTest) -> None:
    assert _rows(app) == 1206
    app.checkbox(key=_family_key("clinical_anomalies")).check().run()
    _ok(app)
    assert _rows(app) == 1190
    app.checkbox(key=_family_key("study_inclusion")).check().run()
    _ok(app)
    assert _rows(app) == 928
    assert len(app.session_state["_data_session"].base) == 1206  # nothing deleted

    app.switch_page(nav.page_path("analyze")).run()
    _ok(app)
    assert any("928 of 1,206 rows (12 criteria)" in i.value for i in app.info)  # the banner

    consent = _widget_key(criterion_id("study_inclusion", "Written consent"))
    app.checkbox(key=consent).uncheck().run()
    _ok(app)
    # 937: the 14 patients without consent come back, except the 5 who also fail another criterion
    assert _rows(app) == 937
    # a family partly ticked shows its box unticked: a click ticks the whole family, a second unticks it
    inclusion = _family_key("study_inclusion")
    assert app.checkbox(key=inclusion).value is False
    app.checkbox(key=inclusion).check().run()
    assert _rows(app) == 928
    app.checkbox(key=inclusion).uncheck().run()
    app.checkbox(key=_family_key("clinical_anomalies")).uncheck().run()
    _ok(app)
    assert _rows(app) == 1206


def test_freezing_keeps_only_the_population_as_one_traced_step(app: AppTest) -> None:
    app.checkbox(key=_family_key("clinical_anomalies")).check().run()
    app.switch_page(nav.page_path("validate")).run()
    _ok(app)
    app.session_state["validate_tab"] = "Analysis population"  # only the open tab runs
    app.run()
    app.checkbox(key="pop_freeze_confirm").check().run()
    app.button(key="pop_freeze").click().run()
    _ok(app)
    session = app.session_state["_data_session"]
    assert len(session.base) == 1190 and session.ticked == ()
    steps = [s["function"] for s in app.session_state["transformation_manager"].history]
    assert steps[-1] == "freeze_population"


def test_validate_shows_the_participant_flow_and_the_reasons_of_each_exclusion(
    app: AppTest,
) -> None:
    app.checkbox(key=_family_key("clinical_anomalies")).check().run()
    app.checkbox(key=_family_key("study_inclusion")).check().run()
    app.switch_page(nav.page_path("validate")).run()
    _ok(app)
    app.session_state["validate_tab"] = "Analysis population"  # only the open tab runs
    app.run()
    assert app.get("plotly_chart")  # the diagram
    assert not any("does not add up" in w.value for w in app.warning)
    details = app.selectbox(key="validate_flow::item")
    assert details.options[0].startswith("Records read in tutorial_cohort")
    details.set_value("stage:study_inclusion").run()
    _ok(app)
    reasons = next(d.value for d in app.dataframe if "Rows it excluded first" in d.value.columns)
    assert reasons["Rows it excluded first"].sum() == 262
    # its definition, with its role, shown as written
    consent = plain("Written consent: excluded unless")
    assert any(c.value.startswith(consent) for c in app.caption)


def test_imputation_on_enrich_is_fitted_on_the_population_only(app: AppTest) -> None:
    app.checkbox(key=_family_key("clinical_anomalies")).check().run()
    app.checkbox(key=_family_key("study_inclusion")).check().run()
    session = app.session_state["_data_session"]
    before = session.base.copy()
    included = session.population(app.session_state["config"]).included
    excluded = before.index.difference(included)

    app.switch_page(nav.page_path("enrich")).run()
    _ok(app)
    next(s for s in app.selectbox if s.label == "Numerical Imputation Method").select(
        "Median"
    ).run()
    next(b for b in app.button if b.label == "Apply Imputer").click().run()
    _ok(app)
    app.run()  # the draft banner at the top of the page shows the staged imputation from the next run
    app.button(key="draft_apply").click().run()
    _ok(app)

    after = app.session_state["_data_session"].base
    numeric = [c for c in after.columns if after[c].dtype.kind == "f"]
    assert not after.loc[included, numeric].isna().any().any()  # the population is filled
    assert after.loc[excluded].isna().equals(before.loc[excluded].isna())  # excluded rows untouched
    step = app.session_state["transformation_manager"].history[-1]
    assert step["function"] == "imputation" and step["population"]["n_included"] == 928
    assert step["params"]["rows"] == "population"


def test_a_single_imputation_recomputes_the_declared_derived_variables(app: AppTest) -> None:
    import numpy as np

    from core.derived import DerivedVariable, with_definitions

    app.checkbox(key=_family_key("clinical_anomalies")).check().run()
    app.checkbox(key=_family_key("study_inclusion")).check().run()
    bmi = DerivedVariable(
        "bmi_kg_m2",
        "weight_kg / (height_cm / 100) ** 2",
        inverses={"height_cm": "100 * (weight_kg / bmi_kg_m2) ** 0.5"},
    )
    phenotype = DerivedVariable(
        "hf_phenotype",
        "np.where(lvef_pct <= 40, 'HFrEF', np.where(lvef_pct <= 49, 'HFmrEF', 'HFpEF'))",
    )
    app.session_state["config"] = with_definitions(app.session_state["config"], [bmi, phenotype])
    app.switch_page(nav.page_path("enrich")).run()
    _ok(app)
    assert app.checkbox(key="global_recompute_derived").value  # on by default
    next(s for s in app.selectbox if s.label == "Numerical Imputation Method").select(
        "Median"
    ).run()
    next(b for b in app.button if b.label == "Apply Imputer").click().run()
    _ok(app)
    app.run()
    app.button(key="draft_apply").click().run()
    _ok(app)

    session = app.session_state["_data_session"]
    included = session.population(app.session_state["config"]).included
    after = session.base.loc[included]
    assert not after[["bmi_kg_m2", "height_cm"]].isna().any().any()
    # every BMI follows its formula, the imputed ones too
    expected = after["weight_kg"] / (after["height_cm"] / 100) ** 2
    assert np.allclose(after["bmi_kg_m2"], expected, atol=0.05 + 1e-9)
    history = app.session_state["transformation_manager"].history
    functions = [s["function"] for s in history]
    first = functions.index("imputation")
    imputation = history[first]
    # the phenotype, complete once filled exactly, informs the imputation as any recorded variable does
    assert imputation["params"]["set_aside"] == ["bmi_kg_m2"]
    assert imputation["params"]["derived"]["recompute"] is True
    assert [b["definition"]["target"] for b in imputation["params"]["derived"]["definitions"]] == [
        "bmi_kg_m2"
    ]
    assert imputation["description"].endswith(
        "set aside, to be computed from the imputed values: bmi_kg_m2"
    )
    # exactly first (34 BMIs, 10 heights), then the imputation, then the 24 BMIs of imputed heights
    fills = [
        (s["params"]["target_variable"], s["params"]["filled_count"])
        for s in history
        if s["function"] == "targeted_imputation"
    ]
    assert fills == [
        ("bmi_kg_m2", 34),
        ("height_cm", 10),
        ("hf_phenotype", 30),
        ("bmi_kg_m2", 24),
    ]
    last = [s for s in history if s["function"] == "targeted_imputation"][-1]
    assert last["description"].endswith("; all from imputed values, so single-imputed")
    assert (
        functions[first - 1] == "targeted_imputation"
        and functions[first + 1] == "targeted_imputation"
    )

    def count(cells, column: str) -> int:
        return int(
            np.isin(cells.get(column, np.array([], dtype=np.int64)), included.to_numpy()).sum()
        )

    assert count(session.exact_imputed, "bmi_kg_m2") == 34
    assert count(session.single_imputed, "bmi_kg_m2") == 24
    assert count(session.exact_imputed, "height_cm") == 10
    assert count(session.single_imputed, "height_cm") == 24

    from core.state import ParseOptions
    from ingest.loaders import read_table
    from manage.reproduction_manager import run_trace
    from ui.state import TUTORIAL_COHORT_PATH

    source = read_table(
        TUTORIAL_COHORT_PATH.read_bytes(), TUTORIAL_COHORT_PATH.name, ParseOptions(dayfirst=True)
    ).df
    replay = run_trace(app.session_state["transformation_manager"].get_trace(), source_df=source)
    assert replay.error is None and replay.divergence is None, replay.warnings
    assert replay.df["bmi_kg_m2"].equals(session.base["bmi_kg_m2"])
