"""The shared analysis population: sidebar checkboxes, banner of the analysis pages, freeze (AppTest)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from core.population import criterion_id
from ui import nav
from ui.components.population import _family_key, _widget_key

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
    assert app.get("plotly_chart")  # the diagram
    assert not any("does not add up" in w.value for w in app.warning)
    details = app.selectbox(key="validate_flow::item")
    assert details.options[0].startswith("Records read in tutorial_cohort")
    details.set_value("stage:study_inclusion").run()
    _ok(app)
    reasons = next(d.value for d in app.dataframe if "Rows it excluded first" in d.value.columns)
    assert reasons["Rows it excluded first"].sum() == 262
    assert any(c.value.startswith("Written consent:") for c in app.caption)  # its definition


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
