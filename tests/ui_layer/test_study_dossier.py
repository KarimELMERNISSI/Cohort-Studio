"""Results added to the study dossier, the participant flow of the dossier page and the Word dossier (M3)."""

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


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "dossier_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    at.checkbox(key=_family_key("clinical_anomalies")).check().run()
    at.checkbox(key=_family_key("study_inclusion")).check().run()
    _ok(at)
    yield at
    dp.DATA_ROOT = previous


def test_a_result_goes_into_the_dossier_and_its_flow(app: AppTest) -> None:
    app.switch_page(nav.page_path("survival")).run()
    app.selectbox(key="cox_time").select("fu_days").run()
    app.selectbox(key="cox_event").select("death").run()
    app.multiselect(key="cox_covs").select("arm").select("hba1c_pct").run()
    app.button(key="cox_run").click().run()
    _ok(app)
    app.button(key="report_add::cox").click().run()
    _ok(app)
    assert any(b.key == "report_remove::cox" for b in app.button)  # now in the dossier
    (item,) = app.session_state["_report"].values()
    assert item.title == "Cox proportional-hazards model" and "hba1c_pct" in item.settings

    app.switch_page(nav.page_path("dossier")).run()
    _ok(app)
    flow_items = app.selectbox(key="dossier_flow::item").options
    assert any(o.startswith("Cox proportional-hazards model (n = ") for o in flow_items)
    assert any("Cox proportional-hazards model" in m.value for m in app.markdown)


def test_a_result_of_another_population_is_marked_stale_then_removed(app: AppTest) -> None:
    app.checkbox(key=_family_key("study_inclusion")).uncheck().run()
    app.switch_page(nav.page_path("dossier")).run()
    _ok(app)
    assert any("computed on another analysis population" in c.value.lower() for c in app.caption)
    flow_items = app.selectbox(key="dossier_flow::item").options
    assert not any(
        o.startswith("Cox proportional-hazards model") for o in flow_items
    )  # not in the flow
    item_id = next(iter(app.session_state["_report"]))
    app.button(key=f"dossier_remove::{item_id}").click().run()
    _ok(app)
    assert not app.session_state["_report"]
    app.checkbox(key=_family_key("study_inclusion")).check().run()
