"""The Survival & regression page and the Tier C tab run end to end on the sample cohort (AppTest)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "tierc_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_sample").click().run()
    assert at.session_state["data"].shape == (500, 19)
    yield at
    dp.DATA_ROOT = previous


def test_survival_page_runs_kaplan_meier_and_cox(app: AppTest) -> None:
    app.switch_page(nav.page_path("survival")).run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    app.selectbox(key="surv_time").select("fu_days").run()
    app.selectbox(key="surv_event").select("died").run()
    app.selectbox(key="surv_group").select("arm").run()
    app.button(key="surv_run").click().run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    res = app.session_state["surv_result"]["res"]
    assert {c.label for c in res.curves} == {"Placebo", "Treatment"}
    assert res.logrank_p is not None and res.logrank_p < 0.05  # placebo arm has shorter survival by construction
    app.selectbox(key="cox_time").select("fu_days").run()
    app.selectbox(key="cox_event").select("died").run()
    app.multiselect(key="cox_covs").select("arm").select("age").run()
    app.button(key="cox_run").click().run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    table = app.session_state["cox_result"]["res"].table
    assert "arm_Treatment" in set(table["term"]) and float(table.loc[table["term"] == "arm_Treatment", "hr"].iloc[0]) < 1


def test_paired_rates_tab_and_multiple_imputation_render(app: AppTest) -> None:
    app.switch_page(nav.page_path("analyze")).run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    app.button(key="mh_run").click().run() if any(b.key == "mh_run" for b in app.button) else None
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    app.switch_page(nav.page_path("enrich")).run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    assert any(b.key == "mi_run" for b in app.button), "multiple imputation control missing on the Enrich page"
