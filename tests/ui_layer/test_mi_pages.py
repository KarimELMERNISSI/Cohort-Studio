"""Pooled analyses on the analysis pages once an imputation exists (M2, AppTest on the whole app)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

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
    at.session_state["username"] = "mi_pages_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    at.switch_page(nav.page_path("enrich")).run()
    _ok(at)
    at.number_input(key="mi_m").set_value(3)
    at.number_input(key="mi_maxit").set_value(2)
    at.button(key="mi_run").click().run()
    _ok(at)
    assert at.session_state["_mi"].mi.m == 3
    yield at
    dp.DATA_ROOT = previous


def _stored(at: AppTest, key: str):
    return at.session_state["_results"][key].value


def test_survival_page_pools_kaplan_meier_and_cox(app: AppTest) -> None:
    app.session_state["analysis_data_mode"] = "multiple"
    app.switch_page(nav.page_path("survival")).run()
    _ok(app)
    assert any("pool **3 imputed datasets**" in i.value for i in app.info)
    app.selectbox(key="surv_time").select("fu_days").run()
    app.selectbox(key="surv_event").select("death").run()
    app.selectbox(key="surv_group").select("arm").run()
    app.button(key="surv_run").click().run()
    _ok(app)
    km = _stored(app, "survival_mi")
    assert (
        km.kind == "survival"
        and km.m == 3
        and set(km.extra["curves"])
        == set(km.complete_case.curves[i].label for i in range(len(km.complete_case.curves)))
    )
    assert km.tests["log_rank"].method == "D2"
    app.selectbox(key="cox_time").select("fu_days").run()
    app.selectbox(key="cox_event").select("death").run()
    app.multiselect(key="cox_covs").select("age").select("hba1c_pct").select("arm").run()
    app.button(key="cox_run").click().run()
    _ok(app)
    cox = _stored(app, "cox_mi")
    hba1c = cox.term("hba1c_pct")
    assert (
        hba1c.scale == "log" and hba1c.natural_estimate is not None and hba1c.fmi > 0
    )  # imputed covariate
    assert (
        cox.complete_case is not None and cox.complete_case.n < 1206
    )  # complete cases drop the missing HbA1c
    assert any("Complete-case analysis" in e.label for e in app.expander)


def test_observed_mode_is_unchanged_and_a_stale_imputation_is_not_used(app: AppTest) -> None:
    app.session_state["analysis_data_mode"] = "observed"
    app.switch_page(nav.page_path("survival")).run()
    _ok(app)
    assert not any("imputed datasets" in i.value for i in app.info)
    app.session_state["analysis_data_mode"] = "multiple"
    app.checkbox(
        key=next(c.key for c in app.checkbox if c.key and c.key.startswith("popfam::"))
    ).check().run()
    _ok(app)  # a narrower population: the imputation covers more rows than it now analyses
    assert any("fitted on more rows" in c.label for c in app.checkbox)
    assert any("analyses below use the observed data" in w.value for w in app.warning)
