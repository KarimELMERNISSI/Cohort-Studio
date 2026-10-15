"""Guidance of Analyze > Hypothesis tests & epidemiology on the sample cohort (AppTest)."""

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
    at.session_state["username"] = "guide_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_sample").click().run()
    at.switch_page(nav.page_path("analyze")).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    yield at
    dp.DATA_ROOT = previous


def _markdown(at: AppTest) -> str:
    return " | ".join(m.value for m in at.markdown)


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _click(at: AppTest, label: str) -> None:
    next(b for b in at.button if b.label == label).click().run()
    _ok(at)


def test_orientation_and_plan_before_running(app: AppTest) -> None:
    assert any(e.label == "Which tab do I need?" for e in app.expander)
    app.selectbox(key="group_col").select("arm").run()
    _ok(app)
    assert "Before you run: the plan" in _markdown(app)
    app.selectbox(key="correction").select("None").run()
    assert "tests without correction" in _markdown(app)
    app.button(key="epi_use_recommended_correction").click().run()
    _ok(app)
    assert app.selectbox(key="correction").value == "Holm (recommended)"


def test_guidance_can_be_hidden(app: AppTest) -> None:
    app.toggle(key="epi_guidance").set_value(False).run()
    _ok(app)
    assert "Before you run: the plan" not in _markdown(app)
    assert not any(e.label == "Which tab do I need?" for e in app.expander)
    app.toggle(key="epi_guidance").set_value(True).run()
    assert "Before you run: the plan" in _markdown(app)


def test_power_from_expected_means_and_proportions(app: AppTest) -> None:
    app.selectbox(key="power_test_type_selector").select("T-Test (Independent)").run()
    app.selectbox(key="es_mode_ttest").select("Expected means and SD").run()
    app.number_input(key="ratio_ttest").set_value(
        1.0
    ).run()  # auto-filled from the cohort's arms otherwise
    _click(app, "Calculate Sample Size")
    success = " ".join(s.value for s in app.success)
    assert "a difference of 5 units (d = 0.50)" in success
    assert "64 participants per group (128 in total)" in success

    app.selectbox(key="power_test_type_selector").select("ANOVA (One-way)").run()
    app.selectbox(key="es_type_anova").select("Expected group means and SD").run()
    _click(app, "Calculate Sample Size")
    assert "per group" in " ".join(s.value for s in app.success)

    app.selectbox(key="power_test_type_selector").select("Chi-Square").run()
    app.selectbox(key="es_type_chi2").select("Expected event proportions").run()
    _click(app, "Calculate Sample Size")
    success = " ".join(s.value for s in app.success)
    assert "an association of w = 0.17" in success and "per group" in success


def test_ancova_model_check(app: AppTest) -> None:
    app.session_state["epi_analysis_mode"] = "Multivariate (ANCOVA/Regression)"
    app.run()
    _ok(app)
    assert "Model check" in _markdown(app)


def test_ancova_runs_on_the_linear_model_and_keeps_its_result(app: AppTest) -> None:
    app.session_state["epi_analysis_mode"] = "Multivariate (ANCOVA/Regression)"
    app.run()
    app.selectbox(key="ancova_group").select("arm").run()
    covariate = app.multiselect(key="ancova_covs").options[0]
    app.multiselect(key="ancova_covs").select(covariate).run()
    app.button(key="ancova_run").click().run()
    _ok(app)
    result = app.session_state["_results"]["ancova"].value
    assert list(result.type2["source"]) == ["arm", covariate, "Residual"]
    assert len(result.adjusted_means) == 2
    assert "Adjusted means" in _markdown(app)
    app.toggle(key="epi_guidance").set_value(False).run()  # another widget: the result stays
    _ok(app)
    assert "Adjusted means" in _markdown(app)
    app.toggle(key="epi_guidance").set_value(True).run()
