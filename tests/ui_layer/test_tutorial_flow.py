"""The tutorial pack works in the app: load it from Start, then validate, apply the inclusion criteria (AppTest)."""

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
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "tutorial_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    yield at
    dp.DATA_ROOT = previous


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _click(at: AppTest, label: str) -> None:
    next(b for b in at.button if b.label == label).click().run()
    _ok(at)


def test_load_the_tutorial_cohort_with_its_configuration(app: AppTest) -> None:
    app.button(key="start_tutorial").click().run()
    _ok(app)
    data = app.session_state["data"]
    assert data.shape == (1206, 38)
    families = app.session_state["config"]["mask_families"]
    assert len([k for k in families["clinical_anomalies"] if k != "operator"]) == 8
    assert len([k for k in families["study_inclusion"] if k != "operator"]) == 4
    assert (
        "data_enrichments" in app.session_state["config"]
    )  # merged into the default configuration


def test_validate_applies_the_inclusion_criteria(app: AppTest) -> None:
    app.switch_page(nav.page_path("validate")).run()
    _ok(app)
    _click(app, "Run Inclusion Criteria Check")
    assert int(app.session_state["data"]["study_inclusion_all"].sum()) == 939
    # the Results panel (and its Apply button) survives any later rerun: it used to vanish, so Apply never worked
    app.run()
    _ok(app)
    app.button(key="inclusion_data_apply_criteria").click().run()
    _ok(app)
    texts = " ".join(m.value for m in app.markdown)
    assert (
        "267 of 1206 rows will be excluded" in texts
    )  # the confirmation dialog (its button runs in the browser)


def test_anomaly_detection_flags_the_planted_errors(app: AppTest) -> None:
    app.switch_page(nav.page_path("start")).run()
    app.button(key="start_tutorial").click().run()  # a fresh copy: the check flags are reset
    app.switch_page(nav.page_path("validate")).run()
    _ok(app)
    assert "anomaly_check_done" not in app.session_state
    _click(app, "Run Anomaly Detection")
    assert int(app.session_state["data"]["clinical_anomalies_any"].sum()) == 16
    app.run()
    app.button(key="anomaly_data_apply_criteria").click().run()
    _ok(app)
    assert "16 of 1206 rows will be excluded" in " ".join(m.value for m in app.markdown)
