"""
Validate's configuration upload, on the page (AppTest): the file is merged once, by the uploader's callback, before
the sidebar is drawn.

Merged in the page on every run while the file stayed in the uploader, it left the sidebar, drawn before the page, one
run behind: "Analysis population: every row" right after a configuration with criteria was loaded, and nothing to
tick until the next click (M5b browser check); and a rule removed from the session after the upload came back from
the file on the next run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav
from ui.components.text import plain

ROOT = Path(__file__).resolve().parents[2]
CRITERION = "At least 2 rules of clinical_anomalies"
THRESHOLD = {
    "mask_families": {
        "clinical_anomalies": {
            "operator": {"threshold": 2},
            "Older than 70": {"numeric": False, "expression": "df['age'] > 70"},
            "High systolic pressure": {"numeric": False, "expression": "df['sbp'] > 150"},
            "Obese": {"numeric": False, "expression": "df['bmi'] >= 30"},
        }
    }
}
REFUSED = {
    "mask_families": {"study_inclusion": {"age": {"numeric": True, "lower_bound": "18 years"}}}
}


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _upload(at: AppTest, content: object | None) -> None:
    data = (
        None
        if content is None
        else ("config.json", json.dumps(content).encode("utf-8"), "application/json")
    )
    at.sidebar.file_uploader[0].set_value(data).run()
    _ok(at)


@pytest.fixture
def validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Validate on the sample cohort, whose configuration starts with no criterion."""
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "upload_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_sample").click().run()
    at.switch_page(nav.page_path("validate")).run()
    _ok(at)
    return at


def test_the_sidebar_lists_the_uploaded_criteria_in_the_run_of_the_upload(
    validate: AppTest,
) -> None:
    before = json.dumps(validate.session_state["config"]["mask_families"], sort_keys=True)
    _upload(validate, REFUSED)
    assert any("This configuration was not loaded" in e.value for e in validate.sidebar.error)
    assert json.dumps(validate.session_state["config"]["mask_families"], sort_keys=True) == before
    _upload(validate, THRESHOLD)
    assert [s.value for s in validate.sidebar.success] == [
        "Configuration loaded and merged successfully!"
    ]
    assert f"{plain(CRITERION)} (52 excluded)" in [box.label for box in validate.sidebar.checkbox]


def test_the_file_is_merged_once_and_forgotten_with_it(validate: AppTest) -> None:
    _upload(validate, THRESHOLD)
    del validate.session_state["config"]["mask_families"]["clinical_anomalies"]["Obese"]
    validate.run()
    assert "Obese" not in validate.session_state["config"]["mask_families"]["clinical_anomalies"]
    _upload(validate, None)  # the file removed from the uploader
    assert not validate.sidebar.success
    assert "validate_config_uploaded" not in validate.session_state
