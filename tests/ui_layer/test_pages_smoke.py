"""Headless smoke test: every registered page renders without exceptions, empty and with the sample cohort."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")
DATA_PAGES = ("validate", "explore", "visualize", "analyze", "enrich", "variable_map", "reproduce")


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any(t.label == "Username" for t in at.text_input), "login form expected"
    # Credential handling is covered by tests/test_authz.py; authenticate through session state.
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "smoke_admin"
    at.session_state["user_role"] = "admin"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    yield at
    dp.DATA_ROOT = previous


@pytest.mark.parametrize("key", [p.key for p in nav.PAGES])
def test_page_renders_without_dataset(app: AppTest, key: str) -> None:
    app.switch_page(nav.page_path(key)).run()
    assert not app.exception, [str(e.value)[:400] for e in app.exception]


def test_sample_cohort_then_data_pages(app: AppTest) -> None:
    app.switch_page(nav.page_path("start")).run()
    app.button(key="start_sample").click().run()
    assert not app.exception, [str(e.value)[:400] for e in app.exception]
    assert app.session_state["data"].shape == (500, 19)
    assert app.session_state["dataset_ref"].origin == "sample"
    for key in DATA_PAGES:
        app.switch_page(nav.page_path(key)).run()
        assert not app.exception, (key, [str(e.value)[:400] for e in app.exception])
