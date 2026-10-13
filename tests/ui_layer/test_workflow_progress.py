"""The Start page follows real progress: Validate and Explore tick their boxes, a data change asks for them again."""

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
    at.session_state["username"] = "progress_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    yield at
    dp.DATA_ROOT = previous


def _start_captions(at: AppTest) -> str:
    at.switch_page(nav.page_path("start")).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return " | ".join(c.value for c in at.caption)


def _marks(at: AppTest) -> dict:
    return at.session_state["workflow_marks"]["marks"]


def test_progress_follows_the_pages_and_the_data(app: AppTest) -> None:
    assert "No dataset loaded yet." in _start_captions(app)

    app.button(key="start_sample").click().run()
    captions = _start_captions(app)
    assert "Dataset loaded: sample_cohort" in captions
    assert "Not checked yet for this dataset." in captions

    # Validate: the quality audit runs on the page and marks the step at the current revision
    app.switch_page(nav.page_path("validate")).run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    assert _marks(app)["validate"].revision == 0
    assert "Quality checked on the current data (quality score" in _start_captions(app)

    # Explore: descriptive statistics count as analysing the current data
    app.switch_page(nav.page_path("explore")).run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    assert _marks(app)["analyse"].detail == "descriptive statistics"
    assert "Analysed on the current data (descriptive statistics)." in _start_captions(app)

    # A change to the data (as the Enrich page records it) makes both steps "run again"
    app.session_state["transformation_manager"].add_step(
        "variable_computation", {"name": "age_over_65"}, "Computed variable age_over_65"
    )
    captions = _start_captions(app)
    assert "Checked before the last 1 change to the data: check the quality again." in captions
    assert "Results predate the last 1 change to the data: run them again." in captions
    assert "1 change applied, last: Computed variable age_over_65." in captions

    # Validating again brings the step back to done on the new revision
    app.switch_page(nav.page_path("validate")).run()
    assert _marks(app)["validate"].revision == 1
    assert "Quality checked on the current data" in _start_captions(app)

    # Loading another dataset starts the workflow afresh (new trace session)
    app.button(key="start_sample").click().run()
    assert "Not checked yet for this dataset." in _start_captions(app)
