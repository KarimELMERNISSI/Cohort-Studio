"""
Versions & traces offers the selected trace as a JSON download (M5): the file as recorded, so its
hash chain verifies when it is uploaded again or replayed elsewhere. The page resolves the trace's
file paths for this machine after reading it; the download must not carry those changes.
"""

import json
import os
from pathlib import Path

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from core.lineage import verify_trace
from core.workspace import UserScope
from manage.transformation_manager import TransformationManager
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")


def test_the_selected_trace_downloads_as_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    tm = TransformationManager(scope=UserScope.for_user("trace_user"))
    tm.initialize_session("cohort.csv")
    tm.add_step(
        "initial_load", {"source_path": "uploads/cohort.csv"}, "Loaded", data={"fp_out": "a"}
    )
    tm.add_step("rename_columns", {"mapping": {"a": "b"}}, "Renamed", data={"fp_in": "a"})
    recorded = Path(tm.get_trace_path()).read_bytes()

    downloads: dict[str, tuple[bytes, str]] = {}
    real = streamlit.download_button

    def capture(label, data=None, *args, **kwargs):
        downloads[label] = (data, kwargs.get("file_name"))
        return real(label, data, *args, **kwargs)

    monkeypatch.setattr(streamlit, "download_button", capture)
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "trace_user"
    at.session_state["user_role"] = "researcher"
    # the source file was found elsewhere on this machine: the page shows that path
    at.session_state["resolved_paths"] = {"cohort.csv": str(tmp_path / "found" / "cohort.csv")}
    at.run()
    at.switch_page(nav.page_path("reproduce")).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    assert any("found" in m.value for m in at.markdown)
    data, name = downloads["Download the trace (JSON)"]
    assert data == recorded and name == Path(tm.get_trace_path()).name
    assert verify_trace(json.loads(data)).status == "verified"
