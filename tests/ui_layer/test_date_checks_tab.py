"""
Validate > Date checks (M6): the text columns that look like dates, how they would be read, and their conversion as one
step of the trace. The conversion sat above Validate's tabs and parsed every such column whenever Validate opened on
new data; a line there now counts them from the column profile, without parsing them, and opens the tab.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from app_pages import data_monitoring as dm
from app_pages import validate_dates
from core.fingerprint import frame_fingerprint
from manage.reproduction_manager import apply_steps
from ui import nav

MAIN = str(Path(__file__).resolve().parents[2] / "main.py")
DATES = ["enrol_date", "last_contact_date"]


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _line(at: AppTest) -> list[str]:
    """The line above the tabs, when it is drawn: its caption."""
    if not [b for b in at.button if b.key == "open_date_checks"]:
        return []
    return [c.value for c in at.caption if "like dates" in c.value]


def test_the_tab_reads_the_dates_on_request_and_records_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    reads: list[int] = []
    preview = validate_dates._date_preview

    def counted(df, analyzer):
        reads.append(1)
        return preview(df, analyzer)

    monkeypatch.setattr(validate_dates, "_date_preview", counted)
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "date_checks_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)

    # the dashboard: the line counts the columns, and nothing is parsed
    at.switch_page(nav.page_path("validate")).run()
    _ok(at)
    assert _line(at) == [":material/calendar_month: 2 text columns look like dates."]
    assert reads == []

    # its button opens the tab, which parses them once, until the data change
    at.button(key="open_date_checks").click().run()
    _ok(at)
    assert at.session_state[dm.TAB_KEY] == "Date checks"
    headings = [s.value for s in at.subheader if "like dates" in s.value]
    assert headings == ["2 text columns look like dates"]
    assert _line(at) == []
    assert reads == [1]
    at.run()
    assert reads == [1]

    # a link (the Start page's) opens the tab on its first run, without the line
    at.session_state[dm.TAB_KEY] = "Data Quality Dashboard"
    at.run()
    assert _line(at)
    at.query_params[nav.TAB_PARAM] = "date_checks"
    at.run()
    _ok(at)
    assert at.session_state[dm.TAB_KEY] == "Date checks"
    assert _line(at) == []

    # the conversion: one step, as Validate recorded it before M6, that replays to the same table
    before = at.session_state["data"].copy()
    at.button(key="convert_dates").click().run()
    _ok(at)
    step = at.session_state["transformation_manager"].history[-1]
    assert step["function"] == "data_type_conversion"
    assert step["params"] == {
        "columns": DATES,
        "new_type": "datetime",
        "method": "smart_parse_dates",
        "notes": dict.fromkeys(DATES, "Strict Structure: Y=P2, M=P1, D=P0"),
        "values_lost": dict.fromkeys(DATES, 0),
    }
    after = at.session_state["data"]
    assert all(pd.api.types.is_datetime64_any_dtype(after[c]) for c in DATES)
    replayed = apply_steps(before, [step])
    assert replayed.error is None and replayed.divergence is None
    assert frame_fingerprint(replayed.df) == step["data"]["fp_out"] == frame_fingerprint(after)

    # nothing left to convert: the data changed, so the open tab parsed again
    assert reads == [1, 1]
    assert "No text column looks like dates." in [c.value for c in at.caption]
    del at.query_params[nav.TAB_PARAM]
    at.session_state[dm.TAB_KEY] = "Data Quality Dashboard"
    at.run()
    _ok(at)
    assert _line(at) == []
