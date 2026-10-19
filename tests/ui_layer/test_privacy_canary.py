"""
A value of the data never reaches what is shared (M5): strings planted in a malformed line and in a cell of an
imported file are absent from the trace, the trace report and the study dossier, and the trace names no absolute
path of the server. The malformed line is still shown on the Import page, where it helps the user.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from docx import Document
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from core.population import without_row_lists

CANARY_LINE = "CANARY-LINE-9F2"
CANARY_CELL = "CANARY-CELL-4D1"


def _csv() -> bytes:
    rows = ["id,age,note"] + [f"{i},{40 + i % 30},routine" for i in range(1, 60)]
    rows.insert(5, f"5,47,{CANARY_LINE},extra")  # one field too many: skipped by the parser
    rows.append(f"60,79,{CANARY_CELL}")  # a value of one participant
    return "\n".join(rows).encode()


def _script() -> None:
    import streamlit as st

    from app_pages.study_dossier import build_dossier_bytes
    from ingest.loaders import read_table
    from ui.results import report_items
    from ui.state import get_state

    state = get_state()
    ss = st.session_state
    data = ss["csv"]
    result = read_table(data, "cohort.csv")
    state.load_parsed(
        result.df, filename="cohort.csv", data=data, options=result.options, name="cohort"
    )
    state.note("parse_report", result.report.to_dict(), "Parse report of the imported file")
    ss["examples"] = list(result.report.bad_line_examples)
    ss["trace"] = state.transformation_manager.get_trace()
    ss["dossier"] = build_dossier_bytes(
        state, report_items(), include_stale=False, title="Canary", author="tester"
    )


@pytest.fixture
def data_root(tmp_path: Path):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path)
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    yield tmp_path
    dp.DATA_ROOT = previous


def _docx_text(data: bytes | io.BytesIO) -> str:
    doc = Document(io.BytesIO(data) if isinstance(data, bytes) else data)
    cells = [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    return "\n".join([p.text for p in doc.paragraphs] + cells)


def test_a_planted_value_never_reaches_a_shared_output(data_root: Path) -> None:
    from manage.trace_documenter import TraceDocumenter

    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["username"] = "canary_user"
    at.session_state["csv"] = _csv()
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    # the malformed line is shown on screen, and only there
    assert any(CANARY_LINE in example for example in at.session_state["examples"])

    trace = at.session_state["trace"]
    trace_text = json.dumps(trace, default=str)
    assert CANARY_LINE not in trace_text and CANARY_CELL not in trace_text
    for path in {str(data_root), str(data_root.resolve()), str(data_root).replace("\\", "\\\\")}:
        assert path not in trace_text  # the trace writes paths relative to the workspace
    load = trace["steps"][0]["params"]
    assert load["source_path"].startswith("traces/") and load["file_hash"].startswith("sha256:")
    assert {"cohortstudio", "python", "pandas"} <= set(load["environment"])

    trace_file = data_root / "trace.json"
    trace_file.write_text(trace_text, encoding="utf-8")
    report = _docx_text(TraceDocumenter(str(trace_file)).generate_report())
    assert CANARY_LINE not in report and CANARY_CELL not in report

    dossier = _docx_text(at.session_state["dossier"])
    assert CANARY_LINE not in dossier and CANARY_CELL not in dossier
    assert "SHA-256" in dossier  # the source file's checksum, with its algorithm


def test_a_configuration_for_sharing_lists_no_row() -> None:
    config = {
        "mask_families": {
            "outliers": {
                "operator": "any",
                "iforest": {"row_ids": [[3, 3], [17, 17]], "columns": ["sbp"]},
                "frozen bounds": {"bounds": {"sbp": [80, 200]}},
            },
            "only_rows": {"operator": "any", "lof": {"row_ids": [[4, 4]]}},
            "study_inclusion": {"operator": "all", "adult": {"expression": "df['age'] >= 18"}},
        }
    }
    shared = without_row_lists(config)
    assert "row_ids" not in json.dumps(shared)
    # a family left empty is dropped
    assert set(shared["mask_families"]) == {"outliers", "study_inclusion"}
    assert "frozen bounds" in shared["mask_families"]["outliers"]
    assert "row_ids" in json.dumps(config)  # the session's configuration is unchanged
