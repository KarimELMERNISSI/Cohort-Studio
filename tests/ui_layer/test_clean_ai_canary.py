"""
What the AI engine is sent or answers on Data > Clean > Categories never reaches a shared output (the trace report, the
dossier, the log), and the trace keeps only what the replay and the methods need: the choices, who proposed each, the
user's reasons. The engine is faked as tests/ui_layer/test_clean_categories.py fakes it, with an Ollama cloud model on
this machine: a cloud recipient, which receives only the values held by 5 rows or more.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
from docx import Document
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp

RARE = "CANARY-RARE-1R"  # 1 row: never sent to a cloud recipient, never offered, in no output
SENT = "CANARY-SENT-6F"  # 6 rows: sent, never chosen: in no record
REASON = "CANARY-REASON-AI"  # the model's words: shown on a pill and as a hint, never recorded
ECHO = (
    "CANARY-ECHO-" + "q" * 300
)  # a value the model invents: never offered; the store keeps 120 characters
TYPED = (
    "CANARY-TYPED-REASON"  # the user's reason for a grouping of values under 5 rows: the trace only
)
MODEL = "canary-model:120b-cloud"  # an Ollama cloud model on this machine: a cloud recipient
FRAME = {
    "diagnosis": ["Myocardial infarction"] * 40
    + ["Stroke"] * 40
    + ["MI"] * 6
    + [SENT] * 6
    + ["Cerebral infarct"] * 8
    + ["AVC"] * 3
    + [RARE]
}
AI = {
    "provider": "ollama",
    "base_url": "http://localhost:11434",
    "model": MODEL,
    "proposals": [
        {"value": "MI", "clean": "Myocardial infarction", "reason": REASON},
        {"value": RARE, "clean": "Stroke", "reason": REASON},
        {"value": ECHO, "clean": "W" * 300, "reason": REASON + " " + "z" * 300},
    ],
}


def _app(frame: dict, ai: dict) -> None:
    import pandas as pd
    import streamlit as st

    if "rag_manager" not in st.session_state:

        class _Engine:
            initialized = True
            confirmed = {
                frozenset({"ollama@ollama.com"})
            }  # the box of Settings > AI engine, ticked

            def __init__(self) -> None:
                self.provider, self.base_url = ai["provider"], ai["base_url"]
                config = type("Config", (), {"model_name": ai["model"]})()
                self.llm = type("Model", (), {"config": config})()
                self.sent: list = []

            def recipients(self) -> frozenset:
                return frozenset({"ollama@ollama.com"})

            def suggest_category_groups(self, column, values, max_proposals=30):
                self.sent.append((column, list(values)))
                return ai["proposals"], None

        st.session_state["rag_manager"] = _Engine()

    from app_pages.clean_categories import render_categories
    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state.setdefault("username", "canary_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        df = pd.DataFrame(frame)
        state.set_dataset(
            df,
            DatasetRef(name="canary", origin="upload", file_hash="c"),
            source_bytes=df.to_csv(index=False).encode("utf-8"),
            source_name="canary.csv",
        )
    render_categories()
    if st.session_state.get("build"):
        from app_pages.study_dossier import build_dossier_bytes
        from ui.results import report_items

        st.session_state["trace"] = state.transformation_manager.get_trace()
        st.session_state["dossier"] = build_dossier_bytes(
            state, report_items(), include_stale=False, title="Canary", author="tester"
        )


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A loaded dataset writes its user's workspace: in a temporary folder, never in the data folder (every test)."""
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _one(widgets, prefix: str):
    found = [w for w in widgets if str(w.key).startswith(prefix)]
    assert len(found) == 1, (prefix, [w.key for w in widgets])
    return found[0]


def _docx_text(data: bytes | io.BytesIO) -> str:
    doc = Document(io.BytesIO(data) if isinstance(data, bytes) else data)
    cells = [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    return "\n".join([p.text for p in doc.paragraphs] + cells)


def test_what_the_ai_engine_is_sent_or_answers_never_reaches_a_shared_output(
    data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from manage.trace_documenter import TraceDocumenter

    caplog.set_level(logging.DEBUG)
    at = AppTest.from_function(_app, kwargs={"frame": FRAME, "ai": AI}, default_timeout=180)
    at.run()
    _ok(at)
    popover = "clean_ai_open::" + str(_one(at.button, "clean_map::").key).split("::", 1)[1]
    at.session_state[popover] = True
    at.run()
    _ok(at)
    # 1. the preview lists exactly what is sent, names the recipient and the values held back
    listed = [
        d.value["Values sent"].tolist()
        for d in at.dataframe
        if list(d.value.columns) == ["Values sent"]
    ]
    expected = sorted(
        ["Cerebral infarct", SENT, "MI", "Myocardial infarction", "Stroke"],
        key=lambda v: (v.casefold(), v),
    )
    assert listed == [expected]
    assert any(
        c.value.startswith("Send to Ollama at ollama") for c in at.caption
    )  # drawn as text: "ollama\.com"
    assert any("2 values held by fewer than 5 rows are not sent" in c.value for c in at.caption)
    _one(at.button, "clean_ai_send::").click().run()
    _ok(at)
    engine = at.session_state["rag_manager"]
    assert engine.sent == [("diagnosis", expected)]  # A to Z, the 1- and 3-row values held back
    # 2. the store keeps at most 120 characters of what the model wrote, never the prompt or the raw answer
    stored = at.session_state["clean_ai"]["diagnosis"]
    keys = {
        "fp",
        "sent",
        "proposals",
        "refused",  # the page's own refusal, apart from the engine's error
        "error",
        "provider",
        "model",
        "where",
        "values_sent",
        "values_held",
    }
    assert set(stored) == keys
    assert all(len(v) <= 120 for p in stored["proposals"] for v in p.values())
    assert (stored["provider"], stored["model"], stored["where"]) == (
        "ollama",
        MODEL,
        "Ollama at ollama.com",
    )
    assert (stored["values_sent"], stored["values_held"]) == (5, 2)
    # 3. only a value sent is offered; the model's words are a hint, never the user's reason
    (ai,) = [w for w in at.pills if "::ai::" in str(w.key)]
    assert ai.options == [
        f"**MI** → **Myocardial infarction** · {REASON.replace('-', chr(92) + '-')} · 6 → 40 rows"
    ]
    ai.select("MI").run()
    _ok(at)
    _one(at.multiselect, "clean_pick::").set_value(["AVC", "Cerebral infarct"]).run()
    _one(at.selectbox, "clean_into::").set_value("Stroke").run()
    _one(at.button, "clean_map::").click().run()
    _ok(at)
    asks = {
        str(w.key).rsplit("::", 1)[1]: w
        for w in at.text_input
        if str(w.key).startswith("clean_reason::")
    }
    assert set(asks) == {"Myocardial infarction", "Stroke"} and all(
        w.value == "" for w in asks.values()
    )
    assert REASON in (asks["Myocardial infarction"].placeholder or "")
    asks["Myocardial infarction"].input("MI abbreviates it").run()
    asks = {
        str(w.key).rsplit("::", 1)[1]: w
        for w in at.text_input
        if str(w.key).startswith("clean_reason::")
    }
    asks["Stroke"].input(TYPED).run()
    _ok(at)
    at.button(key="clean_apply::diagnosis").click().run()
    _ok(at)
    # 4. the record: who proposed each AI choice and where it ran, never the prompt, the answer or a value only sent
    params = at.session_state["transformation_manager"].history[-1]["params"]
    assert params["kinds"] == {"MI": "ai", "AVC": "grouping", "Cerebral infarct": "grouping"}
    assert params["ai_by"] == {
        "MI": {
            "provider": "ollama",
            "model": MODEL,
            "where": "Ollama at ollama.com",
            "values_sent": 5,
            "values_held": 2,
        }
    }
    assert params["groupings"]["Stroke"] == {
        "reason": TYPED,
        "values": ["AVC", "Cerebral infarct", "Stroke"],
        "each_5_rows": False,
    }
    at.session_state["build"] = True
    at.run()
    _ok(at)
    trace_text = json.dumps(at.session_state["trace"], default=str, ensure_ascii=False)
    for absent in (RARE, SENT, "CANARY-REASON", "CANARY-ECHO", "W" * 61):
        assert absent not in trace_text, absent
    assert TYPED in trace_text and MODEL in trace_text  # the replay's record, the provenance
    # 5. the trace report counts: no value under 5 rows, no reason beside them, who proposed by count and where
    trace_file = data_root / "trace.json"
    trace_file.write_text(trace_text, encoding="utf-8")
    report = _docx_text(TraceDocumenter(str(trace_file)).generate_report())
    for absent in (
        RARE,
        SENT,
        "CANARY-REASON",
        "CANARY-ECHO",
        TYPED,
        "AVC",
        "Cerebral infarct",
        "ai_by",
    ):
        assert absent not in report, absent
    assert "3 values grouped (the reason is in the trace)" in report
    assert f"ollama {MODEL} (Ollama at ollama.com): 1 value" in report
    # 6. the dossier and the log: nothing of the data, of the model's answer or of the user's reason
    dossier = _docx_text(at.session_state["dossier"])
    for absent in (RARE, SENT, "CANARY", "AVC", "Cerebral infarct"):
        assert absent not in dossier, absent
    assert "CANARY" not in caplog.text and "Cerebral infarct" not in caplog.text
    # 7. written again without the AI store: who proposed is kept, never asked again
    at.session_state["clean_ai"] = {}
    at.session_state["clean_manual"] = {}
    at.run()
    _ok(at)
    assert at.session_state["clean_manual"]["diagnosis"]["ai_by"] == params["ai_by"]
    assert len(engine.sent) == 1


def test_send_asks_nothing_of_a_recipient_no_box_confirmed() -> None:
    """The gate is Settings' box: a recipient none of the session's boxes named is never asked, whatever the page."""
    unconfirmed = {**AI, "proposals": []}
    at = AppTest.from_function(
        _app, kwargs={"frame": FRAME, "ai": unconfirmed}, default_timeout=180
    )
    at.run()
    at.session_state["rag_manager"].confirmed = set()
    popover = "clean_ai_open::" + str(_one(at.button, "clean_map::").key).split("::", 1)[1]
    at.session_state[popover] = True
    at.run()
    _one(at.button, "clean_ai_send::").click().run()
    _ok(at)
    assert at.session_state["rag_manager"].sent == []
    assert "not confirmed" in at.session_state["clean_ai"]["diagnosis"]["refused"]


def test_send_asks_nothing_when_the_ollama_server_does_not_say_where_it_runs_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asked again at Send, a server that does not answer (late, unreachable) never reads as running the model here:
    nothing is sent, and the page says that it sent nothing, not that the engine could not answer."""
    import manage.rag.providers.ollama_server as server

    def silent(base_url, model):
        raise server.NoAnswer(model)

    monkeypatch.setattr(server, "runs_at", silent)
    at = AppTest.from_function(_app, kwargs={"frame": FRAME, "ai": AI}, default_timeout=180)
    at.run()
    popover = "clean_ai_open::" + str(_one(at.button, "clean_map::").key).split("::", 1)[1]
    at.session_state[popover] = True
    at.run()
    _one(at.button, "clean_ai_send::").click().run()
    _ok(at)
    assert at.session_state["rag_manager"].sent == []
    assert any(
        c.value.startswith(
            "Nothing was sent: the Ollama server did not say where it runs the model"
        )
        for c in at.caption
    )
