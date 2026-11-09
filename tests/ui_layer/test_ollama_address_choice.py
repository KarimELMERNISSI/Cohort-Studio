"""
Settings > AI engine (AppTest): the Ollama address is chosen from the server's list (manage.server_addresses) and never
typed, so that the server sends no request to an address a user writes; an administrator adds an address for every
user, and removes it, below the choice; a value the list does not hold is never asked anything.
"""

import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")
KEY = "rag_ollama_base_url"
CANARY = "CANARY-PASS-51c"
LISTED = ["http://localhost:11434", "http://127.0.0.2:11434"]
ASKED: list[str] = []  # the addresses an Ollama was asked something at


@pytest.fixture(autouse=True)
def _quiet_ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No Ollama is asked anything: the addresses that would have been are kept in ASKED."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    import app_pages.rag_sidebar as sidebar
    import manage.rag.providers.ollama_server as server

    ASKED.clear()
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    monkeypatch.setenv(
        "CMP_OLLAMA_ADDRESSES", "http://127.0.0.2:11434"
    )  # after OLLAMA_BASE_URL's default
    monkeypatch.setattr(server, "runs_at", lambda base_url, model: ASKED.append(str(base_url)))
    monkeypatch.setattr(server, "remote_models", lambda base_url: ASKED.append(str(base_url)) or {})
    monkeypatch.setattr(
        sidebar, "_discover_ollama_models", lambda base_url: ASKED.append(str(base_url)) or []
    )
    sidebar._runs_at.clear()
    sidebar._remote_models.clear()


def _settings(role: str, username: str | None = None) -> AppTest:
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = username or f"{role}_user"
    at.session_state["user_role"] = role
    at.run()
    at.switch_page(nav.page_path("settings")).run()
    at.selectbox(key="rag_provider_select").select("ollama").run()
    _ok(at)
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _texts(at: AppTest) -> str:
    return " | ".join(
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [e.value for e in at.error]
        + [s.value for s in at.success]
        + [i.value for i in at.info]
    )


def _add(at: AppTest, address: str) -> None:
    at.text_input(key=f"{KEY}_new").input(address)
    at.button(key=f"{KEY}_add").click().run()
    _ok(at)


def test_a_researcher_chooses_from_the_server_s_list_and_is_told_whom_to_ask() -> None:
    at = _settings("researcher")
    box = at.selectbox(key=KEY)
    assert box.options == LISTED and box.value == LISTED[0]
    assert not [
        t for t in at.text_input if t.key in (KEY, f"{KEY}_new")
    ]  # nothing to type an address in
    assert not [
        b for b in at.button if b.key == f"{KEY}_add"
    ]  # the administrators' list is not shown
    assert "Ask an administrator of this server to add its address." in _texts(at)
    box.select(LISTED[1]).run()
    _ok(at)
    assert "Your network: data goes to 127.0.0.2" in _texts(at)


def test_an_administrator_adds_an_address_for_every_user_and_removes_it(tmp_path: Path) -> None:
    admin = _settings("admin", "j.smith@lab.example")
    _add(admin, "HTTP://10.0.0.5:11434/")
    assert admin.selectbox(key=KEY).value == "http://10.0.0.5:11434"  # offered, and chosen here
    assert (
        "Added for every user of this server, and chosen here: `http://10.0.0.5:11434`."
        in _texts(admin)
    )
    assert admin.text_input(key=f"{KEY}_new").value == ""
    saved = json.loads((tmp_path / "server" / "ollama_addresses.json").read_text(encoding="utf-8"))
    assert [(e["address"], e["added_by"]) for e in saved["added"]] == [
        ("http://10.0.0.5:11434", "j.smith@lab.example")
    ]

    researcher = _settings("researcher")  # another session: offered to every user
    assert researcher.selectbox(key=KEY).options == [*LISTED, "http://10.0.0.5:11434"]

    _add(admin, "localhost:11434")  # the deployment's: nothing is added, and the page says so
    assert "Already offered: `http://localhost:11434`, now chosen here." in _texts(admin)
    admin.selectbox(key=KEY).select("http://10.0.0.5:11434").run()

    admin.button(key=f"{KEY}_remove_0").click().run()
    _ok(admin)
    box = admin.selectbox(key=KEY)
    assert "http://10.0.0.5:11434" not in box.options and box.value == LISTED[0]
    shown = _texts(admin)
    assert "Removed for every user: `http://10.0.0.5:11434`." in shown
    assert "This page now uses `http://localhost:11434`." in shown
    admin.button(key=f"{KEY}_add_back").click().run()  # a slip, undone at once
    _ok(admin)
    assert admin.selectbox(key=KEY).options[-1] == "http://10.0.0.5:11434"


def test_a_choice_removed_by_an_administrator_is_named_with_the_one_used_instead() -> None:
    admin = _settings("admin")
    _add(admin, "http://10.0.0.5:11434")
    researcher = _settings("researcher")
    researcher.selectbox(key=KEY).select("http://10.0.0.5:11434").run()
    admin.selectbox(key=KEY).select("http://10.0.0.5:11434").run()
    admin.button(key=f"{KEY}_remove_0").click().run()
    researcher.run()
    _ok(researcher)
    assert researcher.selectbox(key=KEY).value == LISTED[0]
    assert (
        "The address you chose, `http://10.0.0.5:11434`, is no longer offered: `http://localhost:11434` is used."
        in _texts(researcher)
    )
    researcher.selectbox(key=KEY).select(LISTED[1]).run()  # another choice: the notice goes
    assert "is no longer offered" not in _texts(researcher)


def test_an_address_with_a_password_is_refused_and_kept_nowhere(tmp_path: Path) -> None:
    admin = _settings("admin")
    for written in (
        f"http://user:{CANARY}@10.0.0.5:11434",
        f"http://user:{CANARY}＠10.0.0.5:11434",
    ):
        _add(admin, written)
        shown = _texts(admin)
        assert "Not added: an address" in shown and CANARY not in shown
    assert admin.selectbox(key=KEY).options == LISTED
    kept = [
        p
        for p in tmp_path.rglob("*")
        if p.is_file() and CANARY in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not kept, kept


def test_a_forged_choice_is_never_asked_anything() -> None:
    """The browser sends the value of a choice: one that is not in the list (a hand-made message) is replaced by the
    first address before any request, here as in the engine's start."""
    from streamlit.proto.WidgetStates_pb2 import WidgetState
    from streamlit.testing.v1.element_tree import Selectbox

    forged_address = "http://169.254.169.254:80"
    at = _settings("researcher")
    original = Selectbox._widget_state

    def forged(self):
        if str(self.id).endswith(KEY):
            state = WidgetState()
            state.id = self.id
            state.string_value = forged_address
            return state
        return original.fget(self)

    ASKED.clear()
    Selectbox._widget_state = property(forged)
    try:
        at.run()
        at.button(key="rag_init_button").click().run()
    finally:
        Selectbox._widget_state = original
    _ok(at)
    assert forged_address not in " ".join(ASKED)
    assert at.selectbox(key=KEY).value == LISTED[0]
