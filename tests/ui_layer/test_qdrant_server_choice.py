"""
Settings > AI engine > Vector database (AppTest): the Qdrant server is chosen from the server's list (manage.server_
addresses), the embedded mode first, never typed; an administrator adds a server for every user and removes it; a page
whose server is removed says it is back on the embedded mode.
"""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")
KEY = "rag_qdrant_url"
SERVER = "https://qdrant.example.org:6333"


@pytest.fixture(autouse=True)
def _settings_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    pytest.importorskip("qdrant_client", reason="the Qdrant engine is in the optional 'ai' extra")
    from manage.vectorstores import qdrant_index

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    for name in ("CMP_QDRANT_URL", "CMP_QDRANT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        qdrant_index, "_client", lambda **kwargs: pytest.fail("Settings asked a Qdrant server")
    )


def _vector_card(role: str) -> AppTest:
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = f"{role}_user"
    at.session_state["user_role"] = role
    at.run()
    at.switch_page(nav.page_path("settings")).run()
    at.segmented_control(key="rag_vector_engine").set_value("qdrant").run()
    _ok(at)
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _texts(at: AppTest) -> str:
    return " | ".join(
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [s.value for s in at.success]
        + [i.value for i in at.info]
    )


def test_a_researcher_chooses_the_embedded_mode_or_a_listed_server() -> None:
    from core.workspace import UserScope
    from manage.server_addresses import QDRANT_ADDRESSES

    QDRANT_ADDRESSES.add(SERVER, by=UserScope.for_user("an_admin", "admin"))
    at = _vector_card("researcher")
    assert not [
        t for t in at.text_input if t.key in (KEY, f"{KEY}_new")
    ]  # nothing to type an address in
    box = at.selectbox(key=KEY)
    assert box.value == "" and len(box.options) == 2  # the embedded mode first
    assert box.options[0] == "Embedded in your workspace (no server)"
    assert (
        "Your Qdrant server is not listed? Ask a CohortStudio administrator to add it."
        in _texts(at)
    )
    box.select(SERVER).run()
    _ok(at)
    assert at.selectbox(key=KEY).value == SERVER
    assert f"rag_qdrant_api_key::{SERVER}" in [t.key for t in at.text_input]  # a key box of its own


def test_an_administrator_adds_a_server_and_removes_it() -> None:
    at = _vector_card("admin")
    at.text_input(key=f"{KEY}_new").input("qdrant.example.org:6333")  # no scheme: refused
    at.button(key=f"{KEY}_add").click().run()
    _ok(at)
    assert (
        "Not added: write https:// (a server outside your network, such as Qdrant Cloud)"
        in " | ".join(e.value for e in at.error)
    )
    at.text_input(key=f"{KEY}_new").input("http://qdrant.example.org:6333")
    at.button(key=f"{KEY}_add").click().run()
    _ok(at)
    assert (
        at.selectbox(key=KEY).value == "http://qdrant.example.org:6333"
    )  # without a scheme or key: http
    assert "Added to the Qdrant servers of every user, and chosen here" in _texts(at)
    at.button(key=f"{KEY}_remove_0").click().run()
    _ok(at)
    assert at.selectbox(key=KEY).value == ""
    assert "This page now uses the embedded Qdrant, in your workspace." in _texts(at)


def test_the_deployment_s_server_is_chosen_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMP_QDRANT_URL", "qdrant.lan:6333")
    at = _vector_card("researcher")
    assert (
        at.selectbox(key=KEY).value == "http://qdrant.lan:6333"
    )  # as the client reads it, without a key: http


def test_a_key_typed_for_one_server_never_goes_to_the_next() -> None:
    from core.workspace import UserScope
    from manage.server_addresses import QDRANT_ADDRESSES

    other = "http://qdrant-b.lan:6333"
    for address in (SERVER, other):
        QDRANT_ADDRESSES.add(address, by=UserScope.for_user("an_admin", "admin"))
    at = _vector_card("researcher")
    at.selectbox(key=KEY).select(SERVER).run()
    at.text_input(key=f"rag_qdrant_api_key::{SERVER}").input("KEY-TYPED-FOR-A").run()
    at.selectbox(key=KEY).select(other).run()
    _ok(at)
    assert at.text_input(key=f"rag_qdrant_api_key::{other}").value == ""


def test_an_index_left_on_a_removed_server_is_named_with_its_collection(tmp_path: Path) -> None:
    """The app sends nothing more to a server an administrator removed, so no rebuild will delete the index there."""
    from manage.vectorstores.base import IndexManifest
    from manage.vectorstores.qdrant_index import server_folder

    directory = tmp_path / "users" / "researcher_user" / "vectors" / "qdrant"
    IndexManifest(
        engine="qdrant", embedding="ollama:e", dim=8, chunks=42, location=f"{SERVER} (cmp_0123abcd)"
    ).save(server_folder(directory, SERVER))
    at = _vector_card("researcher")
    shown = _texts(at)
    assert (
        f"is on `{SERVER}`, which this list does not offer. To use it again, ask a CohortStudio administrator to add "
        "that address, as written here; if it should go instead, ask whoever runs that server to delete the "
        "collection `cmp_0123abcd`." in shown
    )
    # never "build one" without saying the index on the server is not used
    assert (
        f"No index here yet: Initialise / reindex builds one here; your index on `{SERVER}` is not used."
        in shown
    )


def test_removing_a_server_says_its_indexes_stay_there() -> None:
    at = _vector_card("admin")
    at.text_input(key=f"{KEY}_new").input(SERVER)
    at.button(key=f"{KEY}_add").click().run()
    at.button(key=f"{KEY}_remove_0").click().run()
    _ok(at)
    assert "The indexes users built there stay on that server" in _texts(at)
