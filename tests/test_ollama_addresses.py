"""
The Ollama addresses a user may choose (utils.settings.server_address, OLLAMA_BASE_URL then CMP_OLLAMA_ADDRESSES,
manage.server_addresses): a user chooses among the server's and never types one, so that the server sends no request to
an address someone wrote; an administrator adds and removes addresses for every user. A refused address is never
repeated in a message, a log entry or the list's file (it may carry a password); the engine refuses an address the list
does not hold, and stops when its address is removed.
"""

import json
import logging
from pathlib import Path

import pytest

import utils.data_paths as dp
from core.workspace import UserScope
from manage.server_addresses import MAX_ADDED, MAX_BYTES, NotAllowed
from manage.server_addresses import OLLAMA_ADDRESSES as addresses
from utils.settings import SettingsError, get_settings, server_address

ADMIN = UserScope.for_user("admin_user", "admin")
RESEARCHER = UserScope.for_user("research_user", "researcher")
CANARY = "CANARY-PASS-9f3"
FULL_AT = "＠"  # a full-width "@", from an East Asian input method: urlparse refuses it, quoting the address


@pytest.fixture(autouse=True)
def _server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("CMP_OLLAMA_ADDRESSES", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    return tmp_path


@pytest.mark.parametrize(
    ("written", "offered"),
    [
        (
            "localhost:11434",
            "http://localhost:11434",
        ),  # no scheme: http, as Ollama's client reads it
        ("HTTP://Ollama.Example.ORG:11434/", "http://ollama.example.org:11434"),
        ("https://gpu.lab.example/ollama/", "https://gpu.lab.example/ollama"),
        ("  http://10.0.0.5:11434  ", "http://10.0.0.5:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
        ("http://ollama?", "http://ollama"),  # an empty query is no query
        ("http://ollama_gpu:11434", "http://ollama_gpu:11434"),  # a Docker service's name
        (
            "http://xn--llama-3ve.example",
            "http://xn--llama-3ve.example",
        ),  # an international name, as written for DNS
    ],
)
def test_an_address_is_offered_in_one_spelling(written: str, offered: str) -> None:
    assert server_address(written) == offered


@pytest.mark.parametrize(
    ("written", "why"),
    [
        ("", "an address is needed"),
        ("ftp://ollama:21", "starts with http:// or https://"),
        ("file:///etc/passwd", "starts with http:// or https://"),
        (f"http://user:{CANARY}@127.0.0.1:9", "must not hold a user name or password"),
        (f"http://{CANARY}@127.0.0.1:9", "must not hold a user name or password"),
        (f"http://user:{CANARY}{FULL_AT}127.0.0.1:9", "written in ASCII"),
        (f"http://user:{CANARY}／127.0.0.1:9", "written in ASCII"),  # a full-width "/"
        ("http://оllama.lab.example", "written in ASCII"),  # a Cyrillic "o"
        (f"http://ollama:11434/?token={CANARY}", "no query"),
        (f"http://ollama:11434/#{CANARY}", "no query"),
        ("http://ollama:99999", "port is a number from 1 to 65535"),
        ("http://ollama:0", "port is a number from 1 to 65535"),
        ("http://ollama:port", "port is a number from 1 to 65535"),
        (
            "http://localhost:11434,http://x",
            "port is a number from 1 to 65535",
        ),  # two addresses in one
        ("http://local,host:11434", "host is written with letters"),
        ("http://oll%61ma:11434", "host is written with letters"),
        (
            "http://ollama:11434/a`b",
            "path is written with",
        ),  # a backtick, which a code span would drop
        ("http:// ollama:11434", "holds no space"),
        ("http://", "names a host"),
        ("http://" + "a" * 200, "at most 200 characters"),
    ],
)
def test_an_address_that_is_not_a_server_s_is_refused_without_its_value(
    written: str, why: str
) -> None:
    with pytest.raises(ValueError, match=why) as refused:
        server_address(written)
    assert CANARY not in str(refused.value)


def test_the_deployment_offers_ollama_base_url_first(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_settings().ai.ollama_address_list() == [
        "http://localhost:11434"
    ]  # OLLAMA_BASE_URL's default
    monkeypatch.setenv("OLLAMA_BASE_URL", "ollama:11434/")
    assert get_settings().ai.ollama_address_list() == ["http://ollama:11434"]
    monkeypatch.setenv("CMP_OLLAMA_ADDRESSES", "http://b:2, a:1 ,,http://B:2/, ollama:11434")
    # added to OLLAMA_BASE_URL, never in its place: the first address, used by default, stays the deployment's own
    assert get_settings().ai.ollama_address_list() == [
        "http://ollama:11434",
        "http://b:2",
        "http://a:1",
    ]


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("CMP_OLLAMA_ADDRESSES", f"http://localhost:11434,http://user:{CANARY}@10.0.0.9:11434"),
        ("CMP_OLLAMA_ADDRESSES", f"http://user:{CANARY}{FULL_AT}10.0.0.9:11434"),
        ("OLLAMA_BASE_URL", f"http://user:{CANARY}@10.0.0.9:11434"),
        ("OLLAMA_BASE_URL", f"http://user:{CANARY}{FULL_AT}10.0.0.9:11434"),
        (
            "OLLAMA_BASE_URL",
            "http://localhost:11434,http://10.0.0.9:11434",
        ),  # one address, not a list
        (
            "CMP_QDRANT_URL",
            f"http://user:{CANARY}{FULL_AT}qdrant.lan:6333",
        ),  # the same check of credentials
    ],
)
def test_a_wrong_address_stops_start_up_named_without_its_value(
    variable: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(SettingsError) as stopped:
        get_settings()
    assert variable in str(stopped.value) and CANARY not in str(stopped.value)


def test_an_administrator_adds_an_address_for_every_user(_server: Path) -> None:
    assert addresses.listed() == ["http://localhost:11434"]
    assert (
        addresses.add("HTTP://10.0.0.5:11434/", by=ADMIN, account="j.smith@lab.example")
        == "http://10.0.0.5:11434"
    )
    assert addresses.add("10.0.0.5:11434", by=ADMIN) == "http://10.0.0.5:11434"  # once
    assert (
        addresses.add("http://localhost:11434", by=ADMIN) == "http://localhost:11434"
    )  # the deployment's
    assert addresses.listed() == ["http://localhost:11434", "http://10.0.0.5:11434"]
    assert addresses.offered("10.0.0.5:11434/") and not addresses.offered("http://10.0.0.6:11434")
    assert not addresses.offered(None) and not addresses.offered(
        f"http://u:{CANARY}@10.0.0.5:11434"
    )
    saved = json.loads((_server / "server" / "ollama_addresses.json").read_text(encoding="utf-8"))
    (entry,) = saved["added"]
    assert entry["address"] == "http://10.0.0.5:11434"
    assert (
        entry["added_by"] == "j.smith@lab.example"
    )  # the sign-in name, not the workspace folder's
    assert entry["added_at"].endswith("+00:00")
    removed = addresses.remove("http://10.0.0.5:11434", by=ADMIN)
    again = addresses.remove("http://10.0.0.5:11434", by=ADMIN)
    deployed = addresses.remove("http://localhost:11434", by=ADMIN)  # changed in the deployment
    assert removed and not again and not deployed
    assert addresses.listed() == ["http://localhost:11434"]


def test_who_added_or_removed_an_address_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    addresses.add("http://10.0.0.5:11434", by=ADMIN, account="j.smith@lab.example")
    addresses.remove("http://10.0.0.5:11434", by=ADMIN, account="a.martin@lab.example")
    assert (
        "added to the list by the administrator j.smith@lab.example: http://10.0.0.5:11434"
        in caplog.text
    )
    assert (
        "removed from the list by the administrator a.martin@lab.example: http://10.0.0.5:11434"
        in caplog.text
    )


def test_only_an_administrator_changes_the_list(_server: Path) -> None:
    with pytest.raises(NotAllowed):
        addresses.add("http://10.0.0.5:11434", by=RESEARCHER)
    addresses.add("http://10.0.0.5:11434", by=ADMIN)
    with pytest.raises(NotAllowed):
        addresses.remove("http://10.0.0.5:11434", by=RESEARCHER)
    assert addresses.listed() == ["http://localhost:11434", "http://10.0.0.5:11434"]


def test_a_refused_address_is_never_written_nor_logged(
    _server: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    for written in (
        f"http://user:{CANARY}@10.0.0.5:11434",
        f"http://user:{CANARY}{FULL_AT}10.0.0.5:11434",
    ):
        with pytest.raises(ValueError) as refused:
            addresses.add(written, by=ADMIN)
        assert CANARY not in str(refused.value)
    assert not (_server / "server").exists()
    assert CANARY not in caplog.text


def test_a_damaged_list_offers_the_deployment_s_addresses_and_is_not_overwritten(
    _server: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _server / "server" / "ollama_addresses.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"added": [{"address": "http://10.0.0.7:11434"},]' + CANARY, encoding="utf-8")
    assert addresses.listed() == ["http://localhost:11434"]
    assert addresses.unreadable()
    assert CANARY not in caplog.text and "cannot be read" in caplog.text
    with pytest.raises(ValueError, match="repair or remove its file first"):
        addresses.add(
            "http://10.0.0.5:11434", by=ADMIN
        )  # it would write a list of one address over the others
    assert CANARY in path.read_text(encoding="utf-8")  # left as it was
    entries = [
        {"address": "http://10.0.0.5:11434", "added_by": "admin_user"},
        {"address": f"http://user:{CANARY}@10.0.0.6:11434"},  # written by hand: not offered
        {"address": ["not", "text"]},
        {"no address": True},
        "not an entry",
        {"address": "10.0.0.5:11434/"},  # the same address, written otherwise
    ]
    path.write_text(json.dumps({"version": 1, "added": entries}), encoding="utf-8")
    assert addresses.listed() == ["http://localhost:11434", "http://10.0.0.5:11434"]
    assert not addresses.unreadable()
    assert CANARY not in caplog.text
    path.write_text(" " * (MAX_BYTES + 1), encoding="utf-8")
    assert addresses.added() == [] and addresses.unreadable()


def test_the_list_holds_at_most_fifty_addresses(_server: Path) -> None:
    for n in range(MAX_ADDED):
        addresses.add(f"http://10.0.1.{n}:11434", by=ADMIN)
    with pytest.raises(ValueError, match="remove one first"):
        addresses.add("http://10.0.2.1:11434", by=ADMIN)
    assert len(addresses.listed()) == MAX_ADDED + 1


def test_an_engine_refuses_and_stops_on_an_address_the_server_does_not_offer() -> None:
    """The engine asks nothing of an Ollama address the list does not hold (a value forged around Settings), and an
    engine started on an address an administrator then removes stops at the next run of its session."""
    from manage.rag_manager import RAGManager

    off_list = RAGManager._off_list
    assert not off_list("ollama", "http://localhost:11434", "ollama", None)
    assert not off_list(
        "ollama", None, "ollama", None
    )  # Ollama's default, which OLLAMA_BASE_URL offers
    assert off_list("ollama", "http://169.254.169.254", "ollama", None)
    assert off_list("anthropic", None, "ollama", "http://10.0.0.5:11434")  # Claude's embeddings
    assert not off_list("gemini", "http://169.254.169.254", "gemini", None)  # no Ollama: not asked

    addresses.add("http://10.0.0.5:11434", by=ADMIN)
    engine = object.__new__(RAGManager)
    engine.provider, engine.base_url = "ollama", "http://10.0.0.5:11434"
    engine.embedding_provider, engine.embedding_base_url = "ollama", None
    engine.llm, engine.qa_chain, engine.vector_store, engine._output_helper = (
        object(),
        None,
        None,
        None,
    )
    engine.client, engine.initialized = None, True
    assert not engine.stop_if_address_withdrawn()  # still offered
    addresses.remove("http://10.0.0.5:11434", by=ADMIN)
    assert engine.stop_if_address_withdrawn()
    assert not engine.initialized and engine.llm is None
    assert "no longer offered" in engine.stop_reason


def test_the_engine_s_start_asks_nothing_of_an_address_not_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    import manage.rag.providers.ollama_server as server
    from manage.rag_manager import RAGManager

    asked: list[str] = []
    monkeypatch.setattr(server, "runs_at", lambda base_url, model: asked.append(str(base_url)))
    engine = RAGManager(scope=RESEARCHER)
    started, message = engine.initialize_system(
        provider="ollama", base_url="http://169.254.169.254:80"
    )
    assert not started and "not one this server offers" in message
    assert asked == [] and not engine.initialized
