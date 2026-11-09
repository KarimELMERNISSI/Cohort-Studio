"""
The Qdrant servers a user may choose (utils.settings.qdrant_address, CMP_QDRANT_URL, manage.server_addresses
QDRANT_ADDRESSES): chosen from the server's list with the embedded mode, never typed; the server's key goes to its own
address only, however it is written; the engine refuses a server the list does not hold and stops when its server is
removed; the connection test never shows what an address answered.
"""

from pathlib import Path

import pytest

import utils.data_paths as dp
from core.workspace import UserScope
from manage.server_addresses import QDRANT_ADDRESSES as servers
from utils.settings import SettingsError, get_settings, qdrant_address

ADMIN = UserScope.for_user("admin_user", "admin")
CANARY = "INTERNAL-SECRET-BODY-7d1"


@pytest.fixture(autouse=True)
def _server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    for name in ("CMP_QDRANT_URL", "CMP_QDRANT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_an_address_written_without_a_scheme_is_read_as_the_client_reads_it() -> None:
    assert qdrant_address("qdrant.lan:6333") == "http://qdrant.lan:6333"
    assert (
        qdrant_address("qdrant.lan:6333", "a-key") == "https://qdrant.lan:6333"
    )  # with a key: https
    assert (
        qdrant_address("http://Qdrant.LAN:6333/", "a-key") == "http://qdrant.lan:6333"
    )  # a scheme is kept


def test_the_deployment_offers_its_server_as_the_client_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert servers.deployment() == []  # no server set: the embedded mode alone
    monkeypatch.setenv("CMP_QDRANT_URL", "qdrant.lan:6333")
    assert servers.deployment() == ["http://qdrant.lan:6333"]
    monkeypatch.setenv("CMP_QDRANT_API_KEY", "server-key-123456")
    assert servers.deployment() == ["https://qdrant.lan:6333"]
    monkeypatch.setenv("CMP_QDRANT_URL", "http://qdrant.lan:6333/?api-key=123")
    with pytest.raises(SettingsError) as stopped:
        get_settings()
    assert "CMP_QDRANT_URL" in str(stopped.value) and "123" not in str(stopped.value)


def test_the_server_s_key_goes_to_its_own_address_however_it_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manage.vectorstores.selection import VectorStoreSettings

    monkeypatch.setenv("CMP_QDRANT_URL", "qdrant.lan:6333")
    monkeypatch.setenv("CMP_QDRANT_API_KEY", "server-key-123456")
    settings = VectorStoreSettings.from_env()
    assert (
        settings.key_for("https://qdrant.lan:6333") == "server-key-123456"
    )  # as Settings offers it
    assert settings.key_for("qdrant.lan:6333/") == "server-key-123456"  # as written
    assert (
        settings.key_for("http://qdrant.lan:6333") is None
    )  # another server: not the one the client reads
    assert settings.key_for("https://qdrant.example.org:6333") is None


def test_an_administrator_lists_a_server_and_the_engine_follows_the_list() -> None:
    from manage.rag_manager import RAGManager

    assert not RAGManager._qdrant_off_list(None)  # embedded: no server
    assert RAGManager._qdrant_off_list("https://qdrant.example.org:6333")
    assert (
        servers.add("https://Qdrant.Example.org:6333/", by=ADMIN)
        == "https://qdrant.example.org:6333"
    )
    assert not RAGManager._qdrant_off_list("https://qdrant.example.org:6333")
    engine = object.__new__(RAGManager)
    engine.provider, engine.base_url = "ollama", None
    engine.embedding_provider, engine.embedding_base_url = "ollama", None
    engine.qdrant_url = "https://qdrant.example.org:6333"
    engine.llm, engine.qa_chain, engine.vector_store, engine._output_helper = (
        object(),
        None,
        None,
        None,
    )
    engine.client, engine.initialized = None, True
    assert not engine.stop_if_address_withdrawn()
    servers.remove("https://qdrant.example.org:6333", by=ADMIN)
    assert engine.stop_if_address_withdrawn()
    assert (
        not engine.initialized
        and "Qdrant server was removed from CohortStudio's list" in engine.stop_reason
    )


def test_the_connection_test_never_shows_what_an_address_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qdrant-client quotes the first bytes of a reply in its error ("Raw response content"): Test the connection drew
    them on the page, whatever service answered at the address."""
    pytest.importorskip("langchain_core", reason="the vector stores are in the optional 'ai' extra")
    from manage.vectorstores import qdrant_index

    class Unexpected(Exception):
        status_code = 404

    class Answering:  # the client is made; the request meets another service, which answers 404
        def get_collections(self):
            raise Unexpected(
                f"Unexpected Response: 404 (Not Found) Raw response content: b'{CANARY}'"
            )

        collection_exists = get_collections

    def not_made(**kwargs):
        raise Unexpected(f"Raw response content: b'{CANARY}'")

    monkeypatch.setattr(qdrant_index, "_client", lambda **kwargs: Answering())
    index = qdrant_index.QdrantIndex.open(
        tmp_path / "qdrant", username="u", url="http://10.0.0.7:8080"
    )
    ok, message = index.ping()
    assert not ok and CANARY not in message
    assert "the server answered with an error (HTTP status 404)" in message
    with pytest.raises(qdrant_index.VectorStoreError) as failed:
        index.exists()
    assert CANARY not in str(failed.value)
    monkeypatch.setattr(qdrant_index, "_client", not_made)  # the client itself cannot be made
    ok, message = index.ping()
    assert (
        not ok
        and CANARY not in message
        and "the server answered with an error (HTTP status 404)" in message
    )


def test_the_engine_s_start_opens_no_server_the_list_does_not_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    from manage import rag_manager as rm

    monkeypatch.setattr(rm, "open_index", lambda *a, **k: pytest.fail("the server was opened"))
    engine = rm.RAGManager(scope=UserScope.for_user("research_user", "researcher"))
    started, message = engine.initialize_system(
        provider="ollama", vector_engine="qdrant", qdrant_url="http://10.0.0.7:8080"
    )
    assert not started and "This Qdrant server is not in CohortStudio's list" in message


@pytest.mark.parametrize(
    "written",
    [
        "http://qdrant.lan:6333//collector.example.org",  # the client would reach collector.example.org
        "http://qdrant.lan:6333/a/..//collector.example.org",
        "https://gateway.lan//qdrant",
        "http://qdrant.lan:6333/./x",
    ],
)
def test_a_path_that_would_reach_another_host_is_refused(written: str) -> None:
    with pytest.raises(ValueError, match="no empty, '.' or '..' segment"):
        qdrant_address(written)


@pytest.mark.parametrize(
    "written",
    [
        "http://qdrant.lan:6333",
        "https://qdrant.example.org:6333/qdrant",
        "qdrant.lan:6333",
        "http://[::1]:6333",
        "http://qdrant_db:6333/a/b",
    ],
)
def test_the_client_reaches_the_host_the_badge_names(written: str) -> None:
    """The badge and the confirmation name an address's host: the client must connect to that very host."""
    pytest.importorskip("qdrant_client", reason="the Qdrant engine is in the optional 'ai' extra")
    from urllib.parse import urlparse

    from qdrant_client import QdrantClient

    from manage.ai_payload import host_of

    offered = qdrant_address(written)
    client = QdrantClient(url=offered, check_compatibility=False)
    rest_uri = getattr(getattr(client, "_client", None), "rest_uri", None)
    if rest_uri is None:
        pytest.skip("this qdrant-client keeps its address elsewhere")
    assert urlparse(rest_uri).hostname == host_of(offered)


def test_a_stray_double_slash_is_refused() -> None:
    """ "qdrant.lan:6333//" was read as written with a scheme, and the server's key went over http: it holds an empty
    path segment, refused; a final "/" alone is no segment."""
    with pytest.raises(ValueError, match="no empty"):
        qdrant_address("qdrant.lan:6333//", "a-key")
    assert qdrant_address("qdrant.lan:6333/", "a-key") == "https://qdrant.lan:6333"


def test_no_request_of_the_store_shows_what_an_address_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initialise / reindex and a search drew the first bytes of a listed server's reply (an intranet's page, say)."""
    pytest.importorskip("langchain_core", reason="the vector stores are in the optional 'ai' extra")
    pytest.importorskip("qdrant_client", reason="the Qdrant engine is in the optional 'ai' extra")
    from manage.vectorstores import qdrant_index
    from manage.vectorstores.base import ChunkRow

    class Unexpected(Exception):
        status_code = 500

    class Answering:
        def __getattr__(self, name):
            def answer(*args, **kwargs):
                raise Unexpected(f"Raw response content: b'{CANARY}'")

            return answer

    monkeypatch.setattr(qdrant_index, "_client", lambda **kwargs: Answering())
    index = qdrant_index.QdrantIndex.open(
        tmp_path / "qdrant", username="u", url="http://10.0.0.7:6333"
    )
    row = ChunkRow(id="a", text="t", source="s.pdf", page=1, vector=[0.1, 0.2], metadata={})
    for request in (
        lambda: index.reset(2),
        lambda: index.add([row]),
        lambda: index.query([0.1, 0.2], 3),
    ):
        with pytest.raises(qdrant_index.VectorStoreError) as failed:
            request()
        assert CANARY not in str(
            failed.value
        ) and "the server answered with an error (HTTP status 500)" in str(failed.value)


def test_the_workspace_lists_the_indexes_it_built_on_servers(tmp_path: Path) -> None:
    pytest.importorskip("langchain_core", reason="the vector stores are in the optional 'ai' extra")
    from manage.vectorstores.base import IndexManifest
    from manage.vectorstores.qdrant_index import server_folder, server_indexes

    directory = tmp_path / "qdrant"
    url = "https://qdrant.example.org:6333"
    IndexManifest(
        engine="qdrant", embedding="ollama:e", dim=8, chunks=42, location=f"{url} (cmp_0123abcd)"
    ).save(server_folder(directory, url))
    ((url_found, collection, manifest),) = server_indexes(directory)
    assert (url_found, collection, manifest.chunks) == (url, "cmp_0123abcd", 42)
    assert server_indexes(tmp_path / "none") == []


@pytest.mark.parametrize(
    ("kind", "status", "cause", "said"),
    [
        (
            "UnexpectedResponse",
            401,
            None,
            "the server refused the API key (HTTP status 401): check the key",
        ),
        (
            "UnexpectedResponse",
            403,
            None,
            "this key may not list the collections (HTTP status 403)",
        ),
        ("UnexpectedResponse", 502, None, "the server answered with an error (HTTP status 502)"),
        (
            "ResponseHandlingException",
            None,
            "JSONDecodeError",
            "but not as a Qdrant server: check the address and port",
        ),
        (
            "ResponseHandlingException",
            None,
            "ConnectError",
            "no answer: the server is down, or the address or port",
        ),
        (
            "ResponseHandlingException",
            None,
            "SSLError",
            "the https connection failed: the server may take http only",
        ),
        ("SomethingElse", None, None, "the request failed (the server log names the error)"),
    ],
)
def test_a_failure_is_said_in_words(
    kind: str, status: int | None, cause: str | None, said: str
) -> None:
    """A wrong key read "not reachable" with the client's class names: a researcher checked the network instead."""
    pytest.importorskip("langchain_core", reason="the vector stores are in the optional 'ai' extra")
    import ssl

    from manage.vectorstores.qdrant_index import _said

    failure = type(kind, (Exception,), {"status_code": status})(
        f"Raw response content: b'{CANARY}'"
    )
    if cause == "SSLError":
        failure.__cause__ = ssl.SSLError("handshake")
    elif cause:
        failure.__cause__ = type(cause, (Exception,), {})("detail")
    text = _said(failure)
    assert said in text and CANARY not in text and kind not in text


def test_a_qdrant_server_is_added_with_its_scheme() -> None:
    """Written without one, it read http, and the key a researcher typed for it crossed the network in clear."""
    with pytest.raises(ValueError, match="write https://"):
        servers.add("abc.cloud.qdrant.io:6333", by=ADMIN)
    assert (
        servers.add("https://abc.cloud.qdrant.io:6333", by=ADMIN)
        == "https://abc.cloud.qdrant.io:6333"
    )
