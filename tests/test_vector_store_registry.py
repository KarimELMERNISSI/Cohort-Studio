"""The vector databases as a connector family (M5b): the registry keeps the connector contract, and reading it loads
neither the engines nor the AI stack (it runs without the optional 'ai' group)."""

import sys
from pathlib import Path

import pytest

from core.registry import Availability
from manage.vectorstores import VECTOR_STORES
from tests import connector_contract as contract


def test_the_stored_names_never_change() -> None:
    # kept in the users' preferences, the index manifests and CMP_VECTOR_STORE
    contract.check_names(VECTOR_STORES, ("lancedb", "qdrant"))
    assert VECTOR_STORES.valid() == ("auto", "lancedb", "qdrant")


def test_the_registry_keeps_the_connector_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    contract.check_availability(VECTOR_STORES, monkeypatch)
    contract.check_classes(VECTOR_STORES)
    contract.check_precedence(VECTOR_STORES)
    contract.check_unknown_setting(VECTOR_STORES)
    contract.check_unknown_names(VECTOR_STORES)


@pytest.mark.parametrize(("name", "module"), [("lancedb", "lancedb"), ("qdrant", "qdrant_client")])
def test_a_missing_client_names_its_group(
    name: str, module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract.check_missing_package(VECTOR_STORES, name, module, monkeypatch)
    assert VECTOR_STORES.availability(name).extra == "ai"


def test_each_engine_says_what_it_is() -> None:
    # the titles and summaries of the lists they replace (ENGINE_TITLES, ENGINE_SUMMARIES; golden test of 7333f49)
    assert {name: VECTOR_STORES.display_name(name) for name in VECTOR_STORES.names()} == {
        "lancedb": "LanceDB",
        "qdrant": "Qdrant",
    }
    assert VECTOR_STORES.get("lancedb").summary.startswith("Embedded and serverless")
    assert VECTOR_STORES.get("qdrant").summary.startswith("Embedded in-memory for small libraries")


def test_the_engine_follows_the_session_the_preference_the_setting_then_the_recommendation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manage.vectorstores import MachineProfile, VectorStoreSettings, recommend, select_engine

    monkeypatch.setattr(VECTOR_STORES, "availability", lambda name: Availability(True))
    laptop = recommend(MachineProfile(8, 16.0, 8.0, 100.0), chunks=10)  # LanceDB, with its reasons
    server = VectorStoreSettings(engine="qdrant")

    def pick(**choices):
        chosen = select_engine(advice=laptop, **choices)
        return chosen.chosen, chosen.source

    assert pick(settings=server) == ("qdrant", "setting")
    assert pick(settings=server, preference="lancedb") == ("lancedb", "preference")
    assert pick(settings=server, preference="lancedb", session="qdrant") == ("qdrant", "session")
    automatic = select_engine(settings=VectorStoreSettings(), advice=laptop)
    assert (automatic.chosen, automatic.reasons) == ("lancedb", laptop.reasons)
    stale = select_engine(settings=VectorStoreSettings(), preference="chroma", advice=laptop)
    assert stale.chosen == "lancedb" and "`chroma` is not offered here" in stale.warnings[0]


def test_an_unknown_setting_is_refused_without_its_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """It counted as auto; start-up stops on it first (manage.connectors)."""
    from manage.vectorstores import VectorStoreSettings
    from utils.settings import SettingsError

    monkeypatch.setenv("CMP_VECTOR_STORE", " Qdrant ")
    monkeypatch.setenv("CMP_QDRANT_URL", "http://qdrant:6333")
    server = VectorStoreSettings.from_env()
    assert (server.engine, server.qdrant_url) == ("qdrant", "http://qdrant:6333")
    monkeypatch.setenv("CMP_VECTOR_STORE", "zz9-chroma")
    with pytest.raises(SettingsError, match="CMP_VECTOR_STORE is not a vector database") as refused:
        VectorStoreSettings.from_env()
    assert "zz9" not in str(refused.value)


def test_each_engine_opens_its_own_index(tmp_path: Path) -> None:
    from manage.vectorstores.qdrant_index import user_collection

    lance = VECTOR_STORES.get("lancedb").open(tmp_path / "lancedb", url="http://ignored:1")
    assert lance.location == str(tmp_path / "lancedb")  # embedded only: an address does not apply
    qdrant = VECTOR_STORES.get("qdrant")
    embedded = qdrant.open(tmp_path / "qdrant")
    assert embedded.mode == "embedded" and embedded.location == str(tmp_path / "qdrant" / "storage")
    shared = qdrant.open(tmp_path / "qdrant", url="http://qdrant:6333", username="Dr. Alice-Smith")
    assert shared.mode == "server" and shared.collection == user_collection("Dr. Alice-Smith")
    assert user_collection("Dr. Alice-Smith") == "cmp_dr_alice_smith_chunks"


def test_an_engine_that_cannot_run_names_the_group_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from manage.vectorstores import VectorStoreError, open_index

    monkeypatch.setitem(sys.modules, "lancedb", None)
    with pytest.raises(VectorStoreError, match=r"LanceDB cannot be used: .*uv sync --extra ai"):
        open_index("lancedb", base_dir=tmp_path)


def test_the_registry_loads_neither_the_engines_nor_the_ai_stack() -> None:
    contract.check_light(
        "manage.vectorstores:VECTOR_STORES",
        never_loaded=("langchain_core", "lancedb", "qdrant_client", "streamlit"),
    )


def test_the_package_names_load_when_first_used() -> None:
    from manage import vectorstores as package

    assert package.VectorStoreSettings.__name__ == "VectorStoreSettings"
    assert "VectorStoreSettings" in dir(package)
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        package.Nope


def test_a_server_collection_is_never_named_after_the_account(tmp_path: Path) -> None:
    """The server may belong to someone else (Qdrant Cloud): a new collection's name says nothing of the user. One
    named after the account by an earlier version is recognised (the next rebuild deletes it); no request is made."""
    import re

    from manage.vectorstores.qdrant_index import POINTER_NAME, QdrantIndex, user_collection

    server = QdrantIndex(
        tmp_path, url="https://qdrant.example.org:6333", collection=user_collection("alice")
    )
    names = {server._generation_name() for _ in range(3)}
    assert len(names) == 3 and all(re.fullmatch(r"cmp_[0-9a-f]{32}", name) for name in names)
    assert not server.named_after_account  # nothing built yet
    (tmp_path / POINTER_NAME).write_text("cmp_alice_chunks_ab12cd34", encoding="utf-8")
    assert server.named_after_account
    (tmp_path / POINTER_NAME).write_text(names.pop(), encoding="utf-8")
    assert not server.named_after_account
    embedded = QdrantIndex(tmp_path / "local", path=tmp_path / "local" / "storage")
    assert embedded._generation_name().startswith("chunks_") and not embedded.named_after_account


def test_the_server_s_key_goes_to_its_own_address_only() -> None:
    """CMP_QDRANT_API_KEY was attached to any address a user typed (a connection test sent it there)."""
    from manage.vectorstores import VectorStoreSettings

    server = VectorStoreSettings(
        qdrant_url="https://admin.cloud.qdrant.io:6333/", qdrant_api_key="ADMIN-KEY"
    )
    assert server.key_for("https://ADMIN.cloud.qdrant.io:6333") == "ADMIN-KEY"
    assert (
        server.key_for("https://attacker.example.org:6333") is None and server.key_for(None) is None
    )
    # None asked: the server's address; an emptied field: embedded
    assert server.server_for(None) == "https://admin.cloud.qdrant.io:6333/"
    assert server.server_for("") is None and server.server_for(" http://q:6333 ") == "http://q:6333"


class _FakeServer:
    """The calls a Qdrant server receives from reset(), recorded."""

    def __init__(self, collections: list[str]) -> None:
        self.collections = set(collections)
        self.asked: list[str] = []
        self.deleted: list[str] = []

    def collection_exists(self, name: str) -> bool:
        self.asked.append(name)
        return name in self.collections

    def create_collection(self, name: str, vectors_config) -> None:
        self.collections.add(name)

    def create_payload_index(self, *args, **kwargs) -> None:
        pass

    def delete_collection(self, name: str) -> None:
        self.collections.discard(name)
        self.deleted.append(name)


def test_an_earlier_index_moves_to_its_server_and_only_its_rebuild_deletes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier version kept one manifest and one pointer for every place. Opening the engine moves a server's to
    that server's folder: another server is never asked about it, an embedded build keeps it, and a rebuild on its
    server (however its address is spelt) deletes the collection named after the account."""
    import re

    pytest.importorskip("qdrant_client", reason="the Qdrant engine is in the optional 'ai' extra")
    from manage.vectorstores import qdrant_index as qi
    from manage.vectorstores.base import IndexManifest

    first, other = "https://a.example.org:6333", "https://b.example.org:6333"
    legacy = "cmp_alice_chunks_ab12cd34"
    directory = tmp_path / "qdrant"
    IndexManifest(
        engine="qdrant", embedding="ollama:old", dim=8, chunks=2, location=f"{first} ({legacy})"
    ).save(directory)
    (directory / qi.POINTER_NAME).write_text(legacy, encoding="utf-8")
    servers = {
        qi._server_key(first): _FakeServer([legacy]),
        qi._server_key(other): _FakeServer([]),
        "embedded": _FakeServer([]),
    }
    monkeypatch.setattr(
        qi,
        "_client",
        lambda *, path=None, url=None, api_key=None: servers[
            "embedded" if path else qi._server_key(url)
        ],
    )

    elsewhere = qi.QdrantIndex.open(directory, username="alice", url=other)
    assert not elsewhere.named_after_account and elsewhere.manifest() is None
    elsewhere.reset(8)
    assert (
        legacy not in servers[qi._server_key(other)].asked
    )  # never asked about another server's index
    qi.QdrantIndex.open(directory, username="alice").reset(8)  # an embedded build keeps it known
    back = qi.QdrantIndex.open(
        directory, username="alice", url="https://A.example.org:6333/"
    )  # another spelling
    assert back.named_after_account and back.manifest().embedding == "ollama:old"
    back.reset(8)
    assert servers[qi._server_key(first)].deleted == [legacy]
    assert (
        re.fullmatch(r"cmp_[0-9a-f]{32}", back.active_collection) and not back.named_after_account
    )
    assert qi.server_folder(directory, first) != qi.server_folder(
        directory, "a.example.org:6333"
    )  # http


def test_no_server_index_is_assumed_without_a_trace_in_the_workspace(tmp_path: Path) -> None:
    """Two installations can share a server and a user name, and the embedded index's manifest is not a server's:
    without this server's own trace, the old base name (another installation's index, perhaps) is never taken for
    this one's."""
    from manage.vectorstores.base import IndexManifest
    from manage.vectorstores.qdrant_index import NO_INDEX, QdrantIndex, user_collection

    server = QdrantIndex(
        tmp_path, url="https://shared.example.org:6333", collection=user_collection("admin")
    )
    assert not server.named_after_account and server.active_collection != user_collection("admin")
    directory = tmp_path / "qdrant"
    IndexManifest(
        engine="qdrant", embedding="e", dim=8, chunks=1, location=str(directory / "storage")
    ).save(directory)
    opened = QdrantIndex.open(directory, username="admin", url="https://shared.example.org:6333")
    assert opened.active_collection == NO_INDEX and not opened.named_after_account
