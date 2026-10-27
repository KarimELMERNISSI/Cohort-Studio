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
