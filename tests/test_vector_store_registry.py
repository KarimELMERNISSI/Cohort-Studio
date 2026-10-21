"""The vector databases as a connector family (M5b): the registry keeps the connector contract, and reading it loads
neither the engines nor the AI stack (it runs without the optional 'ai' group)."""

import subprocess
import sys
from pathlib import Path

import pytest

from manage.vectorstores import VECTOR_STORES
from tests import connector_contract as contract

ROOT = Path(__file__).resolve().parents[1]


def test_the_stored_names_never_change() -> None:
    # kept in the users' preferences, the index manifests and CMP_VECTOR_STORE
    contract.check_names(VECTOR_STORES, ("lancedb", "qdrant"))
    assert VECTOR_STORES.valid() == ("auto", "lancedb", "qdrant")


def test_the_registry_keeps_the_connector_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    contract.check_availability(VECTOR_STORES, monkeypatch)
    contract.check_classes(VECTOR_STORES)
    contract.check_precedence(VECTOR_STORES)
    contract.check_unknown_setting(VECTOR_STORES)


@pytest.mark.parametrize(("name", "module"), [("lancedb", "lancedb"), ("qdrant", "qdrant_client")])
def test_a_missing_client_names_its_group(
    name: str, module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract.check_missing_package(VECTOR_STORES, name, module, monkeypatch)
    assert VECTOR_STORES.availability(name).extra == "ai"


def test_the_registry_says_what_the_lists_it_replaces_said() -> None:
    # golden: the engines' own titles and summaries equal the lists of selection.py, deleted in the next commit
    from manage.vectorstores.selection import ENGINE_SUMMARIES, ENGINE_TITLES

    names = VECTOR_STORES.names()
    assert {e.value: title for e, title in ENGINE_TITLES.items()} == {
        n: VECTOR_STORES.display_name(n) for n in names
    }
    assert {e.value: summary for e, summary in ENGINE_SUMMARIES.items()} == {
        n: VECTOR_STORES.get(n).summary for n in names
    }


def test_each_engine_says_what_it_is() -> None:
    for name in VECTOR_STORES.names():
        engine = VECTOR_STORES.get(name)
        assert engine.display_name and engine.summary.endswith("."), name
    assert VECTOR_STORES.display_name("lancedb") == "LanceDB"
    assert VECTOR_STORES.display_name("qdrant") == "Qdrant"


def test_the_registry_loads_neither_the_engines_nor_the_ai_stack() -> None:
    code = (
        "import sys\n"
        "from manage.vectorstores import VECTOR_STORES\n"
        "[VECTOR_STORES.availability(name) for name in VECTOR_STORES.names()]\n"
        "print(sorted(m for m in ('langchain_core', 'lancedb', 'qdrant_client', 'streamlit') if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_the_package_names_load_when_first_used() -> None:
    import manage.vectorstores as package

    assert package.VectorStoreSettings.__name__ == "VectorStoreSettings"
    assert "VectorStoreSettings" in dir(package)
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        package.Nope
