"""The session keys shared between tabs and pages are declared, with the lifetime AppState gives them (ui/keys.py)."""

from __future__ import annotations

import re
from pathlib import Path

from ui.keys import KEYS
from ui.state import DERIVED_KEYS, KEPT_ON_DATA_CHANGE

ROOT = Path(__file__).resolve().parents[2]


def _source(module: str) -> str:
    return (ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")


def test_each_key_is_declared_once_with_the_lifetime_appstate_gives_it() -> None:
    names = [key.name for key in KEYS]
    assert len(names) == len(set(names))
    for key in KEYS:
        derived, kept = key.name in DERIVED_KEYS, key.name in KEPT_ON_DATA_CHANGE
        if key.lifetime == "page":  # a widget's own state: AppState clears it only on request
            assert derived == key.reset_on_dataset, key.name
            assert kept or not derived, (
                key.name
            )  # kept when the data change, cleared with another dataset
            continue
        lifetime = "session" if not derived else ("dataset" if kept else "data_change")
        assert key.lifetime == lifetime, key.name


def test_the_owner_and_the_readers_use_the_key() -> None:
    """A declaration that no longer matches the code fails here, rather than misleading the next change."""
    for key in KEYS:
        name = re.escape(key.name)
        used = re.compile(rf"""(["']){name}\1|session_state\.{name}\b""")
        for module in (key.owner, *key.readers):
            assert used.search(_source(module)), f"{module} does not use {key.name!r}"
