"""The licence inventory (documentation/developer/License_Dependencies.md) lists what uv.lock installs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "documentation" / "developer" / "License_Dependencies.md"


def _inventory():
    spec = importlib.util.spec_from_file_location(
        "license_inventory", ROOT / "dev_scripts" / "license_inventory.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_licence_inventory_lists_every_locked_package() -> None:
    """A new dependency must have its licence reviewed; a version bump (Dependabot) need not."""
    groups = _inventory().package_groups()
    text = DOCUMENT.read_text(encoding="utf-8")
    missing = [
        name for packages in groups.values() for name, _ in packages if f"| {name} | " not in text
    ]
    assert not missing, (
        f"{len(missing)} package(s) not in the inventory, e.g. {missing[:5]}: "
        "regenerate it with `uv run python dev_scripts/license_inventory.py`"
    )
    assert "xlrd" in {name for name, _ in groups["core"]}
    assert {name for name, _ in groups["graph-yfiles"]} == {"yfiles-graphs-for-streamlit"}


def test_the_inventory_names_the_project_licence_and_the_proprietary_engine() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    assert "released under the Apache License 2.0" in text
    assert "yfiles-graphs-for-streamlit" in text and "No redistribution" in text
