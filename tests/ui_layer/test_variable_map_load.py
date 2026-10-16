"""Variable map: a saved taxonomy version loads with its formulas in one click (AppTest)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp

TAXONOMY = {
    "weight_kg": {"standard_name": "Body weight", "category": "anthropometry"},
    "height_cm": {"standard_name": "Body height", "category": "anthropometry"},
    "bmi": {"standard_name": "Body mass index", "category": "anthropometry"},
}
FORMULAS = {
    "f_bmi": {
        "output_variable": "bmi",
        "input_variables": ["weight_kg", "height_cm"],
        "formula": "weight_kg / (height_cm / 100) ** 2",
    }
}


def _page() -> None:
    from app_pages.data_insight import app

    app()


@pytest.fixture
def workspace(tmp_path: Path):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path)
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    yield tmp_path
    dp.DATA_ROOT = previous


def _save_version(username: str) -> Path:
    from core.workspace import UserScope

    version = Path(UserScope.for_user(username).taxonomy_dir()) / "v1"
    version.mkdir(parents=True)
    (version / "taxonomy_metadata.json").write_text(json.dumps(TAXONOMY), encoding="utf-8")
    (version / "formulas_metadata.json").write_text(json.dumps(FORMULAS), encoding="utf-8")
    return version / "taxonomy_metadata.json"


def test_loading_a_version_from_the_empty_page_brings_its_formulas(workspace: Path) -> None:
    saved = _save_version("vmap_user")
    at = AppTest.from_function(_page, default_timeout=120)
    at.session_state["username"] = "vmap_user"
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    load = [b for b in at.button if b.label == "Load v1"]
    assert load, [b.label for b in at.button]  # the empty state offers the saved version
    load[0].click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["offline_formulas"] == FORMULAS  # one click, formulas included
    taxonomy = at.session_state["offline_taxonomy"]
    assert taxonomy["bmi"]["related_formula_ids"] == ["f_bmi"]  # links repaired from the registry
    assert sorted(taxonomy["weight_kg"]["related_formula_ids"]) == ["f_bmi"]
    assert Path(at.session_state["loaded_path"]) == saved  # Overwrite targets the loaded version
