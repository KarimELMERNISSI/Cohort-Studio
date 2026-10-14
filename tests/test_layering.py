"""Dependency direction (audit E13): domain and service modules never import Streamlit or page modules at import time."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Modules that must not import Streamlit anywhere (not even lazily).
NO_STREAMLIT = [
    "core", "biostats", "ingest", "graph/model.py", "graph/palette.py", "graph/builders", "explore/correlation_strength.py", "explore/quality_score.py",
    "manage/iam", "manage/vectorstores", "manage/db_manager.py", "manage/transformation_manager.py", "manage/reproduction_manager.py",
    "utils/analysis_utils.py", "utils/path_utils.py", "utils/statistics_utils.py", "utils/visualization_utils.py",
    "utils/multipage.py", "utils/data_paths.py", "enrich/transformation_logic.py",
    "enrich/custom_metrics_and_filters.py", "enrich/formulas.py", "enrich/imputation_service.py",
    "explore/data_quality_auditor.py",
]  # utils/session_scope.py is the one bridge that reads the Streamlit session by design
# Packages whose modules must not import pages (app_pages, ui) at module level.
NO_PAGE_IMPORTS_AT_TOP = ["core", "biostats", "ingest", "graph", "manage", "utils", "enrich", "explore", "monitor"]
# Lazy (function-level) page imports still present in service code: known debt, must not grow.
LAZY_PAGE_IMPORT_CEILING = 2


def _py_files(spec: str):
    path = ROOT / spec
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(tree: ast.AST, *, top_level_only: bool):
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


@pytest.mark.parametrize("spec", NO_STREAMLIT)
def test_domain_modules_never_import_streamlit(spec: str) -> None:
    offenders = []
    for file in _py_files(spec):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        if any(name == "streamlit" or name.startswith("streamlit.") for name in _imports(tree, top_level_only=False)):
            offenders.append(str(file.relative_to(ROOT)))
    assert not offenders, offenders


@pytest.mark.parametrize("package", NO_PAGE_IMPORTS_AT_TOP)
def test_no_page_imports_at_module_level(package: str) -> None:
    offenders = []
    for file in _py_files(package):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for name in _imports(tree, top_level_only=True):
            if name.split(".")[0] in ("app_pages", "ui"):
                offenders.append(f"{file.relative_to(ROOT)} -> {name}")
    assert not offenders, offenders


def test_core_is_the_bottom_layer() -> None:
    """core holds the pure domain (state, population, lineage, session): it imports no layer above it."""
    above = ("app_pages", "ui", "manage", "enrich", "explore", "monitor", "graph", "biostats", "ingest")
    offenders = []
    for file in _py_files("core"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for name in _imports(tree, top_level_only=False):
            if name.split(".")[0] in above:
                offenders.append(f"{file.relative_to(ROOT)} -> {name}")
    assert not offenders, offenders


def test_lazy_page_imports_in_services_are_bounded() -> None:
    lazy = []
    for package in ("manage", "utils", "enrich", "explore", "monitor"):
        for file in _py_files(package):
            tree = ast.parse(file.read_text(encoding="utf-8"))
            top = set(_imports(tree, top_level_only=True))
            for name in _imports(tree, top_level_only=False):
                if name.split(".")[0] == "app_pages" and name not in top:
                    lazy.append(f"{file.relative_to(ROOT)} -> {name}")
    assert len(lazy) <= LAZY_PAGE_IMPORT_CEILING, lazy
