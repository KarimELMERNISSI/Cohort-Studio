"""Source-level quality gates (audit E14): fixed rules and ratchets that must never regress."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("app_pages", "biostats", "core", "enrich", "explore", "graph", "ingest", "manage", "monitor", "ui", "utils", "main.py")


def _files(*dirs: str):
    for spec in dirs:
        path = ROOT / spec
        if path.is_file():
            yield path
        else:
            yield from (p for p in path.rglob("*.py") if "__pycache__" not in p.parts)


def _count(pattern: str, *dirs: str, exclude: tuple[str, ...] = ()) -> list[str]:
    regex = re.compile(pattern, re.MULTILINE)
    hits = []
    for file in _files(*dirs):
        rel = file.relative_to(ROOT).as_posix()
        if any(rel.endswith(x) for x in exclude):
            continue
        for match in regex.finditer(file.read_text(encoding="utf-8", errors="replace")):
            line = file.read_text(encoding="utf-8", errors="replace").count("\n", 0, match.start()) + 1
            hits.append(f"{rel}:{line}")
    return hits


def test_no_eval_outside_the_whitelisted_evaluator() -> None:
    hits = _count(r"(?<![\w.])eval\(", *SOURCE_DIRS, exclude=("core/expressions/safe_eval.py",))
    assert not hits, hits


def test_no_artificial_sleeps_on_pages() -> None:
    assert not _count(r"time\.sleep\(", "app_pages", "ui")


def test_no_hard_coded_data_directories() -> None:
    hits = _count(r"""(["']data/|os\.path\.join\(["']data["'])""", "app_pages", "utils", "manage", "ui", exclude=("utils/data_paths.py",))
    assert not hits, hits


def test_raw_html_is_ratcheted_down() -> None:
    hits = _count(r"unsafe_allow_html=True", "app_pages", "ui")
    assert len(hits) <= 30, f"{len(hits)} raw-HTML blocks (ceiling 30): {hits[:5]}"


def test_print_calls_are_ratcheted_down() -> None:
    hits = _count(r"^\s*print\(", "app_pages", "utils", "manage", "enrich", "explore", "monitor")
    assert len(hits) <= 72, f"{len(hits)} print() calls (ceiling 72); use logging"


def test_version_is_recorded_in_the_changelog() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{version}]" in changelog or "## [Unreleased]" in changelog


def test_tests_are_not_named_like_packages() -> None:
    """A tests/<name>/ directory named like a top-level package shadows it under pytest."""
    packages = {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    clashes = [d.name for d in (ROOT / "tests").iterdir() if d.is_dir() and d.name in packages]
    assert not clashes, clashes
