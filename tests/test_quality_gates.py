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


def test_stored_analysis_results_are_cleared_when_the_dataset_changes() -> None:
    """Every ``*_result(s)`` key a page stores is dropped by ``AppState.set_dataset`` (audit E8)."""
    from ui.state import DERIVED_KEYS

    written = re.compile(r"""session_state(?:\[\s*["'](\w+_results?)["']\s*\]|\.(\w+_results?))\s*=(?!=)""")
    stray = set()
    for file in _files("app_pages", "ui"):
        for match in written.finditer(file.read_text(encoding="utf-8", errors="replace")):
            key = match.group(1) or match.group(2)
            if key not in DERIVED_KEYS:
                stray.add(f"{file.relative_to(ROOT).as_posix()}: {key}")
    assert not stray, sorted(stray)


_DATASET_KEYS = "data|working_df|enriched_df|dataset_ref|current_dataset_name"
_DATASET_WRITE = (
    rf"""session_state\[\s*["']({_DATASET_KEYS})["']\s*\]\s*(=(?!=)|\+=)"""
    rf"""|session_state\.({_DATASET_KEYS})\s*=(?!=)"""
    rf"""|session_state\.(pop|setdefault)\(\s*["']({_DATASET_KEYS})["']"""
    rf"""|del\s+st\.session_state\[\s*["']({_DATASET_KEYS})["']"""
)


def test_only_app_state_writes_the_dataset() -> None:
    """Every change of the session data goes through ui/state.py (AppState): one traced write path (audit E8)."""
    hits = [
        hit
        for hit in _count(_DATASET_WRITE, *SOURCE_DIRS, exclude=("ui/state.py",))
        if not _is_comment(hit)
    ]
    assert not hits, hits


def test_analysis_pages_keep_results_in_the_result_store() -> None:
    """Results are stamped with the data, population and settings they were computed on (ui/results.py)."""
    hits = _count(
        r"""session_state\[\s*["']\w+_results?["']\s*\]\s*=(?!=)""",
        "app_pages/analysis_survival.py",
        "app_pages/analysis_extras.py",
        "app_pages/analysis_diagnostic.py",
    )
    assert not hits, hits


def test_trace_steps_are_written_by_app_state_only() -> None:
    """Pages and helpers record trace steps through AppState (commit_change, stage, note)."""
    hits = [
        hit
        for hit in _count(
            r"\.add_step\(", "app_pages", "ui", "utils", "enrich", "explore", "monitor", exclude=("ui/state.py",)
        )
        if not _is_comment(hit)
    ]
    assert not hits, hits


def test_pages_use_the_session_trace() -> None:
    """A page never starts a trace of its own: the session has one (AppState.transformation_manager)."""
    assert not _count(r"TransformationManager\(", "app_pages")


def _is_comment(hit: str) -> bool:
    path, line = hit.rsplit(":", 1)
    text = (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()[int(line) - 1]
    return text.lstrip().startswith("#")


def test_the_wheel_ships_every_first_party_package() -> None:
    """A built wheel must contain every package the code imports (biostats/core/ingest/ui were once missing)."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    shipped = set(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    first_party = set(pyproject["tool"]["ruff"]["lint"]["isort"]["known-first-party"])
    on_disk = {p.name for p in ROOT.iterdir() if p.is_dir() and any(p.glob("*.py")) and p.name in first_party}
    assert shipped == first_party, {"missing": sorted(first_party - shipped), "extra": sorted(shipped - first_party)}
    assert on_disk == first_party, sorted(first_party - on_disk)


def test_tests_are_not_named_like_packages() -> None:
    """A tests/<name>/ directory named like a top-level package shadows it under pytest."""
    packages = {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    clashes = [d.name for d in (ROOT / "tests").iterdir() if d.is_dir() and d.name in packages]
    assert not clashes, clashes
