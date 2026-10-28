"""Source-level quality gates (audit E14): fixed rules and ratchets that must never regress."""

import ast
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


def test_rules_are_written_through_their_kinds() -> None:
    """
    Every rule the application writes comes from a writer of core/criteria.py, whose definitions read as their kind
    (M5b step 6): the keys of a range and of listed rows appear nowhere else, and no other dict holds both "numeric"
    and "expression" (a rule holding both was refused, after the page had said it was added).
    """
    keys = _count(r"""["'](lower_bound|upper_bound|row_ids)["']""", *SOURCE_DIRS, exclude=("core/criteria.py",))
    assert not keys, keys
    both = []
    for file in _files(*SOURCE_DIRS):
        rel = file.relative_to(ROOT).as_posix()
        if rel == "core/criteria.py":
            continue
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Dict):
                written = {k.value for k in node.keys if isinstance(k, ast.Constant)}
                if {"numeric", "expression"} <= written:
                    both.append(f"{rel}:{node.lineno}")
    assert not both, both


def test_raw_html_is_ratcheted_down() -> None:
    # 31: the graph legend's block moved from graph/ into ui/ (M5b), and graph/ is counted too since then; 30: the
    # Home page's frequent values, cells of the file, shown as text (privacy review, 2026-10-21); 23: Validate's and the
    # dashboard's lists of rules, one card per rule in the flow's words (M5b step 6)
    hits = _count(r"unsafe_allow_html=True", "app_pages", "ui", "graph")
    assert len(hits) <= 23, f"{len(hits)} raw-HTML blocks (ceiling 23): {hits[:5]}"


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


def test_pages_never_build_services() -> None:
    """A page never builds a service of its own: the session has its dataset store, its trace and its AI engine,
    built once by the composition root (manage.services, ui.services; M5b step 5)."""
    assert not _count(r"\b(DBManager|TransformationManager|RAGManager)\(", "app_pages")


_SERVICE = r"(?<!class )\b(DBManager|TransformationManager|RAGManager)\("


def test_the_session_builds_its_services_with_the_users_workspace() -> None:
    """Only the composition root (manage/services.py, ui/services.py) builds the dataset store, the trace and the AI
    engine, always passing the user's scope: these services never ask the session for the user, and without a scope
    they use the shared folders."""
    roots = ("manage/services.py", "ui/services.py")
    assert not _count(_SERVICE, *SOURCE_DIRS, exclude=roots)
    assert _count(_SERVICE, *roots)
    assert not _count(_SERVICE + r"(?![^)]*\bscope=)", *roots)


_SERVICE_KEYS = "_services|db_manager|transformation_manager|rag_manager"
_SERVICE_WRITE = (
    rf"""session_state\[\s*["']({_SERVICE_KEYS})["']\s*\]\s*=(?!=)"""
    rf"""|session_state\.({_SERVICE_KEYS})\s*=(?!=)"""
    rf"""|session_state\.(pop|setdefault)\(\s*["']({_SERVICE_KEYS})["']"""
    rf"""|del\s+st\.session_state\[\s*["']({_SERVICE_KEYS})["']"""
)


def test_only_the_services_write_their_keys() -> None:
    """The session's services are kept under fixed keys, which ui/services.py alone writes: a page that replaced the
    AI engine left the running one, with its connections, out of reach of a withdrawn consent."""
    hits = [
        hit
        for hit in _count(_SERVICE_WRITE, *SOURCE_DIRS, exclude=("ui/services.py",))
        if not _is_comment(hit)
    ]
    assert not hits, hits


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


def test_no_source_file_starts_with_a_byte_order_mark() -> None:
    """Python imports such a file, but ast.parse and the developer-reference generator do not (a Windows editor or
    PowerShell's Out-File adds the mark; dev_scripts/doc_gen.py once dropped a whole page from the reference)."""
    marked = [
        file.relative_to(ROOT).as_posix()
        for file in _files(*SOURCE_DIRS, "prompts", "tests", "dev_scripts")
        if file.read_bytes().startswith(b"\xef\xbb\xbf")
    ]
    assert not marked, marked


def test_tests_are_not_named_like_packages() -> None:
    """A tests/<name>/ directory named like a top-level package shadows it under pytest."""
    packages = {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    clashes = [d.name for d in (ROOT / "tests").iterdir() if d.is_dir() and d.name in packages]
    assert not clashes, clashes


# The files not yet formatted (M5b): the list only shrinks. Lower the ceiling with it.
UNFORMATTED_CEILING = 161


def test_the_list_of_unformatted_files_only_shrinks() -> None:
    """
    Formatting applies to every file but pyproject's [tool.ruff.format] exclude. A listed file that is formatted
    now must leave the list, so that a later change cannot unformat it unseen.
    """
    import subprocess
    import sys

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    listed = pyproject["tool"]["ruff"]["format"]["exclude"]
    assert listed == sorted(set(listed)), "keep the list sorted, without repeats"
    assert len(listed) <= UNFORMATTED_CEILING, "the list only shrinks: do not add a file to it"
    missing = [path for path in listed if not (ROOT / path).exists()]
    assert not missing, f"no longer exist, delete them from the list: {missing}"
    check = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "--output-format", "concise"]
        + ["--config", "format.exclude = []", *listed],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # 0: all formatted, 1: some to reformat; anything else is ruff failing, not a list to empty
    assert check.returncode in (0, 1), check.stderr
    reported = {
        line.split(":")[0].replace("\\", "/")
        for line in check.stdout.splitlines()
        if "unformatted" in line
    }
    formatted = sorted(set(listed) - reported)
    assert not formatted, f"formatted now, delete them from [tool.ruff.format] exclude: {formatted}"


def test_every_package_has_a_coverage_floor() -> None:
    """dev_scripts/check_coverage_floors.py: one floor per package of the wheel, which CI checks."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_coverage_floors", ROOT / "dev_scripts" / "check_coverage_floors.py"
    )
    assert spec is not None and spec.loader is not None
    floors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(floors)
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = set(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    assert set(floors.FLOORS) == wheel, {"no floor": sorted(wheel - set(floors.FLOORS))}


def test_pre_commit_and_ci_format_with_the_same_ruff() -> None:
    """Dependabot bumps ruff in uv.lock; the pre-commit hook would otherwise format with another version."""
    hook = re.search(
        r"ruff-pre-commit\s*\n\s*rev:\s*v?([\d.]+)",
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
    )
    lock = re.search(
        r'\[\[package\]\]\nname = "ruff"\nversion = "([\d.]+)"',
        (ROOT / "uv.lock").read_text(encoding="utf-8"),
    )
    assert hook and lock
    assert hook.group(1) == lock.group(1), "set the ruff-pre-commit rev to the version in uv.lock"
