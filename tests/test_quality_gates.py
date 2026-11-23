"""Source-level quality gates (audit E14): fixed rules and ratchets that must never regress."""

import ast
import re
import tomllib
from pathlib import Path

import pytest

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
    and "expression" (a rule holding both was refused, after the page had said it was added), written as a literal or
    with dict(). This reads the spelling; the tests of the pages that write rules check what they wrote
    (tests/ui_layer/test_criteria_forms.py, test_outlier_rules.py, test_ai_criteria_add.py).
    """
    shapes = {"lower_bound", "upper_bound", "row_ids"}
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
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
                written = {k.arg for k in node.keywords if k.arg}
            else:
                continue
            if {"numeric", "expression"} <= written or written & shapes:
                both.append(f"{rel}:{node.lineno}")
    assert not both, both


def test_raw_html_is_ratcheted_down() -> None:
    # 31: the graph legend's block moved from graph/ into ui/ (M5b), and graph/ is counted too since then; 30: the
    # Home page's frequent values, cells of the file, shown as text (privacy review, 2026-10-21); 23: Validate's and the
    # dashboard's lists of rules, one card per rule in the flow's words (M5b step 6)
    hits = _count(r"unsafe_allow_html=True", "app_pages", "ui", "graph")
    assert len(hits) <= 23, f"{len(hits)} raw-HTML blocks (ceiling 23): {hits[:5]}"


def test_print_calls_are_ratcheted_down() -> None:
    # none since ruff's T20 holds over the whole application (pyproject.toml); this one reads the same folders
    hits = _count(r"^\s*print\(", "app_pages", "utils", "manage", "enrich", "explore", "monitor")
    assert not hits, f"print() calls: {hits}; use logging"


_LOG_CALLS = {"debug", "info", "warning", "error", "exception", "critical", "log"}
_LOGGERS = {"logger", "logging", "log", "_logger", "LOGGER", "_LOGGER"}
# what writes an exception without its values: the three functions of utils.logging_config, and no other
_WITHOUT_VALUES = {"exception_kind", "exception_said", "without_values"}
# what a logging call takes besides its message and the message's arguments
_NOT_TEXT = {"exc_info", "stack_info", "stacklevel"}
# a caught exception read otherwise in a logging call, and why it may be
_EXCEPTION_READ = {
    ("manage/preferences.py", 'str(e["loc"][0]) for e in exc.errors()'): "the names of the fields a file got wrong",
    ("utils/llm_utils.py", 'error["loc"]) for error in e.errors()'): "the fields a reply got wrong, not their values",
    ("manage/workers/wire.py", "it was not used\", str(refused)"): "a global's name, checked by _refused",
    ("manage/workers/pool.py", "did not start: %s\", str(exc)"): "_NotReady's own words: fixed, with numbers only",
    ("manage/workers/pool.py", "did not start (%s%s)\", type(exc).__name__"): "the number of a Windows error",
}


def _reads_exception_text(node: ast.AST, caught: set[str]) -> bool:
    """Whether ``node`` reads a caught exception otherwise than by its kind or through a function that writes it
    without its values."""
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name in _WITHOUT_VALUES or (name == "type" and len(node.args) == 1):
            return False
    if isinstance(node, ast.Name):
        return node.id in caught
    return any(_reads_exception_text(child, caught) for child in ast.iter_child_nodes(node))


def _gives_a_logger(node: ast.AST) -> bool:
    """A call that gives a logger: ``logging.getLogger(__name__)``, ``audit_logger()``."""
    if not isinstance(node, ast.Call):
        return False
    name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
    return name == "getLogger" or name.lower().endswith("logger")


def _logger_names(tree: ast.AST) -> set[str]:
    """The names a file gives a logger (``_audit = audit_logger()``), whatever they are."""
    return {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and _gives_a_logger(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _is_logger(node: ast.AST, named: set[str]) -> bool:
    """A logger: a name of ``_LOGGERS`` or one the file gave a logger, an attribute that ends in "logger"
    (``self.logger``), or one obtained by a call."""
    if isinstance(node, ast.Name):
        return node.id in _LOGGERS or node.id in named
    if isinstance(node, ast.Attribute):
        return node.attr.lower().endswith("logger")
    return _gives_a_logger(node)


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _texts_held(scope: ast.AST) -> set[str]:
    """The names of ``scope`` (a function, or the module) that hold an exception or its text: assigned from an
    expression that reads a caught exception (``last = exc``, ``err = f"failed: {exc}"``), or from such a name."""
    held: set[str] = set()

    def walk(node: ast.AST, caught: set[str]) -> bool:
        grew = False
        if isinstance(node, _SCOPES) and node is not scope:
            return False  # its own scope
        if isinstance(node, ast.ExceptHandler) and node.name:
            caught = caught | {node.name}
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            if _reads_exception_text(node.value, caught | held):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = {n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)}
                grew = not names <= held
                held.update(names)
        for child in ast.iter_child_nodes(node):
            grew = walk(child, caught) or grew
        return grew

    while walk(scope, set()):
        pass
    return held


def _logged_exception_texts(tree: ast.AST, source: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    named = _logger_names(tree)

    def visit(node: ast.AST, caught: set[str], held: set[str]) -> None:
        if isinstance(node, _SCOPES):
            held = _texts_held(node)
        if isinstance(node, ast.ExceptHandler) and node.name:
            caught = caught | {node.name}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LOG_CALLS
            and _is_logger(node.func.value, named)
            and any(
                _reads_exception_text(arg, caught | held)
                for arg in [*node.args, *(k.value for k in node.keywords if k.arg not in _NOT_TEXT)]
            )
        ):
            found.append((node.lineno, " ".join((ast.get_source_segment(source, node) or "").split())))
        for child in ast.iter_child_nodes(node):
            visit(child, caught, held)

    visit(tree, set(), _texts_held(tree))
    return found


def test_a_logging_call_names_a_caught_exception_by_its_kind_or_without_its_values() -> None:
    """
    An exception's text can quote a cell of the data ("could not convert string to float: 'Jean Dupont'") and a
    file's error its path, which names the account: a logging call gives ``type(exc).__name__`` (a failure on the
    data), or ``exception_said(exc)`` (a service's error: the kind and the text without quoted strings and paths).
    Never the exception itself, its ``str`` or an f-string of it, nor a name that was given one (``err = f"...
    {exc}"``, then ``logger.error(err)``): the process-wide cleaning of records (``records_without_values``) cannot
    undo a text already formatted, and a process that did not install it would write the whole text. ``exc_info`` is
    written by that cleaning, and stays allowed. Not seen here: a text that travels through a call or an attribute.
    """
    hits, used = [], set()
    for file in _files(*SOURCE_DIRS, "manage_users.py"):
        rel = file.relative_to(ROOT).as_posix()
        source = file.read_text(encoding="utf-8")
        for line, call in _logged_exception_texts(ast.parse(source), source):
            allowed = [key for key in _EXCEPTION_READ if key[0] == rel and key[1] in call]
            used.update(allowed)
            if not allowed:
                hits.append(f"{rel}:{line}: {call[:120]}")
    assert not hits, "a logging call writes an exception's text:\n" + "\n".join(hits)
    assert used == set(_EXCEPTION_READ), f"entries no call uses: {sorted(set(_EXCEPTION_READ) - used)}"


def test_the_logging_gate_reads_each_way_of_writing_an_exception() -> None:
    source = (
        "try:\n    pass\nexcept ValueError as exc:\n"
        "    logger.warning('a %s', exc)\n"
        "    logger.warning(f'b {exc}')\n"
        "    logger.warning('c %s', str(exc))\n"
        "    logger.warning('d ' + repr(exc))\n"
        "    logging.error('e %s', exc.args[0])\n"
        "    logger.warning('f (%s)', type(exc).__name__)\n"
        "    logger.warning('g %s', exception_said(exc))\n"
        "    logger.debug('h', exc_info=exc)\n"
        "    st.error(f'i {exc}')\n"
        "logger.warning('j %s', exc)\n"
        "def later():\n"
        "    last = said = None\n"
        "    try:\n        pass\n"
        "    except OSError as error:\n"
        "        last = error\n"
        "        text = f'failed: {error}'\n"
        "        said = exception_said(error)\n"
        "        logging.getLogger(__name__).error('k %s', error)\n"
        "        self.logger.error('l %s', error)\n"
        "    logger.error(text)\n"
        "    logger.error('n %s', last)\n"
        "    logger.error('o %s', said)\n"
        "    other.error('p %s', last)\n"
        "    logger.error(msg=f'q {last}')\n"
        "    logger.error('r %s', _reason(last))\n"
        "    logger.error('s %s', exception_kind(last), exc_info=last)\n"
        "    logger.log(30, 't %s', extra={'why': last})\n"
        "    _audit = audit_logger()\n"
        "    _audit.info('u %s', last)\n"
        "    _audit.info('v %s', exception_kind(last))\n"
    )
    found = [line for line, _ in _logged_exception_texts(ast.parse(source), source)]
    assert found == [4, 5, 6, 7, 8, 22, 23, 24, 25, 28, 29, 31, 33]


def test_version_is_recorded_in_the_changelog() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{version}]" in changelog or "## [Unreleased]" in changelog


def _session_writes(text: str) -> set[str]:
    """The session keys a module writes: [...] =, .name =, setdefault(), also through a name bound to the session."""
    alias = re.compile(r"""^\s*(\w+)\s*=\s*st\.session_state\s*$""", re.MULTILINE)  # ss = st.session_state
    holders = "|".join(["session_state", *(re.escape(name) for name in alias.findall(text))])
    written = re.compile(
        rf"""(?:{holders})(?:\[\s*["'](\w+)["']\s*\]\s*=(?!=)|\.setdefault\(\s*["'](\w+)["'])"""
        rf"""|(?:{holders})\.(?!setdefault\b)(\w+)\s*=(?!=)"""
    )
    return {m.group(1) or m.group(2) or m.group(3) for m in written.finditer(text)}


def test_undeclared_session_keys_are_ratcheted_down() -> None:
    """
    A key that the code writes into the session is declared in ui/keys.py, with its owner, its readers and its
    lifetime, from which AppState knows when to clear it (M5b). The keys written without a declaration can only become
    fewer: 54 when the cleared keys were first derived from the declarations, 52 once setdefault() and an alias were
    counted and the six keys AppState and the preferences write were declared. A key is written with [...] =, with
    setdefault(), or through a name the session is bound to (ss = st.session_state).
    """
    from ui.keys import BY_NAME

    sample = 'ss = st.session_state\nss["a"] = 1\nss.setdefault("b", 2)\nst.session_state.c = 3\nst.session_state["d"] = 4\n'
    assert _session_writes(sample) == {"a", "b", "c", "d"}
    undeclared = set()
    for file in _files("app_pages", "ui"):
        written = _session_writes(file.read_text(encoding="utf-8", errors="replace"))
        undeclared |= {key for key in written if key not in BY_NAME}
    assert len(undeclared) <= 52, f"{len(undeclared)} undeclared session keys (ceiling 52): {sorted(undeclared)[:5]}"


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


_SERVICE = r"(?<!class )\b(DBManager|TransformationManager|RAGManager|WorkspaceProtocolStore)\("


def test_the_session_builds_its_services_with_the_users_workspace() -> None:
    """Only the composition root (manage/services.py, ui/services.py) builds the dataset store, the trace, the study
    definitions and the AI engine, always passing the user's scope: these services never ask the session for the user,
    and without a scope they use the shared folders (the study definitions have none)."""
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
UNFORMATTED_CEILING = 153


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


def _floors_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_coverage_floors", ROOT / "dev_scripts" / "check_coverage_floors.py"
    )
    assert spec is not None and spec.loader is not None
    floors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(floors)
    return floors


def test_every_package_has_a_coverage_floor() -> None:
    """dev_scripts/check_coverage_floors.py: one floor per package of the wheel, which CI checks."""
    floors = _floors_script()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = set(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    assert set(floors.FLOORS) == wheel, {"no floor": sorted(wheel - set(floors.FLOORS))}


def test_a_package_off_its_coverage_floor_is_named_in_an_annotation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CI's job log needs admin rights: the annotation names the package where the public API shows it."""
    floors = _floors_script()
    report = tmp_path / "coverage.xml"
    report.write_text(
        '<coverage><packages><package><classes><class filename="app_pages/x.py"><lines>'
        '<line number="1" hits="1"/><line number="2" hits="0"/>'
        "</lines></class></classes></package></packages></coverage>",
        encoding="utf-8",
    )
    assert floors.check(report, {"app_pages": 80.0}, annotate=True) == ["app_pages"]
    assert "::error title=Coverage floor::app_pages 50.0 % (floor 80 %): UNDER THE FLOOR" in (
        capsys.readouterr().out
    )
    assert floors.check(report, {"app_pages": 40.0}, annotate=True) == ["app_pages"]
    assert "RAISE THE FLOOR to 48" in capsys.readouterr().out
    assert floors.check(report, {"app_pages": 48.0}, annotate=True) == []
    assert "::error" not in capsys.readouterr().out


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
