"""
The layers of the code base and the rules between them (architecture review of 2026-10-19, M5b).

| Layer | Packages | Role |
| --- | --- | --- |
| L9 | app_pages | pages: inputs, a service call, a result card |
| L8 | ui | shell, AppState, components |
| L7 | manage | application services, stores and connectors |
| L6 | graph | graph model, builders, renderers |
| L5 | explore, monitor | quality audit and correlations; outliers and comparisons |
| L4 | enrich | formulas, transformations, imputation, merging |
| L3 | ingest | reading files |
| L2 | core | the pure domain kernel and the extension kit (core/registry.py) |
| L1 | biostats, prompts | the statistics library; prompt texts |
| L0 | utils | settings, logging, optional extras, paths, export helpers |

Every ``utils`` module is classified: the platform ones (``UTILS_PLATFORM``) at L0, the legacy ones with the
package they serve (``UTILS_GROUPS``) until a milestone dissolves them. ``manage``'s groups (iam, ai, repro,
storage) are not ordered yet.

- **R1** Imports go strictly down; two groups on the same layer do not import each other.
- **R2** No import cycles. The one exception is the MI engine and its worker module, by design.
- **R3** Nothing below ui reaches Streamlit, through any chain of imports. Graph renderers import it inside their
  ``render`` method only, and a Streamlit component package (yFiles) inside ``translate`` too.
- **R4** The AI providers' SDKs are imported only in ``manage/rag/providers``.
- **R5** The headless entry points (``manage_users``; ``manage.replay`` from M8) reach neither Streamlit nor the
  packages of the ``ai`` extra.

The graph is read from the source:
- imports inside functions count (they run when the function does); those under ``if TYPE_CHECKING:`` do not;
- "from package import submodule" loads the submodule; "from package import name" of a third-party package is
  recorded as both, so ``from google import genai`` counts as ``google.genai``;
- R1 and R2 use the imports as written. R3, R5 and the engine check follow what actually runs: importing
  ``a.b.c`` also runs ``a/__init__`` and ``a/b/__init__``, and a module named in a string (a lazy registry such as
  ``"graph.renderers.plotly"``) or in ``import_module`` is loaded when it is used. A string that equals a
  module's name counts as loading it, which errs on the safe side;
- a Streamlit component package counts as Streamlit, all along the chains.

Today's exceptions are listed below, each with the milestone step that removes it. The lists can only shrink: a
new violation fails, and so does a listed one that no longer occurs, until it is deleted from its list.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections import deque
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path

import pytest

from utils.optional import EXTRAS

ROOT = Path(__file__).resolve().parents[1]

LAYERS = {
    "utils": 0,
    "biostats": 1,
    "prompts": 1,
    "core": 2,
    "ingest": 3,
    "enrich": 4,
    "explore": 5,
    "monitor": 5,
    "graph": 6,
    "manage": 7,
    "ui": 8,
    "app_pages": 9,
}
UTILS_PLATFORM = {
    "utils",
    "utils.config_loader",
    "utils.data_paths",
    "utils.environment",
    "utils.export_utils",
    "utils.logging_config",
    "utils.optional",
    "utils.settings",
}
# legacy utils modules, with the package they serve (a name covers its submodules)
UTILS_GROUPS = {
    "utils.analysis_utils": "explore",
    "utils.clustering_utils": "enrich",
    "utils.epidemiology_utils": "explore",
    "utils.llm_utils": "manage",
    "utils.miss_forest": "enrich",
    "utils.multipage": "ingest",
    "utils.statistics_utils": "explore",
    "utils.visualization_utils": "explore",
}
ENTRY_POINTS = ("main", "manage_users")  # above every layer
# R5; manage.replay arrives with M8, which reads traces outside the app as the replay does today
HEADLESS = ("manage_users", "manage.replay", "manage.reproduction_manager", "manage.services")
AI_SDKS = ("anthropic", "google.genai", "google.generativeai", "mistralai", "ollama", "openai")
AI_PACKAGES = EXTRAS["ai"]
# Streamlit inside their render method only (R3), and in translate a Streamlit component package: a renderer draws
# through Streamlit, and the yFiles primitives it translates into come from a Streamlit component (see _drawing)
RENDERERS = "graph.renderers"
MODULE_NAME = re.compile(r"[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)+")  # a dotted name in a string

# --- today's exceptions (they can only shrink) ------------------------------------------------------------------
# R1, (importer, imported): upward imports
KNOWN_UPWARD: set[tuple[str, str]] = set()
# R2: the cycles allowed, by design
CYCLES_BY_DESIGN = [{"biostats.mi.engine", "biostats.mi.parallel"}]
# R3, (module below ui, the import through which it reaches Streamlit)
KNOWN_STREAMLIT: set[tuple[str, str]] = set()
# R4, modules outside manage/rag/providers that import a provider SDK: none (Gemini's file upload of the document
# graph moved into its connector, read_document)
KNOWN_SDK: set[str] = set()


# --- the import graph ----------------------------------------------------------------------------------------------


@cache
def modules() -> dict[str, Path]:
    """Every module of the layered packages and the entry points, by dotted name."""
    out: dict[str, Path] = {}
    for package in LAYERS:
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            parts = list(path.relative_to(ROOT).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            out[".".join(parts)] = path
    for name in ENTRY_POINTS:
        out[name] = ROOT / f"{name}.py"
    return out


def _type_checking(node: ast.AST) -> bool:
    test = getattr(node, "test", None)
    return isinstance(node, ast.If) and (
        (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
        or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
    )


def _walk(node: ast.AST, function: str | None = None) -> Iterator[tuple[ast.AST, str | None]]:
    """Every node that runs, with the function it runs in (None at module or class level)."""
    for child in ast.iter_child_nodes(node):
        if _type_checking(child):
            for branch in child.orelse:  # the else branch of `if TYPE_CHECKING:` runs
                yield branch, function
                yield from _walk(branch, function)
            continue
        yield child, function
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            yield from _walk(child, child.name)
        elif isinstance(child, ast.Lambda):
            yield from _walk(child, function or "<lambda>")
        else:
            yield from _walk(child, function)


def _resolve(name: str) -> str | None:
    """The first-party module an import name loads (the longest known prefix)."""
    mods = modules()
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in mods:
            return candidate
    return None


def _streamlit(name: str) -> bool:
    """Streamlit, or a Streamlit component package."""
    return "streamlit" in name.split(".")[0]


def _drawing(function: str | None, target: str) -> bool:
    """A renderer's import allowed by R3: Streamlit in ``render``; a Streamlit component package (whose objects the
    renderer translates into, as yFiles' are) in ``translate`` too, never Streamlit itself."""
    if function == "render":
        return _streamlit(target)
    return function == "translate" and _streamlit(target) and target.split(".")[0] != "streamlit"


def _ai(name: str) -> bool:
    return _matches(name, AI_PACKAGES)


def _not_in_engine(name: str) -> bool:
    return _streamlit(name) or _matches(name, NOT_IN_ENGINE)


def _import_module_target(call: ast.Call) -> str | None:
    """The module named in ``importlib.import_module("<name>")`` or ``import_module("<name>")``."""
    func = call.func
    named = (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
        isinstance(func, ast.Name) and func.id == "import_module"
    )
    first = call.args[0] if call.args else None
    if named and isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


@cache
def graph() -> tuple[
    dict[str, set[tuple[str, bool]]], dict[str, set[tuple[str, bool]]], dict[str, set[str]]
]:
    """
    (first-party imports, other imports, modules named in strings) of each module; an import carries whether it is
    lazy (inside a function). A renderer's drawing imports are left out (R3, :func:`_drawing`).
    """
    mods = modules()
    first: dict[str, set[tuple[str, bool]]] = {m: set() for m in mods}
    other: dict[str, set[tuple[str, bool]]] = {m: set() for m in mods}
    named: dict[str, set[str]] = {m: set() for m in mods}
    for name, path in mods.items():
        package = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
        renderer = name.startswith(f"{RENDERERS}.")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, function in _walk(tree):
            lazy = function is not None
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if (
                    MODULE_NAME.fullmatch(node.value.split(":")[0])
                    and node.value.split(":")[0] in mods
                ):
                    named[name].add(node.value.split(":")[0])
                continue
            if isinstance(node, ast.Call):
                target = _import_module_target(node)
                if target is None:
                    continue
                targets = [target]
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = package[: len(package) - node.level + 1]
                    base = ".".join([*anchor, *([node.module] if node.module else [])])
                # "from package import submodule" loads the submodule; for a third-party package both are kept
                targets = []
                for alias in node.names:
                    full = f"{base}.{alias.name}"
                    if full in mods:
                        targets.append(full)
                    elif _resolve(base) is None:
                        targets += [base, full]
                    else:
                        targets.append(base)
            else:
                continue
            for target in targets:
                if renderer and _drawing(function, target):
                    continue  # allowed by design (see RENDERERS)
                resolved = _resolve(target)
                if resolved is None:
                    other[name].add((target, lazy))
                elif resolved != name:
                    first[name].add((resolved, lazy))
    return first, other, named


def _ancestors(module: str) -> list[str]:
    """The packages whose __init__ runs before ``module``."""
    mods = modules()
    parts = module.split(".")
    return [p for p in (".".join(parts[:i]) for i in range(1, len(parts))) if p in mods]


@cache
def runs() -> dict[str, set[tuple[str, bool]]]:
    """
    What importing each module actually loads: its imports and their parent packages, its own parent packages,
    and the modules it names in strings (loaded lazily, when the registry is used).
    """
    first, _, named = graph()
    out: dict[str, set[tuple[str, bool]]] = {}
    for module, targets in first.items():
        edges = set(targets)
        for target, lazy in targets:
            edges |= {(parent, lazy) for parent in _ancestors(target)}
        edges |= {(parent, False) for parent in _ancestors(module)}
        edges |= {(target, True) for target in named[module]}
        out[module] = {(t, lazy) for t, lazy in edges if t != module}
    return out


def group(module: str) -> str:
    for prefix, owner in UTILS_GROUPS.items():
        if module == prefix or module.startswith(f"{prefix}."):
            return owner
    return module.split(".")[0]


def layer(module: str) -> int:
    return LAYERS[group(module)]


def _matches(name: str, packages: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(f"{p}.") for p in packages)


@cache
def _reaching(match: Callable[[str], bool], lazy: bool) -> frozenset[str]:
    """The modules from which a third-party package that ``match`` accepts is loaded, through any chain."""
    _, other, _ = graph()
    edges = runs()
    reach = {m for m, names in other.items() if any(match(n) and (lazy or not z) for n, z in names)}
    queue = deque(reach)
    importers: dict[str, set[str]] = {}
    for module, targets in edges.items():
        for target, is_lazy in targets:
            if lazy or not is_lazy:
                importers.setdefault(target, set()).add(module)
    while queue:
        for importer in importers.get(queue.popleft(), ()):
            if importer not in reach:
                reach.add(importer)
                queue.append(importer)
    return frozenset(reach)


def _chain(start: str, match: Callable[[str], bool], lazy: bool = True) -> list[str] | None:
    """One chain of imports from ``start`` to a third-party package that ``match`` accepts, or None."""
    _, other, _ = graph()
    edges = runs()
    previous: dict[str, str | None] = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        hit = next(
            (n for n, z in sorted(other[current]) if (lazy or not z) and match(n)),
            None,
        )
        if hit:
            chain = [hit, current]
            while previous[chain[-1]] is not None:
                chain.append(previous[chain[-1]])  # type: ignore[arg-type]
            return list(reversed(chain))
        for target, is_lazy in sorted(edges[current]):
            if (lazy or not is_lazy) and target not in previous:
                previous[target] = current
                queue.append(target)
    return None


def _difference(found: set, known: set, *, rule: str) -> str:
    new, fixed = sorted(found - known), sorted(known - found)
    lines = [f"{rule} (tests/test_layering.py)."]
    if new:
        lines.append(f"New: {new}")
    if fixed:
        lines.append(f"No longer occurs, delete it from the list: {fixed}")
    return " ".join(lines)


# --- the rules -----------------------------------------------------------------------------------------------------


def test_every_module_has_a_layer() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert set(LAYERS) == set(wheel), "a package without a layer, or a layer without a package"
    assert {p.stem for p in ROOT.glob("*.py")} == set(ENTRY_POINTS), (
        "a top-level module without a rule"
    )
    utils = {m for m in modules() if m.split(".")[0] == "utils"}
    unclassified = sorted(m for m in utils if m not in UTILS_PLATFORM and group(m) == "utils")
    assert not unclassified, (
        f"classify these utils modules (UTILS_PLATFORM or UTILS_GROUPS): {unclassified}"
    )
    listed = [*UTILS_PLATFORM, *UTILS_GROUPS]
    stale = [p for p in listed if not any(m == p or m.startswith(f"{p}.") for m in utils)]
    assert not stale, f"these utils modules no longer exist: {stale}"
    assert set(UTILS_GROUPS.values()) <= set(LAYERS)


def test_r1_imports_go_down_the_layers() -> None:
    first, _, _ = graph()
    upward = set()
    for importer, targets in first.items():
        if importer in ENTRY_POINTS:
            continue
        for imported, _lazy in targets:
            if group(imported) != group(importer) and layer(imported) >= layer(importer):
                upward.add((importer, imported))
    assert upward == KNOWN_UPWARD, _difference(
        upward,
        KNOWN_UPWARD,
        rule="R1: imports go strictly down, and two groups on one layer do not import each other",
    )


def test_r2_no_import_cycles() -> None:
    first, _, _ = graph()
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[set[str]] = []

    def visit(node: str) -> None:  # Tarjan's strongly connected components
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        for target, _lazy in sorted(first[node]):
            if target not in index:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], index[target])
        if low[node] == index[node]:
            component = set()
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.add(member)
                if member == node:
                    break
            if len(component) > 1:
                cycles.append(component)

    for node in sorted(first):
        if node not in index:
            visit(node)
    unexpected = [sorted(c) for c in cycles if c not in CYCLES_BY_DESIGN]
    assert not unexpected, f"R2: import cycles (tests/test_layering.py): {unexpected}"
    gone = [sorted(c) for c in CYCLES_BY_DESIGN if c not in cycles]
    assert not gone, f"R2: these cycles no longer occur, delete them from CYCLES_BY_DESIGN: {gone}"


def test_r3_nothing_below_ui_reaches_streamlit() -> None:
    _, other, _ = graph()
    edges = runs()
    reach = _reaching(_streamlit, lazy=True)
    found = set()
    for module in modules():
        if module in ENTRY_POINTS or layer(module) >= LAYERS["ui"]:
            continue
        found |= {(module, name) for name, _ in other[module] if _streamlit(name)}
        found |= {(module, target) for target, _ in edges[module] if target in reach}
    assert found == KNOWN_STREAMLIT, _difference(
        found, KNOWN_STREAMLIT, rule="R3: nothing below ui reaches Streamlit"
    ) + "".join(
        f" Chain: {' -> '.join(_chain(m, _streamlit) or [m, h])}."
        for m, h in sorted(found - KNOWN_STREAMLIT)
    )


def test_r4_ai_sdks_only_in_the_providers() -> None:
    _, other, _ = graph()
    found = {
        module
        for module, names in other.items()
        if not _matches(module, ("manage.rag.providers",))
        and any(_matches(name, AI_SDKS) for name, _lazy in names)
    }
    assert found == KNOWN_SDK, _difference(
        found,
        KNOWN_SDK,
        rule="R4: the AI providers' SDKs are imported only in manage/rag/providers",
    )


@pytest.mark.parametrize("entry", HEADLESS)
def test_r5_headless_entry_points_reach_neither_streamlit_nor_ai(entry: str) -> None:
    if entry not in modules():
        pytest.skip(f"{entry} does not exist yet")
    for match in (_streamlit, _ai):
        chain = _chain(entry, match)
        assert chain is None, f"R5: {entry} reaches {chain[0]}: {' -> '.join(chain)}"


# The engine (domain, statistics, import) runs without the web, plotting, AI or vector-database stacks: an install
# of the core dependencies only, or a library use in a notebook, must import it (follow-up programme M4).
ENGINE = ("core", "biostats", "ingest")
NOT_IN_ENGINE = ("streamlit", "plotly", *AI_PACKAGES, "google", "mistralai")


@pytest.mark.parametrize("package", ENGINE)
def test_the_engine_needs_no_ui_plotting_or_ai_library(package: str) -> None:
    # what importing the package loads, through any chain of module-level imports
    offenders = sorted(
        " -> ".join(chain)
        for module in modules()
        if module.split(".")[0] == package
        and (chain := _chain(module, _not_in_engine, lazy=False)) is not None
    )
    assert not offenders, offenders


def test_biostats_imports_only_biostats() -> None:
    """The statistics library stands alone: the app passes it values, never its own objects."""
    first, _, _ = graph()
    found = sorted(
        f"{m} -> {t}"
        for m, targets in first.items()
        if m.split(".")[0] == "biostats"
        for t, _ in targets
        if t.split(".")[0] != "biostats"
    )
    assert not found, found


# --- the environment (M5b step 3) ------------------------------------------------------------------------------------

# Only utils.settings reads the environment, loads .env or asks Streamlit for secrets; the modules below may, for the
# reason given, until the step named. The list can only shrink.
KNOWN_ENVIRONMENT = {
    # its seed: biostats imports no first-party package (parsed as utils.settings parses it; the seed moves into the
    # analysis specs in M8)
    "biostats._types",
    # the environment of the parallel imputation's worker process: a process variable, not a setting
    "biostats.mi.parallel",
    # turns the tracing switches of third-party libraries (LangSmith) off: not settings of the application
    "manage.ai_payload",
}
_ENVIRONMENT_NAMES = {"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"}


def _reads_environment(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr in _ENVIRONMENT_NAMES:
                found.append(f"os.{node.attr}")
            elif node.value.id == "st" and node.attr == "secrets":
                found.append("st.secrets")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            found += [
                f"from os import {a.name}" for a in node.names if a.name in _ENVIRONMENT_NAMES
            ]
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            found += [f"import {n}" for n in names if n == "dotenv" or n.startswith("dotenv.")]
    return found


def test_only_the_settings_read_the_environment() -> None:
    """utils.settings validates every setting at start-up: a module reading the environment itself would see a value
    nobody checked (M5b step 3)."""
    found = {
        name
        for name, path in modules().items()
        if name != "utils.settings"
        and _reads_environment(ast.parse(path.read_text(encoding="utf-8")))
    }
    assert found == KNOWN_ENVIRONMENT, _difference(
        found, KNOWN_ENVIRONMENT, rule="only utils.settings reads the environment"
    )


@pytest.mark.parametrize("package", ["core", "biostats"])
def test_the_engine_never_reads_the_settings(package: str) -> None:
    """core and biostats receive values as parameters: they stay usable without the application's settings."""
    first, _, _ = graph()
    reading = sorted(
        name
        for name, targets in first.items()
        if _matches(name, (package,)) and any(target == "utils.settings" for target, _ in targets)
    )
    assert not reading, reading
