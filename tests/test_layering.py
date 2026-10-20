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
  ``render`` method only.
- **R4** The AI providers' SDKs are imported only in ``manage/rag/providers``.
- **R5** The headless entry points (``manage_users``; ``manage.replay`` from M8) reach neither Streamlit nor the
  packages of the ``ai`` extra.

The graph is read from the source:
- imports inside functions count (they run when the function does); those under ``if TYPE_CHECKING:`` do not;
- "from package import submodule" loads the submodule; "from package import name" of a third-party package is
  recorded as both, so ``from google import genai`` counts as ``google.genai``;
- R1 and R2 use the imports as written. R3, R5 and the engine check follow what actually runs: importing
  ``a.b.c`` also runs ``a/__init__`` and ``a/b/__init__``, and a module named in a string (a lazy registry such as
  ``"graph.renderers.plotly"``) or in ``importlib.import_module`` is loaded when it is used.

Today's exceptions are listed below, each with the milestone step that removes it. The lists can only shrink: a
new violation fails, and so does a listed one that no longer occurs, until it is deleted from its list.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections import deque
from collections.abc import Iterator
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
}
# legacy utils modules, with the package they serve (a name covers its submodules)
UTILS_GROUPS = {
    "utils.analysis_utils": "explore",
    "utils.clustering_utils": "enrich",
    "utils.data_analyzer": "enrich",  # moves into ingest with the date work (M6)
    "utils.date_parser": "ingest",  # moves into ingest (M6)
    "utils.epidemiology_utils": "explore",
    "utils.llm_utils": "manage",
    "utils.miss_forest": "enrich",
    "utils.multipage": "ingest",
    "utils.path_utils": "manage",
    "utils.session_scope": "ui",  # reads the Streamlit session: moves to ui (M5b, composition root)
    "utils.statistics_utils": "explore",
    "utils.visualization_utils": "explore",
}
ENTRY_POINTS = ("main", "manage_users")  # above every layer
HEADLESS = ("manage_users", "manage.replay")  # R5; manage.replay arrives with M8
AI_SDKS = ("anthropic", "google.genai", "google.generativeai", "mistralai", "ollama", "openai")
AI_PACKAGES = EXTRAS["ai"]
RENDERERS = "graph.renderers"  # Streamlit inside their render method only (R3)
MODULE_NAME = re.compile(r"[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)+")  # a dotted name in a string

# --- today's exceptions (they can only shrink) ------------------------------------------------------------------
# R1, (importer, imported): upward imports
KNOWN_UPWARD = {
    # dead shims, deleted with the composition root (M5b)
    ("explore.corr_matrix", "graph.builders.correlation"),
    ("utils.analysis_utils", "app_pages.analysis_config"),
    ("utils.statistics_utils", "app_pages.statistics_guides"),
    # services that ask the Streamlit session for the user: they take the scope explicitly (M5b)
    ("manage.db_manager", "utils.session_scope"),
    ("manage.rag_manager", "utils.session_scope"),
    ("manage.transformation_manager", "utils.session_scope"),
    ("utils.path_utils", "utils.session_scope"),
}
# R2: the cycles allowed, by design
CYCLES_BY_DESIGN = [{"biostats.mi.engine", "biostats.mi.parallel"}]
# R3, (module below ui, the import through which it reaches Streamlit)
KNOWN_STREAMLIT = {
    # graph's Streamlit adapters, imported only by pages: they move to ui (M5b, connectors on one kit)
    ("graph.legend", "streamlit"),
    ("graph.ui", "streamlit"),
    ("graph.ui", "graph.legend"),
    # the yFiles renderer imports its Streamlit component outside render, and the renderer registry names it
    # (M5b, connectors on one kit)
    ("graph.renderers.yfiles", "yfiles_graphs_for_streamlit"),
    ("graph.manager", "graph.renderers.yfiles"),
    # through utils.session_scope: the services take the scope explicitly (M5b, composition root)
    ("manage.db_manager", "utils.session_scope"),
    ("manage.rag_manager", "utils.session_scope"),
    ("manage.reproduction_manager", "manage.db_manager"),
    ("manage.reproduction_manager", "utils.path_utils"),
    ("manage.transformation_manager", "utils.session_scope"),
    ("utils.path_utils", "utils.session_scope"),
    # through the dead shims of the pages (M5b, composition root)
    ("utils.analysis_utils", "app_pages.analysis_config"),
    ("utils.statistics_utils", "app_pages.statistics_guides"),
}
# R4, modules outside manage/rag/providers that import a provider SDK
KNOWN_SDK = {
    "manage.rag_documents",  # Gemini's file upload of the document graph: moves into its provider (M5b, AI providers)
}


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


@cache
def graph() -> tuple[
    dict[str, set[tuple[str, bool]]], dict[str, set[tuple[str, bool]]], dict[str, set[str]]
]:
    """
    (first-party imports, other imports, modules named in strings) of each module; an import carries whether it is
    lazy (inside a function). A renderer's Streamlit imports inside its render method are left out (R3).
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
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
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
            for target in targets:
                if renderer and function == "render" and _streamlit(target):
                    continue  # allowed by design: a renderer draws through Streamlit
                resolved = _resolve(target)
                if resolved is None:
                    other[name].add((target, lazy))
                elif resolved != name:
                    first[name].add((resolved, lazy))
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "import_module"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ):
                target = call.args[0].value
                if _resolve(target) is None:
                    other[name].add((target, True))
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
def _reaching(packages: tuple[str, ...], lazy: bool) -> frozenset[str]:
    """The modules from which one of the third-party ``packages`` is loaded, through any chain of imports."""
    _, other, _ = graph()
    edges = runs()
    reach = {
        m
        for m, names in other.items()
        if any(_matches(n, packages) and (lazy or not z) for n, z in names)
    }
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


def _chain(start: str, packages: tuple[str, ...], lazy: bool = True) -> list[str] | None:
    """One chain of imports from ``start`` to one of ``packages``, or None."""
    _, other, _ = graph()
    edges = runs()
    previous: dict[str, str | None] = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        hit = next(
            (n for n, z in sorted(other[current]) if (lazy or not z) and _matches(n, packages)),
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
    reach = _reaching(("streamlit",), lazy=True) | {
        m for m, names in other.items() if any(_streamlit(n) for n, _ in names)
    }
    found = set()
    for module in modules():
        if module in ENTRY_POINTS or layer(module) >= LAYERS["ui"]:
            continue
        found |= {(module, name) for name, _ in other[module] if _streamlit(name)}
        found |= {(module, target) for target, _ in edges[module] if target in reach}
    assert found == KNOWN_STREAMLIT, _difference(
        found, KNOWN_STREAMLIT, rule="R3: nothing below ui reaches Streamlit"
    ) + "".join(
        f" Chain: {' -> '.join(_chain(m, ('streamlit',)) or [m, h])}."
        for m, h in sorted(found - KNOWN_STREAMLIT)
    )


def test_r4_ai_sdks_only_in_the_providers() -> None:
    _, other, _ = graph()
    found = {
        module
        for module, names in other.items()
        if not module.startswith("manage.rag.providers")
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
    for packages in (("streamlit",), AI_PACKAGES):
        chain = _chain(entry, packages)
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
        and (chain := _chain(module, NOT_IN_ENGINE, lazy=False)) is not None
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
