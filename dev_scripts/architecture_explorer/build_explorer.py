"""
Build the architecture explorer of CohortStudio: one self-contained HTML file (no library, no network).

    python dev_scripts/architecture_explorer/build_explorer.py [<repository root>] [<output file>]

The page is written to ``docs/architecture/index.html`` unless another file is given. It is never committed: the
Pages workflow builds it after CI has passed on ``main`` and publishes it with the site, and
``tests/test_architecture_explorer.py`` runs the same build at each push.

The words come from ``explorer_content.py`` (the bricks, the journeys, the recipes, the matrix) and
``explorer_levels.py`` (the levels of detail, where each connector runs, the catalogues); the overview's cards, its
connector sockets and its storage come from the README's scheme itself (``dev_scripts/make_architecture_diagram.py``).
Everything else is read from the code, with the layering test's own reader of the import graph
(``tests/test_layering.py``):

- the modules of every box at every level, their sizes, what each module says of itself;
- the arrows: A points to B when a module of A imports a module of B (imports inside functions count, those under
  ``if TYPE_CHECKING:`` do not);
- the libraries each box and each row of the matrix imports, with the versions ``uv.lock`` pins;
- the connector families and the catalogues, from their registries;
- the tests, counted in ``tests/``.

The build stops (``NotBuilt``) when the page would not be true:

- a module that belongs to no brick or to two, a pattern that matches nothing, a brick that is not on the layer of
  its modules;
- an arrow that goes up, or that joins two packages of one layer;
- the layering test, the developer guide's layer table, the README's scheme and the bands not placing the same
  packages on the same layers; a caption of the overview that the scheme's script no longer holds;
- a connector registry or a catalogue of the code that the explorer does not describe; a registered connector
  without a place, or a place that contradicts what the connector's class declares;
- a library a module imports that the explorer cannot name;
- a file, a module, a function, a class or a setting named in the words that the code does not hold.

``load(root)`` reads the code, ``build()`` returns the page's data and ``write(data, output)`` writes the page, so
that a test runs the checks and renders the page in a folder of its own.
"""

from __future__ import annotations

import ast
import builtins
import datetime
import html
import importlib
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent


class NotBuilt(Exception):
    """The page would not be true: what is wrong, in words."""


def stop(message: str) -> None:
    raise NotBuilt(message)


# ----------------------------------------------------------------------------- the picture's geometry

WIDTH = 1000
BAND_X, LABEL_W = 8, 132
NODES_X = BAND_X + LABEL_W + 10
NODES_RIGHT = (
    WIDTH - 52
)  # at the right of the boxes: the spine of the arrows that skip a band, then the arrow
SPINE_X = NODES_RIGHT + 14  # "imports go down only"
BAND_RIGHT = NODES_RIGHT + 28
GAP = 8
BAND_GAP = 28  # the gap between two bands is a corridor: the arrows' tracks run in it
BAND_PAD = 7  # above and under the boxes of a band
TITLE_Y, LINE_Y, LINE_H = (
    21,
    37,
    13,
)  # in a box: its title's baseline, its first line's, the step between lines
BOUNDARY_GAP = 18  # under the interface: the line below which Streamlit stays out
TITLE_PX, SUB_PX = (
    7.1,
    5.6,
)  # mean width of a character of a box's title (13 px bold) and of its lines (11 px)
SOCKETS_W, CATALOGUES_W = (
    256,
    190,
)  # at the overview: the room kept at the right of the services and of the engine
PSEUDO = ("tests", "storage")  # boxes that hold no module of the application


def node_height(lines: int) -> int:
    return LINE_Y + LINE_H * (lines - 1) + 10


# a third-party import's top-level name -> (how it is written, its name in uv.lock)
LIBRARIES = {
    "streamlit": ("Streamlit", "streamlit"),
    "pandas": ("pandas", "pandas"),
    "numpy": ("NumPy", "numpy"),
    "scipy": ("SciPy", "scipy"),
    "statsmodels": ("statsmodels", "statsmodels"),
    "sklearn": ("scikit-learn", "scikit-learn"),
    "lifelines": ("lifelines", "lifelines"),
    "plotly": ("Plotly", "plotly"),
    "matplotlib": ("Matplotlib", "matplotlib"),
    "seaborn": ("seaborn", "seaborn"),
    "duckdb": ("DuckDB", "duckdb"),
    "pyarrow": ("PyArrow", "pyarrow"),
    "pydantic": ("Pydantic", "pydantic"),
    "pydantic_settings": ("pydantic-settings", "pydantic-settings"),
    "pyzipper": ("pyzipper", "pyzipper"),
    "bcrypt": ("bcrypt", "bcrypt"),
    "lightgbm": ("LightGBM", "lightgbm"),
    "umap": ("umap-learn", "umap-learn"),
    "prince": ("prince", "prince"),
    "networkx": ("NetworkX", "networkx"),
    "docx": ("python-docx", "python-docx"),
    "openpyxl": ("openpyxl", "openpyxl"),
    "langchain_core": ("LangChain core", "langchain-core"),
    "langchain_text_splitters": ("LangChain text splitters", "langchain-text-splitters"),
    "lancedb": ("LanceDB", "lancedb"),
    "qdrant_client": ("Qdrant client", "qdrant-client"),
    "pypdf": ("pypdf", "pypdf"),
    "rapidfuzz": ("RapidFuzz", "rapidfuzz"),
    "charset_normalizer": ("charset-normalizer", "charset-normalizer"),
    "threadpoolctl": ("threadpoolctl", "threadpoolctl"),
    "httpx": ("HTTPX", "httpx"),
    "requests": ("Requests", "requests"),
    "langsmith": ("LangSmith", "langsmith"),
    "tqdm": ("tqdm", "tqdm"),
    "dotenv": ("python-dotenv", "python-dotenv"),
    "anthropic": ("Anthropic SDK", "anthropic"),
    "openai": ("OpenAI SDK", "openai"),
    "google": ("google-genai", "google-genai"),
    "yfiles_graphs_for_streamlit": ("yFiles Graphs for Streamlit", "yfiles-graphs-for-streamlit"),
    "pytest": ("pytest", "pytest"),
    "ruff": ("ruff", "ruff"),
    "pyright": ("pyright", "pyright"),
}
MOST_LIBRARIES, MOST_FILES, DOC_MOST = 7, 6, 360

# names written in plain text that look like a class of the code and are not one: products, methods, people
NOT_OF_THE_CODE = {
    "CohortStudio",
    "DuckDB",
    "LanceDB",
    "NetworkX",
    "LightGBM",
    "AppTest",
    "CodeQL",
    "DeLong",
    "PyArrow",
    "OpenAI",
    "McNemar",
    "IterativeImputer",
    "LangChain",
    "GitHub",
    "SciPy",
    "NumPy",
    "MissForest",
    "JavaScript",
}

# ----------------------------------------------------------------------------- the code, read by load()

ROOT: Path
content: ModuleType
levels: ModuleType
scheme: ModuleType
tl: ModuleType
MODULES: dict[str, Path] = {}
FIRST_PARTY: dict = {}
THIRD_PARTY: dict = {}
LOCK: dict[str, str] = {}
PROJECT: dict = {}
SIZES: dict[str, int] = {}
TREES: dict[str, ast.Module] = {}
DEFINED: set[str] = set()
MEMBERS: dict[str, set[str]] = {}
TOP: dict[str, set[str]] = {}
SETTINGS_TEXT = ""
SCHEME_TEXT = ""
FILE_NAMES: set[str] = set()
FIRST: set[str] = set()
CARDS: dict = {}
CARD_OF: dict[str, str] = {}
ENTRY_OF: dict[str, str] = {}
PATHLIKE = (".py", ".md", ".yml", ".yaml", ".toml", ".json", ".txt", ".html", ".svg")


def load(root: Path) -> None:
    """Read the code of the repository at ``root``: its modules, its import graph, its definitions."""
    global \
        ROOT, \
        content, \
        levels, \
        scheme, \
        tl, \
        MODULES, \
        FIRST_PARTY, \
        THIRD_PARTY, \
        LOCK, \
        PROJECT, \
        SIZES, \
        TREES
    global \
        DEFINED, \
        MEMBERS, \
        TOP, \
        SETTINGS_TEXT, \
        SCHEME_TEXT, \
        FILE_NAMES, \
        FIRST, \
        CARDS, \
        CARD_OF, \
        ENTRY_OF
    ROOT = Path(root).resolve()
    if str(ROOT) not in sys.path:
        sys.path.insert(
            0, str(ROOT)
        )  # the packages of the application, and the tests' reader of the import graph
    for folder in (
        ROOT / "dev_scripts",
        HERE,
    ):  # after everything else: nothing here shadows another module
        if str(folder) not in sys.path:
            sys.path.append(str(folder))
    content = importlib.import_module("explorer_content")
    levels = importlib.import_module("explorer_levels")
    scheme = importlib.import_module("make_architecture_diagram")
    tl = importlib.import_module("tests.test_layering")
    MODULES = tl.modules()
    FIRST_PARTY, THIRD_PARTY, _named = tl.graph()
    LOCK = {
        package["name"]: package["version"]
        for package in tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    }
    PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    texts = {module: path.read_text(encoding="utf-8") for module, path in MODULES.items()}
    SIZES = {module: text.count("\n") for module, text in texts.items()}
    TREES = {module: ast.parse(text) for module, text in texts.items()}
    DEFINED, MEMBERS, TOP = _definitions()
    SETTINGS_TEXT = (ROOT / "utils" / "settings.py").read_text(encoding="utf-8")
    SCHEME_TEXT = (ROOT / "dev_scripts" / "make_architecture_diagram.py").read_text(
        encoding="utf-8"
    )
    FILE_NAMES = _file_names()
    FIRST = set(tl.LAYERS) | set(tl.ENTRY_POINTS) | {"tests"}
    CARDS = {
        place: getattr(scheme, place) if place != "FOUNDATION" else (scheme.FOUNDATION,)
        for place in levels.OVERVIEW_PLACES
    }
    CARD_OF = {
        package.strip(): card_id(card.name)
        for cards in CARDS.values()
        for card in cards
        for package in card.name.split(scheme.DOT)
    }
    ENTRY_OF = {
        package: entry["id"] for entry in levels.OVERVIEW_ENTRIES for package in entry["packages"]
    }


def entity(target: str):
    """The object a ``"module:name"`` names."""
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def package_of(module: str) -> str:
    """The package a module counts with for the layers: the layering test's own classification of utils."""
    for legacy, served in tl.UTILS_GROUPS.items():
        if module == legacy or module.startswith(legacy + "."):
            return served
    return module.split(".")[0]


def band_of(module: str) -> str:
    """The band of a module: its package's layer, or the entry points' band."""
    return "E" if module in tl.ENTRY_POINTS else f"L{tl.LAYERS[package_of(module)]}"


def first_paragraph(module: str) -> str:
    """The first paragraph of a module's docstring, without its reStructuredText marks (``x``, :mod:`x`)."""
    doc = ast.get_docstring(TREES[module]) or ""
    text = " ".join(doc.split("\n\n")[0].split())
    text = re.sub(r":[a-z]+:`~?([^`]+)`", r"\1", text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    # an ellipsis (U+2026) where the paragraph is cut
    return text if len(text) <= DOC_MOST else text[: DOC_MOST - 1].rsplit(" ", 1)[0] + chr(0x2026)


def tops_of(module: str) -> set[str]:
    return {target.split(".")[0] for target, _lazy in THIRD_PARTY[module]}


def libraries_of(modules: Iterable[str], most: int | None = MOST_LIBRARIES) -> list[dict[str, str]]:
    """The libraries ``modules`` import, the most used first (ties by name, so that two builds agree); ``most``:
    how many are kept, or all of them."""
    seen: Counter[str] = Counter()
    for module in modules:
        for top in tops_of(module):
            if top in LIBRARIES:
                seen[top] += 1
    ranked = sorted(seen, key=lambda top: (-seen[top], LIBRARIES[top][0].lower()))
    return [named(top) for top in ranked[:most]]


def named(top: str) -> dict[str, str]:
    name, locked = LIBRARIES[top]
    return {"name": name, "version": LOCK.get(locked, "")}


def named_library(written: str) -> dict[str, str]:
    for top, (name, _locked) in LIBRARIES.items():
        if name == written:
            return named(top)
    return {"name": written, "version": ""}


def check_libraries() -> None:
    """Every library a module imports has a name here: none is left out of a box without a word."""
    unknown: dict[str, str] = {}
    for module in MODULES:
        for top in tops_of(module):
            if top not in LIBRARIES and top not in sys.stdlib_module_names and top not in FIRST:
                unknown.setdefault(top, module)
    if unknown:
        stop(
            "libraries the explorer cannot name (add them to LIBRARIES): "
            + ", ".join(f"{top} (in {module})" for top, module in sorted(unknown.items()))
        )


# ----------------------------------------------------------------------------- the documents agree


def layer_table() -> dict[str, int]:
    """The developer guide's layer table: each package and its layer."""
    text = (ROOT / "documentation" / "10_Developer_Guide.md").read_text(encoding="utf-8")
    table: dict[str, int] = {}
    for layer, cell in re.findall(r"^\| L(\d) \| (.+?) \|", text, re.MULTILINE):
        for package in re.findall(r"`([a-z_]+)`", cell.split("(")[0]):
            table[package] = int(layer)
    return table


def check_documents() -> None:
    table = layer_table()
    if table != tl.LAYERS:
        stop(
            f"the developer guide's layer table and tests/test_layering.py differ: {sorted(set(table.items()) ^ set(tl.LAYERS.items()))}"
        )
    if scheme.PACKAGES != set(tl.LAYERS):
        stop(
            f"the README's scheme and tests/test_layering.py differ: {sorted(scheme.PACKAGES ^ set(tl.LAYERS))}"
        )
    if set(levels.PACKAGES) != set(tl.LAYERS) | set(tl.ENTRY_POINTS):
        stop(
            f"explorer_levels.PACKAGES and tests/test_layering.py differ: {sorted(set(levels.PACKAGES) ^ (set(tl.LAYERS) | set(tl.ENTRY_POINTS)))}"
        )
    # the bands of the bricks name the packages of their layer
    for band in content.BANDS:
        if band["id"].startswith("L"):
            named_here = set(re.findall(r"([a-z_]+)/", band["path"]))
            on_layer = {
                package for package, layer in tl.LAYERS.items() if f"L{layer}" == band["id"]
            }
            if named_here != on_layer:
                stop(
                    f"the band {band['id']} names {sorted(named_here)}, the layering test places {sorted(on_layer)} there"
                )
    # the overview's captions are the scheme's own
    for band in levels.OVERVIEW_BANDS:
        if not band.get("own"):
            for words in (band["label"], band["title"]):
                if words and f'"{words}"' not in SCHEME_TEXT:
                    stop(
                        f"the overview says {words!r}, which the README's scheme (its script) no longer does"
                    )
    for extra in (levels.OVERVIEW_SOCKETS, levels.OVERVIEW_CATALOGUES):
        if f'"{extra["label"]}"' not in SCHEME_TEXT or extra["says"] not in SCHEME_TEXT:
            stop(f"the overview's {extra['label']} and the README's scheme differ")


def declared_registries() -> tuple[set[str], set[str]]:
    """The connector registries and the catalogues the code builds at module level, as ``"module:NAME"``."""
    families: set[str] = set()
    catalogues: set[str] = set()
    for module, tree in TREES.items():
        for node in tree.body:
            if not isinstance(node, ast.Assign | ast.AnnAssign) or not isinstance(
                node.value, ast.Call
            ):
                continue
            called = node.value.func
            name = (
                called.id
                if isinstance(called, ast.Name)
                else called.attr
                if isinstance(called, ast.Attribute)
                else ""
            )
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name) and name == "ConnectorRegistry":
                    families.add(f"{module}:{target.id}")
                elif isinstance(target, ast.Name) and name == "Catalogue":
                    catalogues.add(f"{module}:{target.id}")
    return families, catalogues


def check_registries() -> None:
    families, catalogues = declared_registries()
    described = {spec["registry"] for spec in levels.FAMILIES}
    if families != described:
        stop(
            f"the connector registries of the code and of explorer_levels.FAMILIES differ: {sorted(families ^ described)}"
        )
    held = {spec["holder"] for spec in levels.CATALOGUES}
    plain = {
        holder for holder in held if isinstance(entity(holder), dict)
    }  # a plain dict (biostats.plan.KINDS)
    if catalogues != held - plain:
        stop(
            f"the catalogues of the code and of explorer_levels.CATALOGUES differ: {sorted(catalogues ^ (held - plain))}"
        )
    slugs = [spec["slug"] for spec in (*levels.FAMILIES, *levels.CATALOGUES)] + [
        node["slug"] for node in content.NODES
    ]
    twice = [slug for slug, count in Counter(slugs).items() if count > 1]
    if twice:
        stop(f"an address names two things: {twice}")


# ----------------------------------------------------------------------------- who owns each module, per level


def brick_owners() -> dict[str, str]:
    """The brick of each module: an exact name before a ``package.*`` pattern, the longest pattern first."""
    exact: dict[str, str] = {}
    patterns: list[tuple[str, str]] = []
    for node in content.NODES:
        for pattern in node["modules"]:
            if pattern.endswith(".*"):
                patterns.append((pattern[:-2], node["id"]))
            elif pattern in exact:
                stop(f"{pattern} is given to {exact[pattern]} and to {node['id']}")
            else:
                exact[pattern] = node["id"]
    for name in exact:
        if name not in MODULES:
            stop(f"{name} is no module of the code base (brick {exact[name]})")
    patterns.sort(key=lambda item: -len(item[0]))
    used: set[str] = set()
    owner: dict[str, str] = {}
    for name in MODULES:
        if name in exact:
            owner[name] = exact[name]
            continue
        for prefix, node_id in patterns:
            if name == prefix or name.startswith(prefix + "."):
                owner[name] = node_id
                used.add(prefix)
                break
        else:
            stop(f"the module {name} belongs to no brick: add it to explorer_content.NODES")
    for prefix, node_id in patterns:
        if prefix not in used:
            stop(f"the pattern {prefix}.* of {node_id} matches no module")
    # a brick sits on the layer of its modules: the arrows then go down exactly when the layering test's do
    band = {node["id"]: node["band"] for node in content.NODES}
    for name, node_id in owner.items():
        if band_of(name) != band[node_id]:
            stop(
                f"the brick {node_id} is drawn on {band[node_id]}, and its module {name} is on {band_of(name)}"
            )
    return owner


def card_id(name: str) -> str:
    return "o_" + re.sub(r"[^a-z]+", "_", name).strip("_")


# ----------------------------------------------------------------------------- words that fit their box


def wrap(text: str, room: float, per_character: float) -> list[str]:
    """``text`` in lines that fit ``room``: cut at its " · " first, then between words."""
    most = max(8, int(room / per_character))
    lines: list[str] = []
    for piece in text.split(" · "):
        candidate = f"{lines[-1]} · {piece}" if lines else piece
        if lines and len(candidate) <= most:
            lines[-1] = candidate
            continue
        current = ""
        for word in piece.split():
            if current and len(current) + 1 + len(word) > most:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        lines.append(current)
    return lines


def too_wide(where: str, text: str, room: float, per_character: float) -> None:
    if len(re.sub(r"<[^>]+>", "", text)) * per_character > room:
        print(
            f"  too wide for its box: {where} ({len(text)} characters, room for {int(room / per_character)})"
        )


# ----------------------------------------------------------------------------- one level


def view(
    level: str,
    bands: list[dict],
    words: dict[str, dict],
    owner: Callable[[str], str],
    *,
    same_layer_check: bool,
    most_lines: int,
    reserved: dict[str, float] | None = None,
    check_arrows: bool = True,
) -> dict:
    """The boxes of one level, laid out, with their arrows. ``reserved``: per band, the width kept free at the right
    of its boxes (the overview's sockets and catalogues). ``check_arrows``: stop on an arrow that goes up or joins
    two packages of one layer (the layering test's rule R1, which a test leaves to the layering test)."""
    reserved = reserved or {}
    members: dict[str, list[str]] = defaultdict(list)
    for module in MODULES:
        members[owner(module)].append(module)
    for node_id in words:
        if node_id not in PSEUDO and not members[node_id]:
            stop(f"{level}: {node_id} holds no module")
    for node_id in members:
        if node_id not in words:
            stop(f"{level}: modules belong to {node_id}, which has no words")
    band_index = {band["id"]: index for index, band in enumerate(bands)}
    by_band: dict[str, list[str]] = defaultdict(list)
    for node_id, word in words.items():
        by_band[word["band"]].append(node_id)

    # widths first (a band's boxes share its width, by their share), then the lines that fit, then the heights
    boxes: dict[str, dict] = {}
    lines: dict[str, list[str]] = {}
    for band in bands:
        ids = by_band[band["id"]]
        if not ids:
            stop(f"{level}: the band {band['id']} holds nothing")
        kept = reserved.get(band["id"], 0)
        room = NODES_RIGHT - NODES_X - (kept + GAP if kept else 0) - GAP * (len(ids) - 1)
        shares = sum(words[node_id].get("share", 1.0) for node_id in ids)
        x = float(NODES_X)
        for node_id in ids:
            width = room * words[node_id].get("share", 1.0) / shares
            boxes[node_id] = {"x": round(x, 1), "w": round(width, 1)}
            lines[node_id] = wrap(words[node_id]["sub"], width - 18, SUB_PX)
            too_wide(
                f"the title of {node_id} at {level}", words[node_id]["short"], width - 18, TITLE_PX
            )
            if len(lines[node_id]) > most_lines:
                print(f"  too many lines for its box: {node_id} at {level} ({len(lines[node_id])})")
            x += width + GAP
    common = max(len(lines[node_id]) for node_id in words if node_id not in PSEUDO)
    placed, boundary = [], {}
    y = 12.0
    for band in bands:
        ids = by_band[band["id"]]
        own = all(node_id in PSEUDO for node_id in ids)
        height = node_height(max(len(lines[node_id]) for node_id in ids) if own else common)
        for node_id in ids:
            boxes[node_id].update({"y": y + BAND_PAD, "h": height})
        placed.append(
            {
                **band,
                "box": {"x": BAND_X, "y": y, "w": BAND_RIGHT - BAND_X, "h": height + 2 * BAND_PAD},
                "inner": {"y": y + BAND_PAD, "h": height, "right": NODES_RIGHT},
            }
        )
        y += height + 2 * BAND_PAD + BAND_GAP
        if band.get("boundary_after"):
            boundary = {"y": y - BAND_GAP + 4, "h": BOUNDARY_GAP - 2}
            y += BOUNDARY_GAP

    found: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"all": set(), "eager": set()}
    )
    for module, imports in FIRST_PARTY.items():
        for target, lazy in imports:
            a, b = owner(module), owner(target)
            if a == b:
                continue
            found[(a, b)]["all"].add(module)
            if not lazy:
                found[(a, b)]["eager"].add(module)
            up = band_index[words[a]["band"]] > band_index[words[b]["band"]]
            same = (
                same_layer_check
                and words[a]["band"] == words[b]["band"]
                and package_of(module) != package_of(target)
            )
            if check_arrows and (up or same):
                kind = "goes up" if up else "joins two packages of one layer"
                stop(f"{level}: the import of {target} by {module} {kind} ({a} -> {b})")
    edges = [
        {"u": a, "v": b, "n": len(s["all"]), "eager": len(s["eager"])}
        for (a, b), s in sorted(found.items())
    ]
    order_of = {node_id: index for index, node_id in enumerate(words)}

    nodes: dict[str, dict] = {}
    for node_id, word in words.items():
        mods = sorted(members[node_id], key=lambda name: -SIZES[name])
        node = {
            "members": mods,
            "modules": len(mods),
            "lines": sum(SIZES[name] for name in mods),
            "files": [
                {"path": relative(MODULES[name]), "lines": SIZES[name]}
                for name in mods[:MOST_FILES]
            ],
            "libs": libraries_of(mods),
            "methods": [],
            "note": "",
            "facts": [],
        }
        node.update(word)  # a pseudo box brings its own counts and files
        node.update(
            {
                "subLines": lines[node_id],
                "box": boxes[node_id],
                "imports": sorted({e["v"] for e in edges if e["u"] == node_id}, key=order_of.get),
                "importedBy": sorted(
                    {e["u"] for e in edges if e["v"] == node_id}, key=order_of.get
                ),
            }
        )
        nodes[node_id] = node
    return {
        "bands": placed,
        "boundary": boundary,
        "height": round(y - BAND_GAP + 12),
        "order": list(words),
        "nodes": nodes,
        "edges": edges,
        "extras": [],
    }


def layers_in_words(layers: Iterable[int]) -> str:
    ordered = sorted(set(layers), reverse=True)
    return ("layers " if len(ordered) > 1 else "layer ") + ", ".join(
        str(layer) for layer in ordered
    )


def overview_bands() -> list[dict]:
    """The overview's bands, each with the layers of its packages written from the layering test."""
    paths = {"E": "browser · terminal", "S": "files · DuckDB", "T": "tests/ · CI"}
    for place, band in levels.OVERVIEW_PLACES.items():
        packages = [
            package.strip() for card in CARDS[place] for package in card.name.split(scheme.DOT)
        ]
        paths[band] = layers_in_words(tl.LAYERS[package] for package in packages)
    return [{**band, "path": paths[band["id"]]} for band in levels.OVERVIEW_BANDS]


def overview_words() -> dict[str, dict]:
    words: dict[str, dict] = {}
    for entry in levels.OVERVIEW_ENTRIES:
        words[entry["id"]] = {
            "band": "E",
            "short": entry["short"],
            "sub": entry["sub"],
            "title": entry["short"],
            "role": entry["role"],
            "rule": entry["rule"],
            "slug": entry["id"][2:],
        }
    for place, band in levels.OVERVIEW_PLACES.items():
        for card in CARDS[place]:
            extra = levels.OVERVIEW_WORDS.get(card.name)
            if extra is None:
                stop(
                    f"the README's card {card.name} has no words in explorer_levels.OVERVIEW_WORDS"
                )
            packages = [package.strip() for package in card.name.split(scheme.DOT)]
            words[card_id(card.name)] = {
                "band": band,
                "short": card.name,
                "sub": " ".join(card.lines),  # the scheme's words, in lines cut again for this box
                "title": card.name,
                "share": card.share,
                "layer": "L"
                + " · ".join(
                    str(layer)
                    for layer in dict.fromkeys(tl.LAYERS[package] for package in packages)
                ),
                "role": extra["role"],
                "rule": extra["rule"],
                "slug": card_id(card.name)[2:],
            }
    return words


def package_words() -> dict[str, dict]:
    words: dict[str, dict] = {}
    for package in tl.ENTRY_POINTS:
        words["p_" + package] = {"band": "E", **levels.PACKAGES[package], "slug": package}
    for package, layer in sorted(tl.LAYERS.items(), key=lambda item: (-item[1], item[0])):
        words["p_" + package] = {"band": f"L{layer}", **levels.PACKAGES[package], "slug": package}
    for word in words.values():
        word.setdefault("title", word["short"])
    return words


def brick_words() -> dict[str, dict]:
    return {
        node["id"]: {
            "band": node["band"],
            "short": node["short"],
            "sub": node["sub"],
            "title": node["title"],
            "role": node["role"],
            "rule": node["rule"],
            "methods": node["methods"],
            "note": node.get("note", ""),
            "slug": node["slug"],
        }
        for node in content.NODES
    }


def tests_words() -> dict:
    test_files = sorted((ROOT / "tests").rglob("test_*.py"))
    texts = [path.read_text(encoding="utf-8") for path in test_files]
    functions = sum(len(re.findall(r"^\s*def test_", text, re.MULTILINE)) for text in texts)
    listed = sum(
        len(v) for v in (tl.KNOWN_UPWARD, tl.CYCLES_BY_DESIGN, tl.KNOWN_STREAMLIT, tl.KNOWN_SDK)
    )
    tests = content.TESTS
    return {
        "band": "T",
        "short": tests["short"],
        "sub": tests["sub"],
        "title": tests["title"],
        "role": tests["role"],
        "rule": tests["rule"],
        "methods": tests["methods"],
        "slug": "tests",
        "facts": [
            f"{len(test_files):,} test files and {functions:,} test functions in <code>tests/</code> (a "
            "parametrised function runs as several tests).",
            (f"{listed} exceptions" if listed else "No exception")
            + " listed to the rules R1 to R4 in "
            f"<code>tests/test_layering.py</code>; {len(tl.KNOWN_ENVIRONMENT)} modules listed there read the "
            "environment themselves, each with its reason.",
        ],
        "modules": len(test_files),
        "lines": sum(text.count("\n") for text in texts),
        "files": [
            {"path": path, "lines": (ROOT / path).read_text(encoding="utf-8").count("\n")}
            for path in tests["files"]
        ],
        "libs": [named_library(name) for name in tests["libs"]],
        "testFiles": len(test_files),
        "testFunctions": functions,
    }


def storage_words() -> dict:
    """The storage panel of the README's scheme, as a box of the overview."""
    storage = levels.OVERVIEW_STORAGE
    items = [" ".join(item) for item in scheme.STORAGE] + [
        f"{scheme.ACCOUNTS[0]}: {' '.join(scheme.ACCOUNTS[1:])}"
    ]
    return {
        "band": "S",
        "short": storage["short"],
        "sub": " · ".join(items),
        "title": storage["title"],
        "role": storage["role"],
        "rule": storage["rule"],
        "slug": "storage",
        "modules": len(storage["files"]),
        "lines": sum(
            (ROOT / path).read_text(encoding="utf-8").count("\n") for path in storage["files"]
        ),
        "files": [
            {"path": path, "lines": (ROOT / path).read_text(encoding="utf-8").count("\n")}
            for path in storage["files"]
        ],
    }


def overview_extras(data: dict, families: list[dict], cats: list[dict]) -> list[dict]:
    """What the scheme draws beside its cards: a socket per connector family in the services, the catalogues in the
    engine. Each is a link to its card of the Connectors & catalogues view."""
    inner = {band["id"]: band["inner"] for band in data["bands"]}
    extras: list[dict] = []
    sockets = inner[levels.OVERVIEW_SOCKETS["band"]]
    left, top = NODES_RIGHT - SOCKETS_W, sockets["y"]
    extras.append(
        {
            "kind": "caption",
            "x": left,
            "y": top + 9,
            "label": levels.OVERVIEW_SOCKETS["label"],
            "says": levels.OVERVIEW_SOCKETS["says"],
        }
    )
    width, height = (SOCKETS_W - 6) / 2, (sockets["h"] - 15 - 6) / 2
    for index, family in enumerate(families):
        label = family["family"][:1].upper() + family["family"][1:] + "s"
        too_wide(f"the socket {label}", label, width - 14, 5.5)
        extras.append(
            {
                "kind": "socket",
                "label": label,
                "to": f"connectors/{family['slug']}",
                "says": ", ".join(i["name"] for i in family["implementations"]),
                "box": {
                    "x": round(left + (index % 2) * (width + 6), 1),
                    "y": top + 15 + (index // 2) * (height + 6),
                    "w": round(width, 1),
                    "h": round(height, 1),
                },
            }
        )
    cat = inner[levels.OVERVIEW_CATALOGUES["band"]]
    box = {"x": NODES_RIGHT - CATALOGUES_W, "y": cat["y"], "w": CATALOGUES_W, "h": cat["h"]}
    extras.append(
        {
            "kind": "catalogues",
            "label": levels.OVERVIEW_CATALOGUES["label"],
            "to": f"connectors/{cats[0]['slug']}",
            "lines": wrap(levels.OVERVIEW_CATALOGUES["says"], box["w"] - 20, 5.2)
            + [f"{len(cats)} catalogues ›"],
            "says": ", ".join(c["what"].lower() for c in cats),
            "box": box,
        }
    )
    return extras


# ----------------------------------------------------------------------------- connectors and catalogues


def connectors() -> dict:
    families = []
    for spec in levels.FAMILIES:
        module, _, name = spec["registry"].partition(":")
        registry = entity(spec["registry"])
        names = list(registry.names())
        if set(spec["places"]) != set(names):
            stop(f"{name}: the places name {sorted(spec['places'])}, the registry holds {names}")
        try:  # no public reader yet for the order, the aliases, the extra: a renamed attribute stops the build
            auto = list(registry._auto or ())
            aliases = dict(registry._catalogue._aliases)
            extras = {
                connector: registry._catalogue.get(connector).extra or "" for connector in names
            }
        except AttributeError as missing:
            stop(
                f"{name}: core/registry.py no longer holds what the explorer reads of a family ({missing})"
            )
        setting_default = _setting_default(registry.setting)  # "auto", or a connector's name
        implementations = []
        for connector in names:
            cls = registry.get(connector)
            # what the code declares of reaching outside: an AI provider's cloud, a renderer's internet
            outside = getattr(cls, "cloud", None)
            if outside is None and getattr(cls, "capabilities", None) is not None:
                outside = getattr(cls.capabilities, "requires_internet", None)
            places = spec["places"][connector]
            for placed in places:
                if placed["place"] not in levels.PLACES:
                    stop(
                        f"{name}: {connector} is placed in {placed['place']}, which is not a place"
                    )
                if outside is True and placed["place"] != "outside":
                    stop(
                        f"{name}: {connector} reaches outside the network and is placed {placed['place']}"
                    )
                if outside is False and placed["place"] == "outside" and not placed["when"]:
                    stop(
                        f"{name}: {connector} does not reach outside the network, and is placed outside without a when"
                    )
            implementations.append(
                {
                    "name": connector,
                    "display": registry.display_name(connector),
                    "target": registry.target(connector),
                    "extra": extras[connector],
                    "aliases": sorted(
                        alias for alias, target in aliases.items() if target == connector
                    ),
                    "places": places,
                    "auto": auto.index(connector) + 1 if connector in auto else 0,
                    "default": connector == setting_default,
                }
            )
        base_module, _, base = spec["base"].partition(":")
        families.append(
            {
                "id": name,
                "slug": spec["slug"],
                "family": registry.family,
                "registry": name,
                "file": relative(MODULES[module]),
                "base": base,
                "baseFile": relative(MODULES[base_module]),
                "setting": registry.setting,
                "settingOnly": bool(registry.setting_only),
                "settingDefault": setting_default or "",
                "auto": auto,
                "brick": spec["brick"],
                "note": spec["note"],
                "implementations": implementations,
            }
        )
    return {
        "lead": levels.CONNECTOR_LEAD,
        "mechanisms": levels.MECHANISMS,
        "places": levels.PLACES,
        "privateNetwork": levels.PRIVATE_NETWORK,
        "contract": levels.CONTRACT,
        "families": families,
    }


def _setting_default(variable: str) -> str | None:
    import utils.settings as settings

    for section in vars(settings).values():
        fields = getattr(section, "model_fields", None)
        if not isinstance(fields, dict):
            continue
        for field in fields.values():
            if getattr(field, "validation_alias", None) == variable:
                return field.default
    return None


def catalogues() -> list[dict]:
    out = []
    for spec in levels.CATALOGUES:
        module, _, name = spec["holder"].partition(":")
        holder = entity(spec["holder"])
        kinds = list(holder.values()) if isinstance(holder, dict) else list(holder)
        legacy = getattr(holder, "is_legacy", lambda _id: False)
        out.append(
            {
                "id": name,
                "slug": spec["slug"],
                "what": spec["what"],
                "holder": name,
                "file": relative(MODULES[module]),
                "mechanism": "a Catalogue (core/registry.py)"
                if not isinstance(holder, dict)
                else "a plain dict",
                "brick": spec["brick"],
                "note": spec["note"],
                "kinds": [
                    {
                        "id": kind.id,
                        "label": str(getattr(kind, spec["label"], "")),
                        "legacy": bool(legacy(kind.id)),
                    }
                    for kind in kinds
                ],
            }
        )
    return out


# ----------------------------------------------------------------------------- the matrix's libraries


def modules_of(comps: str) -> list[str]:
    """The modules a row of the matrix names: a file, a folder (every module under it), or a bare file name in the
    folder of the file before it."""
    found: list[str] = []
    last_dir = ""
    for piece in (part.strip() for part in comps.split(",")):
        candidates = [piece] if "/" in piece or not last_dir else [f"{last_dir}/{piece}", piece]
        if "/" in piece:
            last_dir = piece.rstrip("/") if piece.endswith("/") else piece.rsplit("/", 1)[0]
        for candidate in candidates:
            if candidate.endswith("/"):
                prefix = candidate.rstrip("/").replace("/", ".")
                modules = [m for m in MODULES if m == prefix or m.startswith(prefix + ".")]
            else:
                name = candidate.removesuffix(".py").replace("/", ".")
                modules = [name] if name in MODULES else []
            if modules:
                found += modules
                break
        else:
            stop(f"the matrix names {piece}, which is no module or folder of the code")
    return found


# ----------------------------------------------------------------------------- every name the words give exists


def _definitions() -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Every name defined in the code (functions, classes, assignments, attributes), each class's members, and each
    module's top-level names."""
    defined: set[str] = set()
    members: dict[str, set[str]] = defaultdict(set)
    top: dict[str, set[str]] = defaultdict(set)
    for module, tree in TREES.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    for leaf in ast.walk(target):
                        if isinstance(leaf, ast.Name):
                            defined.add(leaf.id)
                        elif isinstance(leaf, ast.Attribute):
                            defined.add(leaf.attr)
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                        members[node.name].add(item.name)
                    elif isinstance(item, ast.Assign | ast.AnnAssign):
                        for target in (
                            item.targets if isinstance(item, ast.Assign) else [item.target]
                        ):
                            if isinstance(target, ast.Name):
                                members[node.name].add(target.id)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                top[module].add(node.name)
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    for leaf in ast.walk(target):
                        if isinstance(leaf, ast.Name):
                            top[module].add(leaf.id)
            elif isinstance(node, ast.ImportFrom):
                top[module].update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                top[module].update(
                    (alias.asname or alias.name).split(".")[0] for alias in node.names
                )
    return defined, members, top


def _file_names() -> set[str]:
    """The name of every file the repository tracks (git), or of every file under it when git cannot say."""
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=False
    )
    if listed.returncode == 0 and listed.stdout:
        return {line.rsplit("/", 1)[-1] for line in listed.stdout.splitlines()}
    apart = {
        ".venv",
        ".git",
        "scratch",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "data",
    }
    names: set[str] = set()
    for _folder, folders, files in ROOT.walk():
        folders[:] = [folder for folder in folders if folder not in apart]
        names.update(files)
    return names


def _resolve_dotted(name: str) -> bool:
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        module = ".".join(parts[:cut])
        if module in MODULES or (ROOT / Path(*parts[:cut])).with_suffix(".py").is_file():
            rest = parts[cut:]
            if not rest:
                return True
            if module not in MODULES:
                return False
            if rest[0] not in TOP[module]:
                return False
            return len(rest) == 1 or (len(rest) == 2 and rest[1] in MEMBERS.get(rest[0], set()))
    return False


def _outside(path: str) -> bool:
    """A path that is not the repository's: absolute (``/users/``, ``C:/Users``), as part of a placeholder such as
    ``<DATA_ROOT>/users/<username>/`` is. It is never looked up on the machine that builds: ``C:/Users`` exists on
    Windows and ``/users`` does not on Linux, so the build would depend on where it runs."""
    return path.startswith(("/", "\\")) or bool(re.match(r"[A-Za-z]:[\\/]", path))


def _path_exists(piece: str, last_dir: str | None) -> bool:
    """Whether a path the words name is in the repository (only there)."""
    candidates = [piece]
    if "/" not in piece and last_dir:
        candidates.insert(0, f"{last_dir}/{piece}")
    for candidate in candidates:
        if _outside(candidate):
            continue
        if "*" in candidate:
            if any(ROOT.glob(candidate)):
                return True
        elif (ROOT / candidate.rstrip("/")).exists():
            return True
    return "/" not in piece and piece in FILE_NAMES


def _class_like(piece: str) -> bool:
    """Written as a class is: letters and digits, a capital first, another capital and a small letter after (not the
    plural of an abbreviation: SDKs, PDFs)."""
    if re.fullmatch(r"[A-Z0-9]+s", piece):
        return False
    return (
        piece.isalnum()
        and piece[0].isupper()
        and sum(c.isupper() for c in piece) >= 2
        and any(c.islower() for c in piece)
    )


def _known_class(piece: str) -> bool:
    return piece in DEFINED or hasattr(builtins, piece)


def references(text: str, *, whole: bool) -> list[str]:
    """What in ``text`` names the code and is not in it. ``whole``: the text is all code (a step's function, a list
    of files); otherwise its ``<code>`` spans are read as code, and the rest as plain words."""
    spans = [text] if whole else re.findall(r"<code>(.*?)</code>", text)
    missing = []
    for span in spans:
        span = html.unescape(span)
        last_dir = None
        for match in re.finditer(r"[A-Za-z_./*-][\w./*-]*", span):
            piece = match.group(0).rstrip(".,:")
            after = span[match.end() : match.end() + 1]
            if not piece or piece.startswith("-") or after == "=" or _outside(piece):
                continue  # an option, a keyword argument, or a path outside the repository (a placeholder's)
            if "/" in piece or piece.endswith(PATHLIKE):
                if piece.startswith(("http", ".github")) and not (ROOT / piece).exists():
                    continue
                if not _path_exists(piece, last_dir):
                    missing.append(piece)
                if "/" in piece:
                    last_dir = (
                        piece.rsplit("/", 1)[0] if not piece.endswith("/") else piece.rstrip("/")
                    )
                continue
            head = piece.split(".")[0]
            if "." in piece and head in FIRST:
                if not _resolve_dotted(piece):
                    missing.append(piece)
            elif re.fullmatch(r"[A-Z][A-Za-z0-9]+\.[a-z_]\w*", piece):
                cls, member = piece.split(".")
                if member not in MEMBERS.get(cls, set()):
                    missing.append(piece)
            elif _class_like(piece):
                if not _known_class(piece):
                    missing.append(piece)
            elif re.fullmatch(r"CMP_[A-Z0-9_]+", piece):
                if f'"{piece}"' not in SETTINGS_TEXT:
                    missing.append(piece)
            elif re.fullmatch(r"[A-Z][A-Z0-9]*(_[A-Z0-9]+)+", piece) or (
                after == "(" and re.fullmatch(r"[a-z_]\w*", piece)
            ):
                if piece not in DEFINED:
                    missing.append(piece)
            elif (
                not whole
                and re.fullmatch(r"[a-z]+(_[a-z0-9]+)+", piece)
                and span.strip().rstrip("()") == piece
            ):
                if piece not in DEFINED:
                    missing.append(piece)
    return missing if whole else missing + plain_references(text)


def plain_references(text: str) -> list[str]:
    """What the plain words of ``text`` name of the code and the code does not hold: a module or one of its names
    (``core.session.DataSession``), a file or a folder of the repository, a class, a ``CMP_`` setting."""
    plain = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<code>.*?</code>", " ", text)))
    folders = FIRST | {"tests", "dev_scripts", "documentation", "docs", "assets"}
    missing = []
    for match in re.finditer(r"[A-Za-z_][\w./-]*", plain):
        piece = match.group(0).rstrip(".,:;-/")
        head = re.split(r"[./]", piece)[0]
        if _outside(piece):
            continue
        if "/" in piece or piece.endswith(PATHLIKE):
            if (head in folders or piece.endswith(PATHLIKE)) and not _path_exists(piece, None):
                missing.append(piece)
        elif "." in piece and head in FIRST:
            if not _resolve_dotted(piece):
                missing.append(piece)
        elif re.fullmatch(r"CMP_[A-Z0-9_]+", piece):
            if f'"{piece}"' not in SETTINGS_TEXT:
                missing.append(piece)
        elif _class_like(piece) and piece not in NOT_OF_THE_CODE and not _known_class(piece):
            missing.append(piece)
    return missing


def check_references(data: dict) -> int:
    texts: list[tuple[str, str, bool]] = []

    def walk(where: str, value, whole: bool = False) -> None:
        if isinstance(value, str):
            texts.append((where, value, whole))
        elif isinstance(value, dict):
            for key, item in value.items():
                if key != "libs":  # the libraries' names come from uv.lock, not from the words
                    walk(f"{where}.{key}", item, whole or key in ("fn", "comps", "files"))
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                walk(f"{where}[{index}]", item, whole)

    for name in (
        "features",
        "scenarios",
        "recipes",
        "matrix",
        "rules",
        "moreRules",
        "connectors",
        "catalogues",
    ):
        walk(name, data[name])
    for level, view_data in data["views"].items():
        for node_id, node in view_data["nodes"].items():
            for key in ("role", "rule", "methods", "note", "facts", "sub"):
                walk(f"{level}.{node_id}.{key}", node.get(key, ""))
    walk("tests.files", content.TESTS["files"], True)
    walk("storage.files", levels.OVERVIEW_STORAGE["files"], True)
    missing = [
        (where, piece) for where, text, whole in texts for piece in references(text, whole=whole)
    ]
    if missing:
        stop(
            "names the code does not hold:\n  "
            + "\n  ".join(f"{piece}  (in {where})" for where, piece in missing)
        )
    return len(texts)


# ----------------------------------------------------------------------------- the data of the page


def links_of(
    owner: Callable[[str], str], families: list[dict], cats: list[dict]
) -> dict[str, list[dict]]:
    """For each box of a level, the connector families and catalogues whose registry it holds."""
    links: dict[str, list[dict]] = defaultdict(list)
    for family in families:
        module = family["file"].removesuffix("/__init__.py").removesuffix(".py").replace("/", ".")
        names = ", ".join(i["name"] for i in family["implementations"])
        kind = family["family"][:1].upper() + family["family"][1:]
        links[owner(module)].append(
            {"to": f"connectors/{family['slug']}", "label": f"{kind}s: {names}"}
        )
    for cat in cats:
        module = cat["file"].removesuffix(".py").replace("/", ".")
        links[owner(module)].append(
            {"to": f"connectors/{cat['slug']}", "label": f"{cat['what']} ({len(cat['kinds'])})"}
        )
    return links


def build(*, check_arrows: bool = True) -> dict:
    """The page's data. ``check_arrows``: stop on an import that goes up (see ``view``)."""
    check_documents()
    check_registries()
    check_libraries()
    bricks = brick_owners()
    owners = {
        "overview": lambda m: ENTRY_OF.get(m) or CARD_OF[package_of(m)],
        "packages": lambda m: "p_" + (m if m in tl.ENTRY_POINTS else package_of(m)),
        "bricks": bricks.__getitem__,
    }
    tests = tests_words()
    views = {
        "overview": view(
            "overview",
            overview_bands(),
            {**overview_words(), "storage": storage_words(), "tests": tests},
            owners["overview"],
            same_layer_check=False,
            most_lines=3,
            check_arrows=check_arrows,
            reserved={
                levels.OVERVIEW_SOCKETS["band"]: SOCKETS_W,
                levels.OVERVIEW_CATALOGUES["band"]: CATALOGUES_W,
            },
        ),
        "packages": view(
            "packages",
            content.BANDS,
            {**package_words(), "tests": tests},
            owners["packages"],
            same_layer_check=True,
            most_lines=2,
            check_arrows=check_arrows,
        ),
        "bricks": view(
            "bricks",
            content.BANDS,
            {**brick_words(), "tests": tests},
            owners["bricks"],
            same_layer_check=True,
            most_lines=2,
            check_arrows=check_arrows,
        ),
    }
    connector_data = connectors()
    cats = catalogues()
    views["overview"]["extras"] = overview_extras(
        views["overview"], connector_data["families"], cats
    )
    brick_lines = {brick_id: brick["lines"] for brick_id, brick in views["bricks"]["nodes"].items()}
    for level, data in views.items():
        links = links_of(owners[level], connector_data["families"], cats)
        for node_id, node in data["nodes"].items():
            node["links"] = links.get(node_id, [])
            if level == "bricks" or node_id in PSEUDO:
                node["bricks"] = [node_id] if node_id in views["bricks"]["nodes"] else []
            else:
                node["bricks"] = sorted(
                    {bricks[m] for m in node["members"]}, key=lambda b: -brick_lines[b]
                )
    # each brick's place at the coarser levels
    for brick in views["bricks"]["nodes"].values():
        brick["parents"] = {
            level: sorted({owners[level](m) for m in brick["members"]})
            if brick["members"]
            else ["tests"]
            for level in ("overview", "packages")
        }

    modules = {}
    imported_by: dict[str, set[str]] = defaultdict(set)
    for module, imports in FIRST_PARTY.items():
        for target, _lazy in imports:
            if target != module:
                imported_by[target].add(module)
    for module, path in MODULES.items():
        modules[module] = {
            "path": relative(path),
            "lines": SIZES[module],
            "doc": first_paragraph(module),
            "band": band_of(module),
            "package": owners["packages"](module),
            "overview": owners["overview"](module),
            "brick": bricks[module],
            "imports": sorted({t for t, _lazy in FIRST_PARTY[module] if t != module}),
            "importedBy": sorted(imported_by[module]),
            "libs": libraries_of([module], most=None),
        }
    band_rank = {band["id"]: index for index, band in enumerate(content.BANDS)}
    module_order = sorted(
        modules, key=lambda m: (band_rank[modules[m]["band"]], modules[m]["path"])
    )

    known = views["bricks"]["nodes"]
    for feature in content.FEATURES:
        for node_id in [*feature["nodes"], *([feature["focus"]] if feature["focus"] else [])]:
            if node_id not in known:
                stop(f"the feature {feature['id']} names an unknown brick, {node_id}")
        if feature["focus"] and feature["focus"] not in feature["nodes"]:
            stop(f"the feature {feature['id']} puts in focus a brick it does not highlight")
    for recipe in content.RECIPES:
        for node_id in recipe["nodes"]:
            if node_id not in known:
                stop(f"the recipe {recipe['id']} names an unknown brick, {node_id}")
    for row in content.MATRIX:
        if row["node"] not in known:
            stop(f"the matrix row {row['feature']} names an unknown brick, {row['node']}")
    for spec in (*connector_data["families"], *cats):
        if spec["brick"] not in known:
            stop(f"{spec['id']} names an unknown brick")

    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    data = {
        "meta": {
            "version": PROJECT["version"],
            "python": PROJECT.get("requires-python", ""),
            "commit": commit,
            "date": datetime.date.today().strftime("%d %B %Y").lstrip("0"),
            "repository": levels.REPOSITORY,
            "layers": len(set(tl.LAYERS.values())),
            "packages": len(tl.LAYERS),
            "bricks": len(content.NODES),
            "modules": len(MODULES),
            "lines": sum(SIZES.values()),
            "testFiles": tests["testFiles"],
            "testFunctions": tests["testFunctions"],
            "width": WIDTH,
            "nodesX": NODES_X,
            "nodesRight": NODES_RIGHT,
            "spineX": SPINE_X,
            "bandRight": BAND_RIGHT,
            "labelW": LABEL_W,
            "titleY": TITLE_Y,
            "lineY": LINE_Y,
            "lineH": LINE_H,
        },
        "groups": {**content.GROUPS, "storage": "Storage"},
        "levels": levels.LEVELS,
        "views": views,
        "modules": modules,
        "moduleOrder": module_order,
        "connectors": connector_data,
        "catalogues": cats,
        "features": content.FEATURES,
        "actors": content.ACTORS,
        "scenarios": content.SCENARIOS,
        "rules": content.RULES,
        "moreRules": content.MORE_RULES,
        "outside": content.OUTSIDE,
        "packages": {
            package: layer
            for package, layer in sorted(tl.LAYERS.items(), key=lambda item: -item[1])
        },
        "entryPoints": [f"{name}.py" for name in tl.ENTRY_POINTS],
        "headless": list(tl.HEADLESS),
        "olderUtils": dict(tl.UTILS_GROUPS),
        "environmentReaders": sorted(tl.KNOWN_ENVIRONMENT),
        "recipes": content.RECIPES,
        "matrix": [
            {**row, "libs": libraries_of(modules_of(row["comps"]), most=None)}
            for row in content.MATRIX
        ],
    }
    checked = check_references(data)
    print(
        f"  {checked} texts read: what they name of the code (in code type: files, modules, functions, classes, "
        "settings; in plain words: files, modules, classes, settings) is in it"
    )
    return data


TOKEN = '"__EXPLORER_DATA__"'  # the data's place in the template
OUTPUT = Path("docs") / "architecture" / "index.html"  # under the repository's root; ignored by git


def write(data: dict, output: Path) -> None:
    """The page: the template with the data in its place (no ``</`` left in it to end the script early)."""
    template = (HERE / "explorer_template.html").read_text(encoding="utf-8")
    if template.count(TOKEN) != 1:
        stop("explorer_template.html must hold the data's place once")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace(TOKEN, payload), encoding="utf-8", newline="\n")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1]
    output = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root.resolve() / OUTPUT
    try:
        load(root)
        # the direction of the imports is the layering test's (R1), which CI has run before any deploy
        data = build(check_arrows=False)
        write(data, output)
    except NotBuilt as refused:
        sys.exit(f"The explorer was not built: {refused}")
    sizes = {
        level: f"{len(v['nodes'])} boxes, {len(v['edges'])} arrows"
        for level, v in data["views"].items()
    }
    meta = data["meta"]
    print(
        f"{output.name}: {sizes}; {meta['modules']} modules, {meta['lines']:,} lines, "
        f"{len(data['connectors']['families'])} connector families, {len(data['catalogues'])} catalogues, commit "
        f"{meta['commit']}, {output.stat().st_size / 1024:.0f} KB"
    )


if __name__ == "__main__":
    main()
