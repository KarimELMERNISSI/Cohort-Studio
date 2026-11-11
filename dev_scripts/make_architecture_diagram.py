"""
Build the architecture scheme of the README, in a light and a dark version (assets/readme/architecture.svg and
architecture-dark.svg), from the description below.

The scheme shows what the layering test holds true (tests/test_layering.py): the packages on their layers, with
imports going down only; the two entry points (the browser enters through the interface, the command line below
it); the four connector families that plug into the services, with what stays on the server and what reaches
outside it; the catalogues declared in the engine; and what a workspace holds.

Usage:
    python dev_scripts/make_architecture_diagram.py            write the two files
    python dev_scripts/make_architecture_diagram.py --check    exit 1 when a file is not what this script builds

The output is the same on every run and every machine: no date, no random id. tests/test_docs_site.py checks that
the files are current and that every package of the layer table is drawn.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "readme"
FILES = {"light": "architecture.svg", "dark": "architecture-dark.svg"}

WIDTH = 960
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"
DOT = "·"

# the columns: an arrow in the left margin, the layers, and beside the lower ones what the services store
ARROW_X = 32
LEFT, RIGHT = 56, 940
LAYERS_RIGHT = 740
STORE_LEFT = 760
PAD = 12  # between a band's edge and its cards, and between two cards
GAP = 18  # between two bands: room for the arrow from the services to what they store


@dataclass(frozen=True)
class Theme:
    ink: str  # names and titles
    text: str  # descriptions
    muted: str  # notes
    accent: str  # labels and arrows
    band: str
    band_line: str
    engine: str  # the engine's band stands out: it is what runs without the rest
    engine_line: str
    card: str
    card_line: str
    outside: str  # what reaches outside the server
    canvas: (
        str  # behind everything: GitHub's own page colour, so that either file reads on any page
    )


THEMES = {
    "light": Theme(
        ink="#0B2B36",
        text="#2F4A56",
        muted="#5B7480",
        accent="#0A646C",
        band="#F3F8F9",
        band_line="#DCE8EB",
        engine="#E6F3F4",
        engine_line="#9CCDD2",
        card="#FFFFFF",
        card_line="#C4D6DB",
        outside="#8A5300",
        canvas="#FFFFFF",
    ),
    "dark": Theme(
        ink="#E8F2F4",
        text="#B9CBD1",
        muted="#8CA4AD",
        accent="#56C5CE",
        band="#131F27",
        band_line="#263943",
        engine="#11303A",
        engine_line="#2A6470",
        card="#0C151B",
        card_line="#30464F",
        outside="#E5AC5C",
        canvas="#0D1117",
    ),
}


@dataclass(frozen=True)
class Card:
    """A package (or two that share a layer) and what it holds, in lines short enough for its width."""

    name: str
    lines: tuple[str, ...]
    share: float = 1.0  # of the band's width, among the band's cards


@dataclass(frozen=True)
class Socket:
    """A connector family: the implementations that run on the server (the default first), and those that reach
    outside it."""

    title: str
    inside: tuple[str, ...]
    outside: tuple[str, ...] = ()


INTERFACE = (
    Card("app_pages", ("the pages: inputs, one service call,", "a result card")),
    Card("ui", ("shell, navigation and components;", "AppState, the one writer of the data")),
)
SERVICES = (
    Card(
        "manage",
        (
            "traces and replay, study definition, Word report and dossier,",
            "accounts, AI engine, the stores of a workspace",
        ),
        share=1.4,
    ),
    Card("graph", ("knowledge-graph model, builders", "and renderers")),
)
CONNECTORS = (
    Socket("Identity provider", ("local accounts", f"Better Auth sidecar {DOT} none")),
    Socket("AI provider", ("Ollama",), ("Gemini, OpenAI,", "Mistral, Claude")),
    Socket("Vector database", (f"LanceDB {DOT} Qdrant",), ("a remote Qdrant server",)),
    Socket("Graph engine", (f"Plotly {DOT} Graphviz",), ("yFiles (opt-in)",)),
)
DATA_WORK = (
    Card("ingest", ("CSV, Excel and Parquet,", "with a parse report")),
    Card("enrich", ("formulas and derived", "variables, imputation")),
    Card(f"explore {DOT} monitor", ("quality audit and score,", "correlations, changes")),
)
ENGINE = (
    Card(
        "core",
        (
            "data session, analysis population,",
            "criteria, lineage and fingerprints,",
            "trace steps, study definition",
        ),
        share=1.15,
    ),
    Card(
        "biostats",
        (
            "tests, effect sizes, models,",
            "multiple imputation, pooling;",
            "imports no other package",
        ),
    ),
    Card("prompts", ("texts sent to", "a language", "model"), share=0.5),
)
CATALOGUES = f"trace steps {DOT} criterion kinds {DOT} analysis kinds {DOT} checks"
FOUNDATION = Card(
    "utils", (f"settings checked at start-up {DOT} logging {DOT} optional groups {DOT} paths",)
)
STORAGE = (
    ("Parquet versions", "with lineage files"),
    ("Hash-chained traces", "(JSON)"),
    ("Study definition", "and its revisions"),
    ("Documents and their", "vector index"),
    ("Taxonomies and", "knowledge graphs"),
    ("Preferences",),
)
ACCOUNTS = ("DuckDB", "accounts, statistics", "cache")

# every package the scheme draws, for the test that compares them with the layer table
PACKAGES = frozenset(
    name.strip()
    for card in (*INTERFACE, *SERVICES, *DATA_WORK, *ENGINE, FOUNDATION)
    for name in card.name.split(DOT)
)


class Drawing:
    """The elements of one scheme, in the order they are drawn."""

    def __init__(self, theme: Theme) -> None:
        self.t = theme
        self.parts: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        line: str,
        *,
        rx: int = 6,
        dash: str = "",
    ) -> None:
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" fill="{fill}" '
            f'stroke="{line}" stroke-width="1"{dashed}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        words: str,
        *,
        size: float = 14,
        fill: str | None = None,
        bold: bool = False,
        mono: bool = False,
        anchor: str = "",
        spacing: float = 0,
        rotate: bool = False,
    ) -> None:
        extra = ' font-weight="700"' if bold else ""
        extra += f' font-family="{MONO}"' if mono else ""
        extra += f' text-anchor="{anchor}"' if anchor else ""
        extra += f' letter-spacing="{spacing:g}"' if spacing else ""
        extra += f' transform="rotate(-90 {x:g} {y:g})"' if rotate else ""
        self.parts.append(
            f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" fill="{fill or self.t.text}"{extra}>'
            f"{escape(words)}</text>"
        )

    def arrow(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.parts.append(
            f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{self.t.accent}" '
            'stroke-width="1.5" marker-end="url(#head)"/>'
        )

    def leaves(self, x: float, y: float) -> None:
        """A small arrow pointing up and to the right, drawn (no font is asked for it): reaches outside."""
        self.parts.append(
            f'<path d="M{x:g} {y:g} l7 -7 m-5 0 h5 v5" fill="none" stroke="{self.t.outside}" '
            'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    def label(self, x: float, y: float, words: str, note: str = "", *, size: float = 12) -> None:
        """A band's name in capitals and, in the same text, what is true of it: the note follows the name at one
        distance, whatever the font that draws them."""
        after = (
            f'<tspan dx="12" font-size="{size:g}" font-weight="400" letter-spacing="0" '
            f'fill="{self.t.muted}">{escape(note)}</tspan>'
            if note
            else ""
        )
        self.parts.append(
            f'<text x="{x:g}" y="{y:g}" font-size="11" font-weight="700" letter-spacing="1.1" '
            f'fill="{self.t.accent}">{escape(words)}{after}</text>'
        )

    def card(self, x: float, y: float, w: float, h: float, card: Card) -> None:
        self.rect(x, y, w, h, self.t.card, self.t.card_line)
        self.text(x + 14, y + 24, card.name, size=15, fill=self.t.ink, bold=True, mono=True)
        for place, words in enumerate(card.lines):
            self.text(x + 14, y + 44 + 18 * place, words)

    def cards(self, x: float, y: float, width: float, cards: tuple[Card, ...]) -> float:
        """A row of cards across ``width``, each as wide as its share; the height of the row."""
        height = 36 + 18 * max(len(card.lines) for card in cards)
        room = width - PAD * (len(cards) - 1)
        shares = sum(card.share for card in cards)
        for card in cards:
            w = round(room * card.share / shares)
            self.card(x, y, w, height, card)
            x += w + PAD
        return height


def scheme(theme: Theme) -> str:
    """The scheme as an SVG document."""
    d = Drawing(theme)
    t = theme

    # ---- the two ways in
    d.rect(LEFT, 14, 290, 48, t.card, t.card_line)
    d.text(LEFT + 14, 34, "Browser", fill=t.ink, bold=True)
    d.text(LEFT + 14, 52, "the Streamlit application (main.py)", size=12.5, fill=t.muted)
    scripts = RIGHT - 330
    d.rect(scripts, 14, RIGHT - scripts, 48, t.card, t.card_line)
    d.text(scripts + 14, 34, "Command line and scripts", fill=t.ink, bold=True)
    d.text(
        scripts + 14,
        52,
        f"manage_users.py {DOT} headless replay (run_trace)",
        size=12.5,
        fill=t.muted,
    )
    way_in = (STORE_LEFT + RIGHT) / 2

    # ---- the interface
    y = 86
    top = y
    band_at = len(
        d.parts
    )  # a band is drawn under its cards: its place is kept until its height is known
    d.parts.append("")
    d.label(LEFT + PAD, y + 20, "INTERFACE", "the only layer that needs Streamlit")
    height = d.cards(LEFT + PAD, y + 30, LAYERS_RIGHT - LEFT - 2 * PAD, INTERFACE)
    bottom = y + 30 + height + PAD
    d.parts[band_at] = _band(t, LEFT, y, LAYERS_RIGHT - LEFT, bottom - y)
    d.arrow(LEFT + 145, 62, LEFT + 145, y - 2)

    # ---- the services, as wide as the scheme: the command line enters here, the connectors plug in here, and
    # the storage hangs under their right end
    y = bottom + GAP
    services_top = y
    band_at = len(d.parts)
    d.parts.append("")
    d.label(LEFT + PAD, y + 20, "SERVICES", "run without Streamlit, like every layer below")
    height = d.cards(LEFT + PAD, y + 30, RIGHT - LEFT - 2 * PAD, SERVICES)
    y2 = y + 30 + height + 26
    d.label(LEFT + PAD, y2, "CONNECTORS", "one implementation at a time, loaded when it is used")
    socket_w = (RIGHT - LEFT - 2 * PAD - PAD * (len(CONNECTORS) - 1)) / len(CONNECTORS)
    socket_h = 88
    for place, socket in enumerate(CONNECTORS):
        x = LEFT + PAD + place * (socket_w + PAD)
        d.rect(x, y2 + 10, socket_w, socket_h, t.card, t.accent, dash="4 3")
        d.text(x + 14, y2 + 32, socket.title, size=13, fill=t.ink, bold=True)
        row = y2 + 51
        for words in socket.inside:
            d.text(x + 14, row, words, size=13)
            row += 17
        for words in socket.outside:
            d.leaves(x + 14, row - 1)
            d.text(x + 28, row, words, size=13, fill=t.outside)
            row += 17
    legend = y2 + 10 + socket_h + 20
    d.leaves(LEFT + PAD + 2, legend - 1)
    d.text(
        LEFT + PAD + 16,
        legend,
        "reaches outside your server: only after an explicit consent, or an opt-in install",
        size=12,
        fill=t.outside,
    )
    bottom = legend + PAD
    d.parts[band_at] = _band(t, LEFT, y, RIGHT - LEFT, bottom - y)
    # the command line enters below the interface
    d.arrow(way_in, 62, way_in, services_top - 2)
    for place, words in enumerate(("enters below", "the interface")):
        d.text(
            way_in - 10,
            (62 + services_top) / 2 + 2 + 16 * place,
            words,
            size=12,
            fill=t.muted,
            anchor="end",
        )
    services_bottom = bottom

    # ---- the data work
    y = bottom + GAP
    lower_top = y
    band_at = len(d.parts)
    d.parts.append("")
    d.label(LEFT + PAD, y + 20, "DATA WORK")
    height = d.cards(LEFT + PAD, y + 30, LAYERS_RIGHT - LEFT - 2 * PAD, DATA_WORK)
    bottom = y + 30 + height + PAD
    d.parts[band_at] = _band(t, LEFT, y, LAYERS_RIGHT - LEFT, bottom - y)

    # ---- the engine
    y = bottom + GAP
    band_at = len(d.parts)
    d.parts.append("")
    d.label(
        LEFT + PAD, y + 20, "DATA AND STATISTICS ENGINE", "plain Python, used and tested on its own"
    )
    height = d.cards(LEFT + PAD, y + 30, LAYERS_RIGHT - LEFT - 2 * PAD, ENGINE)
    strip = y + 30 + height + PAD
    d.rect(LEFT + PAD, strip, LAYERS_RIGHT - LEFT - 2 * PAD, 32, t.card, t.accent, dash="4 3")
    d.label(
        LEFT + PAD + 14,
        strip + 21,
        "CATALOGUES",
        f"kinds declared once, named in the data: {CATALOGUES}",
        size=12.5,
    )
    bottom = strip + 32 + PAD
    d.parts[band_at] = _band(t, LEFT, y, LAYERS_RIGHT - LEFT, bottom - y, engine=True)

    # ---- the foundation
    y = bottom + GAP
    d.parts.append(_band(t, LEFT, y, LAYERS_RIGHT - LEFT, 44))
    d.text(LEFT + PAD + 14, y + 28, FOUNDATION.name, size=15, fill=t.ink, bold=True, mono=True)
    d.text(LEFT + PAD + 76, y + 28, FOUNDATION.lines[0])
    bottom = y + 44

    # ---- what the services store, beside the lower layers
    d.parts.append(_band(t, STORE_LEFT, lower_top, RIGHT - STORE_LEFT, bottom - lower_top))
    d.arrow(way_in, services_bottom, way_in, lower_top - 2)
    x = STORE_LEFT + 14
    d.label(x, lower_top + 20, "STORAGE")
    d.text(x, lower_top + 41, "One workspace per user", size=13, fill=t.ink, bold=True)
    row = lower_top + 68
    for item in STORAGE:
        d.parts.append(
            f'<rect x="{x:g}" y="{row - 8:g}" width="5" height="5" rx="1" fill="{t.accent}"/>'
        )
        for words in item:
            d.text(x + 13, row, words, size=13)
            row += 17
        row += 9
    # the accounts' database stands at the foot of the panel, under a rule
    row = bottom - 16 - 17 * (len(ACCOUNTS) - 1)
    d.parts.append(
        f'<line x1="{x:g}" y1="{row - 24:g}" x2="{RIGHT - 14:g}" y2="{row - 24:g}" stroke="{t.card_line}" '
        'stroke-width="1"/>'
    )
    d.text(x, row, ACCOUNTS[0], size=13, fill=t.ink, bold=True)
    for words in ACCOUNTS[1:]:
        row += 17
        d.text(x, row, words, size=13)

    # ---- imports go down only
    d.arrow(ARROW_X, top + 4, ARROW_X, bottom - 4)
    d.text(
        ARROW_X - 9,
        (top + bottom) / 2,
        "IMPORTS GO DOWN ONLY",
        size=10.5,
        fill=t.accent,
        bold=True,
        anchor="middle",
        spacing=1.3,
        rotate=True,
    )

    height = bottom + 16
    title = "The architecture of CohortStudio"
    described = (
        "Layers from the interface (app_pages, ui) through the services (manage, graph), the data work (ingest, "
        "enrich, explore, monitor) and the data and statistics engine (core, biostats, prompts) down to utils; "
        "imports go down only. The browser enters through the interface and the command line below it. Four "
        "connector families plug into the services, and the services store each user's workspace."
    )
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" '
        f'height="{height:g}" role="img" aria-labelledby="title desc" font-family="{SANS}">\n'
        f'<title id="title">{escape(title)}</title>\n<desc id="desc">{escape(described)}</desc>\n'
        f'<defs><marker id="head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{t.accent}"/></marker></defs>\n'
        f'<rect width="{WIDTH}" height="{height:g}" rx="10" fill="{t.canvas}"/>\n'
    )
    return head + "\n".join(part for part in d.parts if part) + "\n</svg>\n"


def _band(t: Theme, x: float, y: float, w: float, h: float, *, engine: bool = False) -> str:
    fill, line = (t.engine, t.engine_line) if engine else (t.band, t.band_line)
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="8" fill="{fill}" stroke="{line}" '
        'stroke-width="1"/>'
    )


def build() -> dict[str, str]:
    """The two files' contents, by file name."""
    return {FILES[name]: scheme(theme) for name, theme in THEMES.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="compare with the files instead of writing them"
    )
    args = parser.parse_args(argv)
    stale = []
    for name, content in build().items():
        path = OUT / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    if stale:
        print("not current, run dev_scripts/make_architecture_diagram.py: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
