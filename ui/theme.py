"""
Design tokens, brand assets, the scoped stylesheet and the Plotly template.

The Streamlit colours live in ``.streamlit/config.toml`` (the only place Streamlit reads them),
with separate light and dark themes; ``TOKENS`` mirrors them for charts and components, and a
unit test keeps both in sync. The brand palette is sampled from the application logo (navy
outlines, teal, cyan and mint). Categorical chart colours follow the Okabe-Ito palette
(colour-blind safe).
"""

from __future__ import annotations

import contextlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLES_PATH = Path(__file__).with_name("styles.css")
BRAND_DIR = ROOT / "assets" / "brand"
FALLBACK_LOGO = ROOT / "assets" / "karim-app-logo.png"

BRAND: dict[str, str] = {
    "navy": "#0B2B36",  # logo outline family; hero gradient start
    "teal": "#0E7C86",  # primary (4.9:1 with white text)
    "cyan": "#24E4E4",  # logo highlight
    "mint": "#54CC9C",  # logo accent
}

OKABE_ITO: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

TOKENS: dict[str, str] = {
    "primary": BRAND["teal"],  # light theme primary
    "primary_dark": "#14919B",  # dark theme primary (4.8:1 on the dark background)
    "ink": "#0F2A36",  # light theme text
    "surface": "#F3F8F9",  # light theme sidebar
    "success": "#1B7F5A",
    "warning": "#B7791F",
    "danger": "#B42318",
    "muted": "#5B7480",
}

STATUS_COLORS = {
    "ok": "green",
    "caution": "orange",
    "violated": "red",
    "info": "blue",
    "neutral": "grey",
}

PLOTLY_TEMPLATE_NAME = "cmp"
_registered = False


def config_theme(config_path: Path | None = None) -> dict:
    """The ``[theme]`` tables of ``.streamlit/config.toml`` (empty when the file is missing)."""
    path = config_path or ROOT / ".streamlit" / "config.toml"
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle).get("theme", {})


def brand_asset(name: str) -> str:
    """Path of a generated brand asset, falling back to the raw logo when assets were not generated."""
    path = BRAND_DIR / name
    return str(path if path.exists() else FALLBACK_LOGO)


def plotly_template():
    """Register (once) and return the application's Plotly template."""
    global _registered
    import plotly.graph_objects as go
    import plotly.io as pio

    if not _registered:
        base = pio.templates["plotly_white"]
        template = go.layout.Template(base)
        layout = template.layout
        layout.colorway = list(OKABE_ITO)  # type: ignore[union-attr]
        layout.font = {"family": "Source Sans 3, Source Sans Pro, Segoe UI, sans-serif", "size": 13}  # type: ignore[union-attr]
        layout.title = {"x": 0.0, "xanchor": "left"}  # type: ignore[union-attr]
        layout.margin = {"l": 50, "r": 30, "t": 60, "b": 50}  # type: ignore[union-attr]
        layout.legend = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0}  # type: ignore[union-attr]
        pio.templates[PLOTLY_TEMPLATE_NAME] = template
        pio.templates.default = PLOTLY_TEMPLATE_NAME
        _registered = True
    return pio.templates[PLOTLY_TEMPLATE_NAME]


def inject_styles() -> None:
    """Load the scoped stylesheet (hero banner, login card, spacing). Call once per script run."""
    import streamlit as st

    if STYLES_PATH.exists():
        st.html(STYLES_PATH)


def show_logo() -> None:
    """Brand wordmark in the top bar (light/dark variant), icon when space is short."""
    import streamlit as st

    dark = False
    with contextlib.suppress(Exception):
        dark = st.context.theme.type == "dark"
    st.logo(
        brand_asset("wordmark_dark.png" if dark else "wordmark_light.png"),
        size="large",
        icon_image=brand_asset("icon.png"),
    )


def apply_theme() -> None:
    """Plotly template (idempotent) and the scoped stylesheet; Streamlit colours come from config.toml."""
    plotly_template()
    inject_styles()


__all__ = [
    "BRAND",
    "OKABE_ITO",
    "PLOTLY_TEMPLATE_NAME",
    "STATUS_COLORS",
    "STYLES_PATH",
    "TOKENS",
    "apply_theme",
    "brand_asset",
    "config_theme",
    "inject_styles",
    "plotly_template",
    "show_logo",
]
