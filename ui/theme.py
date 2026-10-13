"""
Design tokens and the Plotly template.

The Streamlit colours live in ``.streamlit/config.toml`` (the only place Streamlit reads them);
``TOKENS`` mirrors them for charts and components, and a unit test keeps both in sync.
Categorical colours follow the Okabe-Ito palette (colour-blind safe).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

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
    "primary": "#1F5F8B",  # clinical blue, WCAG AA on white
    "primary_dark": "#2F81F7",  # primary on dark backgrounds
    "success": "#1B7F3B",
    "warning": "#B7791F",
    "danger": "#B42318",
    "muted": "#667085",
    "surface": "#F4F6F8",
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
    path = config_path or Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle).get("theme", {})


def plotly_template():
    """Register (once) and return the application's Plotly template."""
    global _registered
    import plotly.graph_objects as go
    import plotly.io as pio

    if not _registered:
        base = pio.templates["plotly_white"]
        template = go.layout.Template(base)
        template.layout.colorway = list(OKABE_ITO)
        template.layout.font = {"family": "Segoe UI, Helvetica Neue, Arial, sans-serif", "size": 13}
        template.layout.title = {"x": 0.0, "xanchor": "left"}
        template.layout.margin = {"l": 50, "r": 30, "t": 60, "b": 50}
        template.layout.legend = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0}
        pio.templates[PLOTLY_TEMPLATE_NAME] = template
        pio.templates.default = PLOTLY_TEMPLATE_NAME
        _registered = True
    return pio.templates[PLOTLY_TEMPLATE_NAME]


def apply_theme() -> None:
    """Idempotent: registers the Plotly template. Streamlit colours come from config.toml."""
    plotly_template()


__all__ = [
    "OKABE_ITO",
    "PLOTLY_TEMPLATE_NAME",
    "STATUS_COLORS",
    "TOKENS",
    "apply_theme",
    "config_theme",
    "plotly_template",
]
