"""Design tokens mirror config.toml; the navigation registry is consistent."""

from pathlib import Path

from ui import nav
from ui.theme import (
    BRAND,
    OKABE_ITO,
    STYLES_PATH,
    TOKENS,
    brand_asset,
    config_theme,
    plotly_template,
)


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_tokens_match_streamlit_config() -> None:
    theme = config_theme()
    assert theme["light"]["primaryColor"] == TOKENS["primary"] == BRAND["teal"]
    assert theme["dark"]["primaryColor"] == TOKENS["primary_dark"]
    assert theme["light"]["textColor"] == TOKENS["ink"]
    assert theme["light"]["sidebar"]["backgroundColor"] == TOKENS["surface"]
    assert theme["chartCategoricalColors"] == list(OKABE_ITO)
    assert (
        len(theme["chartSequentialColors"]) == 10
        and len(theme["dark"]["chartSequentialColors"]) == 10
    )


def test_theme_colours_meet_wcag_aa() -> None:
    theme = config_theme()
    light, dark = theme["light"], theme["dark"]
    assert _contrast(light["primaryColor"], "#FFFFFF") >= 4.5  # white text on primary buttons
    assert _contrast(light["textColor"], light["backgroundColor"]) >= 7
    assert _contrast(dark["textColor"], dark["backgroundColor"]) >= 7
    assert _contrast(dark["primaryColor"], dark["backgroundColor"]) >= 3  # non-text UI components


def test_brand_assets_and_stylesheet_exist() -> None:
    for name in ("icon.png", "wordmark_light.png", "wordmark_dark.png"):
        assert Path(brand_asset(name)).name == name, (
            f"run dev_scripts/make_brand_assets.py ({name} missing)"
        )
    css = STYLES_PATH.read_text(encoding="utf-8")
    assert (
        ".st-key-cmp-hero" in css and "@import" not in css and "http" not in css
    )  # offline, scoped


def test_plotly_template_is_registered_once() -> None:
    import plotly.io as pio

    first = plotly_template()
    second = plotly_template()
    assert first is second
    assert pio.templates.default == "cmp"
    assert list(first.layout.colorway)[0] == "#0072B2"


def test_navigation_registry_is_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    keys = [p.key for p in nav.PAGES]
    assert len(keys) == len(set(keys))
    assert sum(p.default for p in nav.PAGES) == 1
    assert len({p.url_path for p in nav.PAGES}) == len(nav.PAGES)
    for p in nav.PAGES:
        assert (root / nav.page_path(p.key)).is_file(), p.key
        if p.chapter:
            assert (root / "documentation" / f"{p.chapter}.md").is_file(), p.chapter
    assert not [p for p in nav.visible_pages(is_admin=False) if p.admin]
    assert [p for p in nav.visible_pages(is_admin=True) if p.admin]
