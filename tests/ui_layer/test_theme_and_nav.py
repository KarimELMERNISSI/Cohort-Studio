"""Design tokens mirror config.toml; the navigation registry is consistent."""

from pathlib import Path

from ui import nav
from ui.theme import TOKENS, config_theme, plotly_template


def test_tokens_match_streamlit_config() -> None:
    theme = config_theme()
    assert theme["primaryColor"] == TOKENS["primary"]
    assert theme["dark"]["primaryColor"] == TOKENS["primary_dark"]
    assert theme["sidebar"]["backgroundColor"] == TOKENS["surface"]


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
