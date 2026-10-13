"""The palette is the single source of truth for colours, styles and legends."""

import re

import pytest

from graph.model import GraphKind, NodeShape
from graph.palette import (
    DARK_OVERRIDES,
    PALETTE,
    edge_categories,
    edge_label,
    edge_style_for,
    legend_entries,
    node_categories,
    node_label,
    node_style_for,
    text_color_on,
)

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_all_tokens_are_valid_hex() -> None:
    assert all(HEX.match(v) for v in PALETTE.values())
    assert all(HEX.match(v) for v in DARK_OVERRIDES.values())
    assert set(DARK_OVERRIDES) <= set(PALETTE)


@pytest.mark.parametrize("kind", list(GraphKind))
def test_every_category_has_a_style_and_a_legend_entry(kind: GraphKind) -> None:
    legend = legend_entries(kind)
    legend_labels = {entry.label for entry in legend}
    for category in node_categories(kind):
        style = node_style_for(kind, category)
        assert HEX.match(style.color) and isinstance(style.shape, NodeShape)
        assert node_label(kind, category) in legend_labels
    for category in edge_categories(kind):
        style = edge_style_for(kind, category)
        assert HEX.match(style.color) and style.width >= 0.5
        assert edge_label(kind, category) in legend_labels


def test_unknown_category_falls_back_without_error() -> None:
    assert (
        node_style_for(GraphKind.TAXONOMY, "???").color
        == node_style_for(GraphKind.TAXONOMY, "input").color
    )
    assert (
        edge_style_for(GraphKind.DOCUMENT, "treats").color
        == edge_style_for(GraphKind.DOCUMENT, "relation").color
    )


def test_correlation_edge_rules_cover_both_signs_and_all_strengths() -> None:
    cats = set(edge_categories(GraphKind.CORRELATION))
    for sign in ("pos", "neg"):
        for slug in ("negligible", "weak", "moderate", "strong", "very_strong"):
            assert f"corr.{sign}.{slug}" in cats
    assert (
        edge_style_for(GraphKind.CORRELATION, "corr.pos.very_strong").width
        > edge_style_for(GraphKind.CORRELATION, "corr.pos.weak").width
    )


def test_dark_theme_overrides_apply() -> None:
    light = node_style_for(GraphKind.TAXONOMY, "category", theme="light").color
    dark = node_style_for(GraphKind.TAXONOMY, "category", theme="dark").color
    assert light != dark


def test_text_color_contrast() -> None:
    assert text_color_on("#FFFFFF") == "#000000"
    assert text_color_on("#00008B") == "#FFFFFF"
    assert text_color_on(None) == "#000000"
