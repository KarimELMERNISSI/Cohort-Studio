"""
Validate's lists of rules and the quality dashboard's show one box per rule, closed until opened, titled with the
rule's name and its kind, and giving the words of the participant flow and the dossier (M5b step 6). They decided a
rule's kind by "numeric" alone: frozen bounds, listed rows and a threshold showed as an "Expression" with "No
expression defined".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from core.criteria import describe_definition, rule_kind
from ui import nav
from ui.components.text import plain

MAIN = str(Path(__file__).resolve().parents[2] / "main.py")

RULES = {
    "Frozen bounds": {"bounds": {"potassium_mmol_l": [2.5, 7.0]}},
    "Listed rows": {
        "row_ids": [[0, 2]],
        "method": "iqr",
        "params": {"multiplier": 1.5},
        "columns": ["potassium_mmol_l"],
    },
    "Two signs": {
        "threshold": 2,
        "members": {
            "old": {"numeric": False, "expression": "df['age'] > 80"},
            "sbp": {
                "numeric": True,
                "column": "sbp_baseline",
                "upper_bound": 180,
                "strategy": "exclude",
            },
        },
    },
    "A *name* with [a link](https://example.org)": {
        "numeric": False,
        "expression": "df['age'] > 110",
    },
}
UNREADABLE = {"numeric": True, "column": "age", "lower_bound": 18, "expression": "df['age'] > 1"}


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "cards_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    at.session_state["config"]["mask_families"]["clinical_anomalies"].update(RULES)
    yield at
    dp.DATA_ROOT = previous


def _open(app: AppTest, tab: str) -> tuple[list[str], list[str], dict[str, bool]]:
    """The tab's code blocks, its captions, and its boxes by title (opened or not)."""
    from app_pages import data_monitoring as dm

    app.switch_page(nav.page_path("validate")).run()
    app.session_state[dm.TAB_KEY] = tab
    app.run()
    _ok(app)
    boxes = {e.label: bool(e.proto.expanded) for e in app.expander}
    return [c.value for c in app.code], [c.value for c in app.caption], boxes


def _title(name: str, definition: dict) -> str:
    return f"{plain(name, limit=200)} · {rule_kind(name, definition).label}"


def test_validate_shows_a_card_per_rule_in_the_flow_s_words(app: AppTest) -> None:
    family = app.session_state["config"]["mask_families"]["clinical_anomalies"]
    # loaded before the checks, or written into the session by hand
    family["Cannot be read"] = UNREADABLE
    try:
        codes, captions, boxes = _open(app, "Handle Implausible Values")
    finally:
        del family["Cannot be read"]
    why = (
        "criterion 'Cannot be read' holds both a numeric range and an expression: keep one of them"
    )
    assert boxes[f"{plain('Cannot be read')} · cannot be read"] is False
    assert plain(f"Cannot be read: {why}", limit=500) in captions
    for name, definition in family.items():
        if name == "operator":
            continue
        # titled with the name, as written, never as Markdown, and the kind; closed until opened
        assert boxes[_title(name, definition)] is False, name
        assert describe_definition(name, definition) in codes, name
    kinds = {title.rsplit(" · ", 1)[1] for title in boxes if " · " in title}
    assert {
        "Frozen bounds",
        "Listed rows",
        "At least k rules",
        "Numeric range",
        "Expression",
    } <= kinds


def test_the_dashboard_shows_the_same_cards(app: AppTest) -> None:
    codes, _, boxes = _open(app, "Data Quality Dashboard")
    family = app.session_state["config"]["mask_families"]["clinical_anomalies"]
    for name, definition in family.items():
        if name != "operator":
            assert describe_definition(name, definition) in codes, name
            assert _title(name, definition) in boxes, name


def test_the_anomaly_report_highlights_the_columns_each_rule_reads() -> None:
    """A rule named like a column was the only one whose cells were highlighted: an expression's never were."""
    from app_pages.data_quality_dashboard import anomaly_columns

    assert anomaly_columns({"operator": "any", **RULES}) == {
        "Frozen bounds": {"potassium_mmol_l"},
        "Listed rows": {"potassium_mmol_l"},
        "Two signs": {"age", "sbp_baseline"},
        "A *name* with [a link](https://example.org)": {"age"},
    }
