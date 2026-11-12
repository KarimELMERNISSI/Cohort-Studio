"""Every tab a link can open (ui.nav.PageDef.tabs) is on its page, which opens on it (AppTest, the tutorial cohort)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav


def _opened(at: AppTest) -> dict[tuple[str, ...], str]:
    """Each row of tabs on the page (its labels) and the tab it opens."""
    found: dict[tuple[str, ...], str] = {}

    def walk(node) -> None:
        if getattr(node, "type", None) == "tab_container":
            labels = tuple(child.label for child in node.children.values())
            found[labels] = labels[node.proto.tab_container.default_tab_index]
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(at.main)
    return found


def _chosen(at: AppTest) -> dict[tuple[str, ...], str]:
    """Each selector on the page (its labels) and the choice it holds: a label after the tabs of a path names one."""
    return {
        tuple(control.options): control.formatted_values[0]
        for control in at.segmented_control
        if control.formatted_values
    }


DECLARED = [(p.key, tab, labels) for p in nav.PAGES for tab, labels in p.tabs]


def test_a_page_declares_tabs_to_link_to() -> None:
    assert ("enrich", "derived_variables", ("Handle Missing Data", "Derived variables")) in DECLARED


@pytest.mark.parametrize(
    ("key", "tab", "labels"), DECLARED, ids=[f"{k}-{t}" for k, t, _ in DECLARED]
)
def test_a_declared_tab_opens_its_page_on_it(
    key: str, tab: str, labels: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    main = Path(__file__).resolve().parents[2] / "main.py"
    at = AppTest.from_file(str(main), default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "tab_links_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()  # some tabs are drawn only with a dataset
    at.switch_page(nav.page_path(key)).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    rows = {row: row[0] for row in _opened(at)}
    assert any(labels[0] in row for row in rows), (key, tab, labels[0], list(rows))
    assert _opened(at) == rows  # without the link, each row opens on its first tab
    at.query_params[nav.TAB_PARAM] = tab
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    # with it, each label of its path is what its row opens on: a row of tabs, or the selector a tab opens on (drawn
    # once its tab is open: only the open tab of such a page runs)
    opened = {**_chosen(at), **_opened(at)}
    for label in labels:
        assert label in {name for row in opened for name in row}, (key, tab, label, list(opened))
        assert next(o for row, o in opened.items() if label in row) == label, (key, tab, label)
