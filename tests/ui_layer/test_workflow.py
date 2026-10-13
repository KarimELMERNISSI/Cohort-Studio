"""The Start-page workflow: core path and optional branches point at real pages; the diagram is well formed."""

from ui import nav
from ui.workflow import BRANCHES, workflow_dot


def test_branches_link_to_registered_pages() -> None:
    keys = {p.key for p in nav.PAGES}
    assert [b.key for b in BRANCHES] == ["enrich", "variable_map", "documents", "reproduce"]
    for branch in BRANCHES:
        assert branch.key in keys
        assert branch.links and all(key in keys for key, _ in branch.links)
        assert branch.title and branch.when and branch.text
    assert {b.key for b in BRANCHES if b.needs_ai} == {"variable_map", "documents"}


def test_diagram_shows_core_path_and_optional_branches() -> None:
    dot = workflow_dot(has_data=False)
    assert dot.startswith("digraph workflow {") and dot.rstrip().endswith("}")
    assert dot.count("{") == dot.count("}")
    assert "load -> validate -> analyse -> report;" in dot
    for edge in (
        "validate -> enrich",
        "enrich -> analyse",
        "docs -> varmap",
        "varmap -> analyse",
        "docs -> enrich",
    ):
        assert f"{edge} [style=dashed" in dot or f"{edge};" in dot.replace(" [style=dashed];", ";")
    assert "(done)" not in dot
    assert "1  Load data (done)" in workflow_dot(has_data=True)
