"""The Start-page workflow: boxes point at real pages; the drawing is well formed and laid out cleanly."""

import xml.dom.minidom

from core.progress import StepMark, evaluate, initial_progress
from ui import nav
from ui.workflow import (
    BOX_H,
    BOX_W,
    BRANCHES,
    CANVAS,
    CORE_STEPS,
    EDGES,
    NEXT_TEXT,
    NODES,
    default_hint,
    edge_points,
    label_box,
    legend_box,
    next_label_box,
    node_box,
    node_page,
    workflow_model,
    workflow_svg,
)

Box = tuple[float, float, float, float]


def _inside(point: tuple[float, float], box: Box, margin: float = 0.0) -> bool:
    x, y = point
    return box[0] - margin < x < box[2] + margin and box[1] - margin < y < box[3] + margin


def _overlap(a: Box, b: Box, margin: float = 0.0) -> bool:
    return (
        a[0] - margin < b[2]
        and b[0] - margin < a[2]
        and a[1] - margin < b[3]
        and b[1] - margin < a[3]
    )


def _cross(p, q, r, s) -> bool:
    """Segments pq and rs intersect (touching counts)."""

    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2, d3, d4 = orient(r, s, p), orient(r, s, q), orient(p, q, r), orient(p, q, s)
    return (d1 * d2 <= 0) and (d3 * d4 <= 0) and not (d1 == d2 == d3 == d4 == 0)


def _in_canvas(box: Box) -> bool:
    return box[0] >= 0 and box[1] >= 0 and box[2] <= CANVAS[0] and box[3] <= CANVAS[1]


def test_branches_link_to_registered_pages() -> None:
    keys = {p.key for p in nav.PAGES}
    assert [b.key for b in BRANCHES] == ["enrich", "variable_map", "documents", "reproduce"]
    for branch in BRANCHES:
        assert branch.key in keys
        assert branch.title and branch.when and branch.text
    assert {b.key for b in BRANCHES if b.needs_ai} == {"variable_map", "documents"}
    enrich = next(b for b in BRANCHES if b.key == "enrich")
    assert [link.tab for link in enrich.links] == [None, "derived_variables"]
    # the core path's cards: one per step of the diagram's core path
    assert [s.key for s in CORE_STEPS] == ["load", "validate", "analyse"]
    assert {s.key for s in CORE_STEPS} <= {n.id for n in NODES if n.kind == "core"}


def test_every_start_page_link_opens_a_registered_page_and_a_tab_it_declares() -> None:
    keys = {p.key for p in nav.PAGES}
    for card in (*CORE_STEPS, *BRANCHES):
        assert card.links and all(link.page in keys for link in card.links), card.title
        assert all(
            link.tab is None or link.tab in dict(nav.page(link.page).tabs) for link in card.links
        ), card.title


def test_a_link_opens_a_tab_its_page_declares() -> None:
    import pytest

    from ui.components import page_link, switch_to

    assert nav.opened_tabs("enrich", "derived_variables") == (
        "Handle Missing Data",
        "Derived variables",
    )
    # no parameter, an old or mistyped one, or another page's: the page opens as usual
    assert nav.opened_tabs("enrich", None) == () == nav.opened_tabs("enrich", "no_such_tab")
    assert nav.opened_tabs("validate", "derived_variables") == ()
    assert nav.tab_query("enrich", "derived_variables") == {nav.TAB_PARAM: "derived_variables"}
    assert nav.tab_query("enrich", None) is None
    # a link or a switch to a tab the page does not declare fails, rather than opening the page as usual
    with pytest.raises(ValueError, match="declares no tab"):
        page_link("validate", "Derived variables", tab="derived_variables")
    with pytest.raises(ValueError, match="declares no tab"):
        switch_to("validate", tab="derived_variables")


def test_every_box_opens_a_registered_page() -> None:
    keys = {p.key for p in nav.PAGES}
    ids = [n.id for n in NODES]
    assert len(ids) == len(set(ids))
    for node in NODES:
        assert node.page in keys
        assert node_page(node.id) == node.page
    assert node_page("unknown") is None
    for edge in EDGES:
        assert edge.src in ids and edge.dst in ids


def test_boxes_fit_the_canvas_without_overlapping() -> None:
    for node in NODES:
        assert 0 <= node.x and node.x + BOX_W <= CANVAS[0]
        assert 0 <= node.y and node.y + BOX_H <= CANVAS[1]
    for i, a in enumerate(NODES):
        for b in NODES[i + 1 :]:
            assert not _overlap(node_box(a), node_box(b), 20.0), (a.id, b.id)


def test_links_never_cross_or_share_an_anchor() -> None:
    curves = [(edge, edge_points(edge)) for edge in EDGES]
    for i, (a, pa) in enumerate(curves):
        for b, pb in curves[i + 1 :]:
            assert pa[0] not in (pb[0], pb[-1]) and pa[-1] not in (pb[0], pb[-1]), (a, b)
            for s in zip(pa, pa[1:], strict=False):
                for t in zip(pb, pb[1:], strict=False):
                    assert not _cross(*s, *t), (a.src, a.dst, b.src, b.dst)


def test_links_and_labels_keep_clear_of_boxes_and_of_each_other() -> None:
    boxes = {n.id: node_box(n) for n in NODES}
    labels = {(e.src, e.dst): label_box(e) for e in EDGES if e.label}
    for edge in EDGES:
        points = edge_points(edge)
        for node_id, box in boxes.items():
            # a link starts and ends on a box border: it may touch its own boxes there, never enter them
            margin = -1.0 if node_id in (edge.src, edge.dst) else 8.0
            assert not any(_inside(p, box, margin) for p in points[1:-1]), (
                edge.src,
                edge.dst,
                node_id,
            )
        for key, box in labels.items():
            assert box is not None
            assert not any(_inside(p, box, 2.0) for p in points), (
                edge.src,
                edge.dst,
                "crosses label",
                key,
            )
    for key, box in labels.items():
        assert box is not None and _in_canvas(box), key
        assert not any(_overlap(box, node, 2.0) for node in boxes.values()), key
        others = [b for k, b in labels.items() if k != key and b is not None]
        assert not any(_overlap(box, other, 4.0) for other in others), key


def test_legend_and_suggested_next_keep_clear_of_the_links() -> None:
    regions = {"legend": legend_box()} | {
        f"next:{n.id}": next_label_box(n) for n in NODES if n.kind == "core"
    }
    for name, region in regions.items():
        assert _in_canvas(region), name
        assert not any(_overlap(region, node_box(n), 2.0) for n in NODES), name
        for edge in EDGES:
            assert not any(_inside(p, region, 2.0) for p in edge_points(edge)), (
                name,
                edge.src,
                edge.dst,
            )
        for edge in EDGES:
            box = label_box(edge)
            assert box is None or not _overlap(region, box, 2.0), (name, edge.label)


NO_DATA = initial_progress(False)
LOADED = initial_progress(True)


def _node_group(svg: str, node_id: str):
    doc = xml.dom.minidom.parseString(svg)
    return next(g for g in doc.getElementsByTagName("g") if g.getAttribute("data-node") == node_id)


def test_drawing_is_valid_svg_with_state_markers() -> None:
    empty = workflow_svg(NO_DATA, ai_on=False)
    xml.dom.minidom.parseString(empty)  # well-formed
    assert empty.count('data-node="') == len(NODES)
    assert empty.count('class="edge') == len(EDGES)
    assert 'tabindex="0"' in empty and 'role="link"' in empty  # keyboard accessible
    assert 'class="check"' not in empty and empty.count("AI off") == 2
    assert NO_DATA.next_step == "load" and 'class="node core next" data-node="load"' in empty
    assert 'data-ring-for="load"' in empty and 'class="legend"' in empty
    # the ring is drawn before every link, so arrow heads are painted over it
    assert empty.index('class="ring"') < empty.index('class="edge')

    loaded = workflow_svg(LOADED, ai_on=True)
    assert 'class="node core done" data-node="load"' in loaded and "AI off" not in loaded
    assert LOADED.next_step == "validate" and 'data-ring-for="validate"' in loaded
    assert f">{NEXT_TEXT['todo']}<" in loaded


def test_progress_ticks_stale_badges_and_the_next_step() -> None:
    steps = [
        {"function": "initial_load"},
        {"function": "imputation", "description": "Imputation (KNN)"},
    ]
    marks = {"validate": StepMark(0, detail="quality score 90/100"), "analyse": StepMark(0)}
    knowledge = {
        "docs": (True, "1 PDF in your library, 1 knowledge graph saved."),
        "varmap": (False, ""),
    }
    progress = evaluate(steps, marks, has_data=True, knowledge=knowledge)
    svg = workflow_svg(progress, ai_on=False)
    xml.dom.minidom.parseString(svg)

    validate = _node_group(svg, "validate")
    assert validate.getAttribute("class") == "node core stale next"
    assert validate.getElementsByTagName("circle")[0].getAttribute("class") == "stale-badge"
    assert f">{NEXT_TEXT['stale']}<" in svg and 'data-ring-for="validate"' in svg
    assert "Checked before the last 1 change" in validate.getAttribute("aria-label")

    assert "done" in _node_group(svg, "enrich").getAttribute("class").split()
    assert "stale" in _node_group(svg, "analyse").getAttribute("class").split()
    # one corner mark per box: documents (done) keep the tick, the variable map (to do) the "AI off" pill
    docs, varmap = _node_group(svg, "docs"), _node_group(svg, "varmap")
    assert docs.getElementsByTagName("circle")[0].getAttribute("class") == "check"
    assert all(r.getAttribute("class") != "pill" for r in docs.getElementsByTagName("rect"))
    assert any(r.getAttribute("class") == "pill" for r in varmap.getElementsByTagName("rect"))
    assert svg.count("AI off") == 1
    assert default_hint(progress).startswith(
        "Suggested next: Validate. Checked before the last 1 change"
    )


def test_no_ring_when_every_step_is_up_to_date() -> None:
    steps = [
        {"function": "initial_load"},
        {"function": "save_version", "description": "Saved version: c_v2"},
    ]
    marks = {"validate": StepMark(0), "analyse": StepMark(0)}
    progress = evaluate(steps, marks, has_data=True)
    svg = workflow_svg(progress, ai_on=True)
    assert progress.next_step is None and 'class="ring"' not in svg and "suggested next" not in svg
    assert svg.count('class="check"') == 4  # load, validate, analyse, report
    assert default_hint(progress).startswith("Every step is up to date")
    assert default_hint(NO_DATA).startswith("Start here")


def test_enrichment_loops_with_validation_and_analysis() -> None:
    """Enrichment is incremental: Validate <-> Enrich <-> Explore & analyse are the only two-way links."""
    assert {(e.src, e.dst) for e in EDGES if e.both} == {
        ("validate", "enrich"),
        ("enrich", "analyse"),
    }
    doc = xml.dom.minidom.parseString(workflow_svg(LOADED, ai_on=True))
    for group in doc.getElementsByTagName("g"):
        classes = group.getAttribute("class").split()
        if "edge" not in classes:
            continue
        path = group.getElementsByTagName("path")[0]
        assert path.getAttribute("marker-end") == "url(#wf-arrow)"
        assert bool(path.getAttribute("marker-start")) == ("both" in classes)
    assert workflow_svg(LOADED, ai_on=True).count(">iterate<") == 2


def test_model_carries_details_for_the_hint_line() -> None:
    model = workflow_model(LOADED, ai_on=False)
    assert set(model["nodes"]) == {n.id for n in NODES}
    assert model["nodes"]["load"]["title"] == "Load data"
    assert model["nodes"]["load"]["state"] == "done"
    assert model["nodes"]["validate"]["status"] == "Not checked yet for this dataset."
    assert model["nodes"]["docs"]["needs_ai"] is True
    assert "Incremental" in model["nodes"]["enrich"]["description"]
    assert ["docs", "varmap"] in model["edges"] and ["enrich", "analyse"] in model["edges"]
    assert model["hint"].startswith("Suggested next: Validate.")
    # The version follows the drawing: stable for a state, different when the state changes.
    assert model["version"] == workflow_model(LOADED, ai_on=False)["version"]
    assert model["version"] != workflow_model(NO_DATA, ai_on=False)["version"]
