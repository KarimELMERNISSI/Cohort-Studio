"""
Interactive workflow map (Streamlit custom component v2).

Every box is a link: hovering highlights the box, the paths that connect to it and shows its purpose;
clicking it (or pressing Enter/Space when it has keyboard focus) returns the box id to Python, which
switches to the matching page in the same session. Colours come from the active Streamlit theme
(``--st-*`` CSS variables), so the map follows light and dark mode.
"""

from __future__ import annotations

import streamlit as st

from ui.workflow import node_page, workflow_model

_HTML = """
<div class="wf">
  <div class="wf-canvas"></div>
  <div class="wf-hint" aria-live="polite"></div>
</div>
"""

_CSS = """
:host { display: block; }
.wf { font-family: var(--st-font); color: var(--st-text-color); }
.wf-canvas svg { display: block; width: 100%; height: auto; overflow: visible; }
.wf-hint {
  min-height: 1.4em;
  margin-top: 0.35rem;
  font-size: 0.9rem;
  color: var(--st-text-color);
  opacity: 0.8;
}
.wf-hint strong { color: var(--st-primary-color); }

/* edges */
.edge path {
  fill: none;
  stroke: var(--st-gray-color);
  stroke-width: 1.6;
  transition: stroke 0.15s ease, opacity 0.15s ease, stroke-width 0.15s ease;
}
.edge.dashed path { stroke-dasharray: 6 5; }
.edge text {
  font-size: 11px;
  fill: var(--st-text-color);
  opacity: 0.7;
  paint-order: stroke;
  stroke: var(--st-background-color);
  stroke-width: 4px;
}
.arrow { fill: var(--st-gray-color); }
.arrow-hi { fill: var(--st-primary-color); }

/* nodes */
.node { cursor: pointer; outline: none; }
.node .box { stroke-width: 1.6; transition: filter 0.15s ease, stroke-width 0.15s ease; }
.node .title { font-size: 14px; font-weight: 700; }
.node .sub { font-size: 11.5px; opacity: 0.85; }
.node.core .box { fill: var(--st-primary-color); stroke: var(--st-primary-color); }
.node.core text { fill: #FFFFFF; }
.node.core.done .box {
  fill: color-mix(in srgb, var(--st-primary-color) 55%, #0B2B36);
  stroke: color-mix(in srgb, var(--st-primary-color) 55%, #0B2B36);
}
.node.optional .box {
  fill: var(--st-secondary-background-color);
  stroke: var(--st-primary-color);
  stroke-dasharray: 7 5;
}
.node.optional text { fill: var(--st-text-color); }
.node.ai-off .box { opacity: 0.75; }
.node:hover .box, .node:focus-visible .box {
  stroke-width: 3;
  filter: drop-shadow(0 6px 10px rgba(11, 43, 54, 0.28));
}
.node:focus-visible .box { stroke: var(--st-text-color); }
.check { fill: var(--st-green-color); }
.tick { fill: none; stroke: #FFFFFF; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.pill { fill: var(--st-orange-background-color); }
.pill-text { font-size: 10.5px; font-weight: 600; fill: var(--st-orange-text-color) !important; }
.next-text { font-size: 11px; font-weight: 600; fill: var(--st-primary-color) !important; }
.ring {
  fill: none;
  stroke: var(--st-primary-color);
  stroke-width: 2;
  transform-box: fill-box;
  transform-origin: center;
  animation: wf-pulse 1.9s ease-out infinite;
}
@keyframes wf-pulse {
  0% { opacity: 0.9; transform: scale(1); }
  100% { opacity: 0; transform: scale(1.08); }
}
@media (prefers-reduced-motion: reduce) { .ring { animation: none; opacity: 0.6; } }

/* legend (bottom-left corner of the drawing) */
.legend path { fill: none; stroke: var(--st-gray-color); stroke-width: 1.6; }
.legend path.dashed { stroke-dasharray: 6 5; }
.legend rect.core { fill: var(--st-primary-color); }
.legend rect.optional {
  fill: var(--st-secondary-background-color);
  stroke: var(--st-primary-color);
  stroke-width: 1.4;
  stroke-dasharray: 4 3;
}
.legend text { font-size: 11.5px; fill: var(--st-text-color); opacity: 0.75; }

/* focus mode while hovering: dim what is unrelated, highlight connected paths */
.focusing .node.dim, .focusing .edge.dim { opacity: 0.32; }
.focusing .ring.dim { animation: none; opacity: 0.15; }
.edge.hi path { stroke: var(--st-primary-color); stroke-width: 2.4; marker-end: url(#wf-arrow-hi); }
.edge.hi.both path { marker-start: url(#wf-arrow-hi); }
.edge.hi text { opacity: 1; }
"""

_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const canvas = parentElement.querySelector(".wf-canvas");
  const hint = parentElement.querySelector(".wf-hint");
  if (!canvas || !hint || !data) return;

  if (canvas.dataset.version !== data.version) {
    canvas.innerHTML = data.svg;
    canvas.dataset.version = data.version;
  }
  const svg = canvas.querySelector("svg");
  if (!svg) return;

  const nodes = data.nodes || {};
  const edges = data.edges || [];
  const defaultHint = data.hint || "";
  hint.textContent = defaultHint;

  const linked = (id) => {
    const set = new Set([id]);
    edges.forEach(([a, b]) => { if (a === id) set.add(b); if (b === id) set.add(a); });
    return set;
  };

  const focusOn = (id) => {
    const keep = linked(id);
    svg.classList.add("focusing");
    svg.querySelectorAll("[data-node]").forEach((el) => {
      el.classList.toggle("dim", !keep.has(el.dataset.node));
    });
    svg.querySelectorAll("[data-ring-for]").forEach((el) => {
      el.classList.toggle("dim", !keep.has(el.dataset.ringFor));
    });
    svg.querySelectorAll(".edge").forEach((el) => {
      const touches = el.dataset.from === id || el.dataset.to === id;
      el.classList.toggle("hi", touches);
      el.classList.toggle("dim", !touches);
    });
    const info = nodes[id];
    if (info) {
      hint.replaceChildren();
      const title = document.createElement("strong");
      title.textContent = info.title;
      hint.append(title, " - " + info.description);
      if (info.needs_ai && !data.ai_on) hint.append(" (needs the AI engine: Settings)");
      hint.append(" Click to open.");
    }
  };

  const clearFocus = () => {
    svg.classList.remove("focusing");
    svg.querySelectorAll(".dim, .hi").forEach((el) => el.classList.remove("dim", "hi"));
    hint.textContent = defaultHint;
  };

  const nodeOf = (target) => (target && target.closest ? target.closest("[data-node]") : null);
  const open = (id) => setTriggerValue("clicked", id);

  // Assigning on* handlers (not addEventListener) keeps exactly one handler across reruns.
  canvas.onclick = (event) => { const n = nodeOf(event.target); if (n) open(n.dataset.node); };
  canvas.onkeydown = (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const n = nodeOf(event.target);
    if (n) { event.preventDefault(); open(n.dataset.node); }
  };
  canvas.onmouseover = (event) => { const n = nodeOf(event.target); if (n) focusOn(n.dataset.node); };
  canvas.onmouseout = (event) => { if (!nodeOf(event.relatedTarget)) clearFocus(); };
  canvas.onfocusin = (event) => { const n = nodeOf(event.target); if (n) focusOn(n.dataset.node); };
  canvas.onfocusout = () => clearFocus();
}
"""


def workflow_map(*, has_data: bool, ai_on: bool, key: str = "cmp_workflow_map") -> str | None:
    """Render the interactive map; returns the navigation key of the clicked box (once), else ``None``."""
    # Registered on every render, not at import: the registry belongs to the running Streamlit runtime,
    # which can be replaced while this module stays imported (server reload, AppTest). Registering an
    # identical definition again is a silent no-op.
    mount = st.components.v2.component("cmp_workflow_map", html=_HTML, css=_CSS, js=_JS)
    result = mount(data=workflow_model(has_data, ai_on), key=key, on_clicked_change=lambda: None)
    clicked = getattr(result, "clicked", None)
    return node_page(clicked) if clicked else None


__all__ = ["workflow_map"]
