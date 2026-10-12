# Graph Rendering Architecture

The application draws three kinds of graphs: the **correlation network** (Main View), the
**clinical knowledge graph** (Data Insight) and the **document knowledge graph** (Documents Insight).
Since October 2026 they share one renderer-agnostic layer, the `graph` package, so the drawing
library can be swapped without touching analytics code or pages.

```
domain data ──► graph.builders.*  ──► GraphSpec ──► graph.ui.render_graph ──► renderer.translate()
(pandas, JSON)   (pure functions)     (model.py)      (Streamlit entry)        └─► renderer.render()
                                           ▲                                        (Plotly | yFiles | Graphviz)
                                   graph.palette (colours, legend)
```

## Layers

| Module | Role | Imports Streamlit? |
|---|---|---|
| `graph/model.py` | `GraphSpec`, `NodeSpec`, `EdgeSpec`, `RenderOptions`, `SelectionResult`, enums (`GraphKind`, `NodeShape`, `LineStyle`, `LayoutHint`). Immutable, JSON round-trippable, integrity-validated. | no |
| `graph/palette.py` | Colour tokens, style rules per graph kind, legend entries. The **only** place colours live. | no |
| `graph/builders/` | `build_correlation_graph`, `build_taxonomy_graph`, `build_document_graph` (+ `merge_document_graphs`, `filter_by_documents`). Deterministic ids and ordering. | no |
| `graph/base.py` | `BaseGraphRenderer` contract (`capabilities`, `availability()`, `translate()`, `render()`, `export_html()`). | no |
| `graph/manager.py` | Backend selection from `CMP_GRAPH_RENDERER` / `CMP_OFFLINE`, lazy registry, `set_renderer()` for tests. | no |
| `graph/renderers/` | `plotly.py` (offline default), `yfiles.py` (opt-in, proprietary), `graphviz.py` (static). | yes (inside `render`) |
| `graph/legend.py`, `graph/ui.py` | `render_legend`, `render_graph`, `renderer_selector`, `render_graph_exports`. | yes |
| `explore/correlation_strength.py` | The one correlation-strength threshold table (Schober 2018) used by the builder, the Excel export and the UI. | no |

## The contract: "the connector translates the information for the library"

`BaseGraphRenderer.translate(spec, options)` is a **pure** function that converts the semantic
model into the library's own primitives (yFiles `Node`/`Edge`/mapping callables and `show()`
arguments; a Plotly `Figure`; a DOT string). `render()` only calls the library and normalises what
comes back into a `SelectionResult`. Because translation has no side effects, each adapter is
unit-tested without a browser (`tests/graph_layer/test_renderer_*.py`); the yFiles test even
replays the translated payload through the library's own mapping engine.

### Semantic model → yFiles 2.x

| Semantic | yFiles primitive |
|---|---|
| `NodeStyle.shape` / `.color` | `node_styles_mapping` → `NodeStyle(color, shape)` |
| `NodeSpec.label` + text colour | `node_label_mapping` → `LabelStyle(text, color, position=CENTER, wrapping=WRAP_WORD_ELLIPSIS)` |
| `NodeSpec.weight` (with `RenderOptions.heat`) | `heat_mapping` (nodes only) |
| `EdgeStyle.line` / `.width` / `.color` | `edge_styles_mapping` → `EdgeStyle(dash_style, thickness, color, directed=True or None)` |
| `EdgeSpec.directed` | `directed_mapping` + `show(directed=all edges directed)` (never `EdgeStyle(directed=False)`, which the serializer drops) |
| `LayoutHint` | `Layout.HIERARCHICAL` / `ORGANIC` / `CIRCULAR` / `RADIAL` / `TREE` / `ORTHOGONAL` (resolved by member name) |
| `RenderOptions.height/key/selectable/focus_node` | `show(height=, key=, sync_selection=, neighborhood=)` |
| errors | `widget._error` → `RenderError` → fallback to Plotly |

### Semantic model → Plotly/NetworkX

Positions: `spring_layout(seed=42)` (organic), `circular_layout`, `shell_layout` around the focus
(radial), or a small layered Sugiyama layout (hierarchical: topological generations + barycentre
ordering). Nodes are a marker+text trace (`customdata` = node ids), edges are line traces grouped by
style, directed edges get arrow annotations, edge labels/hover/selection use an invisible midpoint
trace. Selection uses `st.plotly_chart(on_select="rerun")`.

## Choosing a renderer

| Setting | Effect |
|---|---|
| `CMP_GRAPH_RENDERER=auto` (default) | Plotly (offline). yFiles is **never** chosen automatically. |
| `CMP_GRAPH_RENDERER=yfiles` | yFiles when installed (`uv sync --extra graph-yfiles`); falls back to Plotly otherwise. |
| `CMP_GRAPH_RENDERER=plotly` / `graphviz` | Force a backend. |
| `CMP_OFFLINE=1` | Refuse any backend with `requires_internet=True` (yFiles). |
| In-app "Graph engine" selector | Session-wide override (`graph.ui.renderer_selector`). |

yFiles is distributed under the yWorks *yFiles Streamlit Component Free License*: it loads its
runtime from yworks.com at render time (internet required), reports anonymised graph-structure
usage data to yWorks, must not be redistributed, and requires informing end users (the About page
and a caption under every yFiles graph do that). The "air-gapped" deployment story therefore applies
to the Plotly and Graphviz backends only.

## Adding a renderer

1. Create `graph/renderers/<name>.py` with a class deriving from `BaseGraphRenderer`: set `name`,
   `display_name`, implement `capabilities`, `availability()`, `translate()`, `render()`.
2. Register it in `graph/manager.py` (`GraphRendererType`, `_REGISTRY_SPEC`, and `_AUTO_PREFERENCE`
   if it may be chosen automatically).
3. Add a translation test under `tests/graph_layer/`; the contract test
   (`test_ui_contract.py`) picks the backend up automatically.
4. Document licence and network requirements in `capabilities.license_note`.

## Adding a graph kind

1. Add the kind to `GraphKind`, its node/edge categories to `graph/palette.py` (`_NODE_RULES`,
   `_EDGE_RULES`) so legends are generated automatically.
2. Write a builder in `graph/builders/` returning a `GraphSpec` (use `edge_id()` for ids; keep node
   order deterministic; put domain payloads in `data`).
3. In the page: `spec = build_x(...)`, `selection = render_graph(spec, RenderOptions(key=..., selectable=True))`,
   then use `selection.node_ids` for detail panels and `render_graph_exports(spec, ...)` for downloads.
