# Graph Rendering Architecture

The application draws three kinds of graphs: the **correlation network** (Explore > Statistics & correlations),
the **clinical knowledge graph** (Knowledge > Variable map) and the **document knowledge graph** (Knowledge >
Documents).
Since October 2026 they share one renderer-agnostic layer, the `graph` package, so the drawing
library can be swapped without touching analytics code or pages.

```
domain data ──► graph.builders.*  ──► GraphSpec ──► ui.graph_view.render_graph ──► renderer.translate()
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
| `graph/manager.py` | Backend selection (session choice, saved preference, `CMP_GRAPH_RENDERER`, offline-first order) with the offline rule (`CMP_OFFLINE`), the fallback engine, `set_renderer()` for tests. | no |
| `graph/renderers/__init__.py` | `GRAPH_RENDERERS`, the registry of the engines (a connector family on `core/registry.py`): names, lazy `"module:Class"` targets, optional groups, the automatic order. | no |
| `graph/renderers/` | `plotly.py` (offline default), `yfiles.py` (opt-in, proprietary), `graphviz.py` (static). | yes (inside `translate` and `render` only) |
| `ui/graph_legend.py`, `ui/graph_view.py` | `render_legend`, `render_graph`, `renderer_selector`, `render_graph_exports`. | yes |
| `explore/correlation_strength.py` | The one correlation-strength threshold table (Schober 2018) used by the builder, the Excel export and the UI. | no |

## The contract: "the connector translates the information for the library"

`BaseGraphRenderer.translate(spec, options)` converts the semantic model, with no side effects, into
the library's own primitives (yFiles `Node`/`Edge`/mapping callables and `show()` arguments; a
Plotly `Figure`; a DOT string). `render()` only calls the library and normalises what comes back
into a `SelectionResult`. Because translation has no side effects, each adapter is unit-tested
without a browser (`tests/graph_layer/test_renderer_*.py`); the yFiles test even replays the
translated payload through the library's own mapping engine.

A renderer loads its library inside `translate()` and `render()` only, and Streamlit inside
`render()` only, except that the yFiles primitives come from a Streamlit component, which
`translate()` may load (the layer test's rule R3). `availability()` finds the library without
importing it (`Availability.of_module`) and never raises, so listing the engines loads none of them;
the yFiles adapter checks the component's API when drawing, where a mismatch is a `RenderError` with
the fallback.

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
| `CMP_GRAPH_RENDERER=plotly` / `graphviz` | The server's default engine (a user's own choice comes first). |
| `CMP_OFFLINE=1` | Refuse any backend with `requires_internet=True` (yFiles). |
| In-app "Graph engine" selector | Session-wide override (`ui.graph_view.renderer_selector`, on `ui.components.connector_selector`), kept for the user's next session. Its first option is the server's default: "Automatic: <engine>", or "Server default: <engine>" when `CMP_GRAPH_RENDERER` names one. Only the engines that can draw here are offered; on Settings it says why the engine is used, and lists the engines that cannot draw here with the group to install. |

The choice is the session's selector, then the user's saved preference, then `CMP_GRAPH_RENDERER`,
then the automatic order (Plotly, then Graphviz); "automatic" at a level leaves the choice to the
next one. A choice that cannot draw here falls through to the next one. Settings names a saved engine
or a server setting that cannot draw here, and the engine used instead. A `CMP_GRAPH_RENDERER` that
names no engine stops start-up (`manage.connectors.check_connector_settings`), with the variable and
the valid names, never its value.

yFiles is distributed under the yWorks *yFiles Streamlit Component Free License*: it loads its
runtime from yworks.com at render time (internet required), reports anonymised graph-structure
usage data to yWorks, must not be redistributed, and requires informing end users (the About page
and a caption under every yFiles graph do that). The "air-gapped" deployment story therefore applies
to the Plotly and Graphviz backends only.

## Adding a renderer

1. Create `graph/renderers/<name>.py` with a class deriving from `BaseGraphRenderer`: set `name`,
   `display_name` and `capabilities` as class attributes; implement `availability()` (find the library
   with `Availability.of_module`, without importing it), `translate()` and `render()`.
2. Register it in `graph/renderers/__init__.py`:
   `GRAPH_RENDERERS.register("<name>", "graph.renderers.<name>:<Class>", extra="<optional group>")`,
   and add it to the automatic order there if it may be chosen automatically. The name is stored in
   users' preferences and settings: never rename it.
3. Add the name to the stored names in `tests/graph_layer/test_renderer_registry.py`, and a
   translation test under `tests/graph_layer/`; the contract tests (`test_renderer_registry.py`,
   `test_ui_contract.py`) pick the backend up automatically.
4. Document licence and network requirements in `capabilities.license_note`.

## Adding a graph kind

1. Add the kind to `GraphKind`, its node/edge categories to `graph/palette.py` (`_NODE_RULES`,
   `_EDGE_RULES`) so legends are generated automatically.
2. Write a builder in `graph/builders/` returning a `GraphSpec` (use `edge_id()` for ids; keep node
   order deterministic; put domain payloads in `data`).
3. In the page: `spec = build_x(...)`, `selection = render_graph(spec, RenderOptions(key=..., selectable=True))`,
   then use `selection.node_ids` for detail panels and `render_graph_exports(spec, ...)` for downloads.
