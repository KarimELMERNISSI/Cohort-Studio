# Variable map — Taxonomy (Knowledge Graph)

The **Variable map** page (Knowledge > Variable map) generates, visualises, and refines a **Taxonomy** (knowledge graph) from your dataset's variables. The Taxonomy captures how variables relate to clinical domains, computed formulas, and broader scientific concepts. It is built with retrieval-augmented generation (RAG) by the AI engine chosen in Settings > AI engine (local Ollama by default; cloud providers need an API key and your consent).

Once a taxonomy is open, the page has six tabs: **Knowledge Graph**, **Overview** (counts, the variable table and the enrichment proposals), **Variable Details**, **Formulas Details**, **Population Clustering** and **Refinement**. The sidebar holds the saved versions, the *Graph Actions*, the variable and formula searches, the category filter and the JSON export.

---

## Taxonomy Overview

A Taxonomy is a structured graph where:

| Node Type | Description | Example |
| --- | --- | --- |
| **Category** | Clinical domain or grouping. | Anthropometry, Lipid Panel |
| **Variable** | A dataset column. | `weight_kg`, `hdl_cholesterol` |
| **Concept** | An external-derived clinical or scientific concept. | BMI, HOMA-IR |
| **Formula** | A computable relationship between variables. | `BMI = Weight / Height²` |

Edges link a variable to its category, the input variables of a formula to the formula, the formula to the variable it yields, and related variables to each other.

---

## Taxonomy Generation

### Steps

1. **Load or Create** — Load a saved version (choose it in the sidebar, then **Load**; while no taxonomy is open,
   the page also offers **Load vN**), or generate a new taxonomy from the current dataset with **Generate Taxonomy** (shown while no taxonomy is open) or **New Taxonomy** (sidebar, *Graph Actions*). Generation needs a loaded dataset and an initialised AI engine.
2. **AI Generation** — The AI engine analyses the column names (with simple summaries of their values when you start from **Generate Taxonomy**) and passages retrieved from your indexed documents, to infer standard names, categories, clinical usage and formulas.
3. **Review** — Inspect the generated graph in the interactive visualisation (offline Plotly engine by default; yFiles optional).
4. **Iterate** — Enrich or refine the taxonomy as needed, then save it as a version.

### Generation Options

| Parameter | Description |
| --- | --- |
| **Context Documents** | Optional PDFs to give the AI domain knowledge for better classification. Add them in Settings > AI engine > *Document library* and click **Initialise / reindex**; passages relevant to your columns are retrieved during generation. |

A new generation does not start from an earlier taxonomy. To build on a saved version, load it and use the refinement tools below.

---

## Taxonomy Refinement

After initial generation, several tools are available to improve the taxonomy. They need an initialised AI engine.

### Enrich

- **Enrich Taxonomy** (sidebar, *Graph Actions*) — The AI looks for related external variables and formulas, up to the chosen *Neighborhood Distance* (1 to 3). The proposals appear in the **Overview** tab under *Enrichment Proposals*: tick **Keep?** for the variables to add, then click **Merge Selected**, or **Discard All**.
- **Refine Existing Variable (AI)** (**Refinement** tab) — Select variables, write feedback on their definitions, and click **Refine Selected Variables**: the AI updates them.
- **Add Missing Concept Node (AI)** (**Refinement** tab) — Name a concept or score (e.g. HOMA-IR), optionally restrict the AI to some variables (*Involved Variables*) and choose a *Target Category* (*Auto (AI)* lets the AI choose), click **Generate Concept Suggestions**, then **Add to Graph** for each suggestion you keep. The concept is added with its formula and linked to its input variables.

### Link repair on load

Loading a version, with **Load** in the sidebar or **Load vN** on the empty page, brings its formulas and re-links
each variable to the saved formulas that use or produce it, in one click. There is no separate structure check, JSON repair or manual editor for nodes and edges: change the taxonomy with the tools above.

---

## Taxonomy Versioning

| Action | Description |
| --- | --- |
| **Save New (vN)** | Saves the current taxonomy and its formulas as the next numbered version (v1, v2, ...). |
| **Overwrite** | Replaces the loaded version with the current taxonomy and formulas. |
| **Load** | Restores a saved version chosen in the sidebar list. |

Versions are stored in your workspace, per user; they are not tied to a dataset. There is no comparison between versions.

---

## Visualisation

The taxonomy is rendered as an interactive graph through a library-agnostic connector (see
`documentation/developer/Graph_Rendering.md`). The default engine is Plotly + NetworkX, which works
offline; the proprietary yFiles engine can be enabled as an optional extra.

- **Node Styles** — Colours and shapes per node type (category, input, derived, outcome, external, formula), with a generated legend.
- **Edge Labels** — Relationship type displayed on each connection (input, yields, category membership, related).
- **Layout** — Hierarchical layout (formulas between their inputs and outputs); organic/circular/radial layouts are available with the yFiles engine.
- **Filter** — Show or hide node and edge types; the formula search restricts the graph to matching formulas and their variables.
- **Selection Details** — Select nodes in the graph to open their metadata, description and formula definitions below the graph.
- **Graph engine** — Switch between the offline engine and yFiles (needs internet access) in the graph toolbar.

---

## Export

| Format | Description |
| --- | --- |
| **JSON** | **Export JSON** in the sidebar: the taxonomy (variables and their metadata) as `variable_taxonomy.json`. The formulas are saved with each version, not in this file. |
| **Interactive Graph** | **Download graph (JSON)** under the graph: the nodes, edges and legend of the rendered graph. **Download graph (HTML)**: a standalone page, with the Plotly engine. |
| **Cluster profiles (CSV)** | **Download Profiles** in **Population Clustering** > *Cluster Profiles*. |

---

## Practical Guidance
- Index relevant research papers **before** generating the taxonomy (Settings > AI engine > *Document library*, then **Initialise / reindex**) — the AI uses them to produce more accurate classifications.
- Save a version before and after major refinements to allow easy rollback.
