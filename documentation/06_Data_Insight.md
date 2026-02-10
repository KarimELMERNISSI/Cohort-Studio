# 🧠 Data Insight — Taxonomy (Knowledge Graph)

The **Data Insight** page generates, visualises, and refines a **Taxonomy** (knowledge graph) from your dataset's variables. The Taxonomy captures how variables relate to clinical domains, computed formulas, and broader scientific concepts — powered by RAG and the Gemini LLM.

---

## 🌐 Taxonomy Overview

A Taxonomy is a structured graph where:

| Node Type | Description | Example |
|---|---|---|
| **Category** | Clinical domain or grouping. | Anthropometry, Lipid Panel |
| **Variable** | A dataset column. | `weight_kg`, `hdl_cholesterol` |
| **Concept** | An external-derived clinical or scientific concept. | BMI, HOMA-IR |
| **Formula** | A computable relationship between variables. | `BMI = Weight / Height²` |

Edges represent relationships such as `belongs_to`, `used_in`, and `derived_from`.

---

## ⚙️ Taxonomy Generation

### Steps

1. **Load or Create** — Start a new taxonomy from the current dataset, or load a previously saved version.
2. **AI Generation** — The RAG system analyses column names, data types, value distributions, and uploaded documentation to infer categories, relationships, and computed formulas.
3. **Review** — Inspect the generated graph in the interactive yFiles visualisation.
4. **Iterate** — Refine, enrich, or repair the taxonomy as needed.

### Generation Options

| Parameter | Description |
|---|---|
| **Context Documents** | Optional PDFs to give the LLM domain knowledge for better classification. |
| **Existing Taxonomy** | Optionally provide a prior version as a starting point. |

---

## 🔧 Taxonomy Refinement

After initial generation, several tools are available to improve the taxonomy:

### Repair

- **Fix Malformed JSON** — Automatically repairs structural issues (orphan nodes, missing edges, cycles).
- **Validate Structure** — Checks that all nodes have required fields and edges reference existing nodes.

### Enrich

- **AI Node Enrichment** — Select any node and request an AI-generated description and metadata. The RAG system provides context-aware definitions.
- **Add Missing Concept Nodes** — The AI scans the taxonomy for concepts implied by the variables but not yet present (e.g. computed indices like HOMA-IR). Newly added concepts are automatically assigned to the most appropriate existing category.

### Manual Editing

- Add, rename, or delete nodes and edges directly in the UI.

---

## 🗂️ Taxonomy Versioning

| Action | Description |
|---|---|
| **Save** | Persist the current taxonomy as a named JSON version. |
| **Load** | Restore a previously saved version. |
| **Compare** | View differences between the current taxonomy and a saved version. |

Versions are stored per-user and per-dataset for isolation.

---

## 📊 Visualisation

The taxonomy is rendered as an interactive graph using **yFiles**:

- **Node Styles** — Different colours and shapes per node type (category, variable, concept, formula).
- **Edge Labels** — Relationship type displayed on each connection.
- **Layout Options** — Hierarchical, organic, circular, and other layout algorithms.
- **Search & Filter** — Locate specific nodes or filter by type.
- **Click Details** — Click any node to view its metadata, description, and connected nodes.

---

## 📤 Export

| Format | Description |
|---|---|
| **JSON** | Full taxonomy structure for programmatic use. |
| **CSV** | Flat table of nodes and their attributes. |
| **Interactive Graph** | yFiles HTML export for sharing. |

---

## 💡 Tips

- Upload relevant research papers **before** generating the taxonomy — the RAG system uses them to produce more accurate classifications.
- Use the **Repair** function after any manual edits to ensure structural integrity.
- Save a version before and after major refinements to allow easy rollback.
