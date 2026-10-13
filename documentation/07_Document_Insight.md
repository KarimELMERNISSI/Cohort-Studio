# Documents Insight

The **Documents Insight** page enables you to analyse research PDFs using AI — generate structured summaries, chat with documents via retrieval-augmented Q&A, build knowledge graphs from extracted concepts, and check how well your dataset variables are covered in the literature.

---

## Document Selection

Upload one or more PDF files. Documents are split into chunks, embedded and indexed into your vector database for retrieval.
- Supported format: `.pdf`
- Documents are stored per-user for data isolation.
- Previously uploaded documents can be reused across sessions.

### Choosing the AI provider

Settings > AI engine > **Provider and models** offers local **Ollama** (the default; nothing leaves the server)
and four cloud providers, which need an API key and your explicit data-sharing consent: Google Gemini, OpenAI,
Mistral AI and **Anthropic Claude**. Claude answers questions but has no embedding model, so a second choice
appears: the **embedding provider** that indexes your documents. Keep **Ollama** there to index the whole
library on this server; only the passages retrieved for each question are then sent to Claude. The temperature
slider does not apply to Claude.

### Choosing the vector database

Settings > AI engine > **Vector database** offers two engines; the app recommends one for your machine and says why
(CPU cores, memory, free disk, size of your library, and whether a Qdrant server is available):

| Engine | Choose it when |
| --- | --- |
| **LanceDB** (default) | You work on one machine. Nothing to install or run; vectors stay on disk, so memory use stays low even for large libraries. |
| **Qdrant**, embedded | Your library is small to medium (up to about 20 000 chunks) and you prefer Qdrant; every vector is held in memory. |
| **Qdrant**, server | Several people use the app or the library is very large. Enter the server URL (and API key if secured); each user gets their own collection. |

Changing the engine or the embedding model needs a new index: turn off *Reuse the existing index* and click
**Initialise / reindex**. The card shows what the current index holds (chunks, documents, embedding model, date).
Indexes from earlier versions (ChromaDB) are no longer used; your PDFs are kept and the index is rebuilt from them.

---

## Document Summariser

Generate structured summaries from each uploaded paper.

### Extracted Fields

| Field | Description |
| --- | --- |
| **Objective** | The paper's stated research goal or hypothesis. |
| **Methods** | Study design, population, and analytical methods used. |
| **Key Findings** | Main results, effect sizes, and statistical significance. |
| **Variables** | Clinical and biological variables discussed in the paper. |

Summaries are generated via RAG and displayed in a dedicated tab for quick review.

---

## Knowledge Graph from Documents

Build an interactive knowledge graph from concepts extracted across all uploaded PDFs.

### How It Works

1. **Entity Extraction** — The LLM identifies clinical concepts, variables, and relationships from each document.
2. **Concept Merging** — Similar concepts (e.g. "Heart Failure" and "Heart Failure (HF)") are unified into single nodes with aggregated citations.
3. **Graph Construction** — Nodes represent concepts; edges represent co-occurrence or semantic relationships.
4. **Citation Tracking** — Each node includes page-level citations from all documents where it appears.

### Features

- **Filter by Document** — Show/hide concepts from specific papers.
- **Node Details** — Select a node in the graph (or in the entity list) to view its definition, citations, and connected concepts; toggle *Focus* to show only its neighbourhood.
- **Interactive Layout** — Hierarchical layout rendered by the configured graph engine (offline Plotly by default; yFiles optional). Entities mentioned only inside a relation appear as flagged *Inferred* nodes.

---

## Chat with Documents

Ask questions about the uploaded documents using retrieval-augmented Q&A.

### How It Works

1. Relevant document chunks are retrieved from the vector store based on your question.
2. The LLM generates an answer grounded in the retrieved context.
3. Source citations (document name and page) are displayed with each answer.

### Use Cases

- *"What biomarkers are associated with insulin resistance in these papers?"*
- *"Which studies used a cohort design?"*
- *"Summarise the findings about HDL cholesterol across all documents."*

---

## Dataset Coverage Analysis

Check how well your dataset's variables are represented in the uploaded literature.

### Output

| Column | Coverage | Source |
| --- | --- | --- |
| `hdl_cholesterol` | Covered | Paper A (p. 3), Paper B (p. 7) |
| `homa_ir` | Covered | Paper A (p. 5) |
| `patient_id` | Not covered | — |

This helps identify:

- **Well-supported variables** — backed by literature.
- **Gaps** — variables in your dataset not discussed in any uploaded paper.
- **Opportunities** — concepts in the literature not yet captured in your dataset.

---

## Practical Guidance
- Upload all relevant papers **before** generating the taxonomy on the Data Insight page — the RAG system uses them for better variable classification.
- Use the **Chat** feature for quick literature queries instead of manually searching PDFs.
- The **Coverage Analysis** is especially useful when preparing study protocols or grant applications.
