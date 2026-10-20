# Documents

The **Documents** page (Knowledge > Documents) enables you to analyse research PDFs using AI — generate structured summaries, chat with documents via retrieval-augmented Q&A, build knowledge graphs from extracted concepts, and check how well your dataset variables are covered in the literature.

---

## Document Selection

Add one or more PDF files in Settings > AI engine > **Document library** (*Add literature (PDF)*), then click
**Initialise / reindex**. Documents are split into chunks, embedded and indexed into your vector database for
retrieval. On the Documents page, choose the documents in the sidebar (*Choose Document(s)*) and click
**Generate Graph**; the tabs described below appear once a graph is generated or loaded (*Load Saved Analysis*).
- Supported format: `.pdf`
- Documents are stored per-user for data isolation.
- Previously uploaded documents can be reused across sessions.
- **Remove a document** (under the list of documents) deletes a PDF from your library, after a confirmation. The
  AI engine keeps its current index until it is initialised again.

### Choosing the AI provider

Settings > AI engine > **Provider and models** offers local **Ollama** (the default) and four cloud providers, which
need an API key and your explicit confirmation that data may be sent to them: Google Gemini, OpenAI, Mistral AI and
**Anthropic Claude**. Claude answers questions but has no embedding model, so a second choice appears: the
**embedding provider** that indexes your documents. Keep **Ollama** there to index the whole library on this
server; only the passages retrieved for each question are then sent to Claude. The temperature slider does not
apply to Claude.

Nothing leaves the server with an Ollama on this machine. An Ollama address elsewhere (another machine of your
network) receives the prompts, the text of your documents and the column summaries: the page shows that host.

**What a cloud provider receives about your dataset** (column renaming, the variable map, unit hints, suggested
criteria) is a summary of each column, never a row:
- **every column:** its name and type;
- **numeric columns:** the 5th, 25th, 50th, 75th and 95th percentiles, never the minimum or the maximum, which can
  single out one participant;
- **other columns:** their most frequent values with their counts, only values shared by at least 5 rows;
- **date columns:** the first and the last year.

Your descriptions and criteria, the text of the PDFs you index and your chat messages are sent as you write them.
With Google Gemini, **Document insight** sends the PDF file itself while its graph is extracted; the app then asks
Google to delete it. A cloud **embedding provider** of Claude receives the text of the PDFs you index and every
search text (your questions, column names and hints). The **Data sharing** card of Settings lists all of this and
names every recipient.

**The confirmation.** The box names the providers it confirms: with Claude and a cloud embedding provider, both.
It is yours, for this browser tab only: another tab asks again. It does not replace the approvals of your
data-protection framework, and what was already sent stays with the provider, under its own terms.

- **Untick it** to withdraw it: an engine that uses one of these providers stops at once, so nothing more is sent
  from this tab.
- **Withdraw the confirmation and stop**, at the top of the card, does the same whatever provider is selected below;
  the card says which providers the running engine sends data to.
- Your documents stay indexed: tick the box again, then **Initialise / reindex** reuses the index.

The prompts are never sent to a tracing service: the app switches LangChain's tracing (LangSmith) off at start-up,
whatever `.env` says. A **Qdrant server** (rather than the embedded Qdrant) stores the text of the indexed passages
and keeps it after the engine stops.

### Choosing the vector database

Settings > AI engine > **Vector database** offers two engines; the app recommends one for your machine and says why
(CPU cores, memory, free disk, size of your library, and whether a Qdrant server is available):

| Engine | Choose it when |
| --- | --- |
| **LanceDB** (default) | You work on one machine. Nothing to install or run; vectors stay on disk, so memory use stays low even for large libraries. |
| **Qdrant**, embedded | Your library is small to medium (up to about 20 000 chunks) and you prefer Qdrant; every vector is held in memory. |
| **Qdrant**, server | Several people use the app or the library is very large. Enter the server URL (and API key if secured); each user gets their own collection. |

With *Reuse the existing index* on, the index is reused only when it holds the documents selected: after a
document is added or removed, **Initialise / reindex** rebuilds it. Changing the engine or the embedding model needs a
new index too: turn off *Reuse the existing index* and click **Initialise / reindex**. The card shows what the current index holds (chunks, documents, embedding model, date).
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
| **Clinical Significance** | Clinical or scientific implications of the paper. |
| **Top Concepts** | The three to five most important concepts of the paper. |

Summaries are generated with the knowledge graph (**Generate Graph**), with the inferred title and study type, and
displayed in the *Document Summary* tab for quick review.

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

1. Relevant document chunks are retrieved from the vector store based on your question, from the visible documents only.
2. The LLM generates an answer grounded in the retrieved context; when the answer is not in the documents, it says so.
3. Each retrieved chunk is labelled with its source file, so the answer can name it; page numbers are not added to chat answers. For page-level citations, use the node details of the Knowledge Graph tab.

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
- Add and index all relevant papers **before** generating the taxonomy on the Variable map page (Knowledge > Variable map) — the RAG system uses them for better variable classification.
- Use the **Chat** feature for quick literature queries instead of manually searching PDFs.
- Indexed documents also ground the **AI Variable Suggestions** of Data > Enrich: a formula taken from one of them is marked *Document*, with its file and pages.
- The **Coverage Analysis** is especially useful when preparing study protocols or grant applications.
