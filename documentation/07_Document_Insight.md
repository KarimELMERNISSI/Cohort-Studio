# Documents Insight

The **Documents Insight** page enables you to analyse research PDFs using AI — generate structured summaries, chat with documents via retrieval-augmented Q&A, build knowledge graphs from extracted concepts, and check how well your dataset variables are covered in the literature.

---

## Document Selection

Upload one or more PDF files. Documents are processed and indexed into the RAG vector store (ChromaDB) for retrieval.
- Supported format: `.pdf`
- Documents are stored per-user for data isolation.
- Previously uploaded documents can be reused across sessions.

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
- **Node Details** — Click any node to view its definition, citations, and connected concepts.
- **Interactive Layout** — Powered by yFiles with multiple layout options.

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
