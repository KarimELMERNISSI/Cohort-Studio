# 📄 Documents Insight

## Overview

The **Documents Insight** module allows you to "talk" to your research papers and visualize their content. It extracts concepts from PDFs and builds a merged Knowledge Graph, showing how different documents cite the same or related topics.

## Key Features

### 1. 📂 Document Selection

* **Load Saved Analysis**: Retrieve a previously generated graph.
* **Select Documents**: Choose one or multiple PDF documents (uploaded via the RAG Manager) to analyze.

### 2. 🕸️ Knowledge Graph

* **Merged View**: See a unified graph where nodes represent Concepts (e.g., "Heart Failure") and edges represent relationships found in the text.
* **Traceability**: Click any node to see exactly which documents cite it, including page numbers and source text snippets.
* **Filtering**: Toggle text-based filtering to focus on specific documents within the large graph.

### 3. 💬 Chat with Documents

* **Context-Aware Chat**: Ask questions like "What does the dataset say about exclusion criteria?". The AI answers using *only* the currently visible/selected documents.

### 4. 📊 Dataset Coverage (Tab 3)

* **Gap Analysis**: Automatically compares your current dataset variables against the concepts found in the documents.
* **Coverage Score**: See what percentage of your variables are supported by literature.
* **Match Details**: Review which variable maps to which concept and the confidence score of the match.

## Usage Guide

1. **Login**: Ensure you are authenticated via the sidebar.
2. **Select Documents**: Use the sidebar multiselect to choose the PDFs you want to analyze.
3. **Generate Graph**: Click **✨ Generate Graph** to extract concepts. This may take a minute.
4. **Explore**:
    * **Graph Tab**: Visualize connections. Click nodes to see citations in the right panel.
    * **Formulas Tab**: View any explicit mathematical formulas extracted from the text.
    * **Coverage Tab**: Click **Checking Coverage** to see if your dataset aligns with the papers.
    * **Chat Tab**: Ask free-text questions about the selected documents.
5. **Save**: Give your analysis a name (e.g., "Review 2024") and save it for later quick access.

## Technical Details

* **File**: `app_pages/document_insight.py`
* **Storage**: Graphs are saved in `data/knowledge_graphs/{username}_Graph.json`.
* **AI**: Uses LLMs to extract entities and relations.
