# 🧠 Data Insight

## Overview

The **Data Insight** module converts your dataset into an interactive Knowledge Graph ("Taxonomy"). It helps you understand relationships between your variables (e.g., Inputs, Derived variables, Outcomes) and uses AI to organize this information based on medical knowledge.

## Key Features

### 1. 📂 Data Management & Taxonomy Versioning (Sidebar)

* **Load Taxonomy**: Load previously saved versions of your Knowledge Graph.
* **Save/Overwrite**: Save your current graph state as a new version or overwrite the existing one.
* **AI Enrichment**: Use the RAG system to find "candidates" for new variables or relationships based on external knowledge.

### 2. ⚡ Graph Visualization

Visualize your dataset as a network:

* **Input Variables** (Blue): Raw data points.
* **Derived Variables** (Yellow): Calculated fields.
* **Formulas** (Red Hexagons): Mathematical transformations connecting inputs to outputs.
* **Categories** (Rectangles): Logical groupings of variables.

### 3. 🚀 New Taxonomy Generation

If you don't have a taxonomy yet, you can generate one from your current dataset:

1. **AI Analysis**: The system analyzes your column names and values.
2. **Structure Inference**: It attempts to categorize variables and identify apparent formulas.
3. **Result**: A fully interactive graph is generated.

## Usage Guide

1. **Login**: Ensure you are authenticated via the sidebar.
2. **Load a Dataset**: Ensure you have a dataset loaded in the **Home** page.
3. **Generate or Load**:
    * If starting fresh, click **🚀 New Taxonomy** (requires RAG initialization).
    * If you have a saved graph, select a version (e.g., `v1`) in the sidebar and click **Load**.
4. **Explore**:
    * Zoom and pan the graph widget.
    * Click on nodes to see details like their standard name and description.
5. **Save**: Once satisfied, click **Save New (vX)** to persist your work.

## Technical Details

* **File**: `app_pages/data_insight.py`
* **Storage**: Taxonomies are stored in `data/taxonomy/{username}/vX/taxonomy_metadata.json`.
* **Visualization**: Powered by **yFiles for HTML**.
