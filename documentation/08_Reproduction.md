# 🔁 Reproduce Analysis

## Overview

The **Reproduction** module ensures the reproducibility of analysis workflows. It allows users to replay previously recorded analysis sessions ("traces"), ensuring that results can be verified and recreated exactly as they were originally generated.

## Key Features

### 1. 📜 Trace Management

* **Session Recording**: The application automatically records key actions into JSON trace files.
* **Supported Operations**:
  * **Enrichment**: Merging external datasets.
  * **Variable Computation**: Creating new variables via formulas.
  * **Imputation**: Handling missing values (Mean, Median, MissForest, etc.).
  * **Transformation**: Scaling, Encoding, and Dimensionality Reduction.
* **Trace Selection**: Users can browse and select from a history of analysis sessions.

### 2. ⏯️ Replay Engine

* **Step-by-Step Replay**: Visualizes the sequence of operations performed in a session.
* **Modes**:
  * **Fast Replay**: Uses intermediate snapshots (if available) to quickly jump to a specific state.
  * **Full Replay**: Re-executes every step from the raw source dataset to ensure complete verification.
* **Artifact Links**: Provides direct links to intermediate outputs (e.g., "Categorical Stats", "Correlation Matrix") generated during the original session.
* **Portable Reproduction (Path Resolver)**: Automatically adapts file paths (e.g., `C:/Users/Bob/Data.csv` -> `/home/Alice/Data.csv`) allowing traces to run across different machines and OS (Windows/Linux/Docker).

### 3. 📄 Automated Reporting

* **One-Click Generation**: Create a human-readable Word document (`.docx`) from any analysis trace.
* **Comprehensive Content**:
  * **Session Context**: ID, Source, Date/Time.
  * **Step Breakdown**: Readable descriptions of every operation.
  * **Parameter Details**: Exact values used for imputation, transformations, etc.
* **No Confidentiality Footer**: Clean output suitable for diverse use cases.

## Usage Guide

1. **Select Session**: Choose a trace file from the list (sorted by date).
2. **Review Steps**: Examine the list of operations to understand the workflow.
3. **Reproduce**: Click "Reproduce to Step X" to restore the application state to that specific point in the analysis.
4. **Generate Report**: Click the **📄 Download Transformation Report (.docx)** button at the top of the session view to save a documentation file.

## Technical Details

* **File**: `app_pages/reproduce_analysis.py`
* **Dependencies**: `manage.reproduction_manager`, `json`, `python-docx`
* **Helper Class**: `manage.trace_documenter.TraceDocumenter`
