# 🔄 Reproduction

## Overview
The **Reproduction** module ensures the reproducibility of analysis workflows. It allows users to replay previously recorded analysis sessions ("traces"), ensuring that results can be verified and recreated exactly as they were originally generated.

## Key Features

### 1. 📜 Trace Management
*   **Session Recording**: The application automatically records key actions into JSON trace files.
*   **Supported Operations**:
    *   **Enrichment**: Merging external datasets.
    *   **Variable Computation**: Creating new variables via formulas.
    *   **Imputation**: Handling missing values (Mean, Median, MissForest, etc.).
    *   **Transformation**: Scaling, Encoding, and Dimensionality Reduction.
*   **Trace Selection**: Users can browse and select from a history of analysis sessions.

### 2. ⏯️ Replay Engine
*   **Step-by-Step Replay**: Visualizes the sequence of operations performed in a session.
*   **Modes**:
    *   **Fast Replay**: Uses intermediate snapshots (if available) to quickly jump to a specific state.
    *   **Full Replay**: Re-executes every step from the raw source dataset to ensure complete verification.
*   **Artifact Links**: Provides direct links to intermediate outputs (e.g., "Categorical Stats", "Correlation Matrix") generated during the original session.

## Usage Guide
1.  **Select Session**: Choose a trace file from the list (sorted by date).
2.  **Review Steps**: Examine the list of operations to understand the workflow.
3.  **Reproduce**: Click "Reproduce to Step X" to restore the application state to that specific point in the analysis.

## Technical Details
*   **File**: `app_pages/reproduce_analysis.py`
*   **Dependencies**: `manage.reproduction_manager`, `json`
