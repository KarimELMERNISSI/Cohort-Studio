# 🎓 Tutorials

This section provides step-by-step guides for common workflows in the **Integrated Research Environment**.

---

## 1. Setting Up a Study (Validation First)

**Goal**: Load a raw dataset, define who belongs in the study, and set up your quality rules.

1. **Launch & Load**:
    * Run `streamlit run main.py`.
    * Go to **Main View** (Home).
    * Use the sidebar to **Upload File** (CSV/Excel).
2. **Standardize Names** (Optional):
    * Go to **Main View > Tab 2: Columns Renaming**.
    * Use **AI Suggest Names** to map cryptic codes (e.g., `bp_1`) to standard terms (`Systolic BP`).
    * Click **Apply**.
3. **Define Rules**:
    * Go to **Data Validation & Monitoring** (Sidebar).
    * **Tab 2: Clinical Anomalies**: Define impossible values (e.g., `Age > 120`).
    * **Tab 3: Inclusion Criteria**: Define your study population (e.g., `Status == 'Enrolled'`).
    * *Tip: Use the interactive editor to add rules without writing code.*
4. **Check Health**:
    * Go to **Tab 1: Data Quality Dashboard**.
    * Review your **Health Score** and address any red flags (e.g., High Missingness).
5. **Save Baseline**:
    * In the sidebar ("Dataset Management"), click **Save Version** (e.g., `v1_raw_validated`).

---

## 2. Enrichment & Imputation strategy

**Goal**: Handle missing data and create analytical variables.

1. **Go to Data Enrichment**: Select **Data Enrichment** from the sidebar.
2. **Targeted Imputation (Precision)**:
    * Go to **Tab 1: Handle Missing Data > Sub-tab A: Targeted Imputation**.
    * Select a variable (e.g., `BMI`).
    * Enter a formula (e.g., `Weight / (Height/100)**2`) or ask the **AI Assistant** ("How do I calculate BMI?").
    * Click **Apply Formula** to fill missing values just for that column.
3. **Global Imputation (Bulk)**:
    * Go to **Sub-tab B: Global Imputation**.
    * Select **Numerical Strategy**: `MICE` (Best for accuracy) or `MissForest` (Best for non-linear).
    * Click **Run Imputation**.
4. **Create New Variables**:
    * Go to **Tab 3: Feature Engineering**.
    * Select **Transformation Type**: `Computed Column`.
    * Enter your formula.
5. **Save Enriched Version**: Save as `v2_enriched`.

---

## 3. Generating a Knowledge Graph (Data Insight)

**Goal**: Visualize the relationships between your variables automatically.

1. **Load Data**: Ensure your dataset is loaded.
2. **Go to Data Insight**: Select **Data Insight** from the sidebar.
3. **Generate**:
    * Click **🚀 New Taxonomy**.
    * The system analyzes column names and detects relationships (Formula links, Categories).
4. **Explore**:
    * Blue nodes = Inputs. Yellow nodes = Derived variables.
    * Click a node to see its metadata.
5. **Save**: In the sidebar, save as `v1_graph`.

---

## 4. Analyzing Research Documents (RAG)

**Goal**: Extract concepts from a PDF and check if your dataset covers them.

1. **Go to Documents Insight**: Select **Documents Insight** from the sidebar.
2. **Select Documents**: Check the PDFs you want to analyze in the sidebar.
3. **Generate Graph**: Click **✨ Generate Graph**.
4. **Check Coverage** (Tab 3):
    * Click **Checking Coverage**.
    * Green checks ✅ mean your dataset contains variables that match the concepts in the paper.
5. **Chat** (Tab 4):
    * Ask: *"What are the inclusion criteria mentioned in these papers?"*

---

## 5. Running a Statistical Analysis

**Goal**: Compare two groups (e.g., Treatment vs Control).

1. **Go to Epidemiology**: Select **Epidemiology & Hypothesis** from the sidebar.
2. **Configure Analysis** (Sidebar):
    * **Test Type**: Select "Parametric" or "Non-Parametric".
    * **Correction**: Select "Bonferroni" if you are testing many variables.
3. **Define Groups**:
    * **Group A Filter**: `Treatment == 1`.
    * **Group B Filter**: `Treatment == 0`.
4. **Run Tests**:
    * Select variables (e.g., `Age`, `Outcome`).
    * Click **Run Analysis**.
5. **Check Power** (Tab 2):
    * Input your sample size and observed effect size to see if the result is robust.
