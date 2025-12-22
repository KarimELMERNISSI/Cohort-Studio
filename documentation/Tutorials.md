# 🎓 Tutorials

This section provides step-by-step guides for common workflows in the Cohort Monitoring Package.

---

## 1. Importing and Cleaning a New Dataset

**Goal**: Load a raw CSV/Excel file, standardize column names, and save it as "Version 1".

1. **Launch the App**: Run `streamlit run main.py`.
2. **Go to Home**: In the sidebar, select **Main View**.
3. **Upload**:
    * Look for the **Data Source** section in the sidebar.
    * Choose **Upload File** and select your dataset (CSV or Excel).
4. **Inspect**:
    * The "Dataset Statistics" tab will show you an initial overview.
    * Look for any columns with strange names or obvious errors.
5. **Rename Columns** (Optional but Recommended):
    * Switch to the **Columns Renaming** tab.
    * If the AI is active, click **Suggest Names** to get standardized medical names.
    * Otherwise, manually double-click cells in the "Rename to" column to edit them.
    * Click **Apply Renames** to update the dataset in memory.
6. **Save Version**:
    * In the sidebar, under "Dataset Management", click **Save Version**.
    * This creates a checkpoint (e.g., `v1`) that you can reload anytime.

---

## 2. Generating a Knowledge Graph (Data Insight)

**Goal**: Visualize the relationships between your variables automatically.

1. **Load Data**: Ensure your dataset is loaded (see Tutorial 1).
2. **Go to Data Insight**: Select **Data Insight** from the sidebar.
3. **Generate**:
    * If the screen is empty, look for the **✨ Generate New** column or button.
    * Click **🚀 Generate Taxonomy**.
    * *Note: This requires the RAG/AI system to be initialized.*
4. **Review**:
    * Wait for the progress bar to complete.
    * Explore the graph. Blue nodes are your inputs, yellow are derived.
5. **Save**:
    * In the sidebar, click **💾 Save New (v1)** to store this graph layout.

---

## 3. Analyzing Research Documents

**Goal**: Extract concepts from a PDF and check if your dataset covers them.

1. **Go to Documents Insight**: Select **Documents Insight** from the sidebar.
2. **Select Documents**:
    * In the sidebar, check the boxes for the PDFs you want to analyze.
    * *If no docs appear, ensure you have placed PDFs in the `DOCUMENTS` folder and the system has indexed them.*
3. **Generate Graph**: Click **✨ Generate Graph**.
4. **Check Coverage**:
    * Once the graph appears, switch to the **📊 Dataset Coverage** tab.
    * Click **🔍 Check Coverage**.
    * Review the list. Green checks ✅ mean your dataset contains variables that match the concepts in the paper.
5. **Chat**:
    * Switch to the **💬 Chat** tab.
    * Ask: *"What are the inclusion criteria mentioned in these papers?"*

---

## 4. Running a Statistical Report

**Goal**: Compare two groups (e.g., Treatment vs Control) and export the results.

1. **Go to Epidemiology**: Select **Epidemiology & Hypothesis** from the sidebar.
2. **Define Groups**:
    * Select **Hypothesis Testing**.
    * **Group A**: Define a filter, e.g., `Treatment == 1`.
    * **Group B**: Define a filter, e.g., `Treatment == 0`.
3. **Select Variables**: Choose the variables you want to compare (e.g., `Age`, `BMI`, `Outcome`).
4. **Run Tests**:
    * Click **Run Analysis**.
    * The system will automatically select appropriate tests (T-test or Mann-Whitney) based on normality.
5. **Export**:
    * View the results table.
    * Click the **Download Excel** button to save the full statistical report.
