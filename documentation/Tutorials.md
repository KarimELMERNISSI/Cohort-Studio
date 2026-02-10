# 📘 Tutorials

Step-by-step guides for common workflows in the Cohort Monitoring Package.

---

## Tutorial 1: Setting Up a New Study

**Goal**: Load your dataset, standardise column names, and save an initial version.

1. **Log in** to your account (or sign up and wait for admin activation).
2. Go to the **Main View** page.
3. Click **Upload Dataset** and select your file (CSV, Excel, or Parquet).
4. Review the **Dataset Statistics** panel to check shape, types, and missing values.
5. Open the **Column Renaming** section:
   - For AI-assisted renaming, click **Generate AI Suggestions** and review the proposed names.
   - For manual renaming, edit names directly in the mapping table.
6. Click **Apply Renaming** to update the column names.
7. In the sidebar, click **Save Dataset** to create your first version.

---

## Tutorial 2: Data Quality Check & Cleaning

**Goal**: Identify and handle anomalies, outliers, and incomplete records.

1. Go to **Data Validation & Monitoring**.
2. Review the **Data Quality Score** at the top of the page.
3. In the **Clinical Anomalies** tab:
   - Review flagged values.
   - Decide whether to correct, remove, or retain them.
4. In the **Inclusion Criteria** tab:
   - Create a mask family (e.g. "Adult Patients").
   - Add masks (e.g. `Age >= 18`).
   - Apply the family to filter your cohort.
5. In the **Outlier Detection** tab:
   - Select a detection method (e.g. IQR).
   - Choose columns to analyse.
   - Select a handling strategy (e.g. Tag).
   - Review the outlier summary and visualisation.
6. Save the cleaned dataset as a new version.

---

## Tutorial 3: Imputing Missing Data

**Goal**: Fill missing values using targeted formulas and global strategies.

1. Go to **Data Enrichment** → **Handle Missing Data** tab.
2. **Targeted Imputation**:
   - Select a column with missing values.
   - Click **Generate Formula Suggestions** to get AI-powered formulas.
   - Review suggestions and click **Use Formula** on the best option.
   - Click **Apply Targeted Imputation**.
3. **Global Imputation**:
   - Review the missing data overview.
   - Select a numerical method (e.g. KNN) and a categorical method (e.g. Most Frequent).
   - Click **Apply Global Imputation**.
   - Review the imputation mask to see which values were filled.
4. Save the imputed dataset.

---

## Tutorial 4: Feature Engineering & Clustering

**Goal**: Create new variables and identify population subgroups.

1. Go to **Data Enrichment** → **Create New Variables** tab.
2. **Encoding**:
   - Select "Encoding" as transformation type.
   - Choose "One-Hot Encoding" and select categorical columns.
   - Set a naming pattern and click **Apply Transformation**.
3. **Dimensionality Reduction**:
   - Select "Dimensionality Reduction" → "PCA".
   - Choose numerical columns and set the number of components.
   - Click **Apply Transformation** to create PCA columns.
4. **Clustering**:
   - Select "Clustering" → "K-Means".
   - Choose the columns to cluster on (e.g. PCA components).
   - Set K (number of clusters).
   - Click **Apply Transformation** to create a cluster label column.
5. Visualise the clusters using the **Visualisation** page (scatter plot coloured by cluster label).

---

## Tutorial 5: Generating a Knowledge Graph (Taxonomy)

**Goal**: Build a structured taxonomy from your dataset variables.

1. Go to **Data Insight**.
2. Click **Generate New Taxonomy** (or load an existing version).
3. Optionally upload research PDFs for richer context.
4. Wait for the AI to analyse your variables and generate the graph.
5. Explore the interactive graph:
   - Click nodes to view details.
   - Use the search bar to find specific variables.
6. **Refine** the taxonomy:
   - Use **Repair** to fix structural issues.
   - Use **Enrich** to add AI descriptions to nodes.
   - Use **Add Missing Concepts** to fill in implied clinical indices.
7. **Save** the taxonomy as a named version.

---

## Tutorial 6: Analysing Research Documents

**Goal**: Extract insights from uploaded PDFs and check dataset coverage.

1. Go to **Documents Insight**.
2. **Upload PDFs** of relevant research papers.
3. In the **Summariser** tab:
   - View structured summaries (objective, methods, findings, variables) for each paper.
4. In the **Knowledge Graph** tab:
   - Explore extracted concepts and their relationships across papers.
   - Filter by document to focus on specific papers.
5. In the **Chat** tab:
   - Ask questions like *"What variables are associated with cardiovascular risk?"*
   - Review answers with source citations.
6. In the **Coverage** tab:
   - Check which of your dataset columns are discussed in the literature.
   - Identify gaps and opportunities.

---

## Tutorial 7: Running Statistical Analysis

**Goal**: Perform hypothesis testing and power analysis.

1. Go to **Epidemiology & Hypothesis**.
2. **Hypothesis Testing**:
   - Select an outcome variable and a comparison variable.
   - The system auto-recommends an appropriate test.
   - Review results: test statistic, p-value, effect size, and interpretation.
3. **ANCOVA** (if needed):
   - Select outcome, group variable, and covariates.
   - Review adjusted means and covariate effects.
4. **Power Analysis**:
   - Enter the expected effect size and desired power.
   - View the required sample size.
   - Inspect the power curve.
5. **Z-Score Standardisation**:
   - Select a variable and provide reference values (or use sample-based).
   - The Z-score column is appended to your dataset.

---

## Tutorial 8: Creating Publication-Ready Figures

**Goal**: Generate polished visualisations with statistical annotations.

1. Go to **Visualisation**.
2. Select a chart type (e.g. Box Plot).
3. Choose the outcome variable and a grouping variable.
4. The system automatically applies the appropriate statistical test and overlays significance.
5. Customise:
   - Set a title, axis labels, and colour palette.
   - Adjust font sizes for publication requirements.
6. **Export**:
   - Click the camera icon in the Plotly toolbar for PNG/SVG.
   - Use the HTML export for interactive sharing.

---

## Tutorial 9: Reproducing an Analysis Session

**Goal**: Replay a past analysis session and generate a report.

1. Go to **Reproduce Analysis**.
2. Browse available traces in the history panel.
3. Select a trace to inspect its steps.
4. Choose a replay mode:
   - **Fast Replay** — Re-applies parameters only (requires original dataset).
   - **Full Replay** — Uses embedded snapshots (works independently).
5. Click **Replay** and monitor progress step by step.
6. Click **Download Report** to generate a `.docx` summary of the session.
