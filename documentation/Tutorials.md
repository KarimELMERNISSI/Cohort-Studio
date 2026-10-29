# Tutorials

Step-by-step guides for common workflows in CohortStudio.

> **New here?** Start with the chapter **Full tutorial: from a raw file to a reproducible report**. It uses the
> tutorial cohort and its study configuration (Start > Tutorial cohort), and every step says what you should see.

---

## Tutorial 1: Setting Up a New Study

**Goal**: Load your dataset, standardise column names, and save an initial version.

1. **Log in** to your account (or sign up and wait for admin activation).
2. Go to **Data > Import** (or load the sample cohort from the Start page).
3. Under **1. Choose a file**, select your file (CSV, Excel, or Parquet). Under **2. Check how the file is read**, check the rows, columns, malformed lines and column types. Under **3. Name and load**, click **Load dataset** (it is also saved as a version in your workspace by default).
4. Go to **Explore > Statistics & correlations** and review the **Dataset Statistics** tab to check shape, types, and missing values.
5. Open the **Columns Renaming** tab:
- For AI-assisted renaming, open **AI Renaming Suggestions (RAG)**, click **Generate Suggestions**, and click **Apply** next to each proposed name you accept.
- For manual renaming, edit names directly in the column list.
6. Click **Apply Renaming** to update the column names.
7. In the sidebar dataset card, click **Save** to save a new version.

---

## Tutorial 2: Data Quality Check & Cleaning

**Goal**: Identify implausible values, outliers, and incomplete records, and define the analysis population.

Inclusion and exclusion criteria are filters: ticking them in the sidebar defines the analysis population and
deletes no row.

1. Go to **Data > Validate**.
2. In the **Data Quality Dashboard** tab, review the **Global Quality Score**.
3. In the **Handle Implausible Values** tab:
- Add plausibility rules under **New Plausibility Rule** (e.g. a plausible range for a column).
- Click **Run the Plausibility Check** and review the flagged rows.
- Click **Exclude the rows with implausible values from the analysis population** to tick the plausibility rules in
  the sidebar.
4. In the **Handle Inclusion Criteria** tab:
- Add criteria under **New Inclusion Criterion** (e.g. `df['age'] >= 18`).
- Click **Run Inclusion Criteria Check** and review the rows that meet them.
- Click **Restrict the analysis population to these criteria** to tick them in the sidebar.
5. In the **Handle Outliers** tab:
- Select a detection method (e.g. iqr).
- Choose columns to analyse.
- Select a handling strategy (e.g. tag).
- Review the outlier summary and visualisation, then click **Exclude these N outlier rows from the analysis population** (with the *clip* strategy, the button clips the values in the dataset instead).
6. Review the **Analysis population** tab (participant flow, excluded rows and their reasons), then click **Save** in the sidebar dataset card: the version keeps every row and the ticked criteria. To remove the excluded rows from the data, use **Keep only these rows** on that tab.

---

## Tutorial 3: Imputing Missing Data

**Goal**: Fill missing values, first by an exact formula where other variables determine them (targeted imputation), then by statistical imputation.

1. Go to **Data > Enrich** → **Handle Missing Data** tab.
2. **Targeted Imputation** (*Targeted Imputation (Custom Formula)* tab):
- Select a column with missing values.
- Type the formula that determines it from other columns (e.g. `bmi * (height_cm / 100) ** 2` for a missing weight), or:
- Under **AI Imputation Suggestions (RAG)** (with an initialised AI engine), click **Generate Formula Suggestions** to get AI-powered formulas.
- Review suggestions and click **Use Formula #N** on the best option.
- Tick **Exact formula** when the formula gives the true value (the values then count as observed), then click **Apply Targeted Imputation**.
3. **Global Imputation** (*Global Imputation (single imputation)* tab):
- Review the missing data overview.
- Select a numerical method (e.g. K-nearest neighbours) and a categorical method (e.g. Most frequent value).
- Click **Apply Imputer**.
- Review the preview: the imputed values are highlighted.
- Single imputation makes p-values and confidence intervals too narrow; for inference, prefer complete-case analysis with a sensitivity check, or the **Multiple imputation (for inference)** tab.
4. The imputations go into the Enrich draft: click **Apply to the dataset** in the *Enrich draft* panel, then **Save** in the sidebar dataset card.

---

## Tutorial 4: Feature Engineering & Clustering

**Goal**: Create new variables and identify population subgroups.

1. Go to **Data > Enrich** → **Create New Variables** tab.
2. **Formula** (*Variable Computation*):
- Enter a name and type the formula with the column names (e.g. `weight_kg / (height_cm / 100) ** 2`); click a column under **Column Suggestions** to complete a name.
- With the AI engine on, open **AI Variable Suggestions (RAG)**: search for a concept (e.g. *eGFR*) or leave **Specific Search** off to scan the dataset, click **Generate AI Suggestions**, then **Apply to Editor** on the suggestion you keep.
- Click **Compute & Preview**, read the statistics of the new variable, then **Confirm & Add Variable to Dataset**.
3. **Encoding**:
- Select "Encoding" as transformation type.
- Choose "One-Hot Encoding" and select categorical columns.
- Set a naming pattern and click **Apply Transformation**.
4. **Dimensionality Reduction**:
- Select "Dimensionality Reduction" → "PCA" (PCA needs the optional `ml` dependencies).
- Choose numerical columns and set the number of components.
- Click **Apply Transformation** to create PCA columns.
5. **Clustering**:
- Select "Clustering" → "K-Means".
- Choose the columns to cluster on (e.g. PCA components).
- Set K (number of clusters).
- Click **Apply Transformation** to create a cluster label column.
6. Click **Apply to the dataset** in the *Enrich draft* panel, then visualise the clusters on **Explore > Visualize** (scatter plot grouped by the cluster label).

---

## Tutorial 5: Generating a Knowledge Graph (Taxonomy)

**Goal**: Build a structured taxonomy from your dataset variables.

1. Optionally add research PDFs in Settings > AI engine > *Document library* for richer context, and click **Initialise / reindex** (generation needs an initialised AI engine).
2. Go to **Knowledge > Variable map**.
3. Click **Generate Taxonomy** (or load a saved version from the sidebar).
4. Wait for the AI to analyse your variables and generate the graph.
5. Explore the interactive graph:
- Select nodes to view their details below the graph.
- Use **Search Variables** in the sidebar to find specific variables.
6. **Refine** the taxonomy:
- Use **Enrich Taxonomy** (sidebar, *Graph Actions*) to find related external variables and formulas, then review them in the **Overview** tab and click **Merge Selected**.
- Use **Refine Existing Variable (AI)** (**Refinement** tab) to correct definitions with your feedback.
- Use **Add Missing Concept Node (AI)** (**Refinement** tab) to add implied clinical indices such as HOMA-IR.
7. Click **Save New (vN)** in the sidebar to save the taxonomy as the next numbered version.

---

## Tutorial 6: Analysing Research Documents

**Goal**: Extract insights from uploaded PDFs and check dataset coverage.

1. Add PDFs of relevant research papers in Settings > AI engine > *Document library* and click **Initialise / reindex**.
2. Go to **Knowledge > Documents**, choose the documents in the sidebar and click **Generate Graph**.
3. In the **Document Summary** tab:
- View structured summaries (objective, methods, key findings, clinical significance, top concepts) for each paper.
4. In the **Knowledge Graph** tab:
- Explore extracted concepts and their relationships across papers.
- Filter by document (*Filter Visible Documents* in the sidebar) to focus on specific papers.
5. In the **Chat with Document** tab:
- Ask questions like *"What variables are associated with cardiovascular risk?"*
- Review the answers: they are based on the visible documents only.
6. In the **Dataset Coverage** tab:
- Click **Check Coverage** to see which of your dataset columns are discussed in the literature.
- Identify gaps and opportunities.

---

## Tutorial 7: Running Statistical Analysis

**Goal**: Perform hypothesis testing and power analysis.

1. Go to **Analyze > Hypothesis tests & epidemiology**.
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
5. **Z-Score Standardisation** (*Z-Score & Reference* tab):
- Select variables and a reference (a control group, the whole cohort, or reference values you enter), then click **Calculate Z-Scores**.
- Download the data with the Z-score columns (**Download Z-Score Data**); the dataset itself is not changed.

---

## Tutorial 8: Creating Publication-Ready Figures

**Goal**: Generate polished visualisations with statistical annotations.

1. Go to **Explore > Visualize**.
2. In the sidebar, select a plot category and a plot type (e.g. Box Plot).
3. Choose the numeric variables and a grouping variable (*Group by*).
4. To add significance annotations, tick **Show Statistical Significance** (vertical box or violin plots with a grouping variable). The test is chosen automatically unless you force a parametric or non-parametric test; choose a multiple-testing correction if needed.
5. Customise:
- Under **Customize Plot**, set the title, its font size, the grid and the figure size.
- Choose the theme and colour scheme under **Visual Settings** in the sidebar.
6. **Export**:
- Click the camera icon in the Plotly toolbar for an SVG image.
- Under **Figure Export & Publication Options**, click **Interactive HTML (.html)** for an interactive file that opens offline.

---

## Tutorial 9: Reproducing an Analysis Session

**Goal**: Replay a past analysis session and generate a report.

1. Go to **Reproduce > Versions & traces**.
2. Choose a trace under **Select Analysis Session** (or upload one with **Upload Trace File (.json)**).
3. Inspect its steps and its integrity status.
4. Choose a replay mode:
   - **Fast (Use Snapshots)** — Loads the snapshot saved with the step (works without the original dataset).
   - **Full Replay (From Source)** — Recomputes the steps from the source file and checks each one against its recorded fingerprint.
5. Click **Reproduce to Step N** on the step to reach, and monitor progress.
6. Click **Download Transformation Report (.docx)** to generate a Word summary of the session.

---

## Tutorial 10: Managing Users (Admin Only)

**Goal**: Activate new accounts and manage user access.

1. Log in with an **Administrator** account.
2. Go to **Admin > Users**.
3. **Activate a New User**:
- Find the user with "Pending" status in the list.
- Click the **Activate** button.
- The user status changes to "Active" and they can now log in.
4. **Deactivate a User**:
- Click **Deactivate** to temporarily revoke access.
5. **Reset Password**:
- Under **Reset User Password**, select the user, type the new password twice (at least 12 characters by default) and click **Reset Password**. Give the new password to the user.
6. **Delete User**:
- Click **Delete**, then **Yes, Delete**. Only the account is removed: the user's workspace (datasets, traces, documents) stays on the server.

Administrator accounts cannot be deactivated or deleted from this page.

---

## Tutorial 11: Monitoring RAG Quality (Admin Only)

**Goal**: Check the health and performance of the AI documentation assistant.

1. Go to **Admin > AI quality**.
2. **Quality Overview**:
- Check the average context relevance, faithfulness and answer relevance (0-10, scored by the AI engine acting as judge) and the JSON success rate.
- A warning appears when the hallucination rate exceeds 10%.
3. **Evaluation**:
- Go to the **Run Evaluation** tab (the AI engine must be initialised in Settings > AI engine).
- Under *Quick Evaluation*, select the function to test, enter a test question (e.g. "How is BMI calculated?") and click **Run Evaluation**.
- The system retrieves context, generates an answer and scores it automatically.
- Add test cases in the **Embedding Health** tab and run them with **Run Test Suite**.
4. **Review Metrics**:
- Check the scores in **Quality Overview** and **Evaluation Details** to see if the RAG system needs re-indexing or prompt tuning. What the index holds (chunks, documents, embedding model) is shown in Settings > AI engine > *Vector database*.
