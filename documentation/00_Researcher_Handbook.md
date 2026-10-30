# Researcher Handbook

Welcome to **CohortStudio**. This handbook walks you through the application's philosophy and the recommended workflow for cohort data analysis — from raw data ingestion to publication-ready outputs.

---

## Philosophy

This application is built around three principles:

1. **Transparency** — Every transformation is logged. You can trace exactly how the data changed.
2. **Reproducibility** — Analysis sessions can be replayed end-to-end from trace files.
3. **AI-Augmented, Human-Controlled** — AI suggests; you decide.

---

## Core path and optional paths

The **Start** page shows the workflow as a diagram. Three core steps take you from a raw file to results:
**Load** (Data > Import, a saved version or the sample cohort), **Validate & clean** (Data > Validate checks the
data and chooses the rows; Data > Clean fixes how values are written: its *Categories* tab recodes a variable's values
written several ways into a new column, its *Dates* tab reads text columns as dates; the card links to both) and
**Explore & analyse** (Explore and
Analyze). Optional paths branch off whenever they help:

| Optional path | When | What it adds | Page |
| --- | --- | --- | --- |
| Enrich the cohort | after validation, as often as needed | merge external sources, derive variables (formulas or AI suggestions), impute missing values; enrichment is incremental, so validate and analyse again after each change and come back for the next one (two-headed arrows on the diagram) | Data > Enrich; the card's **Derived variables** link opens Handle Missing Data > Derived variables |
| Map your variables to your field | any time (AI engine) | a taxonomy of your variables grounded in your clinical domain and documents: definitions, units, relationships, formulas | Knowledge > Variable map |
| Learn from your documents | any time (AI engine) | index literature, protocols or data dictionaries; chat with them; extract a knowledge graph that informs the variable map and enrichment suggestions | Knowledge > Documents |
| Report and reproduce | at the end, or any time | saved versions, the trace of every step, step-by-step replay and the Word report | Reproduce > Versions & traces |

The AI engine is optional and local by default (Settings > AI engine; a server can preselect a cloud provider,
which receives none of your data until you confirm it); without it, every core step and the enrichment path still
work.

Your choices in Settings are kept for your next session, in your workspace on the server (`preferences.json`): the AI
provider and its models, the provider that embeds your documents for Claude, the vector database, the graph engine
and the day-first reading of dates. Not kept: API keys, the confirmation that data may leave the server, and the
addresses of an Ollama or Qdrant server, which decide where your data go; you enter and confirm them again in each
session. A graph engine kept from the last session is used again: the yFiles engine then connects to yworks.com
again. A choice this server no longer offers (a model its provider stopped listing, say) is not used, and Settings
says which. To change a choice, choose again; to start from the defaults, delete `preferences.json` from your
workspace folder.

### How the diagram follows your progress

The diagram on the Start page, and the step cards under it, show where you are with the dataset currently loaded:

| Mark | Meaning | Evidence |
| --- | --- | --- |
| Green tick | done on the current data | Load: a dataset is loaded. Validate: the quality dashboard was computed (Data > Validate). Explore & analyse: descriptive statistics (Explore) or a test or model run (Analyze). Report & reproduce: a version saved or the Word report of this session's trace downloaded. Enrich: at least one change applied. Documents: documents indexed in your library, with the AI engine on (without it, the box shows *AI off*). Variable map: a variable map saved in your workspace. |
| Amber circular arrow ("run again") | done, but the data changed afterwards | Every change to the data (merge, computed or transformed variable, imputation, renamed column, date conversion, kept population) is a step of the trace. A step done before the latest change is out of date, and so is an analysis made on another analysis population. |
| Pulsing ring, "suggested next" | the first core step (Validate, Explore & analyse, Report) that is not done on the current data | |

This is the enrichment loop in practice: after you derive or impute a variable, Validate and Explore & analyse turn
amber and Validate becomes the suggested next step; check the quality, run the analyses again, then save a new version.
Hover a box to read its status in a sentence. Loading another dataset starts a new trace, and the diagram starts
afresh.

## Workflow Phases

The application is organised into seven phases, mirroring the natural flow of cohort research. Pages are grouped in the navigation bar at the top (Data, Explore, Analyze, Knowledge, Reproduce, Help, Admin); the sidebar always shows the current dataset (name, version, size) with **Change** and **Save version**, the AI engine status and a link to this handbook. The **Start** page lists the three steps, your saved versions and a bundled **sample cohort** (500 synthetic patients) to try everything before loading your own data.

### Phase 1 — Ingestion & Setup

**Pages**: Data > Import (load a file), then Explore > Statistics & correlations (overview, renaming)

| Step | Action |
| --- | --- |
| 1.1 | **Sign up / Log in** to your account. |
| 1.2 | **Import your dataset** (CSV/TSV, Excel, or Parquet): the import flow proposes the encoding, delimiter and decimal separator, shows a preview and a parse report, and lets you override every option. |
| 1.3 | **Rename columns** manually or with AI assistance for standardisation. |
| 1.4 | **Save an initial version** to create a baseline snapshot. |

See: [01_Home.md](01_Home.md)

### Phase 2 — Quality Control

**Page**: Data > Validate

| Step | Action |
| --- | --- |
| 2.1 | Review the **Data Quality Score** dashboard. |
| 2.2 | **Diagnose missing data** patterns (Completeness) using visual and statistical tools. |
| 2.3 | Check for **implausible values** (impossible or very unlikely values, by the plausibility rules you declare). |
| 2.4 | Define **inclusion criteria**, then tick the criteria that define the **analysis population** (sidebar): they filter every analysis without deleting rows. |
| 2.5 | Detect and handle **outliers** using statistical or ML methods. |
| 2.6 | **Check conformity** (duplicates, consistency, uniformity). |
| 2.7 | **Compare** the cleaned dataset against the original. |

See: [02_Data_Validation.md](02_Data_Validation.md)

### Phase 3 — Enrichment & Preparation

**Page**: Data > Enrich

| Step | Action |
| --- | --- |
| 3.1 | **Import external data** and merge with your main dataset. |
| 3.2 | **Impute missing values**: first impute by an exact formula what other variables determine (targeted imputation), then impute the rest statistically (single imputation for exploration, multiple imputation for inference). |
| 3.3 | **Compute new variables from formulas** on the column names: the editor offers the matching columns and previews the result before you add it. |
| 3.4 | Start from the **AI variable suggestions** (AI engine): indices and scores proposed from your columns, each with its source, to review in the editor. |
| 3.5 | **Transform variables** (encoding, scaling) and apply **dimensionality reduction** (PCA, FAMD, t-SNE, UMAP). |
| 3.6 | Run **population clustering** (K-Means, DBSCAN, GMM). |

See: [03_Data_Enrichment.md](03_Data_Enrichment.md), [09_Clustering.md](09_Clustering.md)

### Phase 4 — Exploration & Visualisation

**Page**: Explore > Visualize

| Step | Action |
| --- | --- |
| 4.1 | Explore **distributions** (histograms, box plots, violin plots). |
| 4.2 | Investigate **relationships** (scatter plots, correlation matrices). |
| 4.3 | Draw **ROC curves** (AUC, Youden threshold). Bland-Altman and Kaplan-Meier plots are on Analyze (phase 5). |
| 4.4 | Use **integrated statistical tests** for quick significance checks. |
| 4.5 | **Export** publication-ready figures. |

See: [04_Visualization.md](04_Visualization.md)

### Phase 5 — Statistical Analysis

**Pages**: Analyze > Which test?, Analyze > Hypothesis tests & epidemiology, Analyze > Survival & regression

| Step | Action |
| --- | --- |
| 5.1 | Answer the four questions of **Which test?** to get the recommended analysis and where to run it. |
| 5.2 | Select and run **hypothesis tests** (auto-recommended). |
| 5.3 | Interpret **effect sizes** and confidence intervals. |
| 5.4 | Run **multivariate analysis** (ANCOVA) with covariates. |
| 5.5 | Assess **diagnostic accuracy and agreement** (sensitivity, specificity, likelihood ratios, Bland-Altman, kappa, ICC). |
| 5.6 | Perform **power & sample size** calculations. |
| 5.7 | Compute **Z-scores** for standardised comparisons. |
| 5.8 | Fit **survival and regression models** on Survival & regression (Kaplan-Meier, log-rank, Cox, logistic, Poisson / negative binomial). |

See: [05_Epidemiology.md](05_Epidemiology.md)

### Phase 6 — Knowledge & Literature

**Pages**: Knowledge > Variable map, Knowledge > Documents

| Step | Action |
| --- | --- |
| 6.1 | **Generate a Taxonomy** (knowledge graph) from your dataset variables. |
| 6.2 | **Refine** the taxonomy with AI enrichment and concept nodes. |
| 6.3 | **Upload research papers** and generate structured summaries. |
| 6.4 | **Chat** with documents for targeted literature queries. |
| 6.5 | Run **dataset coverage analysis** against uploaded papers. |

See: [06_Data_Insight.md](06_Data_Insight.md), [07_Document_Insight.md](07_Document_Insight.md)

### Phase 7 — Reproducibility & Reporting

**Pages**: Reproduce > Versions & traces, Reproduce > Study dossier

| Step | Action |
| --- | --- |
| 7.1 | Review **trace history** of your analysis sessions. |
| 7.2 | **Replay** a trace to reproduce results (Fast or Full mode). |
| 7.3 | **Generate reports** (`.docx`) from trace files. |
| 7.4 | **Share** the reports with collaborators. Traces are saved automatically in your workspace (`traces` folder); the app has no trace download, but a trace file can be uploaded on Versions & traces to replay it. |
| 7.5 | Build the **study dossier**: add results with **Add to the study dossier** on the analysis pages, then download the Word dossier (participant flow, data handling, missing data, results, references). |

See: [08_Reproduction.md](08_Reproduction.md)

---

## System Administration

Administrative features for managing users and monitoring system health.

### User Management

**Page**: Admin > Users — *visible only to administrators*

| Step | Action |
| --- | --- |
| A.1 | **View Users** list and their current status (Pending, Active, Deactivated). |
| A.2 | **Activate** new accounts to grant access. |
| A.3 | **Deactivate** or **Delete** users as needed. |
| A.4 | **Reset Passwords** for users who cannot log in. |

The page manages the accounts of the identity provider the server signs users in with (`CMP_IAM_PROVIDER`): the
local accounts, or Better Auth through its admin API. When sign-in is disabled, there are no accounts to manage.

See: [10_Developer_Guide.md](10_Developer_Guide.md)

### RAG Quality Monitoring

**Page**: Admin > AI quality — *optional*

| Step | Action |
| --- | --- |
| B.1 | Review the **quality scores** an LLM judge gives the AI answers: context relevance, faithfulness and answer relevance (0–10), the JSON success rate and a 7-day trend. |
| B.2 | Keep a **test suite** of queries and their expected concepts (Embedding Health tab). |
| B.3 | Run **manual evaluations** (one query or the test suite) and download the results as JSON. |

---

## Best Practices

1. **Save versions frequently** — After each major transformation, save a new dataset version.
2. **Use AI suggestions as starting points** — Always review and validate AI-generated outputs.
3. **Upload literature early** — Research papers improve taxonomy quality and variable suggestions.
4. **Check quality at every phase** — Return to the Data Quality dashboard after imputation, outlier handling, or enrichment.
5. **Archive the traces with the reports** — Traces are your audit trail. They are saved automatically in your workspace (`traces` folder); keep them with the Word report and the study dossier for long-term archiving.
