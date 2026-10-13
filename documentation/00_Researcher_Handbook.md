# Researcher Handbook

Welcome to the **Cohort Monitoring Package**. This handbook walks you through the application's philosophy and the recommended workflow for cohort data analysis — from raw data ingestion to publication-ready outputs.

---

## Philosophy

This application is built around three principles:

1. **Transparency** — Every transformation is logged. You can trace exactly how the data changed.
2. **Reproducibility** — Analysis sessions can be replayed end-to-end from trace files.
3. **AI-Augmented, Human-Controlled** — AI suggests; you decide.

---

## Core path and optional paths

The **Start** page shows the workflow as a diagram. Three core steps take you from a raw file to results:
**Load** (Data > Import, a saved version or the sample cohort), **Validate** (Data > Validate) and
**Explore & analyse** (Explore and Analyze). Optional paths branch off whenever they help:

| Optional path | When | What it adds | Page |
| --- | --- | --- | --- |
| Enrich the cohort | after validation, as often as needed | merge external sources, derive variables (formulas or AI suggestions), impute missing values; enrichment is incremental, so validate and analyse again after each change and come back for the next one (two-headed arrows on the diagram) | Data > Enrich |
| Map your variables to your field | any time (AI engine) | a taxonomy of your variables grounded in your clinical domain and documents: definitions, units, relationships, formulas | Knowledge > Variable map |
| Learn from your documents | any time (AI engine) | index literature, protocols or data dictionaries; chat with them; extract a knowledge graph that informs the variable map and enrichment suggestions | Knowledge > Documents |
| Report and reproduce | at the end, or any time | saved versions, the trace of every step, step-by-step replay and the Word report | Reproduce > Versions & traces |

The AI engine is optional and local by default (Settings > AI engine); without it, every core step and the
enrichment path still work.

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
| 2.3 | Check for **clinical anomalies** (impossible or suspect values). |
| 2.4 | Define **inclusion criteria** to filter your cohort. |
| 2.5 | Detect and handle **outliers** using statistical or ML methods. |
| 2.6 | **Check conformity** (duplicates, consistency, uniformity). |
| 2.7 | **Compare** the cleaned dataset against the original. |

See: [02_Data_Validation.md](02_Data_Validation.md)

### Phase 3 — Enrichment & Preparation

**Page**: Data > Enrich

| Step | Action |
| --- | --- |
| 3.1 | **Import external data** and merge with your main dataset. |
| 3.2 | **Impute missing values** (targeted formulas or global strategies). |
| 3.3 | **Engineer new variables** (encoding, scaling, computed indices). |
| 3.4 | Apply **dimensionality reduction** (PCA, FAMD, t-SNE, UMAP). |
| 3.5 | Run **population clustering** (K-Means, DBSCAN, GMM). |
| 3.6 | Explore **AI variable suggestions** for computed indices. |

See: [03_Data_Enrichment.md](03_Data_Enrichment.md), [09_Clustering.md](09_Clustering.md)

### Phase 4 — Exploration & Visualisation

**Page**: Explore > Visualize

| Step | Action |
| --- | --- |
| 4.1 | Explore **distributions** (histograms, box plots, violin plots). |
| 4.2 | Investigate **relationships** (scatter plots, correlation matrices). |
| 4.3 | Generate **medical plots** (Bland-Altman, ROC, Kaplan-Meier). |
| 4.4 | Use **integrated statistical tests** for quick significance checks. |
| 4.5 | **Export** publication-ready figures. |

See: [04_Visualization.md](04_Visualization.md)

### Phase 5 — Statistical Analysis

**Page**: Analyze > Hypothesis tests & epidemiology

| Step | Action |
| --- | --- |
| 5.1 | Select and run **hypothesis tests** (auto-recommended). |
| 5.2 | Interpret **effect sizes** and confidence intervals. |
| 5.3 | Run **multivariate analysis** (ANCOVA) with covariates. |
| 5.4 | Perform **power & sample size** calculations. |
| 5.5 | Compute **Z-scores** for standardised comparisons. |

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

**Page**: Reproduce > Versions & traces

| Step | Action |
| --- | --- |
| 7.1 | Review **trace history** of your analysis sessions. |
| 7.2 | **Replay** a trace to reproduce results (Fast or Full mode). |
| 7.3 | **Generate reports** (`.docx`) from trace files. |
| 7.4 | **Share** traces and reports with collaborators. |

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

See: [10_Developer_Guide.md](10_Developer_Guide.md)

### RAG Quality Monitoring

**Page**: Admin > AI quality — *optional*

| Step | Action |
| --- | --- |
| B.1 | Monitor **embedding health** and document chunking statistics. |
| B.2 | Review **retrieval quality** metrics (precision, recall). |
| B.3 | Run **manual evaluations** of the RAG system's answers. |

---

## Best Practices

1. **Save versions frequently** — After each major transformation, save a new dataset version.
2. **Use AI suggestions as starting points** — Always review and validate AI-generated outputs.
3. **Upload literature early** — Research papers improve taxonomy quality and variable suggestions.
4. **Check quality at every phase** — Return to the Data Quality dashboard after imputation, outlier handling, or enrichment.
5. **Export traces before closing** — Traces are your audit trail; download them for long-term archiving.
