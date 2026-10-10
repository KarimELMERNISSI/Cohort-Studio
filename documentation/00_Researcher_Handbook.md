# Researcher Handbook

Welcome to the **Cohort Monitoring Package**. This handbook walks you through the application's philosophy and the recommended workflow for cohort data analysis — from raw data ingestion to publication-ready outputs.

---

## Philosophy

This application is built around three principles:

1. **Transparency** — Every transformation is logged. You can trace exactly how the data changed.
2. **Reproducibility** — Analysis sessions can be replayed end-to-end from trace files.
3. **AI-Augmented, Human-Controlled** — AI suggests; you decide.

---

## Workflow Phases

The application is organised into seven phases, mirroring the natural flow of cohort research. You can navigate freely between phases using the sidebar.

### Phase 1 — Ingestion & Setup

**Page**: Main View 

| Step | Action |
| --- | --- |
| 1.1 | **Sign up / Log in** to your account. |
| 1.2 | **Load your dataset** (CSV, Excel, or Parquet). |
| 1.3 | **Rename columns** manually or with AI assistance for standardisation. |
| 1.4 | **Save an initial version** to create a baseline snapshot. |

See: [01_Home.md](01_Home.md)

### Phase 2 — Quality Control

**Page**: Data Validation & Monitoring 

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

**Page**: Data Enrichment 

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

**Page**: Visualisation 

| Step | Action |
| --- | --- |
| 4.1 | Explore **distributions** (histograms, box plots, violin plots). |
| 4.2 | Investigate **relationships** (scatter plots, correlation matrices). |
| 4.3 | Generate **medical plots** (Bland-Altman, ROC, Kaplan-Meier). |
| 4.4 | Use **integrated statistical tests** for quick significance checks. |
| 4.5 | **Export** publication-ready figures. |

See: [04_Visualization.md](04_Visualization.md)

### Phase 5 — Statistical Analysis

**Page**: Epidemiology & Hypothesis 

| Step | Action |
| --- | --- |
| 5.1 | Select and run **hypothesis tests** (auto-recommended). |
| 5.2 | Interpret **effect sizes** and confidence intervals. |
| 5.3 | Run **multivariate analysis** (ANCOVA) with covariates. |
| 5.4 | Perform **power & sample size** calculations. |
| 5.5 | Compute **Z-scores** for standardised comparisons. |

See: [05_Epidemiology.md](05_Epidemiology.md)

### Phase 6 — Knowledge & Literature

**Pages**: Data Insight , Documents Insight 

| Step | Action |
| --- | --- |
| 6.1 | **Generate a Taxonomy** (knowledge graph) from your dataset variables. |
| 6.2 | **Refine** the taxonomy with AI enrichment and concept nodes. |
| 6.3 | **Upload research papers** and generate structured summaries. |
| 6.4 | **Chat** with documents for targeted literature queries. |
| 6.5 | Run **dataset coverage analysis** against uploaded papers. |

See: [06_Data_Insight.md](06_Data_Insight.md), [07_Document_Insight.md](07_Document_Insight.md)

### Phase 7 — Reproducibility & Reporting

**Page**: Reproduce Analysis 

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

**Page**: Users Management  — *Visible only to Admins*

| Step | Action |
| --- | --- |
| A.1 | **View Users** list and their current status (Pending, Active, Deactivated). |
| A.2 | **Activate** new accounts to grant access. |
| A.3 | **Deactivate** or **Delete** users as needed. |
| A.4 | **Reset Passwords** for users who cannot log in. |

See: [10_Developer_Guide.md](10_Developer_Guide.md)

### RAG Quality Monitoring

**Page**: RAG Quality Monitor — *Optional*

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
