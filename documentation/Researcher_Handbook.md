# 📘 Cohort Monitoring Application: Researcher's Handbook

## 1. Introduction

Welcome to the **Cohort Monitoring Package**, a comprehensive tool designed for epidemiological researchers and data scientists. This application streamlines the entire research workflow, from raw data quality assessment to advanced statistical analysis and reproducibility.

**Key Capabilities:**

* **Data Quality Auditing**: Instant health check of your dataset.
* **Intelligent Enrichment**: LLM-assisted variable creation and imputation.
* **Advanced Statistics**: Automated Univariate and Multivariate (ANCOVA) analysis.
* **Dimensionality Reduction**: PCA and FAMD for complex datasets.
* **Knowledge Graph**: Insight extraction from scientific literature (RAG).
* **Reproducibility**: Full trace tracking of all operations.

---

## 2. Workflow Overview

The application is structured to follow a standard research pipeline:

1. **Import & Load**: Upload CSV/Excel files or load existing snapshots.
2. **Quality Check**: Assess completeness and validity.
3. **Enrichment**: Handle missing data, create new variables, and transform data.
4. **Insight & Analysis**: Visualize distributions, run epidemiological tests, and explore dimensionality.
5. **Reporting**: Export results and save reproduction traces.

---

## 3. Module 1: Data Quality & Validation

Before analysis, ensure your data is "research-ready".

### 📊 Global Quality Score

The dashboard provides a single score (0-100) based on weighted metrics:

* **Completeness**: Percentage of non-missing values.
* **Consistency**: Data types match expected formats (e.g., numbers are numeric).
* **Clinical Validity**: Adherence to biological constraints (e.g., BMI < 15 is flagged).
* **Uniqueness**: No duplicate rows.

### 🕵️ Missing Data Diagnosis

Understanding *why* data is missing is crucial for choosing the right imputation method.

* **Little's MCAR Test**:
  * **Result > 0.05**: Data is **MCAR** (Missing Completely At Random). You can safely drop rows/cols or impute.
  * **Result < 0.05**: Data is **MAR** (Missing At Random) or **MNAR** (Not Random). *Do not just drop rows.* Use the **Dependency Scan** to find which variables predict missingness.
* **Visual Matrix**: Use the "Nullity Matrix" to see if missing values cluster together (e.g., an entire form was skipped).

---

## 4. Module 2: Data Enrichment

Refine your dataset for analysis.

### 🧪 Handling Missing Data

* **Imputation**: Use **MICE** (Multiple Imputation by Chained Equations) for high-precision imputation of numerical data.
* **LLM Suggestions**: The "AI Assistant" can suggest imputation strategies based on variable names.

### 📐 Dimensionality Reduction (PCA / FAMD)

Reduce complex data into interpretable components.

* **PCA (Principal Component Analysis)**: Best for purely numerical data.
* **FAMD (Factor Analysis of Mixed Data)**: **Use this for datasets with both categorical (Sex, Treatment) and numerical (Age, BMI) variables.**
  * *Note*: The "Contributions" table is normalized so that variable contributions sum to 100% per component, making interpretation intuitive.
* **Error Handling**: The system checks for missing values before running these sensitive algorithms. **You must impute missing data first.**

---

## 5. Module 3: Data Insight (Smart Documentation)

The **Data Insight** module acts as your automated "Data Steward," turning static metadata into a navigable knowledge graph.

### 🧠 Data Stewardship & Taxonomy

Maintain a single source of truth for your variables.

* **Taxonomy Management**: Definitions, types (Input, Derived, Outcome), and descriptions are stored centrally.
* **Versioning**: The system tracks changes to variable definitions (v1, v2, etc.), allowing you to audit how a variable evolved over time.

### 🕸️ Interactive Knowledge Graph

Stop guessing how variables are calculated.

* **Visual Lineage**: Click on a variable (e.g., `Hypercholesterolemia`) to see its dependencies (e.g., `LDL`, `Total Cholesterol`, `Statin Use`).
* **Formula Transparency**: See the exact mathematical formula or logic rule used to derive any variable.
* **Orphan Detection**: Identify variables that are defined but never used, or inputs that strictly drive outcomes.

---

## 6. Module 4: Epidemiological Analysis

Perform robust statistical comparison of groups.

### ⚖️ Univariate Analysis

Compares groups (e.g., "Treatment vs Control") one variable at a time.

* **Auto-Selection**: The app automatically chooses the correct test (T-test, ANOVA, Chi-Square, etc.) based on data types and distribution.
* **Effect Size**: Look beyond the P-value.
  * **Large Effect**: Cohen's d > 0.8, Odds Ratio > 1.5. Indicates a clinically meaningful difference.
  * **Small Effect**: Statistically significant but distinct distributions overlap heavily.

### 📈 Multivariate Analysis (ANCOVA)

Answers: *"Is the difference real, or due to a confounder?"*

* **Target**: The outcome (e.g., Blood Pressure).
* **Factor**: The group (e.g., Drug).
* **Covariates**: Confounders (e.g., Age, Baseline BP).
* **Interpretation**: If the Group effect remains significant (p < 0.05) after adjustment, the result is robust.

---

## 7. Module 5: RAG & Document Insight

Leverage your document repository.

* **Knowledge Graph**: Visualizes connections between concepts in your uploaded papers.
* **Q&A**: Ask questions like *"What is the standard deviation of Age in the reference paper?"*.
* **Monitoring**: Check the "RAG Quality" dashboard to verify that AI answers are faithful to the source text (Hallucination Check).

---

## 8. Reproducibility & Best Practices

Science must be reproducible.

### 🔁 Traces

* Every action (filtering, recoding, analysis) is recorded in a **Trace**.
* **Save Trace**: Exits the session with a recipe file (`.json`).
* **Load Trace**: Re-applies all steps to the raw data, ensuring exact replication of results.

### ✅ Best Practices

1. **Format**: Ensure your CSV has a single header row.
2. **Dates**: Format dates as YYYY-MM-DD for automatic recognition.
3. **Variable Names**: Use descriptive names (e.g., `systolic_bp` instead of `var1`) to help the AI assistant.
4. **Save Often**: Download intermediate datasets ("Snapshots") after major enrichment steps.
