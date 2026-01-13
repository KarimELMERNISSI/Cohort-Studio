# 📘 Cohort Monitoring Application: Researcher's Handbook

## 1. Introduction

Welcome to your **Integrated Research Environment**. This application is not just a "monitoring package"; it is a **Scientific Command Center** designed to pilot your clinical and epidemiological research from raw data to publication-ready insights.

It enforces rigorous methodological standards (reproducibility, traceability, validity) while automating the tedious parts of data cleaning, enrichment, and documentation.

### 🌟 Key Philosophy

* **Integrated Workflow**: Data cleaning, analysis, and literature review (RAG) are not separate steps but a unified, continuous process.
* **Traceability First**: Every click is recorded in a "Trace Recipe". Your analysis is reproducible by design, anywhere (Windows, Docker, Linux).
* **Enrichment over Deletion**: Don't just drop rows; use AI and advanced imputation to "rescue" data.
* **Context-Aware**: Your documents (protocols, papers) are active participants, guiding your variable selection and validation.

---

## 2. Phase 1: Getting Started (Diagnosis)

*Goal: Assess if your data is fit for purpose.*

### 📥 Import & Load

* **Supported Formats**: CSV, Excel, Parquet.
* **Snapshots**: Load previous sessions (`.parquet`) to resume work instantly.
* **AI Semantic Renaming**: The RAG assistant can auto-rename cryptic columns (e.g., `dx_t2_yr`) to standardized medical terms (e.g., `Type 2 Diabetes Diagnosis Year`) using literature contextualization.
* **Descriptive Statistics**: Instant summary tables for Quantitative (Mean, SD, Distribution) and Qualitative (Counts, Frequencies) variables.
* **Correlation Preview**: Early detection of relationships between variables.

### 🏥 Quality Audit (Health Check)

Before touching a single variable, check the patient's vital signs:

* **Global Score (0-100)**: A weighted composite index of your dataset's health based on **6 key dimensions**:
  * *Completeness*: Are critical fields empty?
  * *Statistical Validity*: Detection of outliers (Z-Score, Isolation Forest).
  * *Clinical Validity*: Adherence to biological rules (e.g., `BMI > 10`).
  * *Consistency*: Data type checks (e.g. "Age" is numeric, "Date" is datetime).
  * *Uniqueness*: Checks for duplicate rows or ID conflicts.
  * *Uniformity*: Checks for standard formatting (e.g., consistent casing/units).
* **Weights**: You can customize the importance of each dimension in the Scorecard (Clinical Validity is weighted 2x by default).

### 🕵️ Missingness Detective

* **Pattern Analysis**: Is data missing at random (MCAR)?
* **Little's Test**: A p-value > 0.05 suggests you can safely impute without bias.
* **Visual Nullity Matrix**: Uses hierarchical clustering (dendrograms) to group variables that tend to be missing together (e.g., an entire skipped questionnaire).
* **Dependency Scan**: If p < 0.05, finding specific variables that *cause* missingness (e.g., "Sicker patients don't fill out the Quality of Life survey").

---

## 3. Phase 2: Standardization (Study Validation)

*Goal: Define the "Rules of the Game" before playing.*

Instead of a complex central configuration file, you define rules where they naturally belong:

### 🎭 Study Inclusion (In "Data Validation" Tab)

Define who belongs in the study using reusable filters (Masks).

* **Inclusion Criteria**: e.g., `Age >= 18` AND `Consent == True`.
* **Clinical Consistency**: Flag impossible values (e.g., `Heart_Rate > 300`) as anomalies.

* **Range Masks**: e.g., `Adults` = Age [18, 99].
* **Logic Masks**: e.g., `Enrolled` = `Status == 'Active' AND Consent == True`.

### 🔄 Unit Standardization

Automatically unify mixed units to preventing analysis errors:

* **Rule**: "If `Weight_Unit` is 'lbs', divide `Weight` by 2.204."
* **Effect**: Applied automatically during the Enrichment phase.

### ➕ Computed Features (In "Data Enrichment" Tab)

* **Formula**: Use the **Targeted Imputation** or **Computed Columns** sub-tabs to create variables like `BMI = Weight / (Height/100)**2`.
* **Traceability**: All created variables are logged in the Trace Recipe.

### 📉 Statistical Tests (In "Epidemiology" Tab)

* **Analysis Configuration**: Select your test preference (Parametric vs Non-Parametric) and correction method (Bonferroni/FDR) directly in the sidebar.
* **Hypothesis Testing**: Define your groups and inputs on the fly.

---

## 4. Phase 3: Data Enrichment (The Engine Room)

*Goal: Transform raw data into analytical variables. This is where you create value.*

### 🧹 Step 1: Cleaning & Imputation

The "Handle Missing Data" tab is now organized into two focused sub-tabs:

1. **🎯 Targeted Imputation (Custom Formulas)**:
    * *Precision*: Manually fix specific variables using known relationships (e.g., `Weight = BMI * Height^2`).
    * *AI Guided Mode*: The RAG assistant can suggest valid Python formulas (`np.where(...)`) based on your columns and even specific hints (e.g. "Unit conversion").
2. **🤖 Global Imputation (Auto)**:
    * *Strategy*: Apply statistical methods like `MICE` (Iterative), `KNN`, or `Mean/Median` to the entire dataset at once.
    * *Automated*: Configure numerical and categorical strategies separately and let the system handle the rest.

### 🧪 Step 2: Advanced Variable Creation

Move beyond simple columns. Create clinically meaningful indices.

#### A. Computed Columns (Formulas)

* **Formulas**: Write natural mathematical formulas: `BMI = Weight / (Height/100)**2`.
* **Transformations**: Apply aggregations or normalizations with one click:
  * *Aggregations*: Mean, Median, Sum, Min/Max (e.g., "Average BP" from three readings).
  * *Normalizations*: Z-Score, Min-Max Scaling, Log transformations.

#### B. External Data Enrichment

* **Merge**: Import columns from secondary files (e.g., genetic data, environmental exposure) matching on a Patient ID.
* **Safety**: Automatic handling of duplicates and mismatched indices.

#### C. Dimensionality Reduction (Synthetic Variables)

*Concept: Turn 50 correlated variables into 3 "Super-Variables" that capture the underlying trend.*

* **PCA (Principal Component Analysis)**:
  * *Usage*: For purely numerical data (e.g., 30 Lab values).
  * *Output*: Creates `PC1`, `PC2` scores representing the major axes of variance.
* **FAMD (Factor Analysis of Mixed Data)**:
  * *Usage*: The "Medical Standard". Handles **both** categorical (Sex, Treatment) and numerical (Age, BP) data simultaneously.
  * *Interpretation*: The "Contributions" table reveals which real variables drive the synthetic score.
* **t-SNE / UMAP**:
  * *Usage*: Non-linear cluster finding (e.g., finding patient phenotypic subgroups).
  * *Parameters*: Tune `Perplexity` (t-SNE) or `Neighbors` (UMAP) to balance local vs global structure.

---

## 5. Phase 4: Exploration (Visualization)

*Goal: See the patterns.*

### 📊 Plotting Suite

* **Distributions**: Histogram, Density, Violin Plots. (Check: Is my data normal?)
* **Comparisons**: Box Plots with **auto-calculated P-values**.
* **Correlations**: Heatmaps & Scatter plots.
* **Publication-Grade Figures**: Customise themes (ggplot, seaborn) and export high-resolution plots (SVG/PNG) for your manuscript in one click.

### 🏎️ Performance Mode

* **Pandas vs DuckDB**: For datasets >100k rows, run the benchmark. The app will verify results and switch to DuckDB (SQL-based) for instant chart rendering if it's faster.

---

## 6. Phase 5: Confirmatory Analysis (Epidemiology)

*Goal: Prove your hypothesis.*

### ⚖️ Univariate Analysis

Compare groups (e.g., Treatment vs Placebo).

* **Auto-Selector**: The app picks the right test (T-Test, Wilcoxon, ANOVA, Kruskal) based on normality and group count.
* **Correction Methods**:
  * **Bonferroni**: Strict control of Family-Wise Error Rate (FWER).
  * **Benjamini-Hochberg (FDR)**: Balanced approach for exploratory analysis.
* **Effect Size Classified**: Results are automatically tagged:
  * ✅ **Clean**: Significant + Large Effect (Cohen's d > 0.8).
  * ⚠️ **Caution**: Significant but Small Effect (likely due to large N).
  * ❌ **Underpowered**: Non-significant but Large Effect (N too small).

### 📈 Multivariate Analysis (GLM/ANCOVA)

"Adjust for Confounders."

* **Model**: `Outcome ~ Group + Confounder1 + Confounder2`.
* **Result**: See if the treatment effect survives after adjusting for Age and Severity.

### 🔋 Power & Sample Size

*Goal: Ensure your study isn't "Underpowered" (unable to detect a real effect).*

* **Post-hoc Power**: Calculates the statistical power (0-100%) of your observed results.
* **Sample Size Estimation**: Tells you how many more patients you need to reach 80% power.

### 📏 Z-Score & Reference Analysis

*Goal: Standardize patients against a reference population.*

* **Standardization**: Converts raw values (e.g., Height in cm) into Z-Scores (Standard Deviations from the mean).
* **Usage**: Critical for pediatric growth charts or comparing variables with vastly different units.

---

## 7. Phase 6: Knowledge & Insight

*Goal: Contextualize your findings.*

* **Interactive Graph**: visualizes the dependencies between your variables.
* **Traceability**: Click any node to see its full history (Raw -> Imputed -> Transformed).
* **Orphan Check**: Find created variables that you forgot to use in the analysis.

### 📚 Document Insight (Literature Graph)

* **Knowledge Graph**: auto-extracts entities (Diseases, Drugs, Findings) from your PDF library and links them.
* **Summarizer**: Automatically generates structured summaries (Objective, Methods, Results) for every uploaded paper.
* **RAG Chat**: "Talk" to your documents. Ask: "What implies exclusion in these 5 papers?"

### 🛡️ RAG Quality Monitor (Trust Center)

Trust but verify.

* **Hallucination Rate**: Tracks % of answers unsupported by context.
* **Faithfulness**: Score (0-10) measuring adherence to source text.
* **Context Relevance**: Score (0-10) measuring if the retrieved PDF chunks actually answer the question.

---

## 8. Master Class: Reproducibility

*Goal: Science that stands the test of time.*

### 🔁 The Trace System

Every project generates a `.json` "Recipe File" that records every click and parameter.

1. **Fast Mode (Replay)**: Reloads intermediate Snapshots for instant resumption.
2. **Full Reproducibility**: Re-runs the *entire* cleaning pipeline from the raw csv module-by-module.
3. **Portable Reproduction**: Our "Path Resolver" ensures your trace works anywhere—Windows, Linux, or Docker—by automatically locating your data files.

### 💾 Dataset Versioning

* **Snapshots**: Save `V1_Raw`, `V2_Imputed`, `V3_Final`.
* **Rollback**: Made a mistake? One-click restore to yesterday's version.

### ✅ Best Practices Checklist

1. **One Header**: Ensure your CSV has a single header row.
2. **ISO Dates**: Use `YYYY-MM-DD` to help the auto-detector.
3. **Meaningful Names**: `Systolic_BP` is better than `Var12`.
4. **Save Traces**: Always download your Trace JSON at the end of a session.
