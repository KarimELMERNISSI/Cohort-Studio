# 📘 Cohort Monitoring Application: Researcher's Handbook

## 1. Introduction

Welcome to the **Cohort Monitoring Package**. This application is designed to be your "Co-Pilot" for clinical and epidemiological research. It does not just analyze data; it enforces rigorous methodological standards (reproducibility, traceability, validity) while automating the tedious parts of data cleaning and enrichment.

### 🌟 Key Philosophy

* **Traceability**: Every click is recorded. Your analysis is reproducible by design.
* **Enrichment over Deletion**: Don't just drop rows; impute, transform, and understand them.
* **Context-Aware**: The app uses your documents (protocols, papers) to guide your analysis.

---

## 2. Phase 1: Getting Started (Diagnosis)

*Goal: Assess if your data is fit for purpose.*

### 📥 Import & Load

* **Supported Formats**: CSV, Excel, Parquet.
* **Snapshots**: Load previous sessions (`.parquet`) to resume work instantly.
* **Descriptive Statistics**: Instant summary tables for Quantitative (Mean, SD, Distribution) and Qualitative (Counts, Frequencies) variables.
* **Correlation Preview**: Early detection of relationships between variables.

### 🏥 Quality Audit (Health Check)

Before touching a single variable, check the patient's vital signs:

* **Global Score (0-100)**: A weighted composite index of your dataset's health:
  * *Clinical Validity*: **Weighted 2x** (Most critical).
  * *Completeness & Consistency*: Weighted 1x.
* **Completeness**: Are critical fields empty?
* **Consistency**: Do "Dates" look like dates and "Age" look like a number?
* **Statistical Validity**: Detection of univariate outliers (Z-Score) and multivariate anomalies (Isolation Forest).
* **Clinical Validity**: Customizable rules (e.g., `BMI > 10` and `BMI < 60`) flag biologically impossible values.

### 🕵️ Missingness Detective

* **Pattern Analysis**: Is data missing at random (MCAR)?
* **Little's Test**: A p-value > 0.05 suggests you can safely impute without bias.
* **Visual Nullity Matrix**: Uses hierarchical clustering (dendrograms) to group variables that tend to be missing together (e.g., an entire skipped questionnaire).
* **Dependency Scan**: If p < 0.05, finding specific variables that *cause* missingness (e.g., "Sicker patients don't fill out the Quality of Life survey").

---

## 3. Phase 2: Configuration (Study Logic)

*Goal: Define the "Rules of the Game" before playing.*

Use the **Configuration Editor** (`config_form.py`) to standardize your study logic. This ensures that every team member applies the exact same criteria.

### 🎭 Mask Families (Inclusion Criteria)

Define reusable filters for your study population:

* **Range Masks**: e.g., `Adults` = Age [18, 99].
* **Logic Masks**: e.g., `Enrolled` = `Status == 'Active' AND Consent == True`.

### 🔄 Unit Standardization

Automatically unify mixed units to preventing analysis errors:

* **Rule**: "If `Weight_Unit` is 'lbs', divide `Weight` by 2.204."
* **Effect**: Applied automatically during the Enrichment phase.

---

## 4. Phase 3: Data Enrichment (The Engine Room)

*Goal: Transform raw data into analytical variables. This is where you create value.*

### 🧹 Step 1: Cleaning & Imputation

* **Strategy**: Choose between `MICE` (Iterative), `KNN` (Similarity), or `Simple` (Mean/Median).
* **AI Suggestion**: The Assistant analyzes variable names to recommend the statistically appropriate method.

### 🧪 Step 2: Advanced Variable Creation

Move beyond simple columns. Create clinically meaningful indices.

#### A. Computed Columns (Formulas)

* **Syntax**: Write natural mathematical formulas: `BMI = Weight / (Height/100)**2`.
* **Safety**: The system handles type checking and zero-division errors.

#### B. Dimensionality Reduction (Synthetic Variables)

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

---

## 7. Phase 6: Knowledge & Insight

*Goal: Contextualize your findings.*

### 🧠 Data Insight (The Steward)

* **Lineage**: Click any variable to see its history: *Raw -> Imputed -> Transformed*.
* **Orphan Check**: Find created variables that you forgot to use in the analysis.

### 📚 RAG (Document Chat)

* **Upload**: Protocols, Papers, PDFs.
* **Query**: "What is the exclusion criteria for Heart Failure in the attached PDF?"
* **Coverage**: "Does my CSV contain all the variables mentioned in Table 1 of the paper?"

### 🛡️ RAG Quality Monitor (Trust Center)

Trust but verify.

* **Hallucination Rate**: Tracks % of answers unsupported by context.
* **Faithfulness**: Score (0-10) measuring adherence to source text.
* **Context Relevance**: Score (0-10) measuring if the retrieved PDF chunks actually answer the question.

---

## 8. Master Class: Reproducibility

*Goal: Science that stands the test of time.*

### 🔁 The Trace System

Every project generates a `.json` "Recipe File".

1. **Fast Mode (Replay)**: Reloads intermediate Snapshots. fast.
2. **Full Reproducibility**: Re-runs the *entire* cleaning pipeline from the raw csv. Proof of correctness.

### 💾 Dataset Versioning

* **Snapshots**: Save `V1_Raw`, `V2_Imputed`, `V3_Final`.
* **Rollback**: Made a mistake? One-click restore to yesterday's version.

### ✅ Best Practices Checklist

1. **One Header**: Ensure your CSV has a single header row.
2. **ISO Dates**: Use `YYYY-MM-DD` to help the auto-detector.
3. **Meaningful Names**: `Systolic_BP` is better than `Var12`.
4. **Save Traces**: Always download your Trace JSON at the end of a session.
