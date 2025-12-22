# 🧬 Epidemiology & Hypothesis

## Overview

The **Epidemiology** module is dedicated to rigorous statistical analysis. It provides a structured workflow for hypothesis testing, effect size calculation, and multivariate analysis, ensuring that statistical claims are robust and reproducible.

## Key Features

### 1. 🔬 Hypothesis Testing

* **Univariate Analysis**: Compare groups on a continuous outcome.
* **Test Selection**:
  * **Parametric**: Independent T-test, ANOVA (One-way).
  * **Non-Parametric**: Mann-Whitney U, Kruskal-Wallis H.
  * **Auto-Detect**: Automatically chooses the best test based on normality (Shapiro-Wilk) and homogeneity of variance (Levene's test).
* **Post-Hoc Analysis**: Automatically performs post-hoc tests (Tukey HSD, Dunn's) if a global test (ANOVA/Kruskal) is significant.

### 2. 📏 Effect Sizes

* **Calculation**: Automatically computes appropriate effect sizes for each test:
  * **Cohen's d** (T-test)
  * **Rank-Biserial Correlation** (Mann-Whitney)
  * **Eta-Squared** (ANOVA)
  * **Epsilon-Squared** (Kruskal-Wallis)
* **Interpretation**: Provides context for effect sizes (Small, Medium, Large).

### 3. 🧮 Multivariate Analysis

* **ANCOVA**: Analysis of Covariance to compare group means while adjusting for continuous covariates (confounders).
* **Model Summary**: Displays full OLS regression results and Type II ANOVA tables.

### 4. ⚡ Power & Sample Size (Tab 2)

* **Power Calculation**: Estimate the statistical power of your study based on sample size and effect size.
* **Sample Size Estimation**: Calculate the required sample size to achieve a desired power (e.g., 0.8) for future studies.
* **Visualizations**: Interactive Power Curves to visualize the relationship between sample size, effect size, and power.

### 5. 📊 Z-Score & Reference (Tab 3)

* **Standardization**: Convert raw values to Z-scores based on the sample mean/SD or an external reference population.
* **Reference Comparison**: Compare your cohort's distribution against standard reference values (e.g., growth charts, lab norms).

### 6. 🛠️ Data Preprocessing

* **Transformations**: Apply Log (Log1p), Z-Score, or Min-Max normalization to variables before analysis to meet normality assumptions.
* **Multiple Testing Correction**: Apply Bonferroni or Benjamini-Hochberg (FDR) corrections to P-values when testing multiple variables to control error rates.

## Usage Guide

1. **Preprocess**: Use the sidebar to apply transformations if your data is skewed.
2. **Select Variables**: Choose a Grouping variable (Factor) and a Target variable (Outcome).
3. **Configure Analysis**: Select the test type (or leave as Auto) and correction method (Bonferroni/FDR).
4. **Run Analysis**: View the results table, which includes P-values, Test Statistics, and Effect Sizes.
5. **Check Power**: Use the "Power & Sample Size" tab to verify if your study was adequately powered to detect the observed effects.

## Technical Details

* **File**: `app_pages/epidemiology.py`
* **Dependencies**: `scipy.stats`, `statsmodels.api`, `statsmodels.formula.api`, `statsmodels.stats.power`
