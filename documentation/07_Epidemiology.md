# 🧬 Epidemiology & Hypothesis Testing

## Overview

The **Epidemiology** module is dedicated to rigorous statistical analysis. It provides a structured workflow for hypothesis testing, effect size calculation, and multivariate analysis, ensuring that statistical claims are robust and reproducible.

## Key Features

### 1. 🔬 Hypothesis Testing

* **Univariate Analysis**: Compare groups on a continuous or categorical outcome.
* **Test Selection**:
  * **Parametric**: Independent T-test (Student's or Welch's), ANOVA (One-way).
  * **Non-Parametric**: Mann-Whitney U, Kruskal-Wallis H.
  * **Categorical**: Chi-Square, Fisher's Exact (for small samples).
  * **Auto-Detect**: Automatically chooses the best test based on normality (Shapiro-Wilk) and homogeneity of variance (Levene's test).
* **Post-Hoc Analysis**: Automatically performs post-hoc tests (Tukey HSD, Dunn's with Bonferroni) if a global test (ANOVA/Kruskal) is significant.

### 2. 📏 Effect Sizes

* **Calculation**: Automatically computes appropriate effect sizes for each test:
  * **Cohen's d** (T-test) — Small: 0.2, Medium: 0.5, Large: 0.8
  * **Rank-Biserial r** (Mann-Whitney) — Small: 0.1, Medium: 0.3, Large: 0.5
  * **Eta-Squared η²** (ANOVA) — Small: 0.01, Medium: 0.06, Large: 0.14
  * **Eta-Squared H η²H** (Kruskal-Wallis) — Small: 0.01, Medium: 0.06, Large: 0.14
  * **Cramér's V** (Chi-Square) — Small: 0.1, Medium: 0.3, Large: 0.5
  * **Odds Ratio** (Fisher's Exact) — 1 = No effect

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
* **Multiple Testing Correction**: Apply Bonferroni or Benjamini-Hochberg (FDR) corrections to P-values when testing multiple variables.

## Usage Guide

1. **Preprocess**: Use the sidebar to apply transformations if your data is skewed.
2. **Select Variables**: Choose a Grouping variable (Factor) and a Target variable (Outcome).
3. **Configure Analysis**: Select the test type (or leave as Auto) and correction method.
4. **Run Analysis**: View the results table, which includes P-values, Test Statistics, and Effect Sizes.
5. **Check Power**: Use the "Power & Sample Size" tab to verify if your study was adequately powered.

## Key Assumptions

| Test Type | Independence | Normality | Equal Variance | Sample Size |
|-----------|:------------:|:---------:|:--------------:|:-----------:|
| T-test | ✓ | ✓ | ✓ (Student's) | n ≥ 30 ideal |
| Welch's T-test | ✓ | ✓ | Not required | n ≥ 30 ideal |
| ANOVA | ✓ | ✓ | ✓ | n ≥ 30/group |
| Mann-Whitney | ✓ | Not required | Similar shapes | n ≥ 5/group |
| Kruskal-Wallis | ✓ | Not required | Similar shapes | n ≥ 5/group |
| Chi-Square | ✓ | N/A | N/A | Expected ≥ 5 |

## Technical Details

* **File**: `app_pages/epidemiology.py`
* **Dependencies**: `scipy.stats`, `statsmodels.api`, `statsmodels.formula.api`, `statsmodels.stats.power`

## References

* Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
* Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 1(21), 19-25.
* Noether, G. E. (1987). Sample size determination for some common nonparametric tests. *JASA*, 82(398), 645-647.
