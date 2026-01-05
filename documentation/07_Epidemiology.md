# 🧬 Epidemiology & Hypothesis Testing

## Overview

The **Epidemiology** module is dedicated to rigorous statistical analysis for biostatisticians and epidemiologists. It provides a structured workflow for hypothesis testing, effect size calculation, multivariate analysis (ANCOVA), power analysis, and Z-score standardization—ensuring that statistical claims are robust, reproducible, and clinically meaningful.

---

## Key Features

### 1. 🔬 Hypothesis Testing (Univariate Analysis)

Compare groups on continuous or categorical outcomes with automatic test selection.

| Test Type | When Used | Effect Size |
|:----------|:----------|:------------|
| **Student's T-test** | 2 groups, normal, equal variance | Cohen's d |
| **Welch's T-test** | 2 groups, normal, unequal variance | Cohen's d |
| **Mann-Whitney U** | 2 groups, non-normal | Rank-Biserial r |
| **ANOVA** | 3+ groups, normal | Eta² (η²) |
| **Kruskal-Wallis** | 3+ groups, non-normal | Eta²H (η²H) |
| **Chi-Square** | Categorical outcome | Cramér's V |
| **Fisher's Exact** | Categorical, small sample | Odds Ratio |

**Features:**

- **Auto-Detect**: Automatically selects test based on Shapiro-Wilk (normality) and Levene's (homogeneity)
- **Post-Hoc Analysis**: Tukey HSD or Dunn's with Bonferroni correction
- **Multiple Testing Correction**: Bonferroni (FWER) or Benjamini-Hochberg (FDR)
- **4-Quadrant Interpretation**: Automatic classification (Significant + Large Effect, etc.)

---

### 2. 📏 Effect Size Interpretation

| Measure | Small | Medium | Large | Test |
|:--------|:-----:|:------:|:-----:|:-----|
| Cohen's d | 0.2 | 0.5 | 0.8 | T-test |
| Rank-Biserial r | 0.1 | 0.3 | 0.5 | Mann-Whitney |
| Eta² (η²) | 0.01 | 0.06 | 0.14 | ANOVA |
| Eta²H (η²H) | 0.01 | 0.06 | 0.14 | Kruskal-Wallis |
| Cramér's V | 0.1 | 0.3 | 0.5 | Chi-Square |
| Odds Ratio | — | — | — | Fisher's (1 = no effect) |

> **Reference:** Cohen (1988), Tomczak & Tomczak (2014)

---

### 3. 🧮 Multivariate Analysis (ANCOVA)

**Analysis of Covariance** compares group means while adjusting for continuous confounders.

**Features:**

- **Model Formula**: `Target ~ Group + Covariate₁ + Covariate₂ + ...`
- **Type II ANOVA Table**: With partial η² effect size
- **Covariate Effects Table**: β coefficients with significance
- **Visualizations**: Adjusted vs Raw distributions, Q-Q plot, Residuals vs Fitted

**Assumptions (displayed in UI):**

1. Independence of observations
2. Normality of residuals
3. Homogeneity of variance
4. Linearity between covariate and outcome
5. Homogeneity of regression slopes

---

### 4. ⚡ Power & Sample Size Calculator

Calculate statistical power or required sample size for study planning.

| Test | Effect Size Input | Methods |
|:-----|:------------------|:--------|
| T-Test | Cohen's d | Non-central t-distribution |
| Mann-Whitney | P(X<Y), Rank-Biserial r | Noether's formula (1987) |
| ANOVA | Cohen's f, Eta² | Non-central F-distribution |
| Kruskal-Wallis | Cohen's f equivalent | ARE adjustment |
| Chi-Square | Cohen's w, Cramér's V | Non-central χ² |

**Features:**

- Import results from analysis tab
- Auto-fill group structure from dataset
- Interactive power curves
- Type I/II error interpretation guide

---

### 5. 📊 Z-Score Standardization & Reference Analysis

Standardize data against a reference population for cross-variable comparison.

**Formula:** `Z = (X - μ_ref) / σ_ref`

**Reference Methods:**

| Method | Description |
|:-------|:------------|
| Internal Control Group | Use a specific group (e.g., controls) as reference |
| Whole Cohort | Standardize to sample mean/SD (mean ≈ 0, SD ≈ 1) |
| Manual Reference | Enter published norms (e.g., WHO growth charts) |

**Visualizations:**

1. **Forest Plot**: Mean Z-scores by group with 95% CI or ±SD
2. **Heatmap**: All samples × variables (identify outliers & patterns)
3. **Individual Profile**: Single subject's Z-score profile with abnormality flagging

**Interpretation Guide:**

| Z-Score | Interpretation | Percentile |
|:--------|:---------------|:-----------|
| Z > +2 | Abnormally High | > 97.7% |
| +1 < Z ≤ +2 | Mildly Elevated | 84.1% - 97.7% |
| -1 ≤ Z ≤ +1 | Normal Range | 15.9% - 84.1% |
| -2 ≤ Z < -1 | Mildly Low | 2.3% - 15.9% |
| Z < -2 | Abnormally Low | < 2.3% |

---

### 6. 🛠️ Data Preprocessing

- **Transformations**: Log (Log1p), Z-Score, Min-Max normalization
- **Multiple Testing Correction**: Bonferroni (controls FWER) or Benjamini-Hochberg (controls FDR)

---

## Usage Guide

1. **Preprocess**: Apply transformations if data is skewed
2. **Select Variables**: Choose grouping variable and target(s)
3. **Configure Test**: Auto-detect or manually select
4. **Run Analysis**: View results with effect sizes and interpretation
5. **Check Power**: Verify study adequacy in Power tab
6. **Standardize**: Use Z-Score tab for reference comparison

---

## Key Assumptions Table

| Test | Independence | Normality | Equal Variance | Min Sample |
|:-----|:------------:|:---------:|:--------------:|:----------:|
| T-test (Student's) | ✓ | ✓ | ✓ | n ≥ 30 |
| T-test (Welch's) | ✓ | ✓ | ✗ | n ≥ 30 |
| ANOVA | ✓ | ✓ | ✓ | n ≥ 30/group |
| Mann-Whitney | ✓ | ✗ | Similar shapes | n ≥ 5/group |
| Kruskal-Wallis | ✓ | ✗ | Similar shapes | n ≥ 5/group |
| Chi-Square | ✓ | N/A | N/A | Expected ≥ 5 |
| Fisher's Exact | ✓ | N/A | N/A | Any size |

---

## Technical Details

- **File**: `app_pages/epidemiology.py`
- **Dependencies**: `scipy.stats`, `statsmodels.api`, `statsmodels.formula.api`, `statsmodels.stats.power`

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
- Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 1(21), 19-25.
- Noether, G. E. (1987). Sample size determination for some common nonparametric tests. *JASA*, 82(398), 645-647.
