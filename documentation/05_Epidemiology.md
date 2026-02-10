# 🧬 Epidemiology & Hypothesis Testing

The **Epidemiology & Hypothesis** page provides a complete statistical analysis toolkit — from univariate tests to multivariate modelling, power calculations, and Z-score standardisation. The system auto-recommends appropriate tests based on your data characteristics.

---

## 🧪 Hypothesis Testing

### Test Selection

The application guides test selection based on the analysis configuration:

| Factor | Options |
| --- | --- |
| **Outcome Variable** | Continuous or categorical. |
| **Comparison Variable** | Binary, multi-group, or continuous. |
| **Parametric / Non-Parametric** | Chosen based on normality testing or user preference. |

### Available Statistical Tests

#### Univariate

| Test | When Used |
| --- | --- |
| **Student's t-test** | Two-group comparison of means (parametric). |
| **Mann-Whitney U** | Two-group comparison (non-parametric). |
| **One-way ANOVA** | Multi-group comparison of means (parametric). |
| **Kruskal-Wallis** | Multi-group comparison (non-parametric). |
| **Chi-squared** | Categorical vs categorical association. |
| **Fisher's Exact** | Categorical association with small samples. |
| **Pearson Correlation** | Linear relationship between two continuous variables. |
| **Spearman Correlation** | Monotonic relationship (non-parametric). |

#### Post-hoc & Corrections

| Method | Description |
| --- | --- |
| **Tukey HSD** | Post-hoc pairwise comparison after ANOVA. |
| **Dunn's Test** | Post-hoc after Kruskal-Wallis. |
| **Bonferroni** | Conservative multiple testing correction. |
| **FDR (Benjamini-Hochberg)** | False discovery rate correction. |

### Effect Size Interpretation

| Measure | Applies To |
| --- | --- |
| **Cohen's d** | Two-group mean differences. |
| **Eta-squared (η²)** | ANOVA / multi-group comparisons. |
| **Cramér's V** | Chi-squared / categorical associations. |

Results include effect size magnitude labels (small / medium / large) based on standard thresholds.

---

## 📊 Multivariate Analysis (ANCOVA)

Analysis of Covariance adjusts group comparisons for confounding variables.

### Configuration

1. **Outcome Variable** — the dependent variable (continuous).
2. **Group Variable** — the independent factor (categorical).
3. **Covariates** — continuous variables to control for.

### Output

- Adjusted group means.
- F-statistic, p-value, and partial η² for the group effect.
- Covariate coefficients and their significance.
- Residual diagnostics.

---

## ⚡ Power & Sample Size Analysis

Estimate the sample size needed to detect a meaningful effect, or the power of your current sample.

### Parameters

| Parameter | Description |
| --- | --- |
| **Effect Size** | Expected magnitude (Cohen's d or custom). |
| **Alpha (α)** | Significance level (default 0.05). |
| **Power (1 − β)** | Desired statistical power (default 0.80). |
| **Number of Groups** | For multi-group comparisons. |

### Outputs

- **Required sample size** per group for the specified power.
- **Achieved power** given current sample size.
- **Power curve** — plot of power vs sample size.

---

## 📐 Z-Score Standardisation

Convert raw variable values into Z-scores relative to a reference population or the sample itself.

### Modes

| Mode | Description |
| --- | --- |
| **Sample-based** | Z-scores computed from the dataset's own mean and standard deviation. |
| **Reference-based** | User provides external reference mean and standard deviation (e.g. population norms). |

### Output

- New column(s) appended with the Z-score transformation.
- Distribution plot of Z-scores for visual inspection.

---

## 💡 Usage Tips

- Use the **Test Recommendation Helper** to identify the most appropriate test based on your variable types and sample characteristics.
- Always check **normality** (Shapiro-Wilk) before choosing parametric tests.
- For multiple comparisons, always apply a correction method to control false positives.
- Power analysis should be performed **before** data collection when possible (a priori).
