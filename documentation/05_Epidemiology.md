# Epidemiology & Hypothesis Testing

The **Epidemiology & Hypothesis** page provides a complete statistical analysis toolkit — from univariate tests to multivariate modelling, power calculations, and Z-score standardisation. The system auto-recommends appropriate tests based on your data characteristics.

## Guidance on this page

The **Guidance** switch at the top of the page (on by default) adds help where decisions are made; turn it off once
you know the page.

- **Which tab do I need?** — maps your question to the right tab (compare groups, diagnostic accuracy and
  agreement, paired data, rates and stratified analyses, power and sample size, reference z-scores) and links to
  the *Which test?* chooser.
- **Every parameter is explained** — hover the help icon of any input: what it means, typical values and what
  changes when you change it.
- **Before you run: the plan** — once the groups and variables are chosen, the page shows, for each variable,
  the test that will run and why (the same rules as the engine: Welch's test by default, a rank test for small
  and strongly skewed groups, Fisher's exact test for sparse 2x2 tables), the group sizes, missing data, and
  warnings: very small or unbalanced groups, many groups, variables with 20% or more missing values, and tests
  run without a multiple-testing correction. It suggests a correction (none for one test, Holm up to 9 tests,
  Benjamini-Hochberg for screens of 10 or more) and applies it in one click.
- **Detailed view** — choosing a test by hand shows what that test assumes.
- **ANCOVA model check** — complete observations per model parameter (at least 10, better 20), rows excluded
  for missing values, covariates largely explained by the group (possible mediators or collinearity), and a
  reminder of how to choose covariates.
- **Power & sample size** — a four-step plan; the effect can be given directly or from what you expect in the
  outcome's own terms (two means and an SD, several group means and an SD, or event proportions per group);
  results are written as a sentence ("To detect ... with 80% power ... you need ... participants") with a
  reminder to add a dropout margin, and achieved power is put in context (below 50%, below the 80% target).

---

## Hypothesis Testing

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

## Multivariate Analysis (ANCOVA)

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

## Power & Sample Size Analysis

Estimate the sample size needed to detect a meaningful effect, or the power of your current sample.

### Parameters

| Parameter | Description |
| --- | --- |
| **Effect Size** | The smallest effect worth detecting: Cohen's d, f or w directly, or computed from expected means and SD, group means, or event proportions per group. |
| **Alpha (α)** | Significance level (default 0.05). |
| **Power (1 − β)** | Desired statistical power (default 0.80). |
| **Number of Groups** | For multi-group comparisons. |

### Outputs

- **Required sample size** — reported **per group and in total** (the one-way ANOVA and Kruskal-Wallis
  calculators follow statsmodels' convention where the sample size is the total over all groups; the
  Kruskal-Wallis figure divides the ANOVA requirement by the asymptotic relative efficiency, 0.955 for
  normal data or the 0.864 worst case).
- **Achieved power** given the current per-group sample size.
- **Chi-square** — the calculator takes the degrees of freedom, (rows − 1) × (columns − 1), of the
  contingency table (df = 1 for a 2×2 table).

---

## Z-Score Standardisation

Convert raw variable values into Z-scores relative to a reference population or the sample itself.

### Modes

| Mode | Description |
| --- | --- |
| **Whole cohort** | Z-scores computed from the dataset's own mean and standard deviation. |
| **Internal control group** | Mean and SD of a reference group of the cohort (for example controls or the placebo arm); groups coded 0/False are supported. |
| **Manual reference values** | Mean and SD entered per variable (published population norms). |

### Output
- New column(s) appended with the Z-score transformation.
- Distribution plot of Z-scores for visual inspection.

---

## Methodological Guidance
- Use the **Test Recommendation Helper** to identify the most appropriate test based on your variable types and sample characteristics.
- Always check **normality** (Shapiro-Wilk) before choosing parametric tests.
- For multiple comparisons, always apply a correction method to control false positives.
- Power analysis should be performed **before** data collection when possible (a priori).

---

## Survival & regression (Analyze > Survival & regression)

| Method | What it gives | Reference |
| --- | --- | --- |
| Kaplan-Meier | survival curve per group with pointwise 95% Greenwood (log-log) intervals, median survival | Kaplan & Meier 1958 |
| Log-rank test | chi-square test of equal survival across groups | Mantel 1966 |
| Cox proportional hazards | hazard ratios with Wald intervals (Efron ties); Grambsch-Therneau proportional-hazards test and concordance when the `survival` extra (lifelines) is installed | Cox 1972 |
| Logistic regression | adjusted odds ratios with Wald intervals, pseudo-R2, DeLong AUC; warns below 10 events per parameter | Hosmer et al. 2013 |
| Poisson / negative binomial | incidence-rate ratios with an optional person-time offset; over-dispersion check | Cameron & Trivedi 2013 |

Categorical covariates are one-hot coded with their first (alphabetical) level as reference; the page states it.

## Paired, rates & stratified (Analyze > Hypothesis tests & epidemiology)

| Method | What it gives |
| --- | --- |
| Paired t-test / Wilcoxon signed-rank | mean or pseudo-median within-pair difference with 95% CI, Cohen's d_z, matched-pairs rank-biserial r |
| One-sample t / Wilcoxon / exact binomial | difference from a reference value, Wilson interval for a proportion |
| McNemar (exact) | paired binary outcomes from the discordant pairs |
| Incidence rates | exact Poisson intervals, rate ratio with a score interval and exact conditional test |
| Mantel-Haenszel | pooled OR (Robins-Breslow-Greenland interval) and RR (Greenland-Robins), Breslow-Day homogeneity, crude vs adjusted comparison |
| Standardised ratio (SMR/SIR) | observed / expected with the exact Poisson interval |
| Cohen's kappa, ICC(2,1), ICC(3,1) | agreement and reliability with intervals and the usual interpretation scales |

The multiple-imputation diagnostic lives on the Enrich page (Global imputation tab). It pools the imputed
means with Rubin's rules; the analyses on this page are not yet pooled across imputations, so run them on the
complete cases for inference (a singly-imputed dataset gives standard errors and p-values that are too small).
