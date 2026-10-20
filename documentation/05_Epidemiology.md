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
- **Detailed view** — choosing a test by hand shows what that test assumes. The result card lists its
  **assumption checks**, which are diagnostics, not gates: the test runs regardless. Each line names the group it
  concerns (*Normality, diabetes = No (n = 506)*), gives the finding with its test and p-value, and says what it
  means for the test that ran:
  - unequal variances are not required by Welch's test, but undermine Student's t-test and the one-way ANOVA;
  - non-normality matters for a test of means only in small groups (fewer than 30 values), and never for a rank
    test.
- **Figures of the detailed view** — four figures in a two-column grid, one colour per group throughout:
  - the box plot with the significance brackets and n;
  - the mean of each group with its 95% CI;
  - the histograms as the percentage of each group, so that groups of different sizes compare;
  - a normal Q-Q plot of each group, standardised within the group, which matches the normality lines of the
    card.
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
| **Parametric / Non-Parametric** | Welch's test by default (Welch's t-test for two groups, Welch ANOVA for more). A rank test (Mann-Whitney U, Kruskal-Wallis) when you force the non-parametric route, or when the groups are very small (n < 15) and strongly skewed. Normality and equal variances are reported as diagnostics; they do not choose the test. |

### Available Statistical Tests

#### Univariate

| Test | When Used |
| --- | --- |
| **Welch's t-test** | Two-group comparison of means (the default; does not assume equal variances). |
| **Student's t-test** | Two-group comparison of means assuming equal variances (chosen by hand in the detailed view). |
| **Mann-Whitney U** | Two-group comparison (non-parametric). |
| **Welch ANOVA** | Multi-group comparison of means (the default; does not assume equal variances). |
| **One-way ANOVA** | Multi-group comparison of means assuming equal variances (chosen by hand in the detailed view). |
| **Kruskal-Wallis** | Multi-group comparison (non-parametric). |
| **Chi-squared** | Categorical vs categorical association. |
| **Fisher's Exact** | Categorical association with small samples. |
| **Pearson Correlation** | Linear relationship between two continuous variables. |
| **Spearman Correlation** | Monotonic relationship (non-parametric). |

#### Post-hoc & Corrections

| Method | Description |
| --- | --- |
| **Games-Howell** | Post-hoc pairwise comparison after Welch ANOVA (the default route). |
| **Tukey HSD** | Post-hoc pairwise comparison after ANOVA. |
| **Dunn's Test** | Post-hoc after Kruskal-Wallis (tie-corrected, Holm-adjusted). |
| **Holm** | Family-wise error rate correction; the default (*Holm (recommended)*). |
| **Bonferroni** | Conservative multiple testing correction. |
| **FDR (Benjamini-Hochberg)** | False discovery rate correction. |

### Effect Size Interpretation

| Measure | Applies To |
| --- | --- |
| **Hedges' g** | Two-group mean differences (Welch or Student t-test). |
| **Rank-biserial r** | Mann-Whitney U. |
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
- A summary per variable: the reference mean and SD, the mean, SD and range of the Z-scores, and the number and share of values with |Z| > 2.
- Plots: mean Z-scores by group (forest plot), a heatmap across samples and an individual profile.
- **Download Z-Score Data** (the data with one `<variable>_Z` column per variable) and **Download Summary Table** (Excel). The Z-scores are not added to the dataset.

---

## Methodological Guidance
- Use the **Which test?** page (Analyze > Which test?) to identify the most appropriate test: four questions give the recommended analysis, the effect size to report and where to run it.
- **Normality** tests (Shapiro-Wilk) are reported as diagnostics, not used to choose the test: Welch's tests are the default, and a rank test suits very small, strongly skewed groups.
- For multiple comparisons, always apply a correction method to control false positives.
- Power analysis should be performed **before** data collection when possible (a priori).

---

## Survival & regression (Analyze > Survival & regression)

| Method | What it gives | Reference |
| --- | --- | --- |
| Kaplan-Meier | survival curve per group with pointwise 95% Greenwood (log-log) intervals, median survival | Kaplan & Meier 1958 |
| Log-rank test | chi-square test of equal survival across groups | Mantel 1966 |
| Cox proportional hazards | hazard ratios with Wald intervals (Efron ties); Grambsch-Therneau proportional-hazards test and concordance when the `survival` extra (lifelines) is installed | Cox 1972 |
| Logistic regression | adjusted odds ratios with Wald intervals, pseudo-R2, the apparent (in-sample) DeLong AUC, which overstates the discrimination in new patients; warns below 10 events per parameter | Hosmer et al. 2013 |
| Poisson / negative binomial | incidence-rate ratios with an optional person-time offset; over-dispersion check | Cameron & Trivedi 2013 |

Categorical covariates are one-hot coded with their first (alphabetical) level as reference; the page states it.

The event indicator must read 1 (event) or 0 (censored): numbers 0 and 1, booleans, or words such as Yes/No,
Dead/Alive and Event/Censored. Any other code is refused with a message rather than guessed. A coding of 1 =
censored and 2 = event (as in R's survival package) must be recoded first, and a competing event needs its own
analysis.

## Paired, rates & stratified (Analyze > Hypothesis tests & epidemiology)

| Method | What it gives |
| --- | --- |
| Paired t-test / Wilcoxon signed-rank | mean or pseudo-median within-pair difference with 95% CI, Cohen's d_z, matched-pairs rank-biserial r |
| One-sample t / Wilcoxon / exact binomial | difference from a reference value, Wilson interval for a proportion |
| McNemar (exact) | paired binary outcomes from the discordant pairs |
| Incidence rates | exact Poisson intervals; rate ratio with the exact conditional interval and test (Clopper-Pearson, as R's `poisson.test`), which agree: the interval excludes 1 exactly when p < 0.05 |
| Mantel-Haenszel | pooled OR (Robins-Breslow-Greenland interval) and RR (Greenland-Robins), Breslow-Day homogeneity, crude vs adjusted comparison |
| Standardised ratio (SMR/SIR) | observed / expected with the exact Poisson interval |
| Cohen's kappa, ICC(2,1), ICC(3,1) | agreement and reliability with intervals and the usual interpretation scales (on the *Diagnostic accuracy & agreement* tab) |

## Analyses with multiply imputed data

Run the multiple imputation on **Data > Enrich > Handle missing data > Multiple imputation (for inference)**
(chapter 3). The sidebar then offers **Analysis data**: *Observed* or *Imputed (m = ...)*. With *Imputed*, the
analysis pages run each analysis on every imputed dataset of the analysis population and combine the results
with Rubin's rules (van Buuren 2018). The complete-case analysis is shown under each pooled result.

| Page | Pooled analyses |
| --- | --- |
| Survival & regression | Kaplan-Meier curves (pooled on the log(-log) scale, median read from the pooled curve), log-rank test (D2), Cox, logistic and count models (ratios pooled on the log scale, D1 test of each factor) |
| Hypothesis tests > Analysis | Welch t-test (two groups); group means with robust (HC3) variances and a D1 test (three or more groups); odds ratio, risk ratio and risk difference (binary variable); chi-square test of independence (D2); ANCOVA (Type II tests by D1, adjusted means, R2 by Harel's method) |
| Hypothesis tests > Paired, rates & stratified | paired and one-sample t-tests |

**Test selection.** The group comparison pooled over the imputed datasets uses the recommended analysis by
default. Neither recommended analysis of a numeric variable assumes equal variances, as on the observed data:

- **two groups:** Welch's t-test;
- **three groups or more:** a linear model with robust (HC3) variances, tested by D1 (MacKinnon & White 1985;
  Long & Ervin 2000). Welch's ANOVA, used on the observed data, has no pooling rule; the robust model also gives
  each group its own variance. Each group needs two values or more: deselect or merge a group of one value.

With *Choose the method*, a numeric variable in two groups can be compared with Student's t-test instead, and three
groups or more with the equal-variance linear model (a one-way ANOVA), each pooled the same way. The page then
warns that the recommended analysis does not assume equal variances and stays valid when they are equal (Delacre
2017 for Welch's test). The summary and the study dossier mark the analysis as chosen. Categorical variables have
a single pooled analysis, and rank tests and Fisher's exact test have no pooling rule: run them on the observed
data.

How to read a pooled table:

- **df**: Barnard-Rubin degrees of freedom. They are small when the imputed datasets disagree.
- **FMI**, the fraction of missing information: the share of the information about the estimate that the
  missing values cost. A large FMI with a small m calls for more imputations (White, Royston & Wood 2011).
- **RIV**: the relative increase in variance due to the missing values.
- **D1** tests several coefficients at once (a factor) with a pooled Wald test. **D2** combines chi-square
  statistics; it is approximate.

Not pooled: rank tests (Mann-Whitney, Kruskal-Wallis, Wilcoxon), Fisher's exact test, Hodges-Lehmann shifts
and bootstrap intervals give no estimate with a variance to combine; the ICC and Bland-Altman limits have no
validated pooling rule; z-scores and min-max scaling depend on each imputed dataset. The page gives the reason
and the alternative. McNemar, Mantel-Haenszel, diagnostic accuracy and kappa analyse the observed data in this
version.

**Zero cells in a 2x2 table.** A zero cell gets the +0.5 (Haldane-Anscombe) correction in its own imputed
dataset only. When that happens in some imputed datasets but not all, their odds and risk ratios measure
different things, so they are not pooled: the page pools the risk difference and says why.
- The option *Zero cells: add 0.5 to every cell of every 2x2 table* applies the correction everywhere, so that the
  ratios can be pooled.
- Report that run as a sensitivity analysis; the page and the dossier label it so.

An imputation is used only while it fits the analysis population. After a change of the ticked criteria or
of the imputed variables, the pages say why the imputed data are not used and analyse the observed data until
the imputation is run again.

Values filled by **single imputation** (Global imputation, formula imputation) are treated as observed by
every analysis. A result that uses them shows a warning: its standard errors and p-values are too small.
