# Interactive Clinical Visualisation & Publication Export Suite

The **Visualisation Suite** in CohortStudio provides an intuitive, publication-grade clinical graphics and hypothesis testing engine built on Plotly, SciPy, and statsmodels. Every chart couples high-resolution visual exploration with rigorous biostatistical testing, clinical outlier filters, dynamic themes, and instant multi-format figure exports.

---

## 1. Visualisation Architecture

The visualization workflow follows a decoupled, three-stage processing pipeline:

```
[ Active Cohort / DBManager ]
              │
              ▼
 1. Inclusion & Outlier Filtering (Clinical Masks, IQR Bounds, Isolation Forest)
              │
              ▼
 2. Plot Generation & Statistical Overlay (Pairwise Tests, FDR/Bonferroni, ROC AUC, Clustermaps)
              │
              ▼
 3. High-Fidelity Rendering & Fast Export (SVG, Interactive HTML, JSON, CSV, Excel)
```

---

## 2. Complete Clinical Plot Library

### A. Distribution & Modality
* **Histogram with Multiple Representations**: Switch between discrete frequency bins, Frequency Polygons (line connecting bin centers), and Smooth Kernel Density Estimation (KDE) probability density curves across patient strata.
* **Box Plot (Box-and-Whisker)**: Displays five-number summaries (minimum, $Q_1$, median, $Q_3$, maximum) with customizable horizontal or vertical orientation. Can annotate pairwise statistical significance brackets between groups (section 3).
* **Violin Plot**: Hybrids box plots with kernel density estimation to reveal bimodal or multimodal sub-distributions, skewed biomarker levels, and group variance disparities.

### B. High-Dimensional & Relational Exploration
* **Hierarchical Clustermap**:
  - Generates interactive dual row and column dendrograms reorganized by Euclidean or correlation distances.
  - Supports log and Z-score transformations.
  - Incorporates sample group color tracks and feature category annotations.
  - **Minimal Conflict Diagnostics**: A diagnostic of the missing values that identifies the minimal conflicting subsets of variables lacking shared patient observations.
* **Correlation Matrix Heatmap**:
  - Pairwise Pearson, Spearman rank, or Kendall tau correlation coefficients.
  - Hierarchical clustering reordering to group co-regulated clinical markers.
  - Lower, upper or full triangle (when rows and columns are not clustered). Only the coefficients are shown: no $p$-values, and no pair is hidden for being non-significant.
* **Scatter Plot with Linear Regression**:
  - Two-dimensional Cartesian relationships with color stratification.
  - Automated linear regression fit ($y = \beta_0 + \beta_1 x$) displaying slope, intercept, Pearson $r$, coefficient of determination ($R^2$), and regression $p$-value.

### C. Diagnostic & Medical Specialty Charts
* **Receiver Operating Characteristic (ROC) Curve**:
  - Calculates Area Under the Curve (AUC) for binary diagnostic endpoints, with its DeLong 95% confidence interval in the legend.
  - Computes optimal Youden's $J$ statistic threshold ($J = \text{Sensitivity} + \text{Specificity} - 1$).
  - Reports the sensitivity and specificity at that threshold (*Statistical Insights*).

Other diagnostic charts are on other pages:

* **Likelihood ratios** ($LR+$, $LR-$) and the **Bland-Altman agreement plot**: Analyze > Hypothesis tests & epidemiology, *Diagnostic accuracy & agreement* tab (chapter 5).
* **Q-Q plots** of residuals: the detailed view and the ANCOVA diagnostics of Analyze > Hypothesis tests & epidemiology.
* **Nullity matrix** (missingness pattern): Data > Validate, *Completeness (Missing)* tab of the quality dashboard (chapter 2).

### D. Categorical Trajectories & Hierarchies
* **Sankey Flow Diagram**: Maps patient progression across sequential trial stages, treatment switches, or adverse event states, with arrow thickness proportional to patient counts.
* **Sunburst, Icicle & Treemap Charts**: Renders deep nested categorical hierarchies (e.g. ICD-10 chapters $\to$ sub-categories $\to$ patient cohorts).

---

## 3. Integrated Statistical Testing & Significance Brackets

Box and violin plots can show pairwise significance brackets: tick **Show Statistical Significance** (vertical orientation, with a grouping variable). Every pair of groups is compared with the same engine as the Analyze page:

| Test Selection Strategy | Pairwise test |
| :--- | :--- |
| **Auto-Detect** (default) | Welch's $t$-test |
| **Force Parametric (T-test/ANOVA)** | Welch's $t$-test |
| **Force Non-Parametric (Mann-Whitney/Kruskal)** | Mann-Whitney $U$ test |

No omnibus test (ANOVA, Kruskal-Wallis) is run on this page. **Multiple Testing Correction**: None (default), Bonferroni or Benjamini-Hochberg (FDR), over all the pairs of the plot. A bracket is drawn for each pair with an adjusted $p < 0.05$; the *Statistical Results* table under the plot lists every pair with its statistic, its $p$-value (raw and, with a correction, adjusted) and an effect size.

Significance formatting follows standard biomedical conventions:
* `***` : $p_{\text{adj}} < 0.001$
* `**` : $p_{\text{adj}} < 0.01$
* `*` : $p_{\text{adj}} < 0.05$
* `ns` : $p_{\text{adj}} \ge 0.05$ (non-significant)

---

## 4. Visual Themes & Color Customization

### Themes
* **`plotly_dark`**: High-contrast dark theme for screens.
* **`plotly_white`**: White background, suited to print publications and reports.
* **`seaborn`**: Soft blue-gray background with muted gridlines conforming to classical Python data science styling.
* **`ggplot2`**: Gray canvas with white gridlines following R publication standards.

### Color Palettes
* **Colorblind-Safe Palettes**: `Safe`, `Cividis`, `Viridis` perceptually uniform color sequences ensuring clarity for readers with color vision deficiencies.
* **High-Contrast Categorical Sequences**: `Set1`, `Set2`, `Vivid`, `Dark24`, `Light24`, `Alphabet` (up to 24 distinct categorical hues).
* **Sequential & Diverging Gradients**: `Plasma`, `Inferno`, `RdBu` (diverging zero-centered scales), `Blues`, `YlOrRd`.

---

## 5. Multi-Format Export

The camera icon of the plot toolbar downloads the figure; the other formats are under *Figure Export & Publication Options* below the plot.

### Export Modalities

1. **Vector Graphics (SVG)**:
   - The camera icon of the Plotly toolbar downloads the figure as SVG: vector graphics without pixelation, for vector editors, LaTeX and journal submissions. There is no PNG export.
2. **Interactive HTML (`.html`)**:
   - A self-contained file with Plotly.js embedded (a few megabytes) that keeps hover tooltips, zoom, pan and trace toggling.
   - It opens offline in a web browser, without Python or CohortStudio.
3. **Plotly JSON (`.json`)**:
   - The complete figure specification, for programmatic storage or reuse.
4. **Filtered Cohort Data (`.csv` & `.xlsx`)**:
   - The rows behind the figure: the analysis population, after the *Outliers Handler*. The summary statistics shown under *Statistical Insights* are not part of these files.

---

## 6. Python API Quickstart

Visualizations can be generated programmatically using the `utils.visualization_utils` and `utils.export_utils` modules:

```python
import pandas as pd
import utils.visualization_utils as vu
import utils.export_utils as eu

# 1. Load the cohort
df = pd.read_csv("cohort_baseline.csv")

# 2. Pairwise Welch t-tests with a Benjamini-Hochberg correction, the groups in the order
#    the box plot uses (first appearance); returns (i, j, adjusted p) for the significant pairs
groups = df["treatment_arm"].dropna().unique()
values = [df.loc[df["treatment_arm"] == g, "systolic_bp"].dropna().to_numpy() for g in groups]
pairs = vu.compute_significant_pairs(values, test_type="Auto-Detect",
                                     correction_method="Benjamini-Hochberg (FDR)")

# 3. Box plot with a bracket for each significant pair
fig = vu.create_boxplot_with_significance_streamlit(
    df, group_col="treatment_arm", value_col="systolic_bp", significant_pairs=pairs,
    title="Systolic blood pressure by arm", y_label="Systolic BP (mmHg)",
)

# 4. Interactive HTML that opens offline (Plotly.js embedded)
with open("systolic_bp_by_arm.html", "w", encoding="utf-8") as f:
    f.write(eu.fig_to_html_str(fig, include_plotlyjs=True))

# 5. Toolbar configuration for st.plotly_chart: the camera icon downloads an SVG
config = eu.get_publication_plot_config(filename="systolic_bp_by_arm", format_type="svg")
```

`vu.compute_pairwise_stats(values, labels, ...)` returns the full table of pairs (statistic, raw and adjusted $p$-values, estimate and effect size).
