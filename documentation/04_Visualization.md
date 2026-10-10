# Interactive Clinical Visualisation & Publication Export Suite

The **Visualisation Suite** in the Cohort Monitoring Package (CMP) provides an intuitive, publication-grade clinical graphics and hypothesis testing engine built on Plotly, SciPy, and statsmodels. Every chart couples high-resolution visual exploration with rigorous biostatistical testing, clinical outlier filters, dynamic themes, and instant multi-format figure exports.

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
 3. High-Fidelity Rendering & Fast Export (SVG, 300 DPI PNG, Interactive HTML, JSON, Excel)
```

---

## 2. Complete Clinical Plot Library

### A. Distribution & Modality
* **Histogram with Multiple Representations**: Switch seamlessly between discrete frequency bins, Frequency Polygons (line connecting bin centers), and Smooth Kernel Density Estimation (KDE) probability density curves across patient strata.
* **Box Plot (Box-and-Whisker)**: Displays five-number summaries (minimum, $Q_1$, median, $Q_3$, maximum) with customizable horizontal or vertical orientation. Automatically annotates pairwise statistical significance brackets between groups.
* **Violin Plot**: Hybrids box plots with kernel density estimation to reveal bimodal or multimodal sub-distributions, skewed biomarker levels, and group variance disparities.

### B. High-Dimensional & Relational Exploration
* **Exclusive Hierarchical Clustermap**: 
  - Generates interactive dual row and column dendrograms reorganized by Euclidean or correlation distances.
  - Supports log and Z-score transformations.
  - Incorporates sample group color tracks and feature category annotations.
  - **Minimal Conflict Diagnostics**: An exclusive diagnostic algorithm that automatically analyzes high-dimensional missingness and identifies the minimal conflicting subsets of variables lacking shared patient observations.
* **Correlation Matrix Heatmap**:
  - Pairwise Pearson, Spearman rank, or Kendall tau correlation coefficients.
  - Hierarchical clustering reordering to group co-regulated clinical markers.
  - Automatic masking of statistically non-significant correlation pairs ($p \ge 0.05$).
* **Scatter Plot with Linear Regression**:
  - Two-dimensional Cartesian relationships with color/size stratification.
  - Automated linear regression fit ($y = \beta_0 + \beta_1 x$) displaying slope, intercept, Pearson $r$, coefficient of determination ($R^2$), and regression $p$-value.

### C. Diagnostic & Medical Specialty Charts
* **Receiver Operating Characteristic (ROC) Curve**:
  - Calculates Area Under the Curve (AUC) for binary diagnostic endpoints.
  - Computes optimal Youden's $J$ statistic threshold ($J = \text{Sensitivity} + \text{Specificity} - 1$).
  - Reports sensitivity, specificity, positive likelihood ratio ($LR+$), and negative likelihood ratio ($LR-$) at the optimal decision boundary.
* **Bland-Altman Agreement Plot**:
  - Compares two clinical measurement instruments or assays on identical patients.
  - Plots mean measurement against pairwise difference ($A - B$).
  - Draws mean bias reference line and $\pm 1.96 \text{ SD}$ Limits of Agreement (LoA) with 95% confidence intervals.
* **Quantile-Quantile (Q-Q) Plot**:
  - Evaluates empirical distribution quantiles against theoretical standard normal quantiles to validate parametric assumptions before linear modeling.
* **Nullity & Sparsity Matrix**:
  - Displays patient-by-variable missingness flags to identify systematic data dropouts across clinical visits.

### D. Categorical Trajectories & Hierarchies
* **Sankey Flow Diagram**: Maps patient progression across sequential trial stages, treatment switches, or adverse event states, with arrow thickness proportional to patient counts.
* **Sunburst, Icicle & Treemap Charts**: Renders deep nested categorical hierarchies (e.g. ICD-10 chapters $\to$ sub-categories $\to$ patient cohorts).

---

## 3. Integrated Statistical Testing & Significance Brackets

CMP bridges visualization and formal inferential statistics by embedding hypothesis tests directly onto box and violin charts:

| Scenario | Parametric Route | Non-Parametric Route | Automated Significance Brackets |
| :--- | :--- | :--- | :--- |
| **Two Groups** | Student's Independent $t$-test | Mann-Whitney $U$ test | Annotated bracket with adjusted $p$-value |
| **Three or More Groups** | One-Way ANOVA + Tukey HSD | Kruskal-Wallis + Dunn's test | Pairwise comparison brackets |
| **Multiple Testing Correction** | Bonferroni single-step | Benjamini-Hochberg (FDR) | Adjusted $p < 0.05$ marked with significance stars |

Significance formatting follows standard biomedical conventions:
* `***` : $p_{\text{adj}} < 0.001$
* `**` : $p_{\text{adj}} < 0.01$
* `*` : $p_{\text{adj}} < 0.05$
* `ns` : $p_{\text{adj}} \ge 0.05$ (non-significant)

---

## 4. Visual Themes & Color Customization

### Themes
* **`plotly_dark`**: High-contrast dark mode optimized for modern researcher dashboards.
* **`plotly_white`**: Clean, minimalist white canvas ideal for formal print publications and regulatory reports.
* **`seaborn`**: Soft blue-gray background with muted gridlines conforming to classical Python data science styling.
* **`ggplot2`**: Gray canvas with white gridlines following R publication standards.

### Color Palettes
* **Colorblind-Safe Palettes**: `Safe`, `Cividis`, `Viridis` perceptually uniform color sequences ensuring clarity for readers with color vision deficiencies.
* **High-Contrast Categorical Sequences**: `Set1`, `Set2`, `Vivid`, `Dark24`, `Light24`, `Alphabet` (up to 24 distinct categorical hues).
* **Sequential & Diverging Gradients**: `Plasma`, `Inferno`, `RdBu` (diverging zero-centered scales), `Blues`, `YlOrRd`.

---

## 5. Fast, Effortless Multi-Format Export Engine

CMP eliminates the friction of capturing publication-ready graphics by providing both automated browser vector downloads and explicit one-click file exports.

### Export Modalities

1. **Publication Vector Graphics (SVG)**:
   - Configured through the Plotly modebar camera icon.
   - Generates infinite-resolution Scalable Vector Graphics (SVG) without pixelation, ideal for Adobe Illustrator, LaTeX, and high-DPI journals (Nature, NEJM, The Lancet).
2. **High-Resolution Raster (300+ DPI PNG)**:
   - Scaled at $3\times$ resolution ($3600 \times 2400 \text{ px}$) directly from client-side canvas rasterization with zero server round-trip latency.
3. **Standalone Interactive HTML (`.html`)**:
   - Single-click download via `fig_to_html_str(fig, include_plotlyjs="cdn")`.
   - Generates a self-contained HTML document that preserves interactive hover tooltips, zoom, pan, and trace toggling.
   - Ideal for sharing findings with collaborators, principal investigators, or medical monitors without requiring Python or CMP installed.
4. **Machine-Readable Plotly JSON (`.json`)**:
   - Exports complete JSON figure specification for programmatic storage, dashboard integration, or automated report compilation.
5. **Filtered Cohort Data Slice (`.csv` & `.xlsx`)**:
   - Download the exact underlying cohort subset and cross-tabulated summary statistics (Count, Mean, Std Dev, Min, 25th, Median, 75th, Max, Skewness, Kurtosis) for auditability.

---

## 6. Python API Quickstart

Visualizations can be generated programmatically using the `utils.visualization_utils` and `utils.export_utils` modules:

```python
import pandas as pd
import utils.visualization_utils as vu
import utils.export_utils as eu

# 1. Load active clinical cohort
df = pd.read_csv("cohort_baseline.csv")

# 2. Generate publication-grade Box Plot with automated significance brackets
fig = vu.create_boxplot_with_significance_streamlit(
    df=df,
    x_col="treatment_arm",
    y_col="systolic_bp",
    test_type="Auto-Detect",
    correction_method="Benjamini-Hochberg (FDR)",
    alpha=0.05,
    palette="Set1",
)

# 3. Export standalone interactive HTML for clinical study report
html_content = eu.fig_to_html_str(fig)
with open("systolic_bp_by_arm.html", "w", encoding="utf-8") as f:
    f.write(html_content)

# 4. Generate high-resolution modebar configuration for web apps
config = eu.get_publication_plot_config(
    filename="systolic_bp_publication",
    format_type="svg",
    width=1200,
    height=800,
    scale=3
)
```
