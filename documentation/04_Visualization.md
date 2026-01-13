# 📊 Visualization

## Overview

The **Visualization** module is a comprehensive plotting suite powered by Plotly. It allows users to explore data distributions, relationships, and trends with interactive, publication-ready charts. It uniquely integrates statistical testing directly into the visualizations.

## Key Features

### 1. 📈 Chart Types

* **Distribution**: Histograms, Box Plots, Violin Plots (with statistical annotations).
* **Categorical**: Bar Plots, Pie Charts.
* **Relationships**: Scatter Plots, Correlation Matrices, Hierarchical Clustermaps.
* **Medical**: Bland-Altman Plots, ROC Curves, Kaplan-Meier (if applicable).
* **Publication Standards**: All charts are generated largely in vector format (SVGs) or high-res PNGs suitable for manuscript submission.

### 2. 📐 Integrated Statistics

* **Significance Testing**: Box and Violin plots include an option to "Show Statistical Significance".
* **Auto-Detection**: Automatically selects appropriate tests (T-test vs. Mann-Whitney, ANOVA vs. Kruskal-Wallis) based on data characteristics.
* **Visual Annotation**: Significant pairwise differences are highlighted directly on the plot with brackets and p-values.
* **Detailed Results**: A collapsible table below the plot provides full statistical details (Test statistic, P-value, Effect size).

### 3. 🛠️ Data Controls

* **Filtering**:
  * **Anomalies**: Option to exclude rows flagged as clinical anomalies.
  * **Inclusion**: Option to restrict the view to the study population (inclusion criteria).
* **Outlier Handling**: Integrated outlier management to clean data before visualization.

### 4. 🎨 Customization

* **Themes**: Support for multiple Plotly themes (Light, Dark, Seaborn, ggplot2) to match journal styles.
* **Color Palettes**: Extensive selection of discrete and continuous color scales (Viridis, Plasma, Set1).
* **Export**: One-click download of charts as **PNG**, **SVG**, or **HTML** (interactive).
* **Interactive Guides**: Each plot type comes with a built-in guide explaining "When to use", "Inputs", and "What to look for".

## Usage Guide

1. **Filter Data**: Use the top expanders to exclude anomalies or apply inclusion criteria if needed.
2. **Select Category**: Choose a plot category (e.g., Distribution, Relationships) from the sidebar.
3. **Configure Plot**: Select the X, Y, and Grouping variables.
4. **Enable Statistics**: For Box/Violin plots, check "Show Statistical Significance" to perform hypothesis testing.
5. **Customize**: Use the "Visual Settings" expander to adjust colors, themes, and dimensions.

## Technical Details

* **File**: `app_pages/visualization.py`
* **Dependencies**: `plotly.express`, `plotly.graph_objects`, `scipy.stats`, `app_pages.visualization_utils`
