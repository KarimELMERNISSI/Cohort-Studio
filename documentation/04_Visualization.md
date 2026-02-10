# 📊 Visualisation

The **Visualisation** page provides an interactive plotting suite powered by Plotly. Charts include integrated statistical testing, flexible data controls, and customisation options for publication-ready figures.

---

## 📈 Chart Types

### Distribution Plots

| Chart | Description |
| --- | --- |
| **Histogram** | Frequency distribution with optional KDE overlay. |
| **Box Plot** | Quartiles, median, whiskers, and outliers. Supports group comparison with statistical annotations. |
| **Violin Plot** | Combines box plot with kernel density estimation for richer shape information. |
| **Density Plot** | Smoothed distribution curve. |

### Relationship Plots

| Chart | Description |
| --- | --- |
| **Scatter Plot** | Two-variable relationship with optional colour grouping, trendlines, and marginal distributions. |
| **Correlation Matrix** | Heatmap of pairwise correlations across selected variables. |
| **Clustermap** | Hierarchically clustered correlation or similarity matrix. |

### Medical / Specialised Plots

| Chart | Description |
| --- | --- |
| **Bland-Altman Plot** | Agreement between two measurement methods. Shows mean difference and limits of agreement. |
| **ROC Curve** | Receiver Operating Characteristic for classification performance (AUC displayed). |
| **Kaplan-Meier Curve** | Survival analysis with optional group comparison and log-rank test. |

---

## 🧪 Integrated Statistical Testing

Statistical tests are automatically applied and overlaid on applicable charts:

| Chart Type | Auto-applied Test |
| --- | --- |
| **Box Plot** (2 groups) | Mann-Whitney U |
| **Box Plot** (3+ groups) | Kruskal-Wallis |
| **Violin Plot** | Same as Box Plot |
| **Scatter Plot** | Pearson / Spearman correlation coefficient |

Significance brackets and p-values are rendered directly on the plot.

---

## 🎛️ Data Controls

| Control | Description |
| --- | --- |
| **Variable Selection** | Choose outcome and grouping variables from the dataset. |
| **Filter** | Restrict the data to a subset before plotting. |
| **Group By** | Split the chart by a categorical variable. |
| **Aggregation** | For grouped plots, apply mean, median, sum, or count. |

---

## 🎨 Customisation

| Option | Description |
| --- | --- |
| **Title & Labels** | Custom chart title, axis labels, and legend title. |
| **Colour Palette** | Choose from preset palettes or define custom colours. |
| **Orientation** | Horizontal or vertical (for bar/box/violin). |
| **Font Size** | Adjust text sizing for axes, title, and annotations. |

---

## 📤 Export

Charts can be exported in multiple formats:

| Format | How |
| --- | --- |
| **PNG** | Via the Plotly toolbar (camera icon). |
| **SVG** | Via the Plotly toolbar. |
| **HTML** | Interactive standalone file — retains hover and zoom. |

---

## 💡 Tips

- Use **Box Plots** with a grouping variable to quickly identify significant differences between subgroups.
- Enable **statistical annotations** to get p-values directly on the figure — useful for presentations and publications.
- **Bland-Altman** plots require exactly two measurement columns; ensure both are numerical and on the same scale.
- Export as **HTML** to share interactive, zoomable figures with collaborators.
