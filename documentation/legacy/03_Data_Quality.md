# Data Quality

## Overview
The **Data Quality** page provides a deep-dive dashboard into the health of the dataset. Unlike the preparation page, which focuses on fixing issues, this page focuses on **monitoring and scoring** the dataset's quality dimensions using a configurable weighting system.

## Key Features

### 1. Configurable Scoring System
* **Custom Weights**: Users can adjust the importance of different quality dimensions (e.g., giving higher weight to "Clinical Validity" over "Uniqueness") to tailor the Global Score to their specific research needs.
* **Global Quality Score**: A single, aggregate score (0-100) represented by a gauge chart, providing an instant summary of dataset health.

### 2. Health Scorecard
* **Detailed Metrics**: Displays individual scores for:
* **Completeness**
* **Statistical Validity**
* **Clinical Validity**
* **Consistency**
* **Uniformity**
* **Uniqueness**
* **Visual Indicators**: Metrics are color-coded (Green/Orange/Red) based on thresholds (<80%, <95%, >95%) to highlight areas requiring attention.

### 3. AI-Driven Recommendations
* **Actionable Advice**: Automatically generates a prioritized list of "Recommended Actions" (High/Medium/Low severity) to fix specific data issues.
* **Best Practices**: Provides educational context on *why* certain metrics matter (e.g., explaining the importance of Clinical Validity vs. Statistical Validity).

### 4. Detailed Analysis Tabs
* **Completeness**: Visualizes missing data patterns with bar charts and tables.
* **Validity (Outliers)**:
* **Method**: Uses the Interquartile Range (IQR) method to detect statistical outliers in numerical columns.
* **Visualization**: Bar charts showing the count and percentage of outliers per column.
* **Clinical Validity**:
* **Anomaly Report**: A detailed table highlighting specific rows that violate expert-defined clinical rules (e.g., "Age > 120").
* **Download**: Allows exporting the anomaly report as a CSV for offline review.
* **Rules Display**: Shows the active clinical criteria (Numeric bounds or Python expressions) used for validation.
* **Conformity**: Checks adherence to expected data formats and constraints.

## Usage Guide
1. **Configure Weights**: Expand "Score Weights Configuration" to adjust the sliders based on what matters most for your study.
2. **Assess Global Score**: Check the gauge chart to see if the dataset meets the overall quality threshold.
3. **Review Advice**: Check the "AI-Driven Recommendations" for immediate fix suggestions.
4. **Deep Dive**: Use the tabs at the bottom to explore specific issues:
* Go to **Completeness** to see which variables are most affected by missingness.
* Go to **Clinical Validity** to download the list of patients violating protocol rules.

## Technical Details
* **File**: `app_pages/data_quality.py`
* **Dependencies**: `explore.data_quality.DataQualityAuditor`, `plotly.graph_objects`, `plotly.express`
