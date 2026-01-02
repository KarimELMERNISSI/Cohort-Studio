# 🛡️ Data Validation & Monitoring

## Overview

The **Data Validation & Monitoring** module is a comprehensive control center for ensuring data integrity and tracking changes over time. It combines rule-based validation (Clinical Anomalies, Inclusion Criteria) with statistical monitoring (Outliers, Drift Detection).

## Key Features

### 1. 🏥 Clinical Anomalies (Tab 2)

* **Rule Definition**: Define domain-specific rules to flag invalid data.
  * **Numeric Rules**: Set Min/Max bounds for quantitative variables (e.g., "Age must be between 0 and 120").
  * **Expression Rules**: Write custom Python expressions for complex logic (e.g., `systolic_bp > diastolic_bp`).
* **Application**: Apply these rules to create an "Anomaly Mask" and visualize the percentage of irregular rows.

### 2. 🎯 Inclusion Criteria (Tab 3)

* **Cohort Selection**: Define criteria to filter the dataset for the final study population.
* **Logic**: Similar to anomalies, use numeric bounds or expressions to define who *stays* in the study.
* **Metrics**: Automatically calculates the number of included vs. excluded patients.

### 3. 📉 Outlier Handling (Tab 4)

* **Detection Methods**:
  * **Statistical**: Z-Score, IQR (Interquartile Range), Quantile.
  * **Machine Learning**: **Isolation Forest** and **LOF (Local Outlier Factor)** for multivariate anomaly detection.
* **Handling Strategies**:
  * **Remove**: Delete rows with outliers.
  * **Clip**: Cap values at the threshold (winsorization).
  * **Tag**: Create a new boolean column flagging the row as an outlier without removing it.

### 4. 🔄 Dataset Comparison (Tab 5)

* **Version Diffing**: Compare the active dataset with any previously saved snapshot or uploaded file.
* **Change Detection**: Identifies:
  * **Added Rows**: New records present only in the current dataset.
  * **Removed Rows**: Records present only in the reference dataset.
  * **Modified Rows**: Records with the same ID but changed values.
* **Export**: Generates a detailed Excel report highlighting cell-level differences.

### 5. 📊 Data Quality Dashboard (Tab 1)

* **Integrated View**: Embeds the Data Quality Scorecard to provide immediate feedback on how validation rules affect the overall dataset health.

## Usage Guide

1. **Define Rules**: Use Tabs 2 & 3 to set up your clinical constraints and inclusion criteria.
2. **Handle Outliers**: Go to Tab 4 to detect and manage statistical outliers using advanced algorithms like Isolation Forest.
3. **Monitor Changes**: Use Tab 5 to compare your current cleaned dataset against previous versions to ensure consistency.

## Technical Details

* **File**: `app_pages/data_monitoring.py`
* **Dependencies**: `monitor.changes`, `monitor.outliers`, `scipy.stats`, `sklearn.ensemble` (Isolation Forest), `sklearn.neighbors` (LOF)
