# Data Preparation

## Overview
The **Data Preparation** module is the first step in the data processing pipeline. It focuses on assessing the initial quality of the raw data, performing type conversions, and handling missing values to ensure the dataset is ready for downstream analysis.

## Key Features

### 1. Data Quality Scorecard
* **Automated Audit**: Runs a comprehensive audit using the `DataQualityAuditor` class.
* **Key Metrics**:
* **Completeness**: Percentage of non-missing cells.
* **Uniqueness**: Percentage of unique rows (duplicate detection).
* **Statistical Validity**: Score based on the absence of statistical outliers (IQR method).
* **Consistency**: Score based on correct data type usage.
* **Clinical Validity**: Compliance with declared anomaly criteria.
* **Actionable Advice**: Generates a prioritized list of recommendations (High/Medium/Low severity) to improve data quality.

### 2. Data Type Conversion
* **Batch Conversion**: Allows users to select multiple columns and convert them to a specific data type (`int`, `float`, `string`, `category`, `datetime`) in one operation.
* **Validation**: Includes error handling to prevent crashes during invalid type coercions.

### 3. Missing Value Handling (NB: Most of the imputation process is treated in data enrichment part)
* **Detection**: Automatically identifies columns with missing values.
* **Strategies**:
* **Drop Rows**: Remove records with missing data in selected columns.
* **Fill with Mean/Median/Mode**: Statistical imputation for numeric/categorical data.
* **Fill with Value**: Manual imputation with a user-specified constant.

## Usage Guide
1. **Review Scorecard**: Check the Data Quality Scorecard to identify critical issues.
2. **Fix Types**: Use the "Data Type Conversion" expander to ensure all columns have the correct format (e.g., convert ID columns to strings, dates to datetime).
3. **Handle Missingness**: Use the "Handle Missing Values" expander to address gaps in the data before proceeding to more advanced imputation in the Enrichment module.

## Technical Details
* **File**: `app_pages/data_preparation.py`
* **Dependencies**: `explore.data_quality.DataQualityAuditor`
