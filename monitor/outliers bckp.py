######################################## PACKAGES ########################################
import sys # some system functions
import os # for path etc
import pandas as pd # for Dataframes manipulation
import numpy as np # extend some specific Dataframes manipulation
from sklearn.neighbors import LocalOutlierFactor # Local Outlier Factor for outliers detection - density neighborhood based
from sklearn.ensemble import IsolationForest # Isolation Forest for outliers detection - isolation by forest successive splits 
from sklearn.impute import SimpleImputer # basic imputation
from sklearn.pipeline import Pipeline # building pipelines

######################################## STATISTICAL OUTLIERS FUNCTIONS DEFINITION ########################################

def find_outliers_zscore(series, threshold=3):
    """
    Finds outliers in a numerical Series using Z-score.

    Parameters:
    - series (pandas.Series): The Series to find outliers in.
    - threshold (float, optional): The Z-score threshold for identifying outliers.
      Values with absolute Z-score greater than this threshold are considered outliers.
      Default is 3.

    Returns:
    - pandas.Series: A boolean Series indicating outliers, with True for outliers and False otherwise.

    This function computes the Z-score for each value in the input Series. For numerical columns
    (int64 or float64 data types), it calculates the Z-score as (value - mean) / standard deviation.
    Outliers are identified as values with absolute Z-score greater than the specified threshold.
    For non-numeric columns, it returns an empty Series with dtype=bool.
    """
    if series.dtype in ['int64', 'float64']:
        z_scores = (series - series.mean()) / series.std()
        outliers = (z_scores.abs() > threshold)
        return outliers
    else:
        # Return an empty Series for non-numeric columns
        return pd.Series(index=series.index, dtype=bool, data=False)


def find_outliers_boxplot(series, tolerance=1.5):
    """
    Finds potential outliers in a numerical Series using boxplot statistics.

    Parameters:
    - series (pandas.Series): The Series to find potential outliers in.
    - tolerance (float, optional): The IQR tolerance for identifying outliers.
      Higher IQR tolerance leads to less sensitivity in detecting outliers. Increase this value if too much outliers are found with this method.
      Default is 1.5.

    Returns:
    - pandas.Series: A boolean Series indicating potential outliers, with True for outliers and False otherwise.

    This function calculates the lower and upper bounds for potential outliers using the
    interquartile range (IQR) method based on boxplot statistics. For numerical columns
    (int64 or float64 data types), it calculates the lower bound as the first quartile (Q1)
    minus tolerance times the IQR, and the upper bound as the third quartile (Q3) plus tolerance times the IQR.
    Values outside this range are considered potential outliers. For non-numeric columns,
    it returns an empty Series with dtype=bool.
    """
    if series.dtype not in ['int64', 'float64']: 
        # Skip non-numeric columns
        return pd.Series(index=series.index, dtype=bool, data=False)
    else:
        # Use boxplot statistics to identify potential outliers
        lower_bound = series.quantile(0.25) - abs(tolerance) * (series.quantile(0.75) - series.quantile(0.25))
        upper_bound = series.quantile(0.75) + abs(tolerance) * (series.quantile(0.75) - series.quantile(0.25))
        outliers = (series < lower_bound) | (series > upper_bound)
        ## Exclude binary case which are not well handled by this method
        # Convert the expected set to a string representation to handle nan values
        unique_values = series.unique()
        unique_values_str = set(str(value) for value in unique_values)
        expected_set_str = set(str(value) for value in {0.0, 1.0, np.nan})
        if unique_values_str == expected_set_str:
            return pd.Series(index=series.index, dtype=bool, data=False)
        else:  
            return outliers


# réfléchir à une alternative qui s'appuierait sur des quantiles bas de fréquence. Hypothèse d'outliers basée sur une sous-représentation
def find_categorical_outliers(series, threshold=1):
    """
    Finds potential outliers in a categorical Series based on the frequency of categories.

    Parameters:
    - series (pandas.Series): The categorical Series to find potential outliers in.
    - threshold (int, optional): The threshold for identifying potential outliers based on z-scores. 
      Categories with z-scores below the negative threshold are considered potential outliers. 
      Default is 1.

    Returns:
    - pandas.Series: A Series with potential outlier categories marked as None, and non-outlier categories unchanged.

    This function identifies potential outliers in a categorical Series based on the frequency
    of categories. It calculates the z-scores for each category count and identifies categories
    with z-scores below the negative threshold as potential outliers. The original Series is
    mapped, marking potential outlier categories as None and leaving non-outlier categories unchanged.
    """
    # Check if the column is not of dtype 'object' (i.e., not categorical)
    if series.dtype not in ['object']:
        # Skip non-categorical columns by returning an empty Series with the same index
        return pd.Series(index=series.index, dtype=object)

    # Calculate the normalized value counts for each category in the column
    category_counts = series.value_counts(normalize=True)
    
    # Calculate z-scores for each category count
    z_scores = (category_counts - category_counts.mean()) / category_counts.std()
    
    # Print information about z-scores for debugging
    #print("z_scores:", z_scores, " category_counts.mean():", category_counts.mean(), " category_counts.std():", category_counts.std())
    
    # Create a boolean mask for potential outliers based on the specified threshold
    outlier_mask = (z_scores < (-1*threshold))
    
    # Map the original series, marking categories identified as outliers with None
    return series.map(lambda x: x if x in outlier_mask.index[outlier_mask].tolist() else None)


# Local Outlier Factor (LOF)
def find_lof_outliers(data):
    """
    Detects outliers in the given dataset using Local Outlier Factor (LOF).

    Parameters:
    - data (pandas.DataFrame): The input DataFrame containing numerical data.

    Returns:
    - numpy.ndarray: An array of boolean values indicating whether each data point is an outlier.
    """
    # Create a pipeline to handle missing values and apply LOF
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('clf', LocalOutlierFactor())
    ])

    # Fit the pipeline to numerical data
    outlier_tags = pipeline.fit_predict(data) == -1

    return outlier_tags  # Return the outlier tags for the DataFrame


def compute_lof_scores(data):
    """
    Computes the Local Outlier Factor (LOF) scores for the given dataset.

    Parameters:
    - data (pandas.DataFrame): The input DataFrame containing numerical data.

    Returns:
    - numpy.ndarray: An array of LOF scores for each data point in the DataFrame.
    """
    # Create a pipeline to handle missing values and apply LOF
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), 
        ('clf', LocalOutlierFactor())
    ])
    # Fit the pipeline to numerical data
    pipeline.fit(data)
    # Compute LOF scores for the entire DataFrame
    lof_scores = pipeline.named_steps['clf'].negative_outlier_factor_
    return lof_scores  # Return the LOF scores for the DataFrame


# Isolation Forest
def find_isolation_forest_outliers(data, n_estimators=100, max_samples='auto', contamination='auto'):
    """
    Detects outliers in the given dataset using Isolation Forest algorithm.

    Parameters:
    - data (pandas.DataFrame): The input DataFrame containing numerical data.

    Returns:
    - numpy.ndarray: An array of boolean values indicating whether each data point is an outlier.
    """
    # Create a pipeline to handle missing values and apply LOF
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('clf', IsolationForest(n_estimators = n_estimators, max_samples=max_samples, contamination=contamination))
    ])
    # Fit the pipeline to numerical data
    outlier_tags = pipeline.fit_predict(data) == -1
    return outlier_tags  # Return the outlier tags for the DataFrame


def compute_isolation_forest_scores(data, n_estimators=100, max_samples='auto', contamination='auto'):
    """
    Computes the Isolation Forest scores for the given dataset.

    Parameters:
    - data (pandas.DataFrame): The input DataFrame containing numerical data.

    Returns:
    - numpy.ndarray: An array of LOF scores for each data point in the DataFrame.
    """
    # Create a pipeline to handle missing values and apply LOF
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('clf', IsolationForest(n_estimators = n_estimators, max_samples=max_samples, contamination=contamination))
    ])
    # Fit the pipeline to numerical data
    outlier_computer = pipeline.fit(data)
    outlier_scores = outlier_computer.decision_function(data)
    return outlier_scores  # Return the outlier tags for the DataFrame

######################################## OUTLIERS TAGGING FUNCTIONS DEFINITION ######################################## 
# Some pbs with merging it into one function...

def tag_cat_columns(row):
    """
    Create a comma-separated list of categorical columns with outliers in a row.

    Parameters:
    - row (pd.Series): A row in the DataFrame.

    Returns:
    - str: Comma-separated list of columns with outliers.
    """
    outlier_columns = []
    for col in row.index:
        if pd.notna(row[col]): # working only for cat : pd.notna(row[col])
            outlier_columns.append(col)
    return ', '.join(map(str, outlier_columns))


def tag_num_columns(row):
    """
    Create a comma-separated list of numerical columns with outliers in a row.

    Parameters:
    - row (pd.Series): A row in the DataFrame.

    Returns:
    - str: Comma-separated list of columns with outliers.
    """
    outlier_columns = []
    for col in row.index:
        if row[col]:
            outlier_columns.append(col)
    return ', '.join(map(str, outlier_columns))
