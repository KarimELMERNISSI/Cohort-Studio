######################################## PACKAGES ########################################
import sys # some system functions
import os # for path etc
import pandas as pd # for Dataframes manipulation
import numpy as np # extend some specific Dataframes manipulation
from sklearn.neighbors import LocalOutlierFactor # Local Outlier Factor for outliers detection - density neighborhood based
from sklearn.ensemble import IsolationForest # Isolation Forest for outliers detection - isolation by forest successive splits 
from sklearn.impute import SimpleImputer # basic imputation
from sklearn.pipeline import Pipeline # building pipelines
import enrich.data_imputation as edi
from sklearn.preprocessing import OneHotEncoder, StandardScaler # preprocess non numerical variables and scale numerical ones
from sklearn.compose import ColumnTransformer

######################################## DEBUG ########################################
# Helper function for controlled debug printing
def debug_print(*args, debug=False):
    if debug:
        print(*args)

######################################## STATISTICAL OUTLIERS DETECTION FUNCTIONS DEFINITION ########################################

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

######################################## ADVANCED OUTLIERS DETECTION FUNCTIONS DEFINITION ########################################

# Local Outlier Factor (LOF)
def find_lof_outliers(data, n_neighbors=25, contamination='auto',
                      numerical_imputation_method='missforest', categorical_imputation_method='missforest',
                      remainder_columns=None, remainder_threshold=0.4, debug=False):
    """
    Detects outliers in a DataFrame using Local Outlier Factor (LOF) with preprocessing.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing both numerical and categorical features.
    - n_neighbors: int, default=20
        Number of neighbors to use for LOF.
    - contamination: str or float, default='auto'
        Proportion of outliers in the dataset.
    - numerical_imputation_method: str, default='mean'
        Imputation method for numerical columns ('mean', 'knn', or 'missforest').
    - categorical_imputation_method: str, default='most_frequent'
        Imputation method for categorical columns ('most_frequent' or 'missforest').
    - remainder_columns: list, default=None
        Columns to pass through without transformation (e.g., ID columns).
    remainder_threshold (float): Default (0.4), from 0 to 1, filter out non-numeric columns having a ratio distinct modalities / non empty rows higher than the fixed threshold if ramainder columns is set to 'auto'. If it is a 'higher than 1' number, then it defines a constant number which is directly used as threshold to evaluate the number of modalities.
    - debug: bool, default=False
        If True, prints debug information.

    Returns:
    - outlier_tags: np.ndarray
        Boolean array where True indicates an outlier.
    - lof_scores: np.ndarray
        An array of LOF scores for each data point.
    - full_pipeline: sklearn.pipeline.Pipeline
        The fitted pipeline for reuse and inspection.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame with named columns.")

    # Handle remainder columns
    if remainder_columns is None:
        remainder_columns_not_miss = []
    elif remainder_columns == 'auto':
        remainder_columns_not_miss = edi.detect_remainder_columns(data, threshold=remainder_threshold)
    else:
        remainder_columns_not_miss = [col for col in remainder_columns if col in data.columns]

    debug_print(f"Filtered remainder_columns: {remainder_columns}", debug=debug)

    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()

    # Exclude remainder columns from categorical and numerical columns
    categorical_cols = [col for col in categorical_cols if col not in remainder_columns_not_miss]
    numerical_cols = [col for col in numerical_cols if col not in remainder_columns_not_miss]

    debug_print(f"Categorical columns: {categorical_cols}", debug=debug)
    debug_print(f"Numerical columns: {numerical_cols}", debug=debug)

    # Check for missing values
    missing_values = data[categorical_cols + numerical_cols].isna().any().any()

    # Define the preprocessing pipeline
    if missing_values:
        debug_print("Missing values detected. Using imputation.", debug=debug)
        # Call the get_imputer function if there are missing values
        imputer, output_cols = edi.get_imputer(
            numerical_imputation_method=numerical_imputation_method,
            categorical_imputation_method=categorical_imputation_method,
            num_scaler=True, cat_encoder=True,
            data=data, debug=debug,
            remainder_columns=remainder_columns, remainder_strategy='drop', remainder_threshold=remainder_threshold
        )
        df_transformed = imputer.fit_transform(data)
        df_transformed = pd.DataFrame(df_transformed, columns=output_cols)

    else:
        debug_print("No missing values. Using encoding and scaling only.", debug=debug)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first'), categorical_cols)
            ],
            remainder='drop'
        )
        df_transformed = preprocessor.fit_transform(data)
        df_transformed = pd.DataFrame(df_transformed, columns=list(preprocessor.get_feature_names_out()))

    debug_print(f"Transformed data shape: {df_transformed.shape}", debug=debug)

    # Define the full pipeline with preprocessing and Local Outlier Factor
    full_pipeline = Pipeline([
        ('data_preprocessing', imputer if missing_values else preprocessor),
        ('clf', LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False))
    ])

    # Fit the pipeline and get the LOF outlier tags
    outlier_tags = full_pipeline.fit_predict(data) == -1  # Outliers are marked as -1 by LOF
    # Since LOF doesn't have a separate `fit` method when `novelty=False`, we retrieve the scores after fit_predict
    outlier_scores = -full_pipeline.named_steps['clf'].negative_outlier_factor_  # Higher scores mean more outlier-like

    debug_print(f"Outlier detection completed. Number of outliers detected: {sum(outlier_tags)}", debug=debug)

    return outlier_tags, outlier_scores, full_pipeline


### Isolation Forest
def find_isolation_forest_outliers(
    data, 
    n_estimators=100, 
    max_samples='auto', 
    contamination='auto',
    numerical_imputation_method='missforest', 
    categorical_imputation_method='missforest', 
    remainder_columns=None, 
    remainder_threshold=0.4, 
    debug=False
):
    """
    Detects outliers in a DataFrame using Isolation Forest with preprocessing.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing both numerical and categorical features.
    - n_estimators: int, default=100
        Number of base estimators in the ensemble.
    - max_samples: str or int, default='auto'
        Number of samples to draw from data for training each base estimator.
    - contamination: str or float, default='auto'
        Proportion of outliers in the data set.
    - numerical_imputation_method: str, default='missforest'
        Imputation method for numerical columns ('mean', 'knn', or 'missforest').
    - categorical_imputation_method: str, default='missforest'
        Imputation method for categorical columns ('most_frequent' or 'missforest').
    - remainder_columns: list, default=None
        Columns to pass through without transformation (e.g., ID columns).
    - remainder_threshold: float, default=0.8
        Threshold for identifying high cardinality categorical columns.
    - debug: bool, default=False
        If True, prints debug information.

    Returns:
    - outlier_tags: np.ndarray
        Boolean array where True indicates an outlier.
    - outlier_scores: np.ndarray
        An array of anomaly scores for each data point.
    - full_pipeline: sklearn.pipeline.Pipeline
        The fitted pipeline for reuse and inspection.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame with named columns.")

    # Handle remainder columns
    if remainder_columns is None:
        remainder_columns = []
    elif remainder_columns == 'auto':
        remainder_columns = edi.detect_remainder_columns(data, threshold=remainder_threshold)
    else:
        remainder_columns = [col for col in remainder_columns if col in data.columns]

    debug_print(f"Filtered remainder_columns: {remainder_columns}", debug=debug)

    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()

    # Exclude remainder columns from categorical and numerical columns
    categorical_cols = [col for col in categorical_cols if col not in remainder_columns]
    numerical_cols = [col for col in numerical_cols if col not in remainder_columns]

    debug_print(f"Categorical columns: {categorical_cols}", debug=debug)
    debug_print(f"Numerical columns: {numerical_cols}", debug=debug)

    # Check for missing values
    missing_values = data[categorical_cols+numerical_cols].isna().any().any()

    # Define the preprocessing pipeline
    if missing_values:
        debug_print("Missing values detected. Using imputation.", debug=debug)
        # Call the get_imputer function if there are missing values
        preprocessor, output_cols = edi.get_imputer(
            numerical_imputation_method=numerical_imputation_method,
            categorical_imputation_method=categorical_imputation_method,
            num_scaler=True, cat_encoder=True,
            data=data, debug=debug,
            remainder_columns=remainder_columns, remainder_strategy='drop', remainder_threshold=remainder_threshold
        )
        df_transformed = preprocessor.fit_transform(data)
        df_transformed = pd.DataFrame(df_transformed, columns=output_cols)
        debug_print(f"Transformed data shape: {df_transformed.shape}", debug=debug)

    else:
        debug_print("No missing values. Using encoding and scaling only.", debug=debug)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first'), categorical_cols)
            ],
            remainder='drop'
        )
        df_transformed = preprocessor.fit_transform(data)
        output_cols = list(preprocessor.get_feature_names_out())
        df_transformed = pd.DataFrame(df_transformed, columns=output_cols)
        debug_print(f"Transformed data shape: {df_transformed.shape}", debug=debug)
    
    # Identify and select the columns for the Isolation Forest model 
    selected_cols = [col for col in output_cols if col not in remainder_columns]
    debug_print(f"Columns selected for Isolation Forest: {selected_cols}", debug=debug)
        
    # Define the full pipeline with preprocessing and IsolationForest
    full_pipeline = Pipeline([
        ('data_preprocessing', preprocessor),
        ('clf', IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination))
    ])

    full_pipeline.fit(data)
    # Fit and predict using the full pipeline
    outlier_tags = full_pipeline.predict(data) == -1  # Outliers are marked as -1 by IsolationForest
    outlier_scores = full_pipeline.decision_function(data)

    debug_print(f"Outlier detection completed. Number of outliers detected: {sum(outlier_tags)}", debug=debug)

    return outlier_tags, outlier_scores, full_pipeline

    # # Define the Isolation Forest pipeline
    # clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination)
    # clf.fit(df_transformed)

    # # Predict outliers
    # outlier_tags = clf.predict(df_transformed) == -1  # Outliers are marked as -1
    # outlier_scores = clf.decision_function(df_transformed)

    # debug_print(f"Outlier detection completed. Number of outliers detected: {sum(outlier_tags)}", debug=debug)

    # return outlier_tags, outlier_scores, clf

### LOF
# def find_lof_outliers(data, n_neighbors=20, contamination='auto', 
#                       numerical_imputation_method='missforest', categorical_imputation_method='missforest', 
#                       remainder_columns=None, remainder_threshold=0.8, debug=False):
#     """
#     Detects outliers in a DataFrame using Local Outlier Factor (LOF) with preprocessing.

#     Parameters:
#     - data: pd.DataFrame
#         The input DataFrame containing both numerical and categorical features.
#     - n_neighbors: int, default=20
#         Number of neighbors to use for LOF.
#     - contamination: str or float, default='auto'
#         Proportion of outliers in the dataset.
#     - numerical_imputation_method: str, default='mean'
#         Imputation method for numerical columns ('mean', 'knn', or 'missforest').
#     - categorical_imputation_method: str, default='most_frequent'
#         Imputation method for categorical columns ('most_frequent' or 'missforest').
#     - remainder_columns: list, default=None
#         Columns to pass through without transformation (e.g., ID columns).
#     remainder_threshold (float): Default (0.8), from 0 to 1, filter out non-numeric columns having a ratio distinct modalities / non empty rows higher than the fixed threshold if ramainder columns is set to 'auto'. If it is a 'higher than 1' number, then it defines a constant number which is directly used as threshold to evaluate the number of modalities.
#     - debug: bool, default=False
#         If True, prints debug information.

#     Returns:
#     - outlier_tags: np.ndarray
#         Boolean array where True indicates an outlier.
#     - lof_scores: np.ndarray
#         An array of LOF scores for each data point.
#     - full_pipeline: sklearn.pipeline.Pipeline
#         The fitted pipeline for reuse and inspection.
#     """
#     if not isinstance(data, pd.DataFrame):
#         raise ValueError("Input data must be a pandas DataFrame with named columns.")

#     # Filter remainder_columns to only those present in the DataFrame
#     if remainder_columns is None:
#         remainder_columns = []
#     elif remainder_columns == 'auto':
#         debug_print(f"Filtered remainder_columns mode: {remainder_columns}", debug=debug)
#     else:
#         remainder_columns = [col for col in remainder_columns if col in data.columns]
#         debug_print(f"Filtered remainder_columns to present DataFrame columns: {remainder_columns}", debug=debug)

#     # Retrieve the imputer and the column names after preprocessing
#     imputer, output_cols = edi.get_imputer(
#         numerical_imputation_method=numerical_imputation_method, 
#         categorical_imputation_method=categorical_imputation_method, 
#         num_scaler=True, cat_encoder=True, data=data, debug=debug, 
#         remainder_columns=remainder_columns, remainder_strategy='drop', remainder_threshold=remainder_threshold
#     )

#     # Fit-transform the data with imputer
#     df_imputed = imputer.fit_transform(data)
#     df_imputed = pd.DataFrame(df_imputed, columns=output_cols)
#     debug_print(f"Original DataFrame Columns: {data.columns.tolist()}", debug=debug)
#     debug_print(f"Columns After Imputer Processing: {output_cols}", debug=debug)

#     # Select columns for LOF
#     selected_cols = [col for col in output_cols if col not in remainder_columns]
#     debug_print(f"Columns selected for LOF: {selected_cols}", debug=debug)

#     # Define the full pipeline with preprocessing and Local Outlier Factor
#     full_pipeline = Pipeline([
#         ('data_preprocessing', imputer),
#         ('clf', LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False))
#     ])

#     # Fit the pipeline and get the LOF outlier tags
#     outlier_tags = full_pipeline.fit_predict(data) == -1  # Outliers are marked as -1 by LOF
#     # Since LOF doesn't have a separate `fit` method when `novelty=False`, we retrieve the scores after fit_predict
#     lof_scores = -full_pipeline.named_steps['clf'].negative_outlier_factor_  # Higher scores mean more outlier-like

#     debug_print(f"Outlier detection completed. Number of outliers detected: {sum(outlier_tags)}", debug=debug)

#     return outlier_tags, lof_scores, full_pipeline


# # Isolation Forest
# def find_isolation_forest_outliers_old(data, n_estimators=100, max_samples='auto', contamination='auto', 
#                                    numerical_imputation_method='missforest', categorical_imputation_method='missforest', 
#                                    remainder_columns=None, remainder_threshold=0.8, debug=False):
#     """
#     Detects outliers in a DataFrame using Isolation Forest with preprocessing.

#     Parameters:
#     - data: pd.DataFrame
#         The input DataFrame containing both numerical and categorical features.
#     - n_estimators: int, default=100
#         Number of base estimators in the ensemble.
#     - max_samples: str or int, default='auto'
#         Number of samples to draw from data for training each base estimator.
#     - contamination: str or float, default='auto'
#         Proportion of outliers in the data set.
#     - numerical_imputation_method: str, default='mean'
#         Imputation method for numerical columns ('mean', 'knn', or 'missforest').
#     - categorical_imputation_method: str, default='most_frequent'
#         Imputation method for categorical columns ('most_frequent' or 'missforest').
#     - remainder_columns: list, default=None
#         Columns to pass through without transformation (e.g., ID columns).
#         remainder_threshold (float): Default (0.8), from 0 to 1, filter out non-numeric columns having a ratio distinct modalities / non empty rows higher than the fixed threshold if ramainder columns is set to 'auto'. If it is a 'higher than 1' number, then it defines a constant number which is directly used as threshold to evaluate the number of modalities.
#     - debug: bool, default=False
#         If True, prints debug information.

#     Returns:
#     - outlier_tags: np.ndarray
#         Boolean array where True indicates an outlier.
#     - outlier_scores: np.ndarray
#         An array of anomaly scores for each data point.
#     - full_pipeline: sklearn.pipeline.Pipeline
#         The fitted pipeline for reuse and inspection.
#     """
#     if not isinstance(data, pd.DataFrame):
#         raise ValueError("Input data must be a pandas DataFrame with named columns.")

#     # Filter remainder_columns to only those present in the DataFrame
#     if remainder_columns is None:
#         remainder_columns = []
#     elif remainder_columns == 'auto':
#         debug_print(f"Filtered remainder_columns mode: {remainder_columns}", debug=debug)
#     else:
#         remainder_columns = [col for col in remainder_columns if col in data.columns]
#         debug_print(f"Filtered remainder_columns to present DataFrame columns: {remainder_columns}", debug=debug)

#     # Retrieve the imputer and the column names after preprocessing
#     imputer, output_cols = edi.get_imputer(
#         numerical_imputation_method=numerical_imputation_method, 
#         categorical_imputation_method=categorical_imputation_method, 
#         num_scaler=True, cat_encoder=True, 
#         data=data, debug=debug, 
#         remainder_columns=remainder_columns, remainder_strategy='drop', remainder_threshold=remainder_threshold
#     )

#     # Fit-transform the data with imputer
#     df_imputed = imputer.fit_transform(data)
#     df_imputed = pd.DataFrame(df_imputed, columns=output_cols)
#     debug_print(f"Original DataFrame Columns: {data.columns.tolist()}", debug=debug)
#     debug_print(f"Columns After Imputer Processing: {output_cols}", debug=debug)

#     if isinstance(max_samples, int) and max_samples <= 0:
#         raise ValueError("max_samples must be a positive integer or 'auto'.")
#     if isinstance(contamination, float) and not (0 < contamination < 0.5):
#         raise ValueError("contamination must be 'auto' or a float between 0 and 0.5.")

#     # Identify and select the columns for the Isolation Forest model
#     selected_cols = [col for col in output_cols if col not in remainder_columns]
#     debug_print(f"Columns selected for Isolation Forest: {selected_cols}", debug=debug)

#     # Define the full pipeline with preprocessing and IsolationForest
#     full_pipeline = Pipeline([
#         ('data_preprocessing', imputer),
#         ('clf', IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination))
#     ])

#     full_pipeline.fit(data)
#     # Fit and predict using the full pipeline
#     outlier_tags = full_pipeline.predict(data) == -1  # Outliers are marked as -1 by IsolationForest
#     outlier_scores = full_pipeline.decision_function(data)

#     debug_print(f"Outlier detection completed. Number of outliers detected: {sum(outlier_tags)}", debug=debug)

#     return outlier_tags, outlier_scores, full_pipeline

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
