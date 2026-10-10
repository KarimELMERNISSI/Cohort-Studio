# pages/data_monitoring.py
import json
import os
import time
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import monitor.changes as mc
import monitor.outliers as mo
import utils.clustering_utils as cu
from app_pages.data_quality_dashboard import render_dashboard
from enrich import custom_metrics_and_filters as ecm
from manage.db_manager import DBManager
from manage.transformation_manager import TransformationManager
from utils.config_loader import create_empty_config, transform_expression
from utils.data_analyzer import DataAnalyzer
from utils.date_parser import smart_parse_dates
from utils.multipage import load_dataframe

#############################################################################################

def export_comparison_results(df1, df2, comparison_result_filtered, 
                               rows_only_in_df1, rows_only_in_df2, 
                               modified_common_ids, common_id, key_base):
    """
    Efficiently export comparison results to an Excel file
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Prepare and write sheets with consistent styling
        sheets = [
            ("Reference Dataset", df1),
            ("data_right", df2),
            ("summary", comparison_result_filtered)
        ]
        
        for sheet_name, dataframe in sheets:
            styled_df = dataframe.style.apply(
                mc.highlight_modifications_tuples, 
                args=(sheet_name, rows_only_in_df1, rows_only_in_df2, 
                      modified_common_ids, common_id), 
                axis=1
            )
            styled_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    output.seek(0)
    return output

############################# OUTLIER HANDLING ###############################################

class OutlierHandler:
    """Helper class to detect and handle outliers using various methods"""
    def __init__(self, df, config):
        self.df = df.copy()
        self.original_df = df.copy()
        self.masks = None  # Initialize to None
  
    def detect_outliers_zscore(self, column, threshold=3):
        """Detect outliers using Z-score method"""
        return mo.find_outliers_zscore(self.df[column], threshold)
        
    def detect_outliers_iqr(self, column, multiplier=1.5):
        """Detect outliers using IQR method"""
        return mo.find_outliers_boxplot(self.df[column], multiplier)
        
    def detect_outliers_quantile(self, column, lower=0.01, upper=0.99):
        """Detect outliers using quantile method"""
        lower_bound = self.df[column].quantile(lower, interpolation='lower')
        upper_bound = self.df[column].quantile(upper, interpolation='higher')
        return (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
    
    def detect_lof_outliers(self, data, n_neighbors=20, contamination='auto', 
                            numerical_imputation_method='missforest', 
                            categorical_imputation_method='missforest', 
                            remainder_columns='auto', 
                            remainder_threshold=0.3, 
                            debug=True):
        """
        Detect outliers using Local Outlier Factor (LOF)
        
        Parameters:
        -----------
        columns : list
            List of column names to use for outlier detection
        n_neighbors : int, optional (default=20)
            Number of neighbors to use for LOF
        contamination : str or float, optional (default='auto')
            Expected proportion of outliers in the dataset
        numerical_imputation_method : str, optional (default='missforest')
            Method for imputing numerical missing values
        categorical_imputation_method : str, optional (default='missforest')
            Method for imputing categorical missing values
        remainder_columns : str or list, optional (default='auto')
            Columns to keep if not in the selected columns
        remainder_threshold : float, optional (default=0.4)
            Threshold for keeping columns with missing values
        debug : bool, optional (default=True)
            Enable debug mode
        
        Returns:
        --------
        tuple
            Outlier tags, outlier scores, and additional information
        """
        outliers_tag, outliers_scores, additional_info = mo.find_lof_outliers(
            data,
            n_neighbors=n_neighbors,
            contamination=contamination,
            numerical_imputation_method=numerical_imputation_method,
            categorical_imputation_method=categorical_imputation_method,
            remainder_columns=remainder_columns,
            remainder_threshold=remainder_threshold,
            debug=debug
        )
        
        return outliers_tag, outliers_scores, additional_info


    def detect_isolation_forest_outliers(self, data, n_estimators=100, 
                                        max_samples='auto', 
                                        contamination='auto', 
                                        numerical_imputation_method='missforest', 
                                        categorical_imputation_method='missforest', 
                                        remainder_columns='auto', 
                                        remainder_threshold=0.3, 
                                        debug=True):
        """
        Detect outliers using Isolation Forest method
        
        Parameters:
        -----------
        columns : list
            List of column names to use for outlier detection
        n_estimators : int, optional (default=100)
            Number of trees in the forest
        max_samples : str or int, optional (default='auto')
            Number of samples to draw for each base estimator
        contamination : str or float, optional (default='auto')
            Expected proportion of outliers in the dataset
        numerical_imputation_method : str, optional (default='missforest')
            Method for imputing numerical missing values
        categorical_imputation_method : str, optional (default='missforest')
            Method for imputing categorical missing values
        remainder_columns : str or list, optional (default='auto')
            Columns to keep if not in the selected columns
        remainder_threshold : float, optional (default=0.4)
            Threshold for keeping columns with missing values
        debug : bool, optional (default=True)
            Enable debug mode
        
        Returns:
        --------
        tuple
            Outlier tags, outlier scores, and additional information
        """
        outliers_tag, outliers_scores, additional_info = mo.find_isolation_forest_outliers(
            data,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            numerical_imputation_method=numerical_imputation_method,
            categorical_imputation_method=categorical_imputation_method,
            remainder_columns=remainder_columns,
            remainder_threshold=remainder_threshold,
            debug=debug
        )
        
        return outliers_tag, outliers_scores, additional_info


    def detect_dbscan_outliers(self, data, eps=0.5, min_samples=5, 
                              numerical_imputation_method='missforest', 
                              categorical_imputation_method='missforest', 
                              remainder_columns='auto', 
                              remainder_threshold=0.3, 
                              debug=True):
        """
        Detect outliers using DBSCAN method
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        numerical_imputation_method : str
            Imputation method for numerical columns
        categorical_imputation_method : str
            Imputation method for categorical columns
        remainder_columns : str
            How to handle remainder columns
        remainder_threshold : float
            Threshold for remainder columns
        debug : bool
            Enable debug printing

        Returns:
        --------
        tuple
            (outliers_tag, outliers_scores, additional_info)
        """
        outliers_tag, outliers_scores, additional_info = mo.find_dbscan_outliers(
            data,
            eps=eps,
            min_samples=min_samples,
            numerical_imputation_method=numerical_imputation_method,
            categorical_imputation_method=categorical_imputation_method,
            remainder_columns=remainder_columns,
            remainder_threshold=remainder_threshold,
            debug=debug
        )
        return outliers_tag, outliers_scores, additional_info


    
    # def handle_outliers(self, columns, method='zscore', handling_strategy='remove', **kwargs):
    #     """
    #     Handle outliers in specified columns using the chosen method and strategy
        
    #     Parameters:
    #     -----------
    #     columns : list
    #         List of column names to handle outliers in
    #     method : str
    #         'zscore', 'iqr', or 'quantile'
    #     handling_strategy : str
    #         'remove', 'clip, tag'
    #     kwargs : dict
    #         Additional parameters for the chosen method
    #     """
    #     self.df = self.original_df.copy()  # Reset to original data
    #     #self.df = self.df.copy()

    #     if method == 'Isolation Forest':
    #         is_outlier, outlier_score, _ = self.detect_isolation_forest_outliers(data= self.df, remainder_threshold=0.4, **kwargs)
    #     elif method == 'Local Outlier Factor':
    #         is_outlier, outlier_score, _ = self.detect_lof_outliers(data= self.df, remainder_threshold=0.4, **kwargs)
    #     else:
    #         if handling_strategy == "tag":
    #             # Initialize the method-specific tag and list columns
    #             self.df[f"{method}_list"] = ""  # Empty string to hold the list of columns with outliers
    #             self.df[f"{method}_tag"] = False  # Boolean column for tagging rows with outliers

    #         for column in columns:
    #             if self.df[column].dtype not in ['int64', 'float64']:
    #                 continue

    #             # Detect outliers using the specified method
    #             if method == 'zscore':
    #                 threshold = kwargs.get('threshold', 3)
    #                 is_outlier = self.detect_outliers_zscore(column, threshold)
    #             elif method == 'iqr':
    #                 multiplier = kwargs.get('multiplier', 1.5)
    #                 is_outlier = self.detect_outliers_iqr(column, multiplier)
    #             elif method == 'quantile':
    #                 lower = kwargs.get('lower', 0.01)
    #                 upper = kwargs.get('upper', 0.99)
    #                 is_outlier = self.detect_outliers_quantile(column, lower, upper)
            
    #     # Handle outliers using the specified strategy
    #     if handling_strategy == 'remove':
    #         self.df = self.df[~is_outlier]

    #     elif handling_strategy == 'tag' and method not in ['Isolation Forest','Local Outlier Factor']:
    #         # Update the method_list column to include this column name for rows with outliers
    #         self.df.loc[is_outlier, f"{method}_list"] = (self.df.loc[is_outlier, f"{method}_list"] + ", " + column).str.strip(", ")
    #         # Update the method_tag column to mark rows with any outlier detected
    #         self.df[f"{method}_tag"] = self.df[f"{method}_tag"] | is_outlier

    #     elif handling_strategy == 'tag' and method in ['Isolation Forest','Local Outlier Factor']:
    #         print("Condition met. Creating outlier columns.")
    #         # Check if is_outlier and outlier_score are defined correctly
    #         print(f"is_outlier: {is_outlier}, outlier_score: {outlier_score}")
    #         # Add columns to the DataFrame
    #         self.df[f"{method}_tag"] = is_outlier
    #         self.df[f"{method}_score"] = outlier_score

    #     elif handling_strategy == 'clip':
    #         if method == 'zscore':
    #             z_scores = stats.zscore(self.df[column])
    #             self.df.loc[is_outlier, column] = self.df[column].mean() + threshold * self.df[column].std() * np.sign(z_scores[is_outlier])

    #         elif method == 'iqr':
    #             Q1 = self.df[column].quantile(0.25)
    #             Q3 = self.df[column].quantile(0.75)
    #             IQR = Q3 - Q1
    #             lower_bound = Q1 - multiplier * IQR
    #             upper_bound = Q3 + multiplier * IQR
    #             self.df.loc[is_outlier, column] = self.df[column].clip(lower_bound, upper_bound)

    #         elif method == 'quantile':
    #             lower_bound = self.df[column].quantile(lower)
    #             upper_bound = self.df[column].quantile(upper)
    #             self.df.loc[is_outlier, column] = self.df[column].clip(lower_bound, upper_bound)
        
    #     return self.df
    
    # def handle_outliers(self, columns, method='zscore', handling_strategy='remove', **kwargs):
    #     """
    #     Handle outliers in specified columns using the chosen method and strategy
        
    #     Parameters:
    #     -----------
    #     columns : list
    #         List of column names to handle outliers in
    #     method : str
    #         'zscore', 'iqr', 'quantile', 'Isolation Forest', or 'Local Outlier Factor'
    #     handling_strategy : str
    #         'remove', 'clip', or 'tag'
    #     kwargs : dict
    #         Additional parameters for the chosen method
    #     """
    #     self.df = self.df.copy()  # Work on the existing DataFrame to retain prior modifications
        
    #     # Initialize tag columns for 'tag' strategy if necessary
    #     if handling_strategy == "tag" and method not in ['Isolation Forest', 'Local Outlier Factor']:
    #         if f"{method}_list" not in self.df.columns:
    #             self.df[f"{method}_list"] = ""  # Hold the list of columns with outliers
    #         if f"{method}_tag" not in self.df.columns:
    #             self.df[f"{method}_tag"] = False  # Mark rows with any outliers

    #     for column in columns:
    #         if self.df[column].dtype not in ['int64', 'float64']:
    #             continue  # Skip non-numeric columns
            
    #         # Detect outliers
    #         if method == 'zscore':
    #             threshold = kwargs.get('threshold', 3)
    #             is_outlier = self.detect_outliers_zscore(column, threshold)
    #         elif method == 'iqr':
    #             multiplier = kwargs.get('multiplier', 1.5)
    #             is_outlier = self.detect_outliers_iqr(column, multiplier)
    #         elif method == 'quantile':
    #             lower = kwargs.get('lower', 0.01)
    #             upper = kwargs.get('upper', 0.99)
    #             is_outlier = self.detect_outliers_quantile(column, lower, upper)
    #         elif method == 'Isolation Forest':
    #             is_outlier, outlier_score, _ = self.detect_isolation_forest_outliers(data=self.df, **kwargs)
    #             self.df[f"{method}_score"] = outlier_score
    #         elif method == 'Local Outlier Factor':
    #             is_outlier, outlier_score, _ = self.detect_lof_outliers(data=self.df, **kwargs)
    #             self.df[f"{method}_score"] = outlier_score
    #         else:
    #             raise ValueError(f"Unsupported method: {method}")
            
    #         # Handle outliers based on the specified strategy
    #         if handling_strategy == 'remove':
    #             self.df = self.df[~is_outlier]
    #         elif handling_strategy == 'clip':
    #             if method in ['zscore', 'iqr', 'quantile']:
    #                 self.clip_outliers(column, is_outlier, method, **kwargs)
    #         elif handling_strategy == 'tag':
    #             if method not in ['Isolation Forest', 'Local Outlier Factor']:
    #                 self.df.loc[is_outlier, f"{method}_list"] = (
    #                     self.df.loc[is_outlier, f"{method}_list"] + ", " + column
    #                 ).str.strip(", ")
    #                 self.df[f"{method}_tag"] |= is_outlier
    #             else:
    #                 self.df[f"{method}_tag"] = is_outlier

    #     return self.df


    def handle_outliers(self, columns, method='zscore', handling_strategy='none', **kwargs):
        """
        Handle outliers in the DataFrame and generate an outlier matrix.
        
        Parameters:
            columns (list): List of columns to analyze for outliers.
            method (str): Method to detect outliers ('zscore', 'iqr', 'quantile', 'Local Outlier Factor', 'Isolation Forest').
            handling_strategy (str): Strategy for handling outliers ('none', 'remove', 'clip', 'tag').
            **kwargs: Additional parameters for specific outlier detection methods.

        Returns:
            DataFrame: Processed DataFrame with outliers handled.
            DataFrame: Outlier matrix with boolean values indicating outliers.
        """
        # Ensure valid methods
        valid_methods = ['zscore', 'iqr', 'quantile', 'Local Outlier Factor', 'Isolation Forest', 'DBSCAN']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}.")

        # Initialize outlier matrix and scores
        outlier_matrix = pd.DataFrame(index=self.df.index)
        outlier_score = None

        if method in ['zscore', 'iqr', 'quantile']:
            for column in columns:
                # Skip non-numeric columns
                if self.df[column].dtype not in ['int64', 'float64']:
                    continue

                # Detect outliers for the column
                if method == 'zscore':
                    threshold = kwargs.get('threshold', 3)
                    is_outlier = self.detect_outliers_zscore(column, threshold)
                elif method == 'iqr':
                    multiplier = kwargs.get('multiplier', 1.5)
                    is_outlier = self.detect_outliers_iqr(column, multiplier)
                elif method == 'quantile':
                    lower = kwargs.get('lower', 0.01)
                    upper = kwargs.get('upper', 0.99)
                    is_outlier = self.detect_outliers_quantile(column, lower, upper)

                # Add to the outlier matrix
                outlier_matrix[column] = is_outlier

        elif method in ['Local Outlier Factor', 'Isolation Forest', 'DBSCAN']:
            # Apply dataset-wide methods
            data = self.df[columns]
            if method == 'Local Outlier Factor':
                is_outlier, outlier_score, _ = self.detect_lof_outliers(data=data, **kwargs)
            elif method == 'Isolation Forest':
                is_outlier, outlier_score, _ = self.detect_isolation_forest_outliers(data=data, **kwargs)
            elif method == 'DBSCAN':
                is_outlier, outlier_score, _ = self.detect_dbscan_outliers(data=data, **kwargs)

            # Add the dataset-wide outliers to the matrix
            outlier_matrix['overall'] = is_outlier

        # Handle outliers in the main DataFrame based on strategy
        df_processed = self._apply_handling_strategy(columns, outlier_matrix, handling_strategy, method, is_outlier, outlier_score, **kwargs)

        return df_processed, outlier_matrix


    def _apply_handling_strategy(self, columns, outlier_matrix, strategy, method, is_outlier=None, outlier_score=None, **kwargs):
        """
        Apply a handling strategy to the outliers.
        
        Parameters:
            columns (list): Columns analyzed for outliers.
            outlier_matrix (DataFrame): Boolean matrix indicating outliers.
            strategy (str): Handling strategy ('none', 'remove', 'clip', 'tag').
            method (str): Outlier detection method.
            is_outlier (ndarray): Boolean array for dataset-wide methods.
            outlier_score (ndarray): Array of outlier scores for dataset-wide methods.
            **kwargs: Additional parameters for handling strategies.
        
        Returns:
            DataFrame: Processed DataFrame with outliers handled.
        """
        df_processed = self.df.copy()

        if strategy == 'none':
            return df_processed

        if strategy == 'remove':
            # Remove rows with any outliers
            outlier_rows = outlier_matrix.any(axis=1)
            df_processed = df_processed[~outlier_rows]

        elif strategy == 'clip':
            # Clip outliers to thresholds
            for col in columns:
                if col in outlier_matrix:
                    df_processed[col] = self.clip_outliers(column=col, is_outlier=is_outlier, method=method, **kwargs)
        elif strategy == 'tag':
            if method in ['zscore', 'iqr', 'quantile']:
                for column in columns:
                    if column in outlier_matrix:
                        # Initialize tag and list columns if not already present
                        tag_col = f"{method}_tag"
                        list_col = f"{method}_list"
                        if tag_col not in df_processed:
                            df_processed[tag_col] = False
                        if list_col not in df_processed:
                            df_processed[list_col] = ""

                        # Update the tag and list columns
                        df_processed.loc[outlier_matrix[column], tag_col] = True
                        df_processed.loc[outlier_matrix[column], f"{method}_list"] = (df_processed.loc[outlier_matrix[column], f"{method}_list"] + ", " + column).str.strip(", ")
                        

            elif method in ['Isolation Forest', 'Local Outlier Factor', 'DBSCAN']:
                # Initialize score and tag columns for dataset-wide methods
                tag_col = f"{method}_tag"
                score_col = f"{method}_score"

                # Update the DataFrame with the outlier information
                df_processed[tag_col] = is_outlier
                df_processed[score_col] = outlier_score

        return df_processed


    def clip_outliers(self, column, is_outlier, method, **kwargs):
        """
        Helper function to clip outliers based on the specified method.
        
        Parameters:
            column (str): Column name to process.
            is_outlier (pd.Series): Boolean mask indicating outliers.
            method (str): Method for clipping ('zscore', 'iqr', 'quantile').
            kwargs: Additional parameters for each method.
            
        Returns:
            pd.Series: Updated column with outliers clipped.
        """
        df = self.df.copy()
        if method == 'zscore':
            threshold = kwargs.get('threshold', 3)
            z_scores = stats.zscore(df[column], nan_policy='omit')
            df.loc[is_outlier, column] = (
                df[column].mean() + threshold * df[column].std() * np.sign(z_scores[is_outlier])
            )
        elif method == 'iqr':
            multiplier = kwargs.get('multiplier', 1.5)
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'quantile':
            lower = kwargs.get('lower', 0.01)
            upper = kwargs.get('upper', 0.99)
            print(f"Column {column}\n----v----v-----\n lower {lower} upper {upper}\n-----v----v----")
            lower_bound = df[column].quantile(lower)
            upper_bound = df[column].quantile(upper)
            print(f"\n----v----v-----\n lower_bound {lower_bound} upper_bound {upper_bound}\n-----v----v----\n")
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        else:
            raise ValueError("Invalid method. Choose from ['zscore', 'iqr', 'quantile']")
        return df[column]


    def get_outliers_masks(self):
            """
            Get the masks of outliers detected so far.

            Returns:
            --------
            dict or None
                Dictionary of outlier masks if available, else None.
            """
            return self.masks


    def get_outlier_summary(self, columns, method='zscore', **kwargs):
        """
        Get a summary of outliers for each column or the dataset as a whole.
        
        Parameters:
            columns (list): List of columns to analyze for outliers.
            method (str): Method to detect outliers ('zscore', 'iqr', 'quantile', 'Local Outlier Factor', 'Isolation Forest').
            **kwargs: Additional parameters for specific outlier detection methods.
        
        Returns:
            dict: Summary of outliers for each column or the dataset.
        """
        summary = {}

        # Check for valid methods
        valid_methods = ['zscore', 'iqr', 'quantile', 'Local Outlier Factor', 'Isolation Forest', 'DBSCAN']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}.")

        # Column-wise methods
        if method in ['zscore', 'iqr', 'quantile']:
            for column in columns:
                # Ensure the column is numeric
                if not pd.api.types.is_numeric_dtype(self.original_df[column]):
                    continue

                # Exclude NaN values to ensure robust calculations
                valid_data = self.original_df[column].dropna()

                # Skip columns that are empty after dropping NaNs
                if valid_data.empty:
                    continue

                # Detect outliers based on the selected method
                if method == 'zscore':
                    threshold = kwargs.get('threshold', 3)
                    is_outlier = self.detect_outliers_zscore(column, threshold) & self.original_df[column].notna()
                elif method == 'iqr':
                    multiplier = kwargs.get('multiplier', 1.5)
                    is_outlier = self.detect_outliers_iqr(column, multiplier) & self.original_df[column].notna()
                elif method == 'quantile':
                    lower = kwargs.get('lower', 0.01)
                    upper = kwargs.get('upper', 0.99)
                    is_outlier = self.detect_outliers_quantile(column, lower, upper) & self.original_df[column].notna()

                # Calculate outlier statistics
                n_outliers = is_outlier.sum()  # Total number of outliers
                total_valid_values = self.original_df[column].notna().sum()
                pct_outliers = (n_outliers / total_valid_values) * 100 if total_valid_values > 0 else 0

                # Update summary for the column
                summary[column] = {
                    'total_outliers': int(n_outliers),
                    'percentage_outliers': round(pct_outliers, 2),
                    'original_range': (
                        self.original_df[column].min(skipna=True),
                        self.original_df[column].max(skipna=True)
                    ),
                    'outlier_values': self.original_df.loc[is_outlier, column].tolist()
                }

        # Dataset-wide methods
        elif method in ['Local Outlier Factor', 'Isolation Forest', 'DBSCAN']:
            data = self.df[columns]#.dropna()
            # data = self.df[columns].select_dtypes(include=['int64', 'float64']).dropna()
            #if data.empty:
            #    raise ValueError("No numeric data available for outlier detection.")
            
            if method == 'Local Outlier Factor':
                is_outlier, scores, _ = self.detect_lof_outliers(data=data, **kwargs)
            elif method == 'Isolation Forest':
                is_outlier, scores, _ = self.detect_isolation_forest_outliers(data=data, **kwargs)
            elif method == 'DBSCAN':
                is_outlier, scores, _ = self.detect_dbscan_outliers(data=data, **kwargs)

            # Calculate overall statistics
            #n_outliers = sum(is_outlier)
            #pct_outliers = (n_outliers / len(data)) * 100
            # Calculate outlier statistics
            n_outliers = is_outlier.sum()  # Total number of outliers
            total_valid_values = self.original_df[column].notna().sum()
            pct_outliers = (n_outliers / total_valid_values) * 100 if total_valid_values > 0 else 0

            # Update summary for the dataset
            summary['overall'] = {
                'total_outliers': n_outliers,
                'percentage_outliers': pct_outliers,
                'outlier_scores': scores.tolist()
            }

        return summary



##### USE OUTLIERHANDLER ####

# def add_outlier_handling_ui_off(df, numeric_cols):
#     """Add outlier handling UI components"""
#     # Initialize outlier handler
#     #config = st.session_state.config
#     outlier_handler = OutlierHandler(df, st.session_state.config)
    
#     # Outlier detection method
#     col1, col2 = st.columns(2)
#     with col1:
#         detection_method = st.selectbox(
#             "Outlier Detection Method",
#             ["zscore", "iqr", "quantile", "Local Outlier Factor", "Isolation Forest"],
#             help="Z-score: Uses standard deviations from mean\n"
#                  "IQR: Uses interquartile range\n"
#                  "Quantile: Uses percentile thresholds"
#         )
        
#     with col2:
#         handling_strategy = st.selectbox(
#             "Handling Strategy",
#             ["None", "remove", "clip", "tag"],
#             help="Remove: Exclude outliers\n"
#                  "Clip: Cap at threshold values\n"
#                  "Tag: Tag outliers for later identification"
#         )
    
#     # Method-specific parameters
#     if detection_method == "zscore":
#         threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
#         params = {"threshold": threshold}
#     elif detection_method == "iqr":
#         multiplier = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1)
#         params = {"multiplier": multiplier}
#     elif detection_method == "quantile":
#         col1, col2 = st.columns(2)
#         with col1:
#             lower = st.number_input("Lower percentile", 0.0, 0.5, 0.01, 0.01)
#         with col2:
#             upper = st.number_input("Upper percentile", 0.5, 1.0, 0.99, 0.01)
#         params = {"lower": lower, "upper": upper}
#     elif detection_method == "Isolation Forest":
#         params = {'n_estimators': 100, 'max_samples': 'auto', 'numerical_imputation_method':'knn', 'categorical_imputation_method':'most_frequent'}
#     elif detection_method == "Local Outlier Factor":
#         params = {'n_neighbors': 20, 'contamination': 'auto', 'numerical_imputation_method':'knn', 'categorical_imputation_method':'most_frequent'}

#     if detection_method in ["quantile","zscore","iqr"]:
#         # Select columns for outlier handling
#         selected_cols = st.multiselect(
#             "Select columns for outlier handling",
#             numeric_cols,
#             default=numeric_cols[:1] if numeric_cols else []
#         )
#         if not selected_cols:
#             st.warning("Please select at least one column")
#             return df
#     else:
#         analyzer = DataAnalyzer(df)
#         selected_cols = st.multiselect(
#             "Select columns for outlier handling",
#             df.columns.to_list(),
#             default = list(set(analyzer.categorical_cols+analyzer.binary_cols+analyzer.low_cardinality_numeric_cols+analyzer.numeric_cols)) #df.columns.to_list()[:1] 
#         )
    
#     # Get outlier summary
#     summary = outlier_handler.get_outlier_summary(selected_cols, detection_method, **params)

#     # Get outlier masks
#     outlier_masks = outlier_handler.get_outliers_masks()
#     if outlier_masks:
#         st.subheader("Outlier Masks:")
#         st.json(outlier_masks)
#     # Display summary
#     st.subheader("Outlier Summary:")
#     for col, stats in summary.items():
#         st.write(f"---\n**{col}**:")
#         st.write(f"- Total outliers: {stats['total_outliers']}")
#         st.write(f"- Percentage of outliers: {stats['percentage_outliers']:.2f}%")
#         if detection_method in ["zscore","iqr","quantile"]:
#             st.write(f"- Original range: ({stats['original_range'][0]:.2f}, {stats['original_range'][1]:.2f})")
    
#     # Handle outliers if strategy is not 'none'
#     if handling_strategy != "None" and detection_method not in ('Isolation Forest', 'Local Outlier Factor'):
#         df_processed = outlier_handler.handle_outliers(
#             selected_cols,
#             method=detection_method,
#             handling_strategy=handling_strategy,
#             **params
#         )
#     elif handling_strategy != "None" and detection_method in ('Isolation Forest', 'Local Outlier Factor'):
#         df_processed = outlier_handler.handle_outliers(
#             df.columns.tolist(),
#             method=detection_method,
#             handling_strategy=handling_strategy,
#             **params
#         )
#         st.dataframe(df_processed)
#     else:
#         return df
        
#     # Show before/after statistics
#     if handling_strategy != "None" and detection_method not in ('Isolation Forest', 'Local Outlier Factor'):
#         st.write("\n**Before/After Statistics:**")
#         for col in selected_cols:
#             st.write(f"\n**{col}:**")
#             col1, col2 = st.columns(2)

#             if df[col].dtype in ['float64', 'int64']:  # Handle numerical columns
#                 with col1:
#                     st.write("Before:")
#                     st.write(f"- Mean: {df[col].mean():.2f}")
#                     st.write(f"- Std: {df[col].std():.2f}")
#                     st.write(f"- Range: ({df[col].min():.2f}, {df[col].max():.2f})")
#                 with col2:
#                     st.write("After:")
#                     st.write(f"- Mean: {df_processed[col].mean():.2f}")
#                     st.write(f"- Std: {df_processed[col].std():.2f}")
#                     st.write(f"- Range: ({df_processed[col].min():.2f}, {df_processed[col].max():.2f})")
#             else:  # Handle non-numerical columns
#                 unique_before = df[col].nunique()
#                 top_before = df[col].value_counts().idxmax()
#                 freq_before = df[col].value_counts().max()

#                 unique_after = df_processed[col].nunique()
#                 top_after = df_processed[col].value_counts().idxmax()
#                 freq_after = df_processed[col].value_counts().max()

#                 with col1:
#                     st.write("Before:")
#                     st.write(f"- Unique Values: {unique_before}")
#                     st.write(f"- Most Frequent: {top_before} ({freq_before} occurrences)")
#                 with col2:
#                     st.write("After:")
#                     st.write(f"- Unique Values: {unique_after}")
#                     st.write(f"- Most Frequent: {top_after} ({freq_after} occurrences)")
        
#     # Add a button to view outlier values
#     if detection_method not in ('Isolation Forest', 'Local Outlier Factor'):
#         if st.button("View Outlier Values"):
#             outlier_values_dialog(summary)
        
#     return df_processed

######################################## handle data for comparisons

def data_selection(tmp:str, label:str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Handle data selection from various sources.

    Parameters:
    -----------
    tmp : str
        Unique identifier for the session keys.
    label : str
        Label to display in the UI selector.

    Returns:
    --------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        The selected dataframe and its source name/path, or (None, None) if selection failed or nothing selected.
    """
    
    options = ["Upload File", "Enter File Path", "Select from History"]
    if 'data' in st.session_state and st.session_state['data'] is not None:
        options.insert(0, "Current Session Data")
        
    selection_method = st.radio(f"Choose source for {label}:", options, key=f"{tmp}_data_method")
    
    if selection_method == "Current Session Data":
        st.info("Using the dataset currently loaded in the application.")
        return st.session_state['data'].copy(), "Current Session Data"

    elif selection_method == "Select from History":
        tm = TransformationManager()
        datasets = tm.get_available_datasets_from_traces()
        
        if datasets:
            # Create formatted labels for the selectbox
            # Format: {timestamp} | {session_id} | Step {step_number} | {function}
            dataset_options = {
                f"{d['timestamp']} | {d['session_id']} | Step {d['step_number']} | {d['function']}": d 
                for d in datasets
            }
            
            selected_label = st.selectbox(
                f"Select a dataset for {label}", 
                list(dataset_options.keys()), 
                key=f"{tmp}_history_select"
            )
            
            if selected_label:
                selected_data = dataset_options[selected_label]
                file_path = selected_data['file_path']
                
                # Handle relative paths if necessary (assuming paths in trace are relative to project root)
                if not os.path.isabs(file_path):
                    file_path = os.path.abspath(file_path)

                if os.path.exists(file_path):
                    try:
                        # Use load_dataframe from multipage which handles string paths via mf.load_dataframe
                        df = load_dataframe(file_path)
                            
                        if df is not None:
                            st.success(f"Loaded: {selected_label}")
                            st.caption(f"Description: {selected_data['description']}")
                            return df, selected_label
                        else:
                            st.error("Failed to load dataframe.")
                            return None, None
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
                        return None, None
                else:
                    st.error(f"File not found: {file_path}")
                    return None, None
        else:
            st.warning("No datasets found in trace history.")
            return None, None

    elif selection_method == "Upload File":
        uploaded_file = st.file_uploader(
            f"Upload {label}",
            type=['csv', 'xlsx', 'parquet'],
            key=f"{tmp}_data_uploader"
        )
        if uploaded_file is not None:
            try:
                df = load_dataframe(uploaded_file)
                st.success("Dataset successfully loaded.")
                return df, uploaded_file.name
            except Exception as e:
                st.error(f"Error loading file: {e!s}")
                return None, None

    elif selection_method == "Enter File Path":
        folder_path = st.text_input("Enter folder path:", key=f"{tmp}_folder_path")
        file_name = st.text_input("Enter file name:", key=f"{tmp}_data_name")
        file_path = os.path.join(folder_path, file_name) if folder_path and file_name else ""
        file_path_input = st.text_input(f"Enter file path for {label}:", value=file_path, key=f"{tmp}_data_path")
        
        if file_path_input:
            if os.path.exists(file_path_input):
                try:
                    df = load_dataframe(file_path_input)
                    st.success("Dataset successfully loaded.")
                    return df, file_path_input
                except Exception as e:
                    st.error(f"Error loading file: {e!s}")
                    return None, None
            else:
                st.error("File not found.")
                return None, None
                
    return None, None

######################################### test
def add_outlier_handling_ui(df, numeric_cols):
    """Add outlier handling UI components with improved organization and summary display."""
    # Initialize outlier handler
    outlier_handler = OutlierHandler(df, st.session_state.config)
    
    # Step 1: Select Outlier Detection Method and Handling Strategy
    col1, col2 = st.columns(2)
    with col1:
        detection_method = st.selectbox(
            "Outlier Detection Method",
            ["zscore", "iqr", "quantile", "Local Outlier Factor", "Isolation Forest", "DBSCAN"],
            help="Z-score: Uses standard deviations from mean\n"
                 "IQR: Uses interquartile range\n"
                 "Quantile: Uses percentile thresholds"
        )

    if detection_method in ["zscore", "iqr", "quantile"]:
        strat_list = ["None", "remove", "clip", "tag"]
    else:
        strat_list = ["None", "remove", "tag"]

    with col2:
        handling_strategy = st.selectbox(
            "Handling Strategy",
            strat_list,
            help="Remove: Exclude outliers\n"
                 "Clip: Cap at threshold values\n"
                 "Tag: Tag outliers for later identification"
        )

    # Step 2: Configure Method-Specific Parameters
    params = configure_method_params(detection_method)

    # Step 3: Select Columns for Outlier Handling
    selected_cols = select_columns_for_outlier_handling(df, numeric_cols, detection_method)

    if not selected_cols:
        st.warning("Please select at least one column")
        return df

    # Step 4: Handle Outliers Based on Strategy
    df_processed, outlier_matrix = outlier_handler.handle_outliers(
        columns=selected_cols,
        method=detection_method,
        handling_strategy=handling_strategy,
        **params
    )
    #st.dataframe(df_processed)

    # Step 5: Generate Outlier Summary from Matrix
    summary = generate_outlier_summary_from_predictions(df, outlier_matrix, selected_cols, detection_method)

    # Step 6: Display Combined Information for Each Variable
    display_combined_information(df, df_processed, summary, selected_cols, detection_method, handling_strategy)

    # Step 7: Visualization
    with st.expander("Visual Inspection", expanded=True):
         # Pass selected columns instead of all numeric columns to ensure correct data scope
         display_outlier_visualization(df, outlier_matrix, selected_cols, detection_method)

    return df_processed


def configure_method_params(detection_method):
    """Configure parameters based on the selected outlier detection method."""
    if detection_method == "zscore":
        threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
        return {"threshold": threshold}
    elif detection_method == "iqr":
        multiplier = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1)
        return {"multiplier": multiplier}
    elif detection_method == "quantile":
        col1, col2 = st.columns(2)
        with col1:
            lower = st.number_input("Lower percentile", 0.0, 0.5, 0.01, 0.01)
        with col2:
            upper = st.number_input("Upper percentile", 0.5, 1.0, 0.99, 0.01)
        return {"lower": lower, "upper": upper}
    elif detection_method == "Isolation Forest":
        return {'n_estimators': 100, 'max_samples': 'auto'}
    elif detection_method == "Local Outlier Factor":
        return {'n_neighbors': 50, 'contamination': 'auto'}
    elif detection_method == "DBSCAN":
        col1, col2 = st.columns(2)
        with col1:
            eps = st.number_input("Epsilon (eps)", 0.1, 10.0, 0.5, 0.1)
        with col2:
            min_samples = st.number_input("Min Samples", 1, 100, 5, 1)
        return {'eps': eps, 'min_samples': int(min_samples)}
    return {}


def select_columns_for_outlier_handling(df, numeric_cols, detection_method):
    """Select columns for outlier handling based on the detection method."""
    if detection_method in ["zscore", "iqr", "quantile"]:
        return st.multiselect("Select columns for outlier handling", numeric_cols, default=numeric_cols[:1])
    else:
        analyzer = DataAnalyzer(df)
        return st.multiselect(
            "Select columns for outlier handling",
            df.columns.to_list(),
            default=list(set(analyzer.categorical_cols + analyzer.binary_cols + 
                             analyzer.low_cardinality_numeric_cols + analyzer.numeric_cols))
        )


# Determine if we should append the percentage sign
def format_change(change, change_measure):
    if change_measure != "flat":
        return f"{change:.2f}%"  # Add percentage sign if not "flat"
    return f"{change:.2f}"  # No percentage sign if "flat"


def display_combined_information(df, df_processed, summary, selected_cols, detection_method, handling_strategy):
    """Display outlier summary, outlier impact (percentage change), and before/after statistics side by side for each variable."""

    st.subheader("Outlier Detection Summary")
    
    # Streamlit radio button to select the change measure
    change_measure = st.radio(
        "Select Change Measure", 
        options=["flat", "(%) of initial value", "(%) of IQR"], 
        index=1,  # Default to "% of initial value"
        horizontal=True
    )
    # Boundary to separate content
    st.markdown("---")

    # Loop through columns for summary and statistics display
    if detection_method in ["iqr", "quantile", "zscore"]:
        for col in selected_cols:
            st.markdown(f"### {col}")

            # Create 3 columns for side-by-side layout
            cols = st.columns(2)
            
            with cols[0]:
                # Outlier Summary Section
                stats = summary.get(col, {})
                st.markdown("**Outlier Summary**")
                st.markdown(f"- **Total outliers**: {stats.get('total_outliers', 'N/A')}")
                st.markdown(f"- **Percentage of outliers**: {stats.get('percentage_outliers', 0):.2f}%")
            
            with cols[1]:
                # Outlier Impact (percentage change) Section
                st.markdown(f"**Handling Strategy Impact ({change_measure})**")
                impact = calculate_outlier_impact(df, df_processed, [col], change_measure)
                if col in impact:
                    imp = impact[col]
                    if isinstance(imp, dict):
                        mean_change = imp.get('mean_change', 0)
                        std_change = imp.get('std_change', 0)
                        min_change = imp.get('min_change', 0)
                        max_change = imp.get('max_change', 0)
                        
                        # Display the changes in Streamlit
                        st.markdown(f"- **Mean Change ({change_measure})**: {format_change(mean_change, change_measure)}")
                        st.markdown(f"- **Std Change ({change_measure})**: {format_change(std_change, change_measure)}")
                        st.markdown(f"- **Min Change ({change_measure})**: {format_change(min_change, change_measure)}")
                        st.markdown(f"- **Max Change ({change_measure})**: {format_change(max_change, change_measure)}")
                    else:
                        st.markdown("- Impact can't be measured for categorical data.")
            
            st.markdown("---")

    else:
        # For methods like "Local Outlier Factor" or "Isolation Forest"
        st.markdown("### Dataset-Wide Outlier Summary")
        for col, stats in summary.items():
            st.markdown(f"**Outlier Summary for {col}**")
            st.markdown(f"- **Total outliers**: {stats.get('total_outliers', 'N/A')}")
            st.markdown(f"- **Percentage of outliers**: {stats.get('percentage_outliers', 0):.2f}%")
        
        # Boundary after dataset-wide outlier summary
        st.markdown("---")

    if handling_strategy in ["remove","clip"]:
        # Calculate impact (same as before)
        impact = calculate_outlier_impact(df, df_processed, selected_cols, change_measure)

        # Prepare a DataFrame for impact
        impact_data = []
        for col, imp in impact.items():
            if isinstance(imp, dict):
                impact_data.append([
                    col,
                    imp.get('mean_change', 0),
                    imp.get('std_change', 0),
                    imp.get('min_change', 0),
                    imp.get('max_change', 0)
                ])
            else:
                impact_data.append([col, None, None, None, None])

        # Create the DataFrame
        impact_df = pd.DataFrame(
            impact_data,
            columns=["Column", f"Mean Change ({change_measure})", f"Std Change ({change_measure})", f"Min Change ({change_measure})", f"Max Change ({change_measure})"]
        ).set_index("Column")

        # Determine vmin and vmax based on change_measure
        vmin, vmax = calculate_vmin_vmax(impact_df, change_measure)

        # Style the DataFrame
        styled_impact_df = (
            impact_df.style
            .format(
                {
                    f"Mean Change ({change_measure})": lambda x: format_change(x, change_measure),
                    f"Std Change ({change_measure})": lambda x: format_change(x, change_measure),
                    f"Min Change ({change_measure})": lambda x: format_change(x, change_measure),
                    f"Max Change ({change_measure})": lambda x: format_change(x, change_measure)
                }
            )
            .background_gradient(
                cmap="RdBu",  # Diverging colormap: red for negative, blue for positive
                vmin=vmin, vmax=vmax,  # Define range for divergence (-100 to 100 for percentages)
                axis=None,  # Apply gradient to the entire table
                subset=[f"Mean Change ({change_measure})", f"Std Change ({change_measure})", f"Min Change ({change_measure})", f"Max Change ({change_measure})"]
            )
            .set_table_styles(
                [
                    {'selector': 'thead th', 'props': [('text-align', 'center'), ('font-size', '16px')]},
                    {'selector': 'tbody td', 'props': [('text-align', 'center'), ('padding', '8px')]},
                    {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
                    {'selector': 'tbody tr:hover', 'props': [('background-color', '#f1f1f1')]},
                ]
            )
            .set_properties(width='100%', border='1px solid #ddd')
        )

        # Render the table in Streamlit
        container = st.container()
        with container:
            st.write("### Overall Impact of Outliers Handling Operation on the Dataset")
            st.dataframe(styled_impact_df, width='stretch')


def calculate_vmin_vmax(df, change_measure):
    if change_measure != "flat":
        # For percentages, set range from -100 to 100
        return -100, 100
    else:
        # For flat values, calculate the range based on the data
        min_change = df.min().min()  # Get the minimum value from the DataFrame
        max_change = df.max().max()  # Get the maximum value from the DataFrame
        return min_change, max_change

def calculate_outlier_impact(df, df_processed, selected_cols, change_measure='flat'):
    """
    Calculate the impact of outlier handling on each selected column by computing
    the percentage change in mean, std, min, and max values.
    """
    impact = {}
    
    # Ensure selected columns exist in both DataFrames
    missing_cols = [col for col in selected_cols if col not in df.columns or col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing in one of the DataFrames: {missing_cols}")
    
    for col in selected_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Calculate before and after metrics
            before_mean = df[col].mean(skipna=True)
            after_mean = df_processed[col].mean(skipna=True)
            
            before_std = df[col].std(skipna=True)
            after_std = df_processed[col].std(skipna=True)
            
            before_min = df[col].min(skipna=True)
            after_min = df_processed[col].min(skipna=True)
            
            before_max = df[col].max(skipna=True)
            after_max = df_processed[col].max(skipna=True)

            # Calculate IQR for '% of initial value' method
            Q1_before = df[col].quantile(0.25)
            Q3_before = df[col].quantile(0.75)
            IQR_before = Q3_before - Q1_before
            
            # Based on the change_type, calculate the percentage change
            if change_measure == "flat":
                # Absolute difference (flat)
                mean_change = after_mean - before_mean
                std_change = after_std - before_std
                min_change = after_min - before_min
                max_change = after_max - before_max
            elif change_measure == "(%) of initial value":
                # Percentage change based on initial value
                mean_change = (after_mean - before_mean) / before_mean * 100
                std_change = (after_std - before_std) / before_std * 100
                min_change = (after_min - before_min) / before_min * 100
                max_change = (after_max - before_max) / before_max * 100
            elif change_measure == "(%) of IQR":
                mean_change = (after_mean - before_mean) / IQR_before * 100
                std_change = (after_std - before_std) / IQR_before * 100
                min_change = (after_min - before_min) / IQR_before * 100
                max_change = (after_max - before_max) / IQR_before * 100
            else:
                raise ValueError("Invalid change_type. Use 'flat', '(%) of IQR' or '(%) of initial value'.")
            
            # Store results in the dictionary
            impact[col] = {
                'mean_change': mean_change,
                'std_change': std_change,
                'min_change': min_change,
                'max_change': max_change
            }
        else:
            # Placeholder for non-numeric columns
            impact[col] = {
                'mean_change': None,
                'std_change': None,
                'min_change': None,
                'max_change': None
            }
    
    return impact




def generate_outlier_summary_from_predictions(df, outlier_predictions, selected_cols, method):
    """
    Generate outlier summary based on raw predictions.

    Parameters:
        df (DataFrame): Original DataFrame.
        outlier_predictions (dict): Raw outlier predictions for each column.
        selected_cols (list): Columns analyzed for outliers.
        method (str): Outlier detection method.

    Returns:
        dict: Summary of outliers for each column or overall.
    """
    summary = {}

    if method in ['zscore', 'iqr', 'quantile']:
        for col in selected_cols:
            is_outlier = outlier_predictions.get(col, [])
            n_outliers = sum(is_outlier)
            pct_outliers = (n_outliers / len(df)) * 100
            summary[col] = {
                'total_outliers': n_outliers,
                'percentage_outliers': pct_outliers,
                'outlier_values': df.loc[is_outlier, col].tolist()
            }
    elif method in ['Local Outlier Factor', 'Isolation Forest', 'DBSCAN']:
        is_outlier = outlier_predictions.get('overall', [])
        n_outliers = sum(is_outlier)
        pct_outliers = (n_outliers / len(df)) * 100
        summary['overall'] = {
            'total_outliers': n_outliers,
            'percentage_outliers': pct_outliers,
        }

    return summary



def display_outlier_visualization(df, outlier_matrix, selected_cols, method):
    """Display 2D visualization of outliers using PCA, FAMD, t-SNE, or UMAP."""
    st.subheader("Visual Inspection")
    
    # Projection Method Selector
    projection_method = st.selectbox(
        "Select Projection Method",
        ["PCA", "FAMD", "t-SNE", "UMAP"],
        index=0,
        help="PCA: Linear (fast, numeric only)\n"
             "FAMD: Factor Analysis for Mixed Data (numeric + categorical)\n"
             "t-SNE: Non-linear, preserves local structure\n"
             "UMAP: Non-linear, preserves global structure"
    )

    if not selected_cols:
         st.warning("No columns selected for visualization.")
         return

    # Prepare data based on method requirements
    # For PCA/t-SNE/UMAP in clustering_utils, it expects numeric data mainly, 
    # but we should filter DF to selected columns first.
    
    data_viz = df[selected_cols].copy()
    
    # Impute missing values instead of dropping to preserve data points
    # Separate numeric and categorical
    num_cols = data_viz.select_dtypes(include=['number']).columns
    cat_cols = data_viz.select_dtypes(include=['object', 'category']).columns

    if not num_cols.empty:
        # Fill numeric with mean
        data_viz[num_cols] = data_viz[num_cols].fillna(data_viz[num_cols].mean())
    
    if not cat_cols.empty:
        # Fill categorical with mode
        for col in cat_cols:
             if not data_viz[col].mode().empty:
                 data_viz[col] = data_viz[col].fillna(data_viz[col].mode()[0])
             else:
                 data_viz[col] = data_viz[col].fillna("Unknown")

    if data_viz.isna().any().any():
        st.warning("Some missing values could not be imputed. Dropping remaining rows.")
        data_viz = data_viz.dropna()

    if data_viz.empty:
        st.warning("No valid data for visualization (all rows contain NaNs in selected columns).")
        return
    
    # Check if we have enough data
    if len(data_viz) < 5:
         st.warning("Not enough data points for visualization (need at least 5).")
         return

    embeddings = None
    info = {}
    
    try:
        with st.spinner(f"Running {projection_method}..."):
            if projection_method == "PCA":
                # PCA usually requires numeric only, let's filter just in case logic above passed mixed
                # But wait, we want to visualize what was used. 
                # If users used LOF on mixed data, they might want to see mixed data projection.
                # Currently cu.run_pca filters for numbers.
                embeddings, info = cu.run_pca(data_viz, n_components=2)
                ratio = info.get('explained_variance_ratio', [0, 0])
                # Ensure we have at least 2 elements
                if len(ratio) < 2: ratio = ratio + [0] * (2 - len(ratio))
                
                axis_labels = {
                    "x": f"PC1 ({ratio[0]:.1%})", 
                    "y": f"PC2 ({ratio[1]:.1%})"
                }
                title = f"PCA Projection (Explained Variance: {info.get('total_variance_explained', 0):.2%})"
            
            elif projection_method == "t-SNE":
                embeddings, info = cu.run_tsne(data_viz, n_components=2)
                axis_labels = {"x": "Dim 1", "y": "Dim 2"}
                title = f"t-SNE Projection (Perplexity: {info.get('perplexity_used', 'N/A')})"
                
            elif projection_method == "UMAP":
                embeddings, info = cu.run_umap(data_viz, n_components=2)
                axis_labels = {"x": "Dim 1", "y": "Dim 2"}
                title = "UMAP Projection"

            elif projection_method == "FAMD":
                # FAMD handles mixed data, so we need to ensure we pass the right data
                # run_famd in utils should handle it.
                embeddings, info = cu.run_famd(data_viz, n_components=2)
                
                inertia = info.get('explained_inertia', [0, 0])
                if len(inertia) < 2: inertia = inertia + [0] * (2 - len(inertia))
                
                # FAMD from prince (via our utils) returns percentages (e.g. 45.2)
                # But sometimes it might fall back to ratio if using different attr.
                # In our utils, we prioritized 'eigenvalues_summary' which has '% of variance'
                # Let's assume it is percentage numbers.
                
                axis_labels = {
                    "x": f"Dim 1 ({inertia[0]:.1f}%)", 
                    "y": f"Dim 2 ({inertia[1]:.1f}%)"
                }
                title = f"FAMD Projection (Explained Variance: {info.get('total_variance_explained', 0):.2%})"

    except ImportError as e:
        st.error(f"Dependency missing: {e!s}")
        return
    except Exception as e:
        st.error(f"Error running {projection_method}: {e!s}")
        return

    if embeddings is None:
         return

    # Check for dropped columns (e.g. categorical cols in PCA)
    used_features = info.get('feature_names', [])
    if used_features:
        dropped_cols = list(set(data_viz.columns) - set(used_features))
        if dropped_cols:
            st.info(f"**Note**: The following columns were excluded from {projection_method} (incompatible data type): {', '.join(dropped_cols)}")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(embeddings, columns=['Dim1', 'Dim2'], index=data_viz.index)
    
    # Determine outlier status
    # We need to match indices from the visualization data back to the outlier matrix
    is_outlier = pd.Series(False, index=data_viz.index)
    
    if method in ['zscore', 'iqr', 'quantile']:
        # Univariate: Outlier in ANY of the selected columns
        relevant_cols = [c for c in outlier_matrix.columns if c in selected_cols]
        if relevant_cols:
            is_outlier = outlier_matrix.loc[data_viz.index, relevant_cols].any(axis=1)
    else:
        # Multivariate: Check 'overall' column
        if 'overall' in outlier_matrix.columns:
            is_outlier = outlier_matrix.loc[data_viz.index, 'overall']

    plot_df['Outlier'] = is_outlier.map({True: 'Yes', False: 'No'})
    
    # Plot using Plotly
    import plotly.express as px
    fig = px.scatter(
        plot_df, x='Dim1', y='Dim2', color='Outlier',
        color_discrete_map={'Yes': 'red', 'No': 'blue'},
        title=title,
        labels={'Dim1': axis_labels.get('x', 'Dim 1'), 'Dim2': axis_labels.get('y', 'Dim 2')},
        hover_data=[plot_df.index],
        opacity=0.7
    )
    st.plotly_chart(fig, width="stretch")




######################################## test


def outlier_values_dialog(outlier_summary):
    """Display outlier values in a dialog"""
    st.write("Outlier Values:")
    for col, stats in outlier_summary.items():
        st.subheader(col)
        st.write(stats['outlier_values'])


def display_results_with_outliers(filtered_df: pd.DataFrame, st):
    """Display the results of the enrichment process."""
    st.subheader("Outlier Detection Results")
    
    # Find potential tag columns
    tag_cols = [col for col in filtered_df.columns if col.endswith('_tag')]
    bool_column = tag_cols[-1] if tag_cols else None

    # Preview the enriched dataset
    display_results(df=filtered_df, bool_column=bool_column, title=None, key_base="outliers_data", st=st)

#############################################################################################

################################ ANOMALY AND INCLUSION HANDLING ############################################################


##### COMMON PART

def ensure_family_exists(config: dict[str, Any], family_name: str):
    """Ensure the specified family exists in the configuration."""
    if family_name not in config["mask_families"]:
        config["mask_families"][family_name] = {}


def validate_mask_name(config: dict[str, Any], family_name: str, mask_name: str, st_container):
    """Validate the mask name and handle duplicates."""
    if not mask_name:
        st_container.warning("Please provide a mask name.")
        return False
    if mask_name in config["mask_families"][family_name]:
        overwrite = st_container.radio(
            f"Mask '{mask_name}' already exists in family '{family_name}'. Overwrite?",
            ["No", "Yes"],
            index=0,
            help="Select 'Yes' to overwrite the existing mask.",
            horizontal=True
        )
        return overwrite == "Yes"
    return True


def handle_numeric_mask_input(st_container, base_key:str):
    """Collect numeric mask inputs from the user."""
    st_container.markdown("Define the range of values to flag or filter:")
    
    col1, col2 = st_container.columns(2)
    with col1:
        is_lower_bound = st_container.checkbox("Need lower bound?", key=f"{base_key}_lower_bound", help="Check if there is a minimum value limit.")
    with col2:
        is_upper_bound = st_container.checkbox("Need upper bound?", key=f"{base_key}_upper_bound", help="Check if there is a maximum value limit.")
        
    lower_bound = st_container.number_input(
        "Lower Bound", value=0.0, disabled=not is_lower_bound, key=f"{base_key}_lower_bound_input",
        help="Values below this may be excluded or flagged depending on the strategy."
    ) if is_lower_bound else None
    
    upper_bound = st_container.number_input(
        "Upper Bound", value=0.0, disabled=not is_upper_bound, key=f"{base_key}_upper_bound_input",
        help="Values above this may be excluded or flagged depending on the strategy."
    ) if is_upper_bound else None
    
    strategy = st_container.selectbox(
        "Strategy",
        ["exclude", "include"],
        disabled=not (is_lower_bound and is_upper_bound),
        key=f"{base_key}_strategy",
        help=(
            "**include**: The range [Lower, Upper] is the *target* (e.g., normal range). Everything OUTSIDE is an anomaly.\n"
            "**exclude**: The range [Lower, Upper] is *forbidden*. Everything INSIDE is an anomaly."
        )
    ) if is_lower_bound and is_upper_bound else None

    return {
        "numeric": True,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "strategy": strategy,
    }


def handle_expression_mask_input(st_container, base_key: str):
    """Collect expression mask inputs from the user."""
    
    with st_container.expander("Expression Help & Examples", expanded=False):
        st.markdown("""
        Write a Python boolean expression. Use `df` to refer to the dataframe.
        
        **Examples:**
        - `df['Age'] < 18` (Selects rows where Age is under 18)
        - `(df['Systolic_BP'] > 140) & (df['Diastolic_BP'] > 90)` (Complex condition)
        - `df['Status'].isin(['Active', 'Pending'])` (Categorical check)
        - `df['Lab_Val'].isna()` (Check for missing values)
        """)

    expression = st_container.text_area(
        "Expression", 
        placeholder="e.g. df['Age'] > 100", 
        key=f"{base_key}_txt_area",
        help="Enter a condition that evaluates to True/False for each row."
    )
    use_simplified_mask_expression = st_container.checkbox(
        "Use Simplified Mask Expression", 
        key=f"{base_key}_simpl",
        help="Enable this if you want to use a simplified syntax (if available via configuration)."
    )
    
    if not expression.strip():
        # st_container.warning("Expression is empty.") # Don't error immediately, let them type
        return None
        
    if use_simplified_mask_expression:
        expression = transform_expression(expression)  # Example placeholder function
    return {"numeric": False, "expression": expression}


def handle_family_operator(st_container, base_key: str):
    """Handle family operator selection and its parameters."""
    operator = st_container.selectbox(
        "Operator for Mask Family",
        ["any", "all", "threshold"],
        key=f"{base_key}_operator",
        help=(
            "Define how the masks in this family should be evaluated:\n"
            "- `any`: At least one mask is satisfied.\n"
            "- `all`: All masks must be satisfied.\n"
            "- `threshold`: A minimum count of masks must be satisfied."
        )
    )
    
    threshold = None
    if operator == "threshold":
        threshold = st_container.number_input(
            "Threshold (minimum number of masks to satisfy)",
            min_value=1,
            step=1,
            help="Specify the minimum number of masks that must be satisfied.",
            key=f"{base_key}_threshold_input"
        )
    return {"operator": operator, "threshold": threshold}


def add_mask_family(config: dict[str, Any], st_container):
    st_container.subheader("Add New Mask Family")

    # Family Selection
    family_name = st_container.text_input("Family Name", placeholder="Enter family name")
    if not family_name:
        st_container.warning("Please provide a family name.")
        return
    ensure_family_exists(config, family_name)

    # Operator Selection
    family_operator = handle_family_operator(st_container)
    config["mask_families"][family_name]["operator"] = family_operator

    # Mask Addition
    mask_name = st_container.text_input("Mask Name", placeholder="Enter mask name")
    if not validate_mask_name(config, family_name, mask_name, st_container):
        return

    is_numeric = st_container.checkbox("Is Numeric")
    if is_numeric:
        mask_details = handle_numeric_mask_input(st_container)
    else:
        mask_details = handle_expression_mask_input(st_container)

    if mask_details and st_container.button("Add Mask"):
        config["mask_families"][family_name][mask_name] = mask_details
        st_container.success(f"Mask '{mask_name}' added to family '{family_name}'.")
    
    st_container.json(config["mask_families"][family_name])


# Define a recursive function to display the JSON as collapsible sections
def display_mask_family(config):
    """
    Create a color-coded display of mask families configuration.
    """
    # Ensure config is a dictionary
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            st.error(f"Invalid JSON configuration: {config}")
            return

    # Prepare dynamic grid layout
    #num_items = len(config)
    cols_per_row = 3  # Adjustable number of columns per row
    grid = st.columns(cols_per_row)

    # Iterate through config items
    for i, (key, value) in enumerate(config.items()):
        if not isinstance(value, dict):
            continue

        # Determine configuration type
        is_numeric = value.get('numeric', False)

        # Set color based on type
        color = '#3498db' if is_numeric else '#2ecc71'
        #border_color = f"2px solid {color}"

        with grid[i % cols_per_row]:
            if is_numeric:
                # Expander for numeric configuration
                with st.expander(f"{key}", expanded=False):
                    # Numeric details
                    st.markdown(f"**Type:** <span style='color:{color};'>Numeric</span>", unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        lower_bound = value.get('lower_bound', None)
                        if lower_bound is not None:
                            st.metric(label="Lower Bound", value=lower_bound)
                    with col2:
                        upper_bound = value.get('upper_bound', None)
                        if upper_bound is not None:
                            st.metric(label="Upper Bound", value=upper_bound)

                    strategy = value.get('strategy', None)
                    if strategy is not None:
                        st.markdown(f"**Validation Strategy:** {strategy}")
                    st.success("Numeric Validation Active")

                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                # Expander for non-numeric configuration
                with st.expander(f"{key}", expanded=False):
                    # Non-numeric details
                    st.markdown(f"**Type:** <span style='color:{color};'>Expression</span>", unsafe_allow_html=True)

                    expression = value.get('expression', "No expression defined")
                    #st.markdown("**Validation Expression:**")
                    st.code(expression, language='python')
                    st.warning("Custom Expression Validation")

                    st.markdown("</div>", unsafe_allow_html=True)


def display_mask_family_alt(config):
    """
    Display mask families configuration in a single top-level expander without nesting.
    """
    st.title("Data Masking Configuration")

    # Ensure config is a dictionary
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            st.error(f"Invalid JSON configuration: {config}")
            return

    # Top-level expander for all configurations
    with st.expander("View Mask Families Configuration", expanded=False):
        for i, (key, value) in enumerate(config.items()):
            # Skip invalid entries
            if not isinstance(value, dict):
                continue

            # Determine type and color
            is_numeric = value.get('numeric', False)
            color = '#3498db' if is_numeric else '#2ecc71'
            border_color = f"2px solid {color}"

            # Section header
            st.markdown(
                f"""
                <div style="border: {border_color}; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                    <h4 style="margin: 0;">{'' if is_numeric else ''} {key}</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Numeric Configuration
            if is_numeric:
                col1, col2 = st.columns(2)
                with col1:
                    lower_bound = value.get('lower_bound', "N/A")
                    st.metric(label="Lower Bound", value=lower_bound)
                with col2:
                    upper_bound = value.get('upper_bound', "N/A")
                    st.metric(label="Upper Bound", value=upper_bound)

                strategy = value.get('strategy', "None")
                st.markdown(f"**Validation Strategy:** {strategy}")
                st.success("Numeric Validation Active")

            # Non-Numeric Configuration
            else:
                expression = value.get('expression', "No expression defined")
                st.markdown("**Validation Expression:**")
                st.code(expression, language='python')
                st.warning("Custom Expression Validation")

            st.divider()  # Divider between items


def apply_family_mask(df: pd.DataFrame, family_name: str, family_config: dict, st_container):
    """
    Apply a single mask family from JSON configuration to the dataset.
    
    Parameters:
    - df: DataFrame to which the mask family is applied.
    - family_name: Name of the mask family to apply.
    - family_config: Configuration for the specific mask family.
    - st_container: Streamlit container for displaying results.
    
    Returns:
    - Updated DataFrame with mask family tags and lists.
    - The names of the columns related to the masks family
    """
    masks_family_cols = []

    print(f"Processing Family Operator for: {family_name}")
    
    # Step 1: Get the operator configuration, ensuring it is interpreted correctly
    operator_config = family_config.get("operator", "any")
    print(f"operator_config: {operator_config}")

    # Handle different types of operator_config
    if isinstance(operator_config, str):
        # If it's a string, set operator_type directly and no threshold
        operator_type = operator_config
        threshold = None
    elif isinstance(operator_config, dict):
        # If it's a dictionary, extract operator type and optional threshold
        operator_type = "threshold"
        threshold = operator_config.get("threshold", None)
    else:
        # Handle unexpected types (fallback or raise error)
        print("Invalid operator configuration format.")
        raise ValueError(f"Unsupported operator configuration type: {type(operator_config)}")

    print(f"Resolved operator_type: {operator_type}, threshold: {threshold}")

    
    # Step 2: Generate masks_zip using zip_masks function
    print(f"Processing Family Masks for: {family_name}")
    masks_zip = ecm.zip_masks(df, family_config)
    print(f"Mask Configuration for '{family_name}':", family_config)
    
    # Step 3: Create masks DataFrame
    masks_df = pd.DataFrame(index=df.index)

    for mask_name, mask_values in masks_zip:
        #st_container.write(f"Processing Mask: {mask_name}")
        masks_df[mask_name] = mask_values

    st_container.dataframe(masks_df)

    # Step 4: Add family-level tags to the DataFrame
    # List of activated masks per row
    df[f"{family_name}_list"] = masks_df.apply(ecm.tag_masks, axis=1)
    masks_family_cols.append(f"{family_name}_list")

    # Apply family operator
    if operator_type == "any":
        # Tag rows where any mask in the family is satisfied
        df[f"{family_name}_any"] = masks_df.any(axis=1)
        masks_family_cols.append(f"{family_name}_any")
    elif operator_type == "all":
        # Tag rows where all masks in the family are satisfied
        df[f"{family_name}_all"] = masks_df.all(axis=1)
        masks_family_cols.append(f"{family_name}_all")
    elif operator_type == "threshold":
        if threshold is not None:
            df[f"{family_name}_above_{threshold}"] = masks_df.sum(axis=1) >= threshold
            masks_family_cols.append(f"{family_name}_above_{threshold}")
        else:
            st_container.error(f"Threshold not set for family '{family_name}'.")
            return df, masks_family_cols  # Return early if no threshold is defined
    else:
        st_container.error(f"Unknown operator '{operator_type}' in family '{family_name}'.")
        return df, masks_family_cols  # Return early for unknown operator type

    return df, masks_family_cols




# Convert DataFrame to Excel for download using openpyxl
def to_excel(df):
    """
    Convert a DataFrame to an Excel file in binary format.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to convert.

    Returns:
    --------
    bytes
        The Excel file content as bytes.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
    return output.getvalue()


def display_results(df: pd.DataFrame, bool_column: str, title: str, key_base: str, st):
    """Display the results of the enrichment process."""
    if bool_column is not None:
        mask = df[bool_column] == True
        # Extract the prefix from the target column
        prefix = '_'.join(bool_column.split('_')[:-1])
        # Select columns that start with the same prefix
        matching_columns = [col for col in df.columns if col.startswith(prefix)]
        print(matching_columns)
        styled_df = df[mask].style.apply(color_columns, axis=0,columns_to_highlight=matching_columns)
    else:
        styled_df = df

    # Preview the enriched dataset
    if title is not None:
        st.subheader(title)
    st.dataframe(styled_df, width='stretch')

    col1, col2, col3 = st.columns(3)
    with col1:
        # Download button
        st.download_button(
            label="Download Dataset",
            key= f"{key_base}_dl_bt",
            data=to_excel(styled_df),
            file_name=f"{key_base}_dataset.xlsx",
            mime="text/xlsx",
            width='stretch'
        )
    
    with col2:
        # Save to session state
        if bool_column is not None:
            if key_base in ["anomaly_data","outliers_data"]:
                df_masked = df[~mask]
            elif key_base == "inclusion_data":
                df_masked = df[mask]
        else:
            df_masked = df       


        if f"{key_base}_applied" not in st.session_state:
            st.session_state[f"{key_base}_applied"] = False

        if st.button("Apply Filtering Criteria for Further Analysis", width='stretch', key=f"{key_base}_apply_criteria"):
            # Ensure df_masked exists and is valid
            #st.write("Masked DataFrame Preview:")
            #st.dataframe(df_masked)
                
            # Update session state
            st.session_state.data = df_masked
            st.session_state.working_df = df_masked
                
            st.success("Dataset saved and ready for further analysis!")
            # Debug outputs to confirm updates
            #st.write("Updated working DataFrame:")
            #st.dataframe(st.session_state.working_df)

    with col3:
        # Save configuration
        st.download_button(
            label="Download Configuration File",
            data=json.dumps(st.session_state.config, indent=4),
            file_name="config.json",
            mime="application/json",
            key=f"{key_base}_config",
            width='stretch'
        )


# Function to apply column-wide styling
def color_columns(s, columns_to_highlight):
    """
    Applies a background color to specific columns.
    """
    if s.name in columns_to_highlight:
        return ['background-color: lightblue'] * len(s)
    else:
        return [''] * len(s)


def deep_merge_dicts(original, new):
    """
    Deeply merges two dictionaries. Adds new keys and values from `new` to `original`
    without modifying existing keys in `original`.
    """
    for key, value in new.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            # Recursively merge dictionaries
            deep_merge_dicts(original[key], value)
        elif key not in original:
            # Only add the key if it doesn't already exist in original
            original[key] = value

##### ANOMALY PART

def render_ai_criteria_assistant(st_container, config, family_name, mode="anomaly"):
    """
    Renders the AI assistant for generating criteria.
    """
    if 'rag_manager' in st.session_state and st.session_state.rag_manager.initialized:
        with st_container.expander("AI Assistant (Beta)", expanded=False):
            st.markdown(
                "Describe what you want to flag or include using natural language. "
                "The AI will generate the Python expressions for you."
            )
            
            col_input, col_btn = st.columns([3, 1])
            with col_input:
                description = st.text_area(
                    "Describe Criteria", 
                    placeholder="e.g. Patients with Systolic BP > 140 or Age under 18",
                    key=f"ai_desc_{family_name}",
                    label_visibility="collapsed"
                )
            with col_btn:
                generate_clicked = st.button("Generate", key=f"ai_gen_{family_name}", type="primary")

            if generate_clicked:
                if 'data' not in st.session_state or st.session_state.data is None:
                    st.error("Please load a dataset first.")
                elif not description:
                    st.warning("Please enter a description.")
                else:
                    with st.spinner("Generating criteria..."):
                        columns = list(st.session_state.data.columns)
                        # Pass sample data for smart inference
                        sample_df = st.session_state.data.head(3) if 'data' in st.session_state else None
                        
                        criteria_list, error = st.session_state.rag_manager.suggest_anomaly_criteria(
                            description, columns, sample_df, mode
                        )
                    
                    if error:
                        st.error(f"AI Error: {error}")
                    elif not criteria_list:
                        st.warning("No criteria suggested. Try a more specific description.")
                    else:
                        st.session_state[f"ai_suggestions_{family_name}"] = criteria_list
            
            # Display Suggestions from Session State
            if f"ai_suggestions_{family_name}" in st.session_state:
                st.write("---")
                st.write("### AI Suggestions")
                
                start_idx = 0
                # Pagination or simple list
                suggestions = st.session_state[f"ai_suggestions_{family_name}"]
                
                for i, item in enumerate(suggestions):
                    with st.container():
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(f"**{item.get('name', 'Unnamed')}**")
                            st.code(item.get('expression', ''), language="python")
                            st.caption(f"_{item.get('explanation', '')}_")

                        # Validation & Proxy Logic
                        missing_vars = item.get('missing_variables', [])
                        if missing_vars:
                            st.warning(f"Missing variables: {', '.join(missing_vars)}")
                            c_fix1, c_fix2 = st.columns([1,1])
                            with c_fix1:
                                proxy_key = f"proxy_{family_name}_{i}_{missing_vars[0]}"
                                if st.button(f"Find Proxy for '{missing_vars[0]}'", key=proxy_key):
                                    with st.spinner("Finding proxy..."):
                                        columns = list(st.session_state.data.columns)
                                        res = st.session_state.rag_manager.suggest_proxy_variable(missing_vars[0], columns)
                                        if res.get('proxy_found'):
                                            st.success(f"Found: {res.get('proxy_name')}")
                                            # Auto-fix: Replace in expression
                                            old_var = missing_vars[0]
                                            new_var = res.get('proxy_name')
                                            item['expression'] = item['expression'].replace(f"'{old_var}'", f"'{new_var}'").replace(f'"{old_var}"', f'"{new_var}"')
                                            # Remove from missing list
                                            item['missing_variables'].remove(old_var)
                                            st.rerun()
                                        else:
                                            st.error("No proxy found.")

                        with c2:
                            # Disable Add if missing variables exist
                            if missing_vars:
                                st.button("Add", key=f"ai_add_{family_name}_{i}", disabled=True, help="Fix missing variables first")
                            elif st.button("Add", key=f"ai_add_{family_name}_{i}"):
                                mask_name = item.get('name', f"criteria_{i}")
                                expression = item.get('expression', '')
                                
                                # Add to config
                                ensure_family_exists(config, family_name)
                                config["mask_families"][family_name][mask_name] = {
                                    "numeric": False,
                                    "expression": expression
                                }
                                st.success("Added!")
                                time.sleep(0.5)
                                st.rerun()
                        st.divider()

def add_domain_expert_based_anomaly_mask(config: dict[str, Any], st_container):
    st_container.subheader("Define New Anomaly Criterion")

    st_container.info(
        """
        **What is a Clinical Anomaly?**
        
        An anomaly is a data point considered *incorrect* or *suspicious* based on domain knowledge.
        * **Action:** Rows matching these criteria are **flagged** but kept in the dataset.
        
        **How to use:**
        * **Numeric Mode:** Use this for simple range checks on a single column.
          * **IMPORTANT:** The 'Anomaly Mask Name' **MUST** be the exact name of the column you want to check.
        * **Custom Expression:** Use this for complex logic involving multiple columns or specific conditions.
          * The 'Anomaly Mask Name' can be anything you like.
        """
    )
    
    # AI Assistant
    render_ai_criteria_assistant(st_container, config, "clinical_anomalies", mode="anomaly")

    family_name = "clinical_anomalies"
    ensure_family_exists(config, family_name)

    # Operator Selection
    family_operator = "any"
    config["mask_families"][family_name]["operator"] = family_operator

    # Mask Addition
    mask_name = st_container.text_input("Anomaly Mask Name", placeholder="Enter mask name (Exact Column Name if Numeric!)")
    if not validate_mask_name(config, family_name, mask_name, st_container):
        return

    is_numeric = st_container.checkbox("Is Numeric", key="is_num_anomaly")
    if is_numeric:
        mask_details = handle_numeric_mask_input(st_container, base_key=family_name)
    else:
        mask_details = handle_expression_mask_input(st_container, base_key=family_name)

    if mask_details and st_container.button("Add Anomaly Criterion"):
        config["mask_families"][family_name][mask_name] = mask_details
        #st.session_state.config["mask_families"][family_name][mask_name] = mask_details
        st_container.success(f"Anomaly mask '{mask_name}' added to family '{family_name}'.")
    
    #st_container.json(config["mask_families"][family_name])


def display_results_with_anomaly(df: pd.DataFrame, anomaly_col: str, st):
    """Display the results of the enrichment process."""
    st.subheader("Anomaly Detection Results")
    filtered_df = df[df[anomaly_col] == True]
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Irregular Rows (%)", 
            delta=len(filtered_df), 
            value=f"{(len(filtered_df) / len(df)) * 100:.2f}%",
            help=(
                "This metric indicates the total number of rows identified as anomalies and their percentage relative to the entire dataset. "
                "The 'Value' represents the percentage of anomalies compared to all rows, while 'Delta' shows the count of anomalous rows."
            )
        )

    with col2:
        st.metric(
            label="Valid Rows", 
            value=len(df) - len(filtered_df),
            delta=f"-{len(filtered_df)}",
            help=(
                "This metric shows the number of rows not flagged as anomalies. "
                "The 'Value' indicates the count of valid rows, while 'Delta' represents "
                "the decrease in rows due to anomalies detected."
            )
        )
    
    # Preview the enriched dataset
    display_results(df=df, bool_column=anomaly_col, title="Rows with anomalies", key_base="anomaly_data", st=st)


##### INCLUSION PART

def add_study_inclusion_mask(config: dict[str, Any], st_container):
    st_container.subheader("Define New Study Inclusion Criterion")

    st_container.info(
        """
        **What is an Inclusion Criterion?**

        Conditions that must be *met* for a row to be included in the study.
        * **Action:** Rows *failing* to match are **excluded** from the analysis.

        **How to use:**
        * **Numeric Mode:** Use for simple range values on a single column (e.g., Age 18-99).
          * **IMPORTANT:** The 'Inclusion Mask Name' **MUST** be the exact name of the column.
        * **Custom Expression:** Use for conditions like `Consent == True` or multi-column logic.
          * The 'Inclusion Mask Name' can be anything (e.g., 'Consent_Check').
        """
    )

    # AI Assistant
    render_ai_criteria_assistant(st_container, config, "study_inclusion", mode="inclusion")

    family_name = "study_inclusion"
    ensure_family_exists(config, family_name)

    # Operator Selection
    family_operator = "all"
    config["mask_families"][family_name]["operator"] = family_operator

    # Mask Addition
    mask_name = st_container.text_input("Inclusion Mask Name", placeholder="Enter mask name (Exact Column Name if Numeric!)")
    if not validate_mask_name(config, family_name, mask_name, st_container):
        return

    is_numeric = st_container.checkbox("Is Numeric", key="is_num_inclusion")
    if is_numeric:
        mask_details = handle_numeric_mask_input(st_container, base_key=family_name)
    else:
        mask_details = handle_expression_mask_input(st_container, base_key=family_name)

    if mask_details and st_container.button("Add Inclusion Criterion"):
        config["mask_families"][family_name][mask_name] = mask_details
        st_container.success(f"Inclusion mask '{mask_name}' added to family '{family_name}'.")


def display_results_with_inclusion(df: pd.DataFrame, inclusion_col: str, st):
    """Display the results of the enrichment process."""
    st.subheader("Data Inclusion Results")

    # Filter rows based on inclusion column
    filtered_df = df[df[inclusion_col] == True]
    included_count = len(filtered_df)
    total_count = len(df)
    excluded_count = total_count - included_count
    inclusion_percentage = (included_count / total_count) * 100 if total_count > 0 else 0
    exclusion_percentage = (excluded_count / total_count) * 100 if total_count > 0 else 0

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Rows to include in the study (%)",
            value=f"{inclusion_percentage:.2f}%",
            delta=included_count,
            help=(
                "This metric indicates the percentage of rows to include in the study compared to the total number of rows. "
                "'Value' shows the inclusion percentage, and 'Delta' represents the count of rows to include."
            )
        )

    with col2:
        st.metric(
            label="Rows to exclude from the study (%)",
            value=f"{exclusion_percentage:.2}%",
            delta=f"-{excluded_count}",
            help=(
                "This metric indicates the percentage of rows to remove from the study compared to the total number of rows. "
                "'Value' shows the exclusion percentage, and 'Delta' represents the decrease in rows due to exclusion."
            )
        )
    
    # Preview the enriched dataset
    display_results(df=df, bool_column=inclusion_col, title="Rows Meeting Inclusion Criteria", key_base="inclusion_data", st=st)


####################### APP PART ###############################

def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Process dataset using DataAnalyzer to fix types."""
    if df is None:
        return None
    
    # Create analyzer instance
    analyzer = DataAnalyzer(df)
    
    # Convert numeric columns
    cols_to_convert = list(set(analyzer.low_cardinality_numeric_cols + analyzer.numeric_cols))
    if cols_to_convert:
        df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    # Convert date columns
    for col in analyzer.date_cols:
        fmt = analyzer.date_formats.get(col)
        if fmt:
             df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
        else:
             df[col] = pd.to_datetime(df[col], errors='coerce')
        
    return df





def app():
    """
    Main application function for the Data Monitoring page.

    Handles:
    1. Data Quality Dashboard (Data Validity, Completeness, etc.).
    2. Clinical Anomalies Management (Add/Apply Masks).
    3. Study Inclusion Criteria Management (Add/Apply Masks).
    4. Outlier Handling (Univariate & Multivariate).
    5. Dataset Comparison (Changes tracking).
    """

    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = create_empty_config()

    # Initialize DB Manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DBManager()
    
    # Sidebar options
    action = st.sidebar.radio(
        "Action",
        ["Load Existing Config", "Create New Config"]
    )
    
    if action == "Load Existing Config":
        uploaded_file = st.sidebar.file_uploader("Upload configuration file", type=['json'])

        if uploaded_file is not None:
            try:
                # Parse the uploaded JSON file
                new_config = json.load(uploaded_file)

                # Merge the new configuration with the existing one
                deep_merge_dicts(st.session_state.config, new_config)

                st.sidebar.success("Configuration loaded and merged successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please upload a valid configuration file.")
            
    
    # Load or select dataset (assuming df is already loaded into session state) 
    df = st.session_state.get('data', None)
    if df is not None:
        analyzer = DataAnalyzer(st.session_state.get('data', None))
        
        # Standardize Date Columns to ensure consistency across the app
        date_cols_standardized = []
        parsing_report = {}
        
        for col in analyzer.date_cols:
            original_values = df[col].copy()
            
            # SMART DATE PARSING (Heuristic-Based)
            parsed_series, used_notes = smart_parse_dates(df[col])
            
            # Check for data loss
            original_nans = df[col].isna().sum()
            new_nans = parsed_series.isna().sum()
            loss = new_nans - original_nans
            
            df[col] = parsed_series
            date_cols_standardized.append(f"{col} ({', '.join(used_notes)})")
            
            if loss > len(df) * 0.05: # >5% loss
                st.warning(f"High data loss in '{col}': {loss} rows could not be parsed.")

            # IMPORTANT: Since we modified the data in place (potentially fixing thousands of rows), 
            # we MUST invalidate any cached statistics that might have been computed on the "bad" data.
            # We do this UNCONDITIONALLY if we processed a date column.
            if 'current_dataset' in st.session_state:
                st.session_state.db_manager.clear_stats(st.session_state.current_dataset)
            if 'dataset_name' in st.session_state: # Another common key
                    st.session_state.db_manager.clear_stats(st.session_state.dataset_name)
            
            # Also clear Streamlit's data cache if possible, though 'st.cache_data.clear()' is global.
            # We rely on DBManager clearing the parquet files.

            # Note: Removed redundant 'else' block and 'res_primary' logic.
            
        if date_cols_standardized:
            st.info(f"**Date Consistency Applied**: {', '.join(date_cols_standardized)}")
            
        if parsing_report:
            with st.expander("Date Parsing Issues Detected", expanded=True):
                st.warning("Some values could not be converted to dates. Please check the examples below:")
                for col, examples in parsing_report.items():
                    st.write(f"**{col}**: Failed to parse strings like `{examples}`")
    
    
    # Create tabs for different visualization aspects
    tab_dashboard, tab1, tab2, tab3, tab4 = st.tabs(["Data Quality Dashboard", "Handle Clinical Anomalies", "Handle Inclusion Criteria", "Handle Outliers", "Compare Data Files"])

    with tab_dashboard:
        if df is not None:
            render_dashboard(df, st.session_state.config)
        else:
            st.warning("Please upload a dataset in the 'Main View' first.")

    with tab1:
        anomaly_family = "clinical_anomalies"
        with st.expander("New Anomaly Criterion", expanded=True):
            add_domain_expert_based_anomaly_mask(st.session_state.config, st)

        #with st.expander("Current Mask Families", expanded=True):
        st.subheader("Declared Anomaly Criteria")
        display_mask_family(st.session_state.config["mask_families"][anomaly_family])

        with st.expander("Apply Anomaly Detection based on Domain-Expert Criteria"):
            
            family_config = st.session_state.config["mask_families"].get(anomaly_family, {})
            #st.json(family_config)
            if family_config:
                detect = st.button("Run Anomaly Detection")

                if detect:
                    df, anom_cols = apply_family_mask(df, anomaly_family, family_config, st)
                    print(f"Columns containing anomaly detection information: {anom_cols}")
                    # Increase the max elements limit for Pandas Styler
                    pd.set_option("styler.render.max_elements", 600000)
                    styled_df = df.style.apply(
                                            color_columns, 
                                            axis=0, 
                                            columns_to_highlight=anom_cols
                                        )
                    st.write("Updated Dataset:")
                    st.dataframe(styled_df, width="stretch")
            else:
                st.error(f"Family '{anomaly_family}' not found in the configuration.")
        
        if 'anom_cols' in locals() and detect:
            with st.expander("Results", expanded=False):
                display_results_with_anomaly(df=df,anomaly_col=f"{anomaly_family}_any", st=st)
            
    with tab2:
        inclusion_family = "study_inclusion"
        with st.expander("New Inclusion Criterion", expanded=True):
            add_study_inclusion_mask(st.session_state.config, st)
        
        st.subheader("Declared Inclusion Criteria")
        display_mask_family(st.session_state.config["mask_families"][inclusion_family])

        with st.expander("Apply Data Validation based on Inclusion Criteria"):
            
            family_config = st.session_state.config["mask_families"].get(inclusion_family, {})
            #st.json(family_config)
            if family_config:
                check_inc = st.button("Run Inclusion Criteria Check")

                if check_inc:
                    df, inc_cols = apply_family_mask(df, inclusion_family, family_config, st)
                    print(f"Columns containing anomaly detection information: {inc_cols}")
                    styled_df = df.style.apply(
                                            color_columns, 
                                            axis=0, 
                                            columns_to_highlight=inc_cols
                                        )
                    st.write("Updated Dataset:")
                    st.dataframe(styled_df, width="stretch")
            else:
                st.error(f"Family '{inclusion_family}' not found in the configuration.")
        
        if 'inc_cols' in locals() and check_inc:
            with st.expander("Results", expanded=False):
                display_results_with_inclusion(df=df,inclusion_col=f"{inclusion_family}_all", st=st)


    with tab3:
        # Outliers Handler
        with st.expander("Outliers Handler"):
            df = add_outlier_handling_ui(df, analyzer.numeric_cols)
        
        with st.expander("Results", expanded=False):
            display_results_with_outliers(filtered_df=df, st=st)
    #    st.json(st.session_state.config["mask_families"])
    #    add_mask_family(st.session_state.config, st)

    with tab4:
        # 1. Add External Data
        with st.expander("Upload de Data Files To Compare", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Reference Dataset")
                data_c1, c1_file = data_selection("c1", "Reference Dataset")
                if data_c1 is not None:
                    data_c1 = process_dataset(data_c1)
                    st.dataframe(data_c1.head(), width='stretch')
                    st.info(f"Rows: {len(data_c1)} | Columns: {len(data_c1.columns)}")

            with col2:
                st.subheader("New Dataset")
                data_c2, c2_file = data_selection("c2", "New Dataset")
                if data_c2 is not None:
                    data_c2 = process_dataset(data_c2)
                    st.dataframe(data_c2.head(), width='stretch')
                    st.info(f"Rows: {len(data_c2)} | Columns: {len(data_c2.columns)}")


        with st.expander("Process Comparison", expanded=False):
            if data_c1 is not None and data_c2 is not None:
                st.write("---------------\n PROCESS COMPARISON \n --------------")
                
                common_columns = list(set(data_c1.columns) & set(data_c2.columns)) 

                if not common_columns:
                    st.warning("No common columns found between datasets!")
                else:  
                    row_ids = st.multiselect(
                        "Select a Subset Among Common identifiers",
                        options=common_columns,
                        default=None,
                        help="Choose the ids that exists in both datasets",
                        key="common_ids_comparator"
                    )

                    column_to_check = st.multiselect(
                        "Select a Subset Among Common Columns",
                        options=set(common_columns) - set(row_ids),
                        default=set(common_columns) - set(row_ids),
                        help="Choose the column that exists in both datasets",
                        key="common_columns_comparator"
                    )
                    
                    if not row_ids:
                        st.warning("Please select at least one common identifier (e.g., subject_id) to proceed.")
                    else:
                        run_comp = st.button("Run Comparison", key="run_comparison_bt", width='stretch')


        with st.expander("Results", expanded=True):
            if 'run_comp' in locals() and run_comp:
                #st.write(f"column_to_check size: {len(column_to_check)}")
                result_comp = mc.process_comparison_st(comparison_name='comp1',
                                                    df1=data_c1,
                                                    df2=data_c2,
                                                    common_id=row_ids,
                                                    common_cols=column_to_check
                                                    #output_file=
                                                )
                if result_comp:
                    df1, df2, common_id, comparison_result_filtered, rows_only_in_df1, rows_only_in_df2, modified_common_ids, _ = result_comp
                    # st.write("### Comparison Result")
                    # st.dataframe(comparison_result_filtered)
                    # st.write(f"len df: {comparison_result_filtered.shape}")
                    # st.write(f"Rows only in DataFrame 1: {len(rows_only_in_df1)}, {rows_only_in_df1}")
                    # st.write(f"Rows only in DataFrame 2: {len(rows_only_in_df2)}, {rows_only_in_df2}")
                    # st.write(f"Modified Rows: {common_id, modified_common_ids}")
                    # st.write(f"output_file: {output_file}")
                    # Create a new column order
                    # List of columns to move to the front
                    priority_columns = ['deletion', 'addition', 'modification', 'differing_columns']
                    new_column_order = (priority_columns + [col for col in comparison_result_filtered.columns if col not in priority_columns])

                    # Reorder the columns
                    comparison_result_filtered = comparison_result_filtered[new_column_order]
                    styled_df = comparison_result_filtered.style.apply(mc.highlight_modifications_tuples, args=("summary", rows_only_in_df1, rows_only_in_df2, modified_common_ids, common_id), axis=1)
                    st.dataframe(styled_df, width="stretch")

                    # Prepare Excel writer
                    if st.download_button(
                        label="Export Comparison Results",
                        key="compare_export_download_bt",
                        data=export_comparison_results(
                            df1, df2, comparison_result_filtered, 
                            rows_only_in_df1, rows_only_in_df2, 
                            modified_common_ids, common_id, "compare"
                        ),
                        file_name="comparison_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch'
                    ):
                        st.success("Comparison results exported successfully!")
                            
                    else:
                        st.error("Failed to process the comparison.")
                else:
                    st.warning("Please upload both datasets first.")

