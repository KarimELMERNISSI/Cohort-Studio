# pages/data_monitoring.py
import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np
from scipy import stats
from page_files.home import DataAnalyzer
from page_files.config_form import transform_expression, create_empty_config
from enrich import custom_metrics_and_filters as ecm
import monitor.outliers as mo
import monitor.changes as mc
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import os
import re
import json
from multipage import load_dataframe, get_file_hash

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
        z_scores = stats.zscore(self.df[column])
        return abs(z_scores) > threshold
        
    def detect_outliers_iqr(self, column, multiplier=1.5):
        """Detect outliers using IQR method"""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
    def detect_outliers_quantile(self, column, lower=0.01, upper=0.99):
        """Detect outliers using quantile method"""
        lower_bound = self.df[column].quantile(lower)
        upper_bound = self.df[column].quantile(upper)
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
        valid_methods = ['zscore', 'iqr', 'quantile', 'Local Outlier Factor', 'Isolation Forest']
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

        elif method in ['Local Outlier Factor', 'Isolation Forest']:
            # Apply dataset-wide methods
            data = self.df[columns]
            if method == 'Local Outlier Factor':
                is_outlier, outlier_score, _ = self.detect_lof_outliers(data=data, **kwargs)
            elif method == 'Isolation Forest':
                is_outlier, outlier_score, _ = self.detect_isolation_forest_outliers(data=data, **kwargs)

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
                    lower = kwargs.get('lower', self.df[col].quantile(0.01))
                    upper = kwargs.get('upper', self.df[col].quantile(0.99))
                    df_processed[col] = self.df[col].clip(lower, upper)

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
                        

            elif method in ['Isolation Forest', 'Local Outlier Factor']:
                # Initialize score and tag columns for dataset-wide methods
                tag_col = f"{method}_tag"
                score_col = f"{method}_score"

                # Update the DataFrame with the outlier information
                df_processed[tag_col] = is_outlier
                df_processed[score_col] = outlier_score

        return df_processed


    def clip_outliers(self, column, is_outlier, method, **kwargs):
        """Helper function to clip outliers based on the specified method."""
        if method == 'zscore':
            threshold = kwargs.get('threshold', 3)
            z_scores = stats.zscore(self.df[column])
            self.df.loc[is_outlier, column] = (
                self.df[column].mean() + threshold * self.df[column].std() * np.sign(z_scores[is_outlier])
            )
        elif method == 'iqr':
            multiplier = kwargs.get('multiplier', 1.5)
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            self.df.loc[is_outlier, column] = self.df[column].clip(lower_bound, upper_bound)
        elif method == 'quantile':
            lower = kwargs.get('lower', 0.01)
            upper = kwargs.get('upper', 0.99)
            lower_bound = self.df[column].quantile(lower)
            upper_bound = self.df[column].quantile(upper)
            self.df.loc[is_outlier, column] = self.df[column].clip(lower_bound, upper_bound)


    def get_outliers_masks(self):
            """Get outliers masks"""
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
        valid_methods = ['zscore', 'iqr', 'quantile', 'Local Outlier Factor', 'Isolation Forest']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}.")

        # Column-wise methods
        if method in ['zscore', 'iqr', 'quantile']:
            for column in columns:
                # Ensure the column is numeric
                if self.original_df[column].dtype not in ['int64', 'float64']:
                    continue

                # Detect outliers based on the selected method
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

                # Calculate outlier statistics
                n_outliers = sum(is_outlier)
                pct_outliers = (n_outliers / len(self.original_df)) * 100

                # Update summary for the column
                summary[column] = {
                    'total_outliers': n_outliers,
                    'percentage_outliers': pct_outliers,
                    'original_range': (self.original_df[column].min(), self.original_df[column].max()),
                    'outlier_values': self.original_df.loc[is_outlier, column].tolist()
                }

        # Dataset-wide methods
        elif method in ['Local Outlier Factor', 'Isolation Forest']:
            data = self.df[columns]#.dropna()
            # data = self.df[columns].select_dtypes(include=['int64', 'float64']).dropna()
            #if data.empty:
            #    raise ValueError("No numeric data available for outlier detection.")
            
            if method == 'Local Outlier Factor':
                is_outlier, scores, _ = self.detect_lof_outliers(data=data, **kwargs)
            elif method == 'Isolation Forest':
                is_outlier, scores, _ = self.detect_isolation_forest_outliers(data=data, **kwargs)

            # Calculate overall statistics
            n_outliers = sum(is_outlier)
            pct_outliers = (n_outliers / len(data)) * 100

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

def data_upload(tmp:str) -> Optional[Union[pd.DataFrame, str]]:
    """Handle the upload or path input for the enrichment dataset."""
    upload_method = st.radio("Choose upload method for Enrichment Dataset:", ["Upload File", "Enter File Path"], key=f"{tmp}_data_method")
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Drag and drop your enrichment dataset here",
            type=['csv', 'xlsx'],
            key=f"{tmp}_data_uploader"
        )
        if uploaded_file is not None:
            try:
                st.write(f"file_name: {uploaded_file.name}")
                file_path = uploaded_file.name
                df = load_dataframe(uploaded_file)
                st.success("✅ Dataset successfully loaded from upload.")
                return df, file_path
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None, None
    else:
        # Get folder path and file name from user input
        folder_path = st.text_input("Enter folder path for the dataset:", key=f"{tmp}_folder_path")
        file_name = st.text_input("Enter file name for the dataset:", key=f"{tmp}_data_name")

        # Dynamically set file_path based on folder_path and file_name
        file_path = os.path.join(folder_path, file_name) if folder_path and file_name else ""

        # Display file path input field with computed value
        file_path_input = st.text_input("Enter file path for the enrichment dataset:", value=file_path, key=f"{tmp}_data_path")
        if file_path_input:
            try:
                df = load_dataframe(file_path_input)
                st.success("✅ Dataset successfully loaded from provided path.")
                return df, file_name
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
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
            ["zscore", "iqr", "quantile", "Local Outlier Factor", "Isolation Forest"],
            help="Z-score: Uses standard deviations from mean\n"
                 "IQR: Uses interquartile range\n"
                 "Quantile: Uses percentile thresholds"
        )
    with col2:
        handling_strategy = st.selectbox(
            "Handling Strategy",
            ["None", "remove", "clip", "tag"],
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

def display_combined_information(df, df_processed, summary, selected_cols, detection_method, handling_strategy):
    """Display outlier summary, outlier impact (percentage change), and before/after statistics side by side for each variable."""

    st.subheader("Outlier Detection Summary")

    # Boundary to separate content
    st.markdown("---")

    # Loop through columns for summary and statistics display
    if detection_method in ["zscore", "iqr", "quantile"]:
        for col in selected_cols:
            st.markdown(f"### {col}")

            # Create 3 columns for side-by-side layout
            cols = st.columns(2)
            
            with cols[0]:
                # Outlier Summary Section
                stats = summary.get(col, {})
                st.markdown(f"**Outlier Summary**")
                st.markdown(f"- **Total outliers**: {stats.get('total_outliers', 'N/A')}")
                st.markdown(f"- **Percentage of outliers**: {stats.get('percentage_outliers', 0):.2f}%")
            
            with cols[1]:
                # Outlier Impact (percentage change) Section
                st.markdown("**Outlier Impact (percentage change)**")
                impact = calculate_outlier_impact(df, df_processed, [col])
                if col in impact:
                    imp = impact[col]
                    if isinstance(imp, dict):
                        mean_change_percentage = imp.get('mean_change_percentage', 0)
                        std_change_percentage = imp.get('std_change_percentage', 0)
                        min_change_percentage = imp.get('min_change_percentage', 0)
                        max_change_percentage = imp.get('max_change_percentage', 0)
                        
                        st.markdown(f"- **Mean Change (%)**: {mean_change_percentage:.2f}%")
                        st.markdown(f"- **Std Change (%)**: {std_change_percentage:.2f}%")
                        st.markdown(f"- **Min Change (%)**: {min_change_percentage:.2f}%")
                        st.markdown(f"- **Max Change (%)**: {max_change_percentage:.2f}%")
                    else:
                        st.markdown(f"- Impact can't be measured for categorical data.")
            
            # with cols[2]:
            #     # Outlier Impact (Before/After Statistics)
            #     st.markdown("**Before/After Statistics**")
            #     if pd.api.types.is_numeric_dtype(df[col]):
            #         st.write(f"- **Before:** Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}, "
            #                 f"Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
            #         st.write(f"- **After:** Mean: {df_processed[col].mean():.2f}, Std: {df_processed[col].std():.2f}, "
            #                 f"Min: {df_processed[col].min():.2f}, Max: {df_processed[col].max():.2f}")
            #     else:
            #         # For categorical columns, show unique values and most frequent occurrences
            #         unique_before = df[col].nunique()
            #         top_before = df[col].value_counts().idxmax()
            #         freq_before = df[col].value_counts().max()

            #         unique_after = df_processed[col].nunique()
            #         top_after = df_processed[col].value_counts().idxmax()
            #         freq_after = df_processed[col].value_counts().max()

            #         st.write(f"- **Before:** Unique Values: {unique_before}, Most Frequent: {top_before} ({freq_before} occurrences)")
            #         st.write(f"- **After:** Unique Values: {unique_after}, Most Frequent: {top_after} ({freq_after} occurrences)")

            # Boundary for better section separation
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
        impact = calculate_outlier_impact(df, df_processed, selected_cols)

        # Prepare a DataFrame for impact
        impact_data = []
        for col, imp in impact.items():
            if isinstance(imp, dict):
                impact_data.append([
                    col,
                    imp.get('mean_change_percentage', 0),
                    imp.get('std_change_percentage', 0),
                    imp.get('min_change_percentage', 0),
                    imp.get('max_change_percentage', 0)
                ])
            else:
                impact_data.append([col, None, None, None, None])

        # Create the DataFrame
        impact_df = pd.DataFrame(
            impact_data,
            columns=["Column", "Mean Change (%)", "Std Change (%)", "Min Change (%)", "Max Change (%)"]
        ).set_index("Column")

        # Style the DataFrame
        styled_impact_df = (
            impact_df.style
            .format(
                {
                    "Mean Change (%)": "{:.2f}%", 
                    "Std Change (%)": "{:.2f}%", 
                    "Min Change (%)": "{:.2f}%", 
                    "Max Change (%)": "{:.2f}%"
                }
            )
            .background_gradient(
                cmap="RdBu",  # Diverging colormap: red for negative, blue for positive
                vmin=-100, vmax=100,  # Define range for divergence (-100 to 100 for percentages)
                axis=None,  # Apply gradient to the entire table
                subset=["Mean Change (%)", "Std Change (%)", "Min Change (%)", "Max Change (%)"]
            )
            .set_table_styles(
                [
                    {'selector': 'thead th', 'props': [('text-align', 'center'), ('font-size', '16px')]},
                    {'selector': 'tbody td', 'props': [('text-align', 'center'), ('padding', '8px')]},
                    {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
                    {'selector': 'tbody tr:hover', 'props': [('background-color', '#f1f1f1')]},
                ]
            )
            .set_properties(**{'width': '100%', 'border': '1px solid #ddd'})
        )

        # Render the table in Streamlit
        container = st.container()
        with container:
            st.write("### Overall Impact of Outliers Handling Operation on the Dataset")
            st.dataframe(styled_impact_df, use_container_width=True)



def calculate_outlier_impact(df, df_processed, selected_cols):
    """
    Calculate the impact of outlier handling on each selected column by computing
    the percentage change in mean, std, min, and max values.
    """
    impact = {}
    
    for col in selected_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Calculate before and after means, standard deviations, min, and max
            before_mean = df[col].mean()
            after_mean = df_processed[col].mean()
            
            before_std = df[col].std()
            after_std = df_processed[col].std()
            
            before_min = df[col].min()
            after_min = df_processed[col].min()
            
            before_max = df[col].max()
            after_max = df_processed[col].max()
            
            # Calculate the percentage change in mean, std, min, and max
            mean_change_percentage = ((after_mean - before_mean) / before_mean) * 100 if before_mean != 0 else 0
            std_change_percentage = ((after_std - before_std) / before_std) * 100 if before_std != 0 else 0
            min_change_percentage = ((after_min - before_min) / before_min) * 100 if before_min != 0 else 0
            max_change_percentage = ((after_max - before_max) / before_max) * 100 if before_max != 0 else 0
            
            # Store the percentage changes in a dictionary
            impact[col] = {
                'mean_change_percentage': mean_change_percentage,
                'std_change_percentage': std_change_percentage,
                'min_change_percentage': min_change_percentage,
                'max_change_percentage': max_change_percentage
            }
        else:
            impact[col] = "Impact can't be measured for categorical data."
    
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
    elif method in ['Local Outlier Factor', 'Isolation Forest']:
        is_outlier = outlier_predictions.get('overall', [])
        n_outliers = sum(is_outlier)
        pct_outliers = (n_outliers / len(df)) * 100
        summary['overall'] = {
            'total_outliers': n_outliers,
            'percentage_outliers': pct_outliers,
        }

    return summary




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
    # TBD
    # Preview the enriched dataset
    display_results(df=filtered_df, bool_column=None, title=None, key_base="outliers_data", st=st)

#############################################################################################

################################ ANOMALY AND INCLUSION HANDLING ############################################################


##### COMMON PART

def ensure_family_exists(config: Dict[str, Any], family_name: str):
    """Ensure the specified family exists in the configuration."""
    if family_name not in config["mask_families"]:
        config["mask_families"][family_name] = {}


def validate_mask_name(config: Dict[str, Any], family_name: str, mask_name: str, st_container):
    """Validate the mask name and handle duplicates."""
    if not mask_name:
        st_container.warning("⚠️ Please provide a mask name.")
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
    is_lower_bound = st_container.checkbox("Need lower bound?", key=f"{base_key}_lower_bound")
    is_upper_bound = st_container.checkbox("Need upper bound?", key=f"{base_key}_upper_bound")
    lower_bound = st_container.number_input(
        "Lower Bound", value=0.0, disabled=not is_lower_bound, key=f"{base_key}_lower_bound_input"
    ) if is_lower_bound else None
    upper_bound = st_container.number_input(
        "Upper Bound", value=0.0, disabled=not is_upper_bound, key=f"{base_key}_upper_bound_input"
    ) if is_upper_bound else None
    strategy = st_container.selectbox(
        "Strategy",
        ["exclude", "include"],
        disabled=not (is_lower_bound and is_upper_bound),
        key=f"{base_key}_strategy"
    ) if is_lower_bound and is_upper_bound else None

    return {
        "numeric": True,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "strategy": strategy,
    }


def handle_expression_mask_input(st_container, base_key: str):
    """Collect expression mask inputs from the user."""
    expression = st_container.text_area(
        "Expression", placeholder="Enter a valid mask expression", key=f"{base_key}_txt_area"
    )
    use_simplified_mask_expression = st_container.checkbox("Use Simplified Mask Expression", key=f"{base_key}_simpl")
    if not expression.strip():
        st_container.error("⚠️ Expression cannot be empty.")
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


def add_mask_family(config: Dict[str, Any], st_container):
    st_container.subheader("Add New Mask Family")

    # Family Selection
    family_name = st_container.text_input("Family Name", placeholder="Enter family name")
    if not family_name:
        st_container.warning("⚠️ Please provide a family name.")
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
        st_container.success(f"✅ Mask '{mask_name}' added to family '{family_name}'.")
    
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
                with st.expander(f"🔢 {key}", expanded=False):
                    # Numeric details
                    st.markdown(f"**Type:** 🔢 <span style='color:{color};'>Numeric</span>", unsafe_allow_html=True)

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
                    st.success("✅ Numeric Validation Active")

                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                # Expander for non-numeric configuration
                with st.expander(f"📝 {key}", expanded=False):
                    # Non-numeric details
                    st.markdown(f"**Type:** 📝 <span style='color:{color};'>Expression</span>", unsafe_allow_html=True)

                    expression = value.get('expression', "No expression defined")
                    #st.markdown("**Validation Expression:**")
                    st.code(expression, language='python')
                    st.warning("⚠️ Custom Expression Validation")

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
    with st.expander("📂 View Mask Families Configuration", expanded=False):
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
                    <h4 style="margin: 0;">{'🔢' if is_numeric else '📝'} {key}</h4>
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
                st.success("✅ Numeric Validation Active")

            # Non-Numeric Configuration
            else:
                expression = value.get('expression', "No expression defined")
                st.markdown("**Validation Expression:**")
                st.code(expression, language='python')
                st.warning("⚠️ Custom Expression Validation")

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
    st.dataframe(styled_df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        # Download button
        st.download_button(
            label="Download Dataset",
            key= f"{key_base}_dl_bt",
            data=to_excel(styled_df),
            file_name=f"{key_base}_dataset.xlsx",
            mime="text/xlsx",
            use_container_width=True
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

        if st.button("Apply Filtering Criteria for Further Analysis", use_container_width=True, key=f"{key_base}_apply_criteria"):
            # Ensure df_masked exists and is valid
            #st.write("Masked DataFrame Preview:")
            #st.dataframe(df_masked)
                
            # Update session state
            st.session_state.data = df_masked
            st.session_state.working_df = df_masked
                
            st.success("✅ Dataset saved and ready for further analysis!")
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
            use_container_width=True
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

def add_domain_expert_based_anomaly_mask(config: Dict[str, Any], st_container):
    st_container.subheader("Define New Anomaly Criterion")

    family_name = "clinical_anomalies"
    ensure_family_exists(config, family_name)

    # Operator Selection
    family_operator = "any"
    config["mask_families"][family_name]["operator"] = family_operator

    # Mask Addition
    mask_name = st_container.text_input("Anomaly Mask Name", placeholder="Enter mask name")
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
        st_container.success(f"✅ Anomaly mask '{mask_name}' added to family '{family_name}'.")
    
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

def add_study_inclusion_mask(config: Dict[str, Any], st_container):
    st_container.subheader("Define New Study Inclusion Criterion")

    family_name = "study_inclusion"
    ensure_family_exists(config, family_name)

    # Operator Selection
    family_operator = "all"
    config["mask_families"][family_name]["operator"] = family_operator

    # Mask Addition
    mask_name = st_container.text_input("Inclusion Mask Name", placeholder="Enter mask name")
    if not validate_mask_name(config, family_name, mask_name, st_container):
        return

    is_numeric = st_container.checkbox("Is Numeric", key="is_num_inclusion")
    if is_numeric:
        mask_details = handle_numeric_mask_input(st_container, base_key=family_name)
    else:
        mask_details = handle_expression_mask_input(st_container, base_key=family_name)

    if mask_details and st_container.button("Add Inclusion Criterion"):
        config["mask_families"][family_name][mask_name] = mask_details
        st_container.success(f"✅ Inclusion mask '{mask_name}' added to family '{family_name}'.")


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
            value=f"{exclusion_percentage:.2f}%",
            delta=f"-{excluded_count}",
            help=(
                "This metric indicates the percentage of rows to remove from the study compared to the total number of rows. "
                "'Value' shows the exclusion percentage, and 'Delta' represents the decrease in rows due to exclusion."
            )
        )
    
    # Preview the enriched dataset
    display_results(df=df, bool_column=inclusion_col, title="Rows Meeting Inclusion Criteria", key_base="inclusion_data", st=st)


####################### APP PART ###############################

def app():

    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = create_empty_config()
    
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
    
    
    # Create tabs for different visualization aspects
    tab1, tab2, tab3, tab4 = st.tabs(["Handle Clinical Anomalies", "Handle Inclusion Criteria", "Handle Outliers", "Compare Data Files"])

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
                    styled_df = df.style.apply(
                                            color_columns, 
                                            axis=0, 
                                            columns_to_highlight=anom_cols
                                        )
                    st.write("Updated Dataset:")
                    st.dataframe(styled_df)
            else:
                st.error(f"Family '{anomaly_family}' not found in the configuration.")
        
        if 'anom_cols' in locals() and detect:
            with st.expander("📊 Results", expanded=False):
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
                    st.dataframe(styled_df)
            else:
                st.error(f"Family '{inclusion_family}' not found in the configuration.")
        
        if 'inc_cols' in locals() and check_inc:
            with st.expander("📊 Results", expanded=False):
                display_results_with_inclusion(df=df,inclusion_col=f"{inclusion_family}_all", st=st)


    with tab3:
        # Outliers Handler
        with st.expander("Outliers Handler"):
            df = add_outlier_handling_ui(df, analyzer.numeric_cols)
        
        with st.expander("📊 Results", expanded=False):
            display_results_with_outliers(filtered_df=df, st=st)
    #    st.json(st.session_state.config["mask_families"])
    #    add_mask_family(st.session_state.config, st)

    with tab4:
        # 1. Add External Data
        with st.expander("📥 Upload de Data Files To Compare", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Reference Dataset")
                data_c1, c1_file = data_upload("c1")
                if data_c1 is not None:
                    st.dataframe(data_c1.head(), use_container_width=True)
                    st.info(f"Rows: {len(data_c1)} | Columns: {len(data_c1.columns)}")

            with col2:
                st.subheader("New Dataset")
                data_c2, c2_file = data_upload("c2")
                if data_c2 is not None:
                    st.dataframe(data_c2.head(), use_container_width=True)
                    st.info(f"Rows: {len(data_c2)} | Columns: {len(data_c2.columns)}")


        with st.expander("🔄 Process Comparison", expanded=False):
            if data_c1 is not None and data_c2 is not None:
                st.write("---------------\n PROCESS COMPARISON \n --------------")
                
                common_columns = list(set(data_c1.columns) & set(data_c2.columns)) 

                if not common_columns:
                    st.warning("No common columns found between datasets!")
                    return None
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
                run_comp = st.button("Run Comparison", key="run_comparison_bt", use_container_width=True)


        with st.expander("📊 Results", expanded=True):
            if 'run_comp' in locals():
                if run_comp:           
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
                        st.dataframe(styled_df)

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
                            use_container_width=True
                        ):
                            st.success("Comparison results exported successfully!")
                            
                    else:
                        st.error("Failed to process the comparison.")
                else:
                    st.warning("Please upload both datasets first.")
    
