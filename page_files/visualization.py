import streamlit as st
import re
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from plotly.subplots import make_subplots
from matplotlib.colors import to_rgba
from page_files import data_monitoring as dm


# class OutlierHandler:
#     """Helper class to detect and handle outliers using various methods"""
#     def __init__(self, df, config):
#         self.df = df.copy()
#         self.original_df = df.copy()
#         self.masks = None  # Initialize to None
  
#     def detect_outliers_zscore(self, column, threshold=3):
#         """Detect outliers using Z-score method"""
#         z_scores = stats.zscore(self.df[column])
#         return abs(z_scores) > threshold
        
#     def detect_outliers_iqr(self, column, multiplier=1.5):
#         """Detect outliers using IQR method"""
#         Q1 = self.df[column].quantile(0.25)
#         Q3 = self.df[column].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - multiplier * IQR
#         upper_bound = Q3 + multiplier * IQR
#         return (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
#     def detect_outliers_quantile(self, column, lower=0.01, upper=0.99):
#         """Detect outliers using quantile method"""
#         lower_bound = self.df[column].quantile(lower)
#         upper_bound = self.df[column].quantile(upper)
#         return (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
    
#     def handle_outliers(self, columns, method='zscore', handling_strategy='remove', **kwargs):
#         """
#         Handle outliers in specified columns using the chosen method and strategy
        
#         Parameters:
#         -----------
#         columns : list
#             List of column names to handle outliers in
#         method : str
#             'zscore', 'iqr', or 'quantile'
#         handling_strategy : str
#             'remove', 'clip'
#         kwargs : dict
#             Additional parameters for the chosen method
#         """
#         self.df = self.original_df.copy()  # Reset to original data
        
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
            
#             # Handle outliers using the specified strategy
#             if handling_strategy == 'remove':
#                 self.df = self.df[~is_outlier]
#             elif handling_strategy == 'clip':
#                 if method == 'zscore':
#                     z_scores = stats.zscore(self.df[column])
#                     self.df.loc[is_outlier, column] = self.df[column].mean() + threshold * self.df[column].std() * np.sign(z_scores[is_outlier])
#                 elif method == 'iqr':
#                     Q1 = self.df[column].quantile(0.25)
#                     Q3 = self.df[column].quantile(0.75)
#                     IQR = Q3 - Q1
#                     lower_bound = Q1 - multiplier * IQR
#                     upper_bound = Q3 + multiplier * IQR
#                     self.df.loc[is_outlier, column] = self.df[column].clip(lower_bound, upper_bound)
#                 elif method == 'quantile':
#                     lower_bound = self.df[column].quantile(lower)
#                     upper_bound = self.df[column].quantile(upper)
#                     self.df.loc[is_outlier, column] = self.df[column].clip(lower_bound, upper_bound)
#             # elif handling_strategy == 'winsorize':
#             #     if method == 'zscore':
#             #         self.df[column] = stats.mstats.winsorize(self.df[column], limits=[0.05, 0.05])
#             #     else:
#             #         if method == 'iqr':
#             #             lower_pct = 0.25 - multiplier * 0.25
#             #             upper_pct = 0.75 + multiplier * 0.25
#             #         else:  # quantile
#             #             lower_pct = lower
#             #             upper_pct = upper
#             #         self.df[column] = stats.mstats.winsorize(self.df[column], limits=[lower_pct, 1-upper_pct])
        
#         return self.df
    
#     def get_outliers_masks(self, mask_name="outliers_masks"):
#             """Get outliers masks"""
#             return self.masks

#     def get_outlier_summary(self, columns, method='zscore', **kwargs):
#         """Get summary of outliers for each column"""
#         summary = {}
#         for column in columns:
#             if self.original_df[column].dtype not in ['int64', 'float64']:
#                 continue
                
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
            
#             n_outliers = sum(is_outlier)
#             pct_outliers = (n_outliers / len(self.original_df)) * 100
            
#             summary[column] = {
#                 'total_outliers': n_outliers,
#                 'percentage_outliers': pct_outliers,
#                 'original_range': (self.original_df[column].min(), self.original_df[column].max()),
#                 'outlier_values': self.original_df.loc[is_outlier, column].tolist()
#             }
        
#         return summary

# ##### USE OUTLIERHANDLER ####

# def add_outlier_handling_ui(df, numeric_cols):
#     """Add outlier handling UI components"""
#     # Initialize outlier handler
#     #config = st.session_state.config
#     outlier_handler = OutlierHandler(df, st.session_state.config)
    
#     # Select columns for outlier handling
#     selected_cols = st.multiselect(
#         "Select columns for outlier handling",
#         numeric_cols,
#         default=numeric_cols[:1] if numeric_cols else []
#     )
    
#     if not selected_cols:
#         st.warning("Please select at least one column")
#         return df
        
#     # Outlier detection method
#     col1, col2 = st.columns(2)
#     with col1:
#         detection_method = st.selectbox(
#             "Outlier Detection Method",
#             ["zscore", "iqr", "quantile"],
#             help="Z-score: Uses standard deviations from mean\n"
#                  "IQR: Uses interquartile range\n"
#                  "Quantile: Uses percentile thresholds"
#         )
        
#     with col2:
#         handling_strategy = st.selectbox(
#             "Handling Strategy",
#             ["None", "remove", "clip"],
#             help="Remove: Exclude outliers\n"
#                  "Clip: Cap at threshold values"
#         )
    
#     # Method-specific parameters
#     if detection_method == "zscore":
#         threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
#         params = {"threshold": threshold}
#     elif detection_method == "iqr":
#         multiplier = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1)
#         params = {"multiplier": multiplier}
#     else:  # quantile
#         col1, col2 = st.columns(2)
#         with col1:
#             lower = st.number_input("Lower percentile", 0.0, 0.5, 0.01, 0.01)
#         with col2:
#             upper = st.number_input("Upper percentile", 0.5, 1.0, 0.99, 0.01)
#         params = {"lower": lower, "upper": upper}
    
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
#         st.write(f"\n**{col}**:")
#         st.write(f"- Total outliers: {stats['total_outliers']}")
#         st.write(f"- Percentage of outliers: {stats['percentage_outliers']:.2f}%")
#         st.write(f"- Original range: ({stats['original_range'][0]:.2f}, {stats['original_range'][1]:.2f})")
    
#     # Handle outliers if strategy is not 'none'
#     if handling_strategy != "none":
#         df_processed = outlier_handler.handle_outliers(
#             selected_cols,
#             method=detection_method,
#             handling_strategy=handling_strategy,
#             **params
#         )
        
#         # Show before/after statistics
#         st.write("\n**Before/After Statistics:**")
#         for col in selected_cols:
#             st.write(f"\n{col}:")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.write("Before:")
#                 st.write(f"- Mean: {df[col].mean():.2f}")
#                 st.write(f"- Std: {df[col].std():.2f}")
#                 st.write(f"- Range: ({df[col].min():.2f}, {df[col].max():.2f})")
#             with col2:
#                 st.write("After:")
#                 st.write(f"- Mean: {df_processed[col].mean():.2f}")
#                 st.write(f"- Std: {df_processed[col].std():.2f}")
#                 st.write(f"- Range: ({df_processed[col].min():.2f}, {df_processed[col].max():.2f})")
        
#         # Add a button to view outlier values
#         if st.button("View Outlier Values"):
#             outlier_values_dialog(summary)
        
#         return df_processed
        
#     return df


# def outlier_values_dialog(outlier_summary):
#     """Display outlier values in a dialog"""
#     st.write("Outlier Values:")
#     for col, stats in outlier_summary.items():
#         st.subheader(col)
#         st.write(stats['outlier_values'])

#############################################################################################

class DataAnalyzer:
    """Helper class to analyze and validate data columns"""
    def __init__(self, df):
        self.df = df
        self.analyze_columns()
    

    def analyze_columns(self):
        """Analyze and categorize columns by data type"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = self._detect_date_columns()
        self.binary_cols = self._detect_binary_columns()
        self.low_cardinality_numeric_cols = self._detect_low_cardinality_numeric_columns()
        self.non_binary_low_cardinality_numeric_cols = [col for col in self.low_cardinality_numeric_cols if col not in self.binary_cols]
        self.timedelta_cols = self.df.select_dtypes(include=[np.timedelta64]).columns.tolist()
        
        # Remove low-cardinality numeric columns from numeric_cols and add to categorical_cols
        for col in self.low_cardinality_numeric_cols:
            if col in self.numeric_cols:
                self.numeric_cols.remove(col)
            if col not in self.categorical_cols and col not in self.binary_cols:
                self.categorical_cols.append(col)
        
        # Remove date and time interval columns from categorical_cols and numeric_cols
        for col in self.date_cols + self.timedelta_cols:
            if col in self.categorical_cols:
                self.categorical_cols.remove(col)
            if col in self.numeric_cols:
                self.numeric_cols.remove(col)


        self.high_cardinality_cat_cols = self._detect_high_cardinality_cat_columns()
        

    def _detect_date_columns(self):
        """Detect columns that are likely dates, with additional checks for accuracy."""
        date_cols = []
        # Expanded list of formats for date detection, including date-time
        formats = [
            "%d/%m/%Y",                # e.g., 31/12/2021
            "%m/%d/%Y",                # e.g., 12/31/2021
            "%Y-%m-%d",                # e.g., 2021-12-31
            "%d-%m-%Y",                # e.g., 31-12-2021
            "%m-%d-%Y",                # e.g., 12-31-2021
            "%Y-%m-%d %H:%M:%S",       # e.g., 2021-12-31 23:59:59
            "%Y-%m-%d %H:%M:%S.%f",    # e.g., 2021-12-31 23:59:59.123456
            "%d %b %Y",                # e.g., 31 Dec 2021
            "%b %d, %Y",               # e.g., Dec 31, 2021
            "%Y/%m/%d",                # e.g., 2021/12/31
            "%d-%b-%Y",                # e.g., 31-Dec-2021
            "%Y-%m-%d %H:%M:%S",       # e.g., 2021-12-31 23:59:59
            "%Y/%m/%d %H:%M:%S",       # e.g., 2021/12/31 23:59:59
            "%Y-%m-%d %H:%M:%S.%f",    # e.g., 2021-12-31 23:59:59.123456
            "%Y-%m-%d %H:%M:%S",       # Timestamps like 1959-05-01 00:00:00
            "%Y-%m-%d %H:%M:%S.%f",    # Timestamps like 1959-05-01 00:00:00.000000
            "%H:%M:%S",                # Time-only values, e.g., 23:59:59
        ]
        
        for col in self.df.columns:
            # Add proper date format columns
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                date_cols.append(col)

            # Proceed only if column is of object or string type, likely to contain dates
            if pd.api.types.is_object_dtype(self.df[col]):
                is_date_column = False
                
                # Check if values look like dates using regex (including times)
                non_null_values = self.df[col].dropna().astype(str)
                sample_size = min(10, len(non_null_values))
                sample_values = non_null_values.sample(n=sample_size, replace=(sample_size > len(non_null_values)))
                #sample_values = self.df[col].dropna().astype(str).sample(min(10, len(self.df[col])))  # Check a sample
                if sample_values.str.match(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}[-/]\d{1,2}[-/]\d{1,2} \d{1,2}:\d{2}(:\d{2})?)').all():
                    # Try parsing each format to confirm it's a date column
                    for fmt in formats:
                        try:
                            converted = pd.to_datetime(self.df[col], format=fmt, errors='coerce')
                            non_na_dates = converted.notna().sum()
                            if non_na_dates > len(self.df[col]) * 0.5:  # More than 50% are dates
                                date_cols.append(col)
                                is_date_column = True
                                break
                        except Exception:
                            continue
                    
                    if not is_date_column:
                        # Fallback with general parsing
                        try:
                            converted = pd.to_datetime(self.df[col], errors='coerce')
                            non_na_dates = converted.notna().sum()
                            if non_na_dates > len(self.df[col]) * 0.5:
                                date_cols.append(col)
                        except Exception:
                            continue
        
        return date_cols

    
    def _detect_binary_columns(self):
        """Detect columns with only two unique values, including numeric columns"""
        return [col for col in self.df.columns if self.df[col].nunique() == 2]
    
    def _detect_low_cardinality_numeric_columns(self):
        """Detect numeric columns with fewer or eq to 5 unique values to treat as categorical"""
        return [col for col in self.numeric_cols if self.df[col].nunique() <= 5]
    
    def _detect_high_cardinality_cat_columns(self):
        """Detect categorical columns with many unique values"""
        return [col for col in self.categorical_cols if self.df[col].nunique() > 0.5 * len(self.df)]
    
    def get_suitable_columns(self, plot_type):
        """Get suitable columns for different plot types"""
        # Ensure numeric columns are unique and sorted
        num_vars = sorted(set(self.numeric_cols))
        date_cols = sorted(set(self.date_cols+self.timedelta_cols))
        # Combine categorical and binary columns, excluding high cardinality columns
        cat_vars = sorted(set(col for col in self.categorical_cols + self.binary_cols + self.low_cardinality_numeric_cols if col not in self.high_cardinality_cat_cols))

        if plot_type == "Histogram":
            return {
                "x": num_vars,
                "color": cat_vars
            }
        elif plot_type == "Box Plot":
            return {
                "y": num_vars,
                "x": cat_vars
            }
        elif plot_type == "Violin Plot":
            return {
                "y": num_vars,
                "x": cat_vars
            }
        elif plot_type == "Scatter Plot":
            return {
                "x": num_vars,
                "y": num_vars,
                "color": cat_vars,
                "size": num_vars
            }
        elif plot_type == "Line Plot":
            return {
                "x": date_cols,
                "y": num_vars,
                "color": cat_vars
            }
        elif plot_type == "Bar Plot":
            return {
                "x": cat_vars,
                "y": num_vars,
                "color": cat_vars
            }
        elif plot_type == "Pie Chart":
            return {
                "names": [col for col in self.categorical_cols if self.df[col].nunique() <= 10],
                "values": num_vars
            }
        elif plot_type == "Correlation Matrix":
            return {
                "numeric": num_vars,
                "categorical": [col for col in cat_vars if all(pd.api.types.is_numeric_dtype(val) for val in self.df[col].unique())]
            }
        return {}

#############################################################################################

def adjust_color_to_rgba(color, transparency=0.2):
    """
    Converts a color (hex or rgb format) to an RGBA string with the specified transparency.
    """
    if isinstance(color, str):
        if color.startswith('#'):  # Hex format
            rgba = to_rgba(color, transparency)
        elif color.startswith('rgb'):  # RGB format (e.g., "rgb(228,26,28)")
            rgb_values = list(map(int, re.findall(r'\d+', color)))
            rgba = (*[v / 255 for v in rgb_values], transparency)
        else:
            rgba = to_rgba(color, transparency)  # Named color
    else:
        rgba = to_rgba(color, transparency)

    return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"


def get_figure_column_names(fig, df):
    """
    Extracts the column names or data-related attributes from a Plotly figure
    and checks if they match any columns in the provided DataFrame (df).
    This function scans through all traces in the Plotly figure and attempts to match 
    the columns referenced in the figure with those in the provided DataFrame.

    Parameters:
    fig (plotly.graph_objects.Figure): The figure from which column names will be extracted.
    df (pandas.DataFrame): The DataFrame to check column names against.

    Returns:
    set: A set of column names or data-related attributes used in the figure that match df's columns.
    """

    # Initialize an empty set to store matching column names
    columns = set()

    # Convert the columns of the DataFrame to a set for efficient comparison
    df_columns = set(df.columns)
    print("Figure data traces:", fig.data)

    # Loop through each trace in the Plotly figure
    for trace in fig.data:
        # Check if 'x', 'y', 'z', 'text', 'labels', or 'values' attributes exist in the trace
        # These are typical data-related attributes in Plotly figures
        for attr in ['x', 'y', 'z', 'text', 'labels', 'values']:
            if attr in trace:
                # Extract the data corresponding to the attribute (e.g., x or y values)
                data = trace[attr]
                
                # If the data is a list, numpy array, or pandas Series, we proceed to check its content
                if isinstance(data, (list, np.ndarray, pd.Series)):
                    # Check if the data is a pandas Series
                    # If it's a Series, check if its name matches any column in the DataFrame
                    if isinstance(data, pd.Series):
                        if data.name in df_columns:
                            columns.add(data.name)
                            print(f"Found matching column for '{attr}': {data.name}")
                    
                    # If the data is a list or numpy array, we need to check if it matches any DataFrame column
                    elif isinstance(data, (list, np.ndarray)):
                        # Loop through each column in the DataFrame to compare the data
                        for col in df_columns:
                            # Check if the data matches the values in a DataFrame column
                            if np.array_equal(data, df[col].values):
                                columns.add(col)
                                print(f"Found matching data for '{attr}' with column: {col}")
                                break  # Exit the loop once a match is found

        # Check for the 'name' attribute in the trace, which can correspond to a categorical column
        if 'name' in trace and trace['name'] in df_columns:
            columns.add(trace['name'])
            print(f"Found matching column for 'name': {trace['name']}")

    # Return the set of columns found in the figure that also exist in the DataFrame
    return columns


def add_custom_hovertemplate(fig, df):
    """
    Adds a custom hovertemplate to an existing Plotly figure, showing only selected columns on hover.

    Parameters:
    fig (plotly.graph_objects.Figure): The existing Plotly figure to modify.
    df (pandas.DataFrame): The DataFrame with data corresponding to the figure.
    """

    default_cols = list(get_figure_column_names(fig,df=df))
    print(f"default_cols : {default_cols}")
    # User selects columns to display on hover, sorted alphabetically
    selected_cols = st.multiselect(
        "Select columns to display on hover",
        options=sorted(df.columns),  # Alphabetically sorted options
        #default_cols=list(default_cols)
    )

    print("Selected additional columns for hover:", selected_cols)

    if len(selected_cols)>=1:
        # Create hovertemplate with NaN handling
        for trace in fig.data:
            if trace['hovertemplate'] is not None:
                hovertemplate = trace['hovertemplate']
            else:
                hovertemplate = ""

            for col in selected_cols: 
                hovertemplate += f"<br>{col}: %{{customdata[{selected_cols.index(col)}]}}"
        
        

        # Prepare customdata for hover, replacing NaN with a placeholder like 'Not Available'
        customdata = df[selected_cols].map(lambda x: 'Not Available' if pd.isna(x) else x)

        # Update figure traces with customdata and hovertemplate
        fig.update_traces(
            customdata=customdata.values,  # Attach customdata with NaN handled
            hovertemplate=hovertemplate  # Set the custom hovertemplate
        )
    return fig


# Function to handle datetime and timedelta conversion
def convert_to_datetime_or_timedelta(df, col):
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    elif pd.api.types.is_timedelta64_dtype(df[col]):
        df[col] = pd.to_timedelta(df[col], errors='coerce')
    return df

# Function to handle resampling based on time granularity
def resample_data(df, x_col, y_col, color_col, time_granularity, agg):
    if color_col != "None":
        df_grouped = df.set_index(x_col).groupby(color_col)
    else:
        df_grouped = df.set_index(x_col)

    agg_mapping = {
        "None": None,
        "Mean": "mean",
        "Median": "median",
        "Sum": "sum",
        "Count": "count"
    }

    if agg_mapping[agg] is None:
        agg_result = df
    else:
        if pd.api.types.is_timedelta64_dtype(df[x_col]):
            # Use fixed-duration frequencies for timedelta
            if time_granularity == "Hour":
                freq = 'H'
            elif time_granularity == "Day":
                freq = 'D'
            elif time_granularity == "Month":
                freq = '30D'  # Approximate month duration
            elif time_granularity == "Year":
                freq = '365D'  # Approximate year duration
        else:
            # Use calendar-based frequencies for datetime
            if time_granularity == "Hour":
                freq = 'h'
            elif time_granularity == "Day":
                freq = 'D'
            elif time_granularity == "Month":
                freq = 'ME'
            elif time_granularity == "Year":
                freq = 'YE'

        def custom_agg(series):
            if agg == "Count":
                return series.count()
            elif len(series) < 2:
                return series.iloc[0] if not series.empty else np.nan
            else:
                return series.agg(agg_mapping[agg])

        if color_col != "None":
            agg_result = df_grouped.resample(freq).agg({y_col: custom_agg}).reset_index()
        else:
            agg_result = df_grouped.resample(freq).agg({y_col: custom_agg}).reset_index()

    return agg_result


def app():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    # Initialize data analyzer
    #if "working_df" in st.session_state:
    #    df = st.session_state["working_df"]
    #else:
    df_full = st.session_state["data"]
    analyzer = DataAnalyzer(df_full)
    # Apply pd.to_datetime with infer_datetime_format=True for each date column
    df_full[analyzer.date_cols] = df_full[analyzer.date_cols].apply(pd.to_datetime, errors='coerce') #, infer_datetime_format=True)


    with st.expander("Apply Inclusion and Anomaly Criteria",expanded=False):
        col_inc, col_ano = st.columns(2)
        # Checkbox for filtering out anomalies
        with col_ano:
            filter_out_anomaly = st.checkbox(
                label="Exclude Rows with Anomalies.",
                value=False,
                help="Check this box to exclude rows flagged as anomalies from the dataset."
            )

        # Checkbox for limiting rows based on inclusion criteria
        with col_inc:
            filter_on_inclusion = st.checkbox(
                label="Restrict to Inclusion Criteria.",
                value=False,
                help="Check this box to limit the displayed rows to those meeting the inclusion criteria."
            )


        # Base DataFrame
        df = df_full.copy()

        # Apply filters based on checkbox states
        if filter_on_inclusion:
            if "study_inclusion_all" not in df.columns:
                st.warning(f"The inclusion criteria are missing and need to be defined first.")
            else:
                df = df[df['study_inclusion_all'] == True]

        if filter_out_anomaly:
            if "clinical_anomalies_any" not in df.columns:
                st.warning(f"The anomaly criteria are missing and need to be defined first.")
            else:
                df = df[df['clinical_anomalies_any'] == False]

        st.dataframe(df.head(), use_container_width=True, hide_index=True)
        st.write(f"📊 **Filtered Data Overview:** {df.shape[0]:,} rows and {df.shape[1]:,} columns selected.")

    # Outliers Handler
    with st.expander("Outliers Handler"):
        df = dm.add_outlier_handling_ui(df, analyzer.numeric_cols)
    
    # Create tabs for different visualization aspects
    tabs = st.tabs(["Basic Visualizations", "..."])
    
    with tabs[0]:
        st.subheader("Basic Data Visualization")
        
        # Plot categories with requirements
        plot_categories = {
            "Distribution": {
                "plots": ["Histogram", "Box Plot", "Violin Plot"],
                "requirements": {"numeric": 1}
            },
            "Categorical": {
                "plots": ["Bar Plot", "Pie Chart"],
                "requirements": {"categorical": 1, "numeric": 1}
            },
            "Time Series": {
                "plots": ["Line Plot"],
                "requirements": {"date": 1,"numeric": 1}
            },
            "Relationships": {
                "plots": ["Scatter Plot", "Correlation Matrix"],
                "requirements": {"numeric": 2}
            },
            "Statistical": {
                "plots": [],
                "requirements": {"numeric": 2}
            }
        }
        
        # Filter available categories based on data
        available_categories = []
        for category, info in plot_categories.items():
            requirements_met = True
            for req_type, req_count in info["requirements"].items():
                if req_type == "numeric" and len(analyzer.numeric_cols) < req_count:
                    requirements_met = False
                elif req_type == "categorical" and len(analyzer.categorical_cols) < req_count:
                    requirements_met = False
                elif req_type == "date" and len(analyzer.date_cols) < req_count:
                    requirements_met = False
            if requirements_met:
                available_categories.append(category)
        
        if not available_categories:
            st.error("Your data doesn't contain enough suitable columns for visualization.")
            return
        
        # Sidebar configurations
        st.sidebar.subheader("Visualization Options")
        plot_category = st.sidebar.selectbox("Plot Category", available_categories)
        plot_type = st.sidebar.selectbox("Plot Type", plot_categories[plot_category]["plots"])
        
        # Get suitable columns for the selected plot type
        suitable_cols = analyzer.get_suitable_columns(plot_type)
        
        # Visual customization options
        with st.sidebar.expander("Visual Settings"):
            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["plotly", "plotly_dark", "plotly_white", "seaborn", "ggplot2"]
            )

            # Discrete color palettes
            color_sequences = {
                "Set1 - Default qualitative color palette": px.colors.qualitative.Set1,
                "Pastel1 - Soft pastel color palette": px.colors.qualitative.Pastel1,
                "Safe - Colorblind-friendly color palette": px.colors.qualitative.Safe,
                "Vivid - Vibrant color palette": px.colors.qualitative.Vivid,
                "Set2 - Alternative qualitative color palette": px.colors.qualitative.Set2,
                "Set3 - Another qualitative color palette": px.colors.qualitative.Set3,
                "Dark24 - Dark color palette with 24 distinct colors": px.colors.qualitative.Dark24,
                "Light24 - Light color palette with 24 distinct colors": px.colors.qualitative.Light24,
                "Alphabet - Wide range of colors for categorical data": px.colors.qualitative.Alphabet,
                "T10 - 10 distinct colors for smaller datasets": px.colors.qualitative.T10,
            }

            # Continuous color palettes
            continuous_color_sequences = {
                "Viridis - Perceptually uniform, colorblind-friendly": px.colors.sequential.Viridis,
                "Cividis - Perceptually uniform, colorblind-friendly": px.colors.sequential.Cividis,
                "Plasma - Smooth transition from dark to light": px.colors.sequential.Plasma,
                "Inferno - Smooth transition from dark to light": px.colors.sequential.Inferno,
                "RdBu - Diverging color map": px.colors.sequential.RdBu,
                "Blues - Sequential color map with shades of blue": px.colors.sequential.Blues,
                "Greens - Sequential color map with shades of green": px.colors.sequential.Greens,
                "Greys - Sequential color map with shades of grey": px.colors.sequential.Greys,
                "YlOrBr - Sequential color map with shades of yellow, orange, and brown": px.colors.sequential.YlOrBr,
                "YlOrRd - Sequential color map with shades of yellow, orange, and red": px.colors.sequential.YlOrRd,
            }

            # Determine available color schemes based on plot type
            if plot_type in ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Line Plot", "Bar Plot", "Pie Chart"]:
                color_scheme_type = "Discrete"
            else:
                color_scheme_type = "Continuous"

            # Unified color scheme selection
            if color_scheme_type == "Discrete":
                color_scheme = st.selectbox("Color Scheme", list(color_sequences.keys()))
                selected_color = color_sequences[color_scheme]  # Use discrete color scheme
            else:
                color_scheme = st.selectbox("Color Scheme", list(continuous_color_sequences.keys()))
                selected_color = continuous_color_sequences[color_scheme]  # Use continuous color scheme


            #color_scheme = st.selectbox("Discrete Color Scheme", list(color_sequences.keys()))
            #continuous_color_scheme = st.selectbox("Continuous Color Scheme", list(continuous_color_sequences.keys()), index=0)
        
        # Main visualization area
        try:
            fig = None  # Initialize figure variable
            
            if plot_type == "Histogram":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Select Variable", suitable_cols["x"])
                with col2:
                    color_col = st.selectbox("Group by (optional)", 
                                           ["None"] + suitable_cols.get("color", []))
                
                nbins = st.slider("Number of Bins", 5, 100, 20)
                
                fig = px.histogram(
                    df,
                    x=x_col,
                    color=None if color_col == "None" else color_col,
                    nbins=nbins,
                    #template=theme,
                    color_discrete_sequence=selected_color
                )
           
            elif plot_type == "Box Plot":
                col1, col2 = st.columns(2)

                with col1:
                    # Allow multiple selection for numeric variables
                    y_cols = st.multiselect("Select Numeric Variables", suitable_cols["y"], default=[suitable_cols["y"][0]])

                with col2:
                    # Group by option remains the same
                    x_col = st.selectbox("Group by", ["None"] + suitable_cols["x"])

                # Add orientation option
                orientation = st.radio("Orientation", ["Vertical", "Horizontal"], index=0)  # Default to Vertical

                # Color scheme selection
                selected_colors = color_sequences[color_scheme]

                # Ensure at least one y variable is selected
                if len(y_cols) > 0:
                    # Create subplots for each selected y variable
                    fig = make_subplots(rows=1, cols=len(y_cols), subplot_titles=y_cols)

                    # Loop over each selected numeric variable to add traces
                    for i, y_col in enumerate(y_cols):
                        if x_col == "None":
                            # If no grouping variable, add a single box for each y variable
                            fig.add_trace(
                                go.Box(
                                    y=df[y_col] if orientation == "Vertical" else None,
                                    x=None if orientation == "Vertical" else df[y_col],
                                    name=y_col,
                                    orientation='v' if orientation == "Vertical" else 'h',
                                    marker=dict(color=selected_colors[i % len(selected_colors)])  # Cycle through colors
                                ),
                                row=1,
                                col=i+1
                            )
                        else:
                            # If grouping by x_col, add a separate box for each category in x_col
                            unique_categories = df[x_col].unique()
                            for j, category in enumerate(unique_categories):
                                # Only show the legend for the first variable in y_cols to avoid duplication
                                show_legend = i == 0  # Show legend only for the first y_col
                                fig.add_trace(
                                    go.Box(
                                        y=df[df[x_col] == category][y_col] if orientation == "Vertical" else None,
                                        x=None if orientation == "Vertical" else df[df[x_col] == category][y_col],
                                        name=category,  # Single legend entry per category
                                        orientation='v' if orientation == "Vertical" else 'h',
                                        marker=dict(color=selected_colors[j % len(selected_colors)]),  # Cycle through colors
                                        showlegend=show_legend  # Show legend only for the first column
                                    ),
                                    row=1,
                                    col=i+1
                                )

                    # Update layout
                    fig.update_layout(title_text="Box Plots")#, template=theme)

                    # Apply x-axis and y-axis settings across all subplots
                    for i in range(1, len(y_cols) + 1):
                        # Update x-axes and y-axes individually
                        fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)
                        fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)

                    # Rotate x-axis labels if vertical and grouping variable is selected
                    if orientation == "Vertical" and x_col != "None":
                        fig.update_layout(xaxis_tickangle=-45)


            elif plot_type == "Violin Plot":
                col1, col2 = st.columns(2)

                with col1:
                    # Allow multiple selection for numeric variables
                    y_cols = st.multiselect("Select Numeric Variables", suitable_cols["y"], default=[suitable_cols["y"][0]])

                with col2:
                    # Group by option remains the same
                    x_col = st.selectbox("Group by", ["None"] + suitable_cols["x"])

                # Add orientation option
                orientation = st.radio("Orientation", ["Vertical", "Horizontal"], index=0)  # Default to Vertical

                # Color scheme selection
                selected_colors = color_sequences[color_scheme]

                # Ensure at least one y variable is selected
                if len(y_cols) > 0:
                    # Create subplots for each selected y variable
                    fig = make_subplots(rows=1, cols=len(y_cols), subplot_titles=y_cols)

                    # Loop over each selected numeric variable to add traces
                    for i, y_col in enumerate(y_cols):
                        if x_col == "None":
                            # If no grouping variable, add a single box for each y variable
                            fig.add_trace(
                                go.Violin(
                                    y=df[y_col] if orientation == "Vertical" else None,
                                    x=None if orientation == "Vertical" else df[y_col],
                                    name=y_col,
                                    orientation='v' if orientation == "Vertical" else 'h',
                                    marker=dict(color=selected_colors[i % len(selected_colors)])  # Cycle through colors
                                ),
                                row=1,
                                col=i+1
                            )
                        else:
                            # If grouping by x_col, add a separate box for each category in x_col
                            unique_categories = df[x_col].unique()
                            for j, category in enumerate(unique_categories):
                                # Only show the legend for the first variable in y_cols to avoid duplication
                                show_legend = i == 0  # Show legend only for the first y_col
                                fig.add_trace(
                                    go.Violin(
                                        y=df[df[x_col] == category][y_col] if orientation == "Vertical" else None,
                                        x=None if orientation == "Vertical" else df[df[x_col] == category][y_col],
                                        name=category,  # Single legend entry per category
                                        orientation='v' if orientation == "Vertical" else 'h',
                                        marker=dict(color=selected_colors[j % len(selected_colors)]),  # Cycle through colors
                                        showlegend=show_legend  # Show legend only for the first column
                                    ),
                                    row=1,
                                    col=i+1
                                )

                    # Update layout
                    fig.update_layout(title_text="Violin Plots")#, template=theme)

                    # Apply x-axis and y-axis settings across all subplots
                    for i in range(1, len(y_cols) + 1):
                        # Update x-axes and y-axes individually
                        fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)
                        fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)

                    # Rotate x-axis labels if vertical and grouping variable is selected
                    if orientation == "Vertical" and x_col != "None":
                        fig.update_layout(xaxis_tickangle=-45)
            
            
            elif plot_type == "Scatter Plot":
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X Variable", suitable_cols["x"])
                with col2:
                    y_col = st.selectbox("Y Variable", suitable_cols["y"])
                with col3:
                    color_col = st.selectbox("Color by (optional)", 
                                           ["None"] + suitable_cols["color"])
                
                show_trendline = st.checkbox("Show Trendline")
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=None if color_col == "None" else color_col,
                    trendline="ols" if show_trendline else None,
                    #template=theme,
                    color_discrete_sequence=selected_color
                )
            
            # elif plot_type == "Line Plot":
            #     col1, col2, col3, col4 = st.columns(4)
            #     with col1:
            #         x_col = st.selectbox("Time Variable", suitable_cols["x"])
            #     with col2:
            #         y_col = st.selectbox("Quantitative Variable", suitable_cols["y"])
            #     with col3:
            #         agg = st.selectbox("Aggregate", ["None", "Mean", "Median", "Sum", "Count", ], index=0)  # Default to None
            #     with col4:
            #         color_col = st.selectbox("Group by (optional)", ["None"] + suitable_cols["color"])


            #     # Select time granularity for aggregation
            #     time_granularity = st.selectbox("Time Granularity", ["Hour","Day", "Month", "Year"], index=1)

            #     # Mapping the aggregation options to real pandas aggregation functions
            #     agg_mapping = {
            #         "None": None,
            #         "Mean": "mean",
            #         "Median": "median",
            #         "Sum": "sum",
            #         "Count": "count"
            #     }

            #     # Convert x_col to datetime if necessary
            #     #df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            #     df = convert_to_datetime_or_timedelta(df, x_col)

            #     # Handle color_col by grouping before resampling
            #     if agg_mapping[agg] is None:
            #         agg_result = df
            #     else:
            #         # Group by both x_col and color_col (if color_col is selected)
            #         if color_col != "None":
            #             df_grouped = df.set_index(x_col)
            #         else:
            #             df_grouped = df
                    
            #         # st.write('test1')
            #         # dft = df.set_index(x_col)
            #         # # First group by color_col, then resample by x_col (date column)
            #         # df_res = dft.groupby(color_col).resample("YE").agg({y_col: agg_mapping[agg]}).reset_index()
            #         # st.dataframe(df_res)
            #         # st.write(f"SUM: {df_res[y_col].sum()}, M: {df_res[df_res['sex'] == 'male'][y_col].sum()}, F: {df_res[df_res['sex'] == 'female'][y_col].sum()}")
                    
            #         # Now resample based on x_col
            #         if time_granularity == "Hour":
            #             if color_col != "None":
            #                 agg_result = df_grouped.groupby(color_col).resample("H").agg({y_col: agg_mapping[agg]}).reset_index()
            #             else:
            #                 agg_result = df_grouped.resample("H", on=x_col).agg({y_col: agg_mapping[agg]}).reset_index()
            #         elif time_granularity == "Day":
            #             if color_col != "None":
            #                 agg_result = df_grouped.groupby(color_col).resample("D").agg({y_col: agg_mapping[agg]}).reset_index()
            #             else:
            #                 agg_result = df_grouped.resample("D", on=x_col).agg({y_col: agg_mapping[agg]}).reset_index()
            #         elif time_granularity == "Month":
            #             if color_col != "None":
            #                 agg_result = df_grouped.groupby(color_col).resample("ME").agg({y_col: agg_mapping[agg]}).reset_index()
            #             else:
            #                 agg_result = df_grouped.resample("ME", on=x_col).agg({y_col: agg_mapping[agg]}).reset_index()
            #         elif time_granularity == "Year":
            #             if color_col != "None":
            #                 agg_result = df_grouped.groupby(color_col).resample("YE").agg({y_col: agg_mapping[agg]}).reset_index()
            #             else:
            #                 agg_result = df_grouped.resample("YE", on=x_col).agg({y_col: agg_mapping[agg]}).reset_index()
            #         else:
            #             # If "None" is selected, no resampling
            #             agg_result = df_grouped

            #     # Add confidence interval as shaded area
            #     show_ci = st.checkbox("Add Confidence Interval", False)

            #     # Plotting
            #     if color_col != "None":
            #         if show_ci:
            #             fig = go.Figure()
            #         else:
            #             fig = px.line(
            #                 agg_result,  # Use the aggregated result
            #                 x=x_col,
            #                 y=y_col,  # Use the column name in y since `agg_result` is already processed
            #                 color=color_col,  # Use color_col for grouping
            #                 color_discrete_sequence=selected_color,  # Default color or use `selected_color`
            #             )     
            #     else:
            #         if show_ci:
            #             fig = go.Figure()
            #         else:
            #             fig = px.line(
            #                 agg_result,  # Use the aggregated result
            #                 x=x_col,
            #                 y=y_col,  # Use the column name in y since `agg_result` is already processed
            #                 color_discrete_sequence=selected_color,  # Default color
            #             )


            #     # Check if confidence interval should be shown
            #     if show_ci:
            #         # Confidence level input
            #         confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=99, value=95, step=1)
                    
            #         # Calculate the z-score based on the selected confidence level
            #         alpha = 1 - (confidence_level / 100)
            #         z_score = stats.norm.ppf(1 - alpha / 2)

            #         # Calculate standard error and confidence interval bounds for each group if color_col is specified
            #         if color_col != "None":
            #             agg_result["y_upper"] = agg_result.groupby(color_col)[y_col].transform(
            #                 lambda y: y + z_score * (np.std(y) / np.sqrt(len(y)))
            #             )
            #             agg_result["y_lower"] = agg_result.groupby(color_col)[y_col].transform(
            #                 lambda y: y - z_score * (np.std(y) / np.sqrt(len(y)))
            #             )
            #         else:
            #             # Single group, so calculate y_upper and y_lower for entire y_col
            #             std_err = np.std(agg_result[y_col]) / np.sqrt(len(agg_result[y_col]))
            #             agg_result["y_upper"] = agg_result[y_col] + z_score * std_err
            #             agg_result["y_lower"] = agg_result[y_col] - z_score * std_err

            #         # Create a mapping between groups and colors using selected_color
            #         unique_groups = agg_result[color_col].unique() if color_col != "None" else [None]
            #         color_map = {group: selected_color[i % len(selected_color)] for i, group in enumerate(unique_groups)}
                    
            #         # Plot with confidence interval as shaded area
            #         for i, (group_name, group_df) in enumerate(agg_result.groupby(color_col) if color_col != "None" else [(None, agg_result)]):
            #             # Retrieve the color for each group from selected_color
            #             line_color = color_map[group_name]
                        
            #             # Adjust color for fill with increased transparency
            #             fillcolor = adjust_color_to_rgba(line_color, transparency=0.2)

            #             #Main line trace
            #             fig.add_trace(go.Scatter(
            #                 x=group_df[x_col],
            #                 y=group_df[y_col],
            #                 mode="lines",
            #                 name=str(group_name) if group_name else y_col,
            #                 line=dict(color=line_color, width=2)
            #             ))

            #             # Confidence interval area (shaded)
            #             fig.add_trace(go.Scatter(
            #                 x=pd.concat([group_df[x_col], group_df[x_col][::-1]]),
            #                 y=pd.concat([group_df["y_upper"], group_df["y_lower"][::-1]]),
            #                 fill="toself",
            #                 fillcolor=fillcolor,  # Use lighter RGBA color for fill
            #                 line=dict(width=0),
            #                 name=f"{group_name} Confidence Interval" if group_name else "Confidence Interval"
            #             ))

            #         # Update layout if needed
            #         fig.update_layout(
            #             title="Line Plot with Confidence Interval",
            #             xaxis_title=x_col,
            #             yaxis_title=y_col,
            #             template="plotly_white"
            #         )

            if plot_type == "Line Plot":
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x_col = st.selectbox("Time Variable", suitable_cols["x"])
                with col2:
                    y_col = st.selectbox("Quantitative Variable", suitable_cols["y"])
                with col3:
                    agg = st.selectbox("Aggregate", ["None", "Mean", "Median", "Sum", "Count"], index=0)  # Default to None
                with col4:
                    color_col = st.selectbox("Group by (optional)", ["None"] + suitable_cols["color"])

                # Select time granularity for aggregation
                time_granularity = st.selectbox("Time Granularity", ["Hour", "Day", "Month", "Year"], index=1)

                # Convert x_col to datetime or timedelta if necessary
                df = convert_to_datetime_or_timedelta(df, x_col)

                # Define the aggregation mapping
                agg_mapping = {
                    "None": None,
                    "Mean": "mean",
                    "Median": "median",
                    "Sum": "sum",
                    "Count": "count"
                }

                # Handle color_col by grouping before resampling
                if agg_mapping[agg] is None:
                    agg_result = df
                else:
                    agg_result = resample_data(df, x_col, y_col, color_col, time_granularity, agg)

                # Add confidence interval as shaded area
                show_ci = st.checkbox("Add Confidence Interval", False)

                # Plotting
                if color_col != "None":
                    if show_ci:
                        fig = go.Figure()
                    else:
                        fig = px.line(
                            agg_result,  # Use the aggregated result
                            x=x_col,
                            y=y_col,  # Use the column name in y since `agg_result` is already processed
                            color=color_col,  # Use color_col for grouping
                            color_discrete_sequence=selected_color,  # Default color or use `selected_color`
                        )
                else:
                    if show_ci:
                        fig = go.Figure()
                    else:
                        fig = px.line(
                            agg_result,  # Use the aggregated result
                            x=x_col,
                            y=y_col,  # Use the column name in y since `agg_result` is already processed
                            color_discrete_sequence=selected_color,  # Default color
                        )

                # Check if confidence interval should be shown
                if show_ci:
                    # Confidence level input
                    confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=99, value=95, step=1)

                    # Calculate the z-score based on the selected confidence level
                    alpha = 1 - (confidence_level / 100)
                    z_score = stats.norm.ppf(1 - alpha / 2)

                    # Calculate standard error and confidence interval bounds for each group if color_col is specified
                    if color_col != "None":
                        agg_result["y_upper"] = agg_result.groupby(color_col)[y_col].transform(
                            lambda y: y + z_score * (np.std(y) / np.sqrt(len(y)))
                        )
                        agg_result["y_lower"] = agg_result.groupby(color_col)[y_col].transform(
                            lambda y: y - z_score * (np.std(y) / np.sqrt(len(y)))
                        )
                    else:
                        # Single group, so calculate y_upper and y_lower for entire y_col
                        std_err = np.std(agg_result[y_col]) / np.sqrt(len(agg_result[y_col]))
                        agg_result["y_upper"] = agg_result[y_col] + z_score * std_err
                        agg_result["y_lower"] = agg_result[y_col] - z_score * std_err

                    # Create a mapping between groups and colors using selected_color
                    unique_groups = agg_result[color_col].unique() if color_col != "None" else [None]
                    color_map = {group: selected_color[i % len(selected_color)] for i, group in enumerate(unique_groups)}

                    # Plot with confidence interval as shaded area
                    for i, (group_name, group_df) in enumerate(agg_result.groupby(color_col) if color_col != "None" else [(None, agg_result)]):
                        # Retrieve the color for each group from selected_color
                        line_color = color_map[group_name]

                        # Adjust color for fill with increased transparency
                        fillcolor = adjust_color_to_rgba(line_color, transparency=0.2)

                        # Main line trace
                        fig.add_trace(go.Scatter(
                            x=group_df[x_col],
                            y=group_df[y_col],
                            mode="lines",
                            name=str(group_name) if group_name else y_col,
                            line=dict(color=line_color, width=2)
                        ))

                        # Confidence interval area (shaded)
                        fig.add_trace(go.Scatter(
                            x=pd.concat([group_df[x_col], group_df[x_col][::-1]]),
                            y=pd.concat([group_df["y_upper"], group_df["y_lower"][::-1]]),
                            fill="toself",
                            fillcolor=fillcolor,  # Use lighter RGBA color for fill
                            line=dict(width=0),
                            name=f"{group_name} Confidence Interval" if group_name else "Confidence Interval"
                        ))

                    # Update layout if needed
                    fig.update_layout(
                        title="Line Plot with Confidence Interval",
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        template="plotly_white"
                    )

            elif plot_type == "Bar Plot":
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("Category", suitable_cols["x"])
                with col2:
                    y_col = st.selectbox("Value", suitable_cols["y"])
                with col3:
                    color_col = st.selectbox("Group by (optional)",
                                           ["None"] + suitable_cols["color"])
                
                orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
                
                fig = px.bar(
                    df,
                    x=x_col if orientation == "Vertical" else y_col,
                    y=y_col if orientation == "Vertical" else x_col,
                    color=None if color_col == "None" else color_col,
                    #template=theme,
                    color_discrete_sequence=selected_color,
                    orientation="v" if orientation == "Vertical" else "h"
                )
            
            elif plot_type == "Pie Chart":
                col1, col2 = st.columns(2)
                with col1:
                    names_col = st.selectbox("Categories", suitable_cols["names"])
                with col2:
                    values_col = st.selectbox("Values", suitable_cols["values"])
                
                fig = px.pie(
                    df,
                    names=names_col,
                    values=values_col,
                    #template=theme,
                    color_discrete_sequence=selected_color
                )
            
            elif plot_type == "Correlation Matrix":
                if len(suitable_cols["numeric"]) < 2:
                    st.error("Need at least 2 numeric columns for correlation matrix!")
                    return

                # Checkbox to include binary and numeric-like categorical columns in selection
                include_binary_and_categorical = st.checkbox(
                    "Include Binary and Numeric-like Categorical Columns", value=False
                )
                
                # Determine columns for selection based on the checkbox
                if include_binary_and_categorical:
                    # Combine numeric, binary, and numeric categorical columns
                    all_numeric_options = (suitable_cols["numeric"] + suitable_cols["categorical"])
                    corr_methods = ["kendall", "spearman"]  # Only non-parametric options
                else:
                    all_numeric_options = suitable_cols["numeric"]
                    corr_methods = ["pearson", "kendall", "spearman"]  # Full options

                # Multiselect for selecting columns
                selected_cols = st.multiselect(
                    "Select Columns for Correlation",
                    all_numeric_options,
                    default=all_numeric_options[:4]
                )

                if len(selected_cols) < 2:
                    st.warning("Please select at least 2 columns")
                    return

                # Allow the user to select the correlation method, restricted based on checkbox
                corr_method = st.selectbox(
                    "Select Correlation Method",
                    options=corr_methods,
                    index=0  # Default to the first available method
                )

                # Compute the correlation matrix with the selected method
                corr_matrix = df[selected_cols].corr(method=corr_method)

                fig = px.imshow(
                    corr_matrix,
                    #template=theme,
                    #aspect="auto",  # This helps the matrix to fit the display
                    color_continuous_scale=selected_color,
                    labels=dict(color="Correlation Coefficient"),
                    title=f"Correlation Matrix ({corr_method.capitalize()})"
                )
            
            # Only proceed with customization and display if a figure was created
            if fig is not None:
                # Plot customization options
                with st.expander("Customize Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        title = st.text_input("Plot Title", "")
                        title_size = st.slider("Title Font Size", 10, 50, 20)
                        show_grid = st.checkbox("Show Grid", True)
                    with col2:
                        height = st.slider("Plot Height", 400, 2000, 700)
                        width = st.slider("Plot Width", 400, 2000, 1000)
                    
                    fig.update_layout(
                        title=dict(text=title, font=dict(size=title_size), x=0.5),
                        height=height,
                        width=width
                    )

                     # Check if we have multiple subplots with y_cols, or a single y_col
                    if 'y_cols' in locals() and len(y_cols) > 1:  # Multiple y_cols, indicating subplots
                        for i in range(1, len(y_cols) + 1):
                            fig.update_xaxes(showgrid=show_grid, gridcolor='lightgray', row=1, col=i)
                            fig.update_yaxes(showgrid=show_grid, gridcolor='lightgray', row=1, col=i)
                    else:  # Single y_col or no y_cols defined
                        fig.update_xaxes(showgrid=show_grid, gridcolor='lightgray')
                        fig.update_yaxes(showgrid=show_grid, gridcolor='lightgray')
                

                # apply theme update
                fig.update_layout(template=theme)

                # Apply theme-specific layout settings
                if theme == "plotly":
                    fig.update_layout(
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        font=dict(color="black", size=14)
                    )
                elif theme == "plotly_dark":
                    fig.update_layout(
                        paper_bgcolor="#1e1e1e",  # Dark gray matching plotly_dark
                        plot_bgcolor="#1e1e1e",
                        font=dict(color="white", size=14)
                    )
                elif theme == "plotly_white":
                    fig.update_layout(
                        paper_bgcolor="#f5f5f5",  # Slightly off-white for a softer effect
                        plot_bgcolor="#f5f5f5",
                        font=dict(color="black", size=14)
                    )
                elif theme == "seaborn":
                    fig.update_layout(
                        paper_bgcolor="#eaf2f8",  # Very light blue-gray to match seaborn styles
                        plot_bgcolor="#eaf2f8",
                        font=dict(color="#333333", size=14)
                    )
                elif theme == "ggplot2":
                    fig.update_layout(
                        paper_bgcolor="#ebebeb",  # Light gray typical in ggplot2
                        plot_bgcolor="#ebebeb",
                        font=dict(color="black", size=14)
                    )

                fig = add_custom_hovertemplate(fig,df=df)
                

                st.plotly_chart(fig) #, use_container_width=True)


                
                # Statistical insights
                with st.expander("Statistical Insights"):
                    if plot_type == "Histogram":
                        # Create a DataFrame to hold summary statistics
                        summary_statistics = []

                        # Check if there's a categorical variable
                        if color_col != 'None':
                            has_categorical = df[color_col].nunique() > 1
                            print(f"has_categorical {has_categorical}!!!")
                        else:
                            has_categorical = False
                            print(f"has_categorical {has_categorical}!!!")

                        if df[x_col].dtype in ['int64', 'float64']:
                            # Global statistics for y_col
                            global_stats = {
                                'Category': 'All',
                                'Variable': x_col,
                                'Count': df[x_col].count(),
                                'Mean': df[x_col].mean(),
                                'Std Dev': df[x_col].std(),
                                'Min': df[x_col].min(),
                                '25th Percentile': df[x_col].quantile(0.25),
                                'Median': df[x_col].median(),
                                '75th Percentile': df[x_col].quantile(0.75),
                                'Max': df[x_col].max(),
                                'Skewness': df[x_col].skew(),
                                'Kurtosis': df[x_col].kurtosis(),
                                }
                            summary_statistics.append(global_stats)
                        
                            if has_categorical:
                                for category in df[color_col].unique():
                                    category_data = df[df[color_col] == category][x_col]
                                    # Calculate summary statistics
                                    sum_stats = {
                                        'Category': category,
                                        'Variable': x_col,
                                        'Count': category_data.count(),
                                        'Mean': category_data.mean(),
                                        'Std Dev': category_data.std(),
                                        'Min': category_data.min(),
                                        '25th Percentile': category_data.quantile(0.25),
                                        'Median': category_data.median(),
                                        '75th Percentile': category_data.quantile(0.75),
                                        'Max': category_data.max(),
                                        'Skewness': category_data.skew(),
                                        'Kurtosis': category_data.kurtosis(),
                                        }
                                        
                                    summary_statistics.append(sum_stats)
                        else:
                            st.error(f"{x_col} is None or not in DataFrame columns.")                  
                                    
                        # Create a summary DataFrame
                        summary_df = pd.DataFrame(summary_statistics)

                        # Pivot the DataFrame to create a cross table
                        pivot_df = summary_df.pivot_table(
                            index='Variable',
                            columns='Category',
                            values=['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis'],
                            aggfunc='first'
                        )

                        if color_col != 'None':
                            summary_df = pivot_df[['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis']]
                        else:
                            summary_df = summary_df.drop(columns='Category').set_index('Variable')

                        summary_df = (
                            summary_df.style
                            .format("{:.2f}")  # Format all numeric values to 2 decimal places
                            .set_table_attributes('style="font-size: 14px; border-collapse: collapse; width: 100%;"')  # Custom table styles
                            .set_caption("Summary Statistics by Variable and Category")  # Add a caption
                            #.highlight_max(color='lightgreen', axis=0)  # Highlight max values for each category
                            #.highlight_min(color='salmon', axis=0)  # Highlight min values for each category
                            .apply(lambda x: ['background: lightgrey' if i % 2 == 0 else '' for i in range(len(x))], axis=0)  # Alternate row colors
                        )

                        # Display the pivot table
                        st.write(f"Summary Statistics for {x_col}:")
                        st.dataframe(summary_df, use_container_width=True)
                        
                    elif plot_type in ["Box Plot", "Violin Plot"]:
                        # Create a DataFrame to hold summary statistics
                        summary_statistics = []

                        # Check if there's a categorical variable
                        if x_col != 'None':
                            has_categorical = df[x_col].nunique() > 1
                            print(f"has_categorical {has_categorical}!!!")
                        else:
                            has_categorical = False
                            print(f"has_categorical {has_categorical}!!!")

                        # Calculate global statistics if there's no categorical variable or include it in the summary
                        for y_col in y_cols:
                            if df[y_col].dtype in ['int64', 'float64']:
                                # Global statistics for y_col
                                global_stats = {
                                    'Category': 'All',
                                    'Variable': y_col,
                                    'Count': df[y_col].count(),
                                    'Mean': df[y_col].mean(),
                                    'Std Dev': df[y_col].std(),
                                    'Min': df[y_col].min(),
                                    '25th Percentile': df[y_col].quantile(0.25),
                                    'Median': df[y_col].median(),
                                    '75th Percentile': df[y_col].quantile(0.75),
                                    'Max': df[y_col].max(),
                                    'Skewness': df[y_col].skew(),
                                    'Kurtosis': df[y_col].kurtosis(),
                                }
                                summary_statistics.append(global_stats)
                            else:
                                st.error(f"{y_col} is None or not in DataFrame columns.")
                                
                        if has_categorical:
                            # For each y_col, calculate statistics based on x_col categories
                            for y_col in y_cols:
                                if df[y_col].dtype in ['int64', 'float64']:
                                    for category in df[x_col].unique():
                                        category_data = df[df[x_col] == category][y_col]

                                        # Calculate summary statistics
                                        sum_stats = {
                                            'Category': category,
                                            'Variable': y_col,
                                            'Count': category_data.count(),
                                            'Mean': category_data.mean(),
                                            'Std Dev': category_data.std(),
                                            'Min': category_data.min(),
                                            '25th Percentile': category_data.quantile(0.25),
                                            'Median': category_data.median(),
                                            '75th Percentile': category_data.quantile(0.75),
                                            'Max': category_data.max(),
                                            'Skewness': category_data.skew(),
                                            'Kurtosis': category_data.kurtosis(),
                                        }
                                        summary_statistics.append(sum_stats)

                        # Create a summary DataFrame
                        summary_df = pd.DataFrame(summary_statistics)

                        # Pivot the DataFrame to create a cross table
                        pivot_df = summary_df.pivot_table(
                            index='Variable',
                            columns='Category',
                            values=['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis'],
                            aggfunc='first'
                        )

                        if x_col != 'None':
                            summary_df = pivot_df[['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis']]
                        else:
                            summary_df = summary_df.drop(columns='Category').set_index('Variable')

                        summary_df = (
                            summary_df.style
                            .format("{:.2f}")  # Format all numeric values to 2 decimal places
                            .set_table_attributes('style="font-size: 14px; border-collapse: collapse; width: 100%;"')  # Custom table styles
                            .set_caption("Summary Statistics by Variable and Category")  # Add a caption
                            #.highlight_max(color='lightgreen', axis=0)  # Highlight max values for each category
                            #.highlight_min(color='salmon', axis=0)  # Highlight min values for each category
                            .apply(lambda x: ['background: lightgrey' if i % 2 == 0 else '' for i in range(len(x))], axis=0)  # Alternate row colors
                        )

                        # Display the pivot table
                        st.write("Summary Statistics (Cross-Tabulated):")
                        st.dataframe(summary_df, use_container_width=True)
                    
                    elif plot_type == "Scatter Plot":
                        if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
                            correlation = df[x_col].corr(df[y_col])
                            st.write(f"Correlation coefficient: {correlation:.2f}")
                            
                            slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
                            st.write(f"R-squared: {r_value**2:.2f}")
                            st.write(f"P-value: {p_value:.4f}")


        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try different variables or plot settings.")

                # V1 # Check if confidence interval should be shown
                # if show_ci:
                #     # Confidence level input
                #     confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=99, value=95, step=1)
                    
                #     # Calculate the z-score based on the selected confidence level
                #     alpha = 1 - (confidence_level / 100)
                #     z_score = stats.norm.ppf(1 - alpha / 2)  # z_score should increase with higher confidence

                #     # Calculate standard error and confidence interval bounds for each group if color_col is specified
                #     if color_col != "None":
                #         agg_result["y_upper"] = agg_result.groupby(color_col)[y_col].transform(
                #             lambda y: y + z_score * (np.std(y) / np.sqrt(len(y)))
                #         )
                #         agg_result["y_lower"] = agg_result.groupby(color_col)[y_col].transform(
                #             lambda y: y - z_score * (np.std(y) / np.sqrt(len(y)))
                #         )
                #     else:
                #         # Single group, so calculate y_upper and y_lower for entire y_col
                #         std_err = np.std(agg_result[y_col]) / np.sqrt(len(agg_result[y_col]))
                #         agg_result["y_upper"] = agg_result[y_col] + z_score * std_err
                #         agg_result["y_lower"] = agg_result[y_col] - z_score * std_err

                #     # Plot with confidence interval as shaded area
                #     for group_name, group_df in agg_result.groupby(color_col) if color_col != "None" else [(None, agg_result)]:
                #         # Upper bound trace (no line width for boundaries)
                #         fig.add_trace(go.Scatter(
                #             x=group_df[x_col],
                #             y=group_df["y_upper"],
                #             mode="lines",
                #             line=dict(width=0),
                #             showlegend=False,
                #             name=f"{group_name} Upper Bound" if group_name else "Upper Bound"
                #         ))

                #         # Lower bound trace with shading between upper and lower bounds
                #         fig.add_trace(go.Scatter(
                #             x=group_df[x_col],
                #             y=group_df["y_lower"],
                #             mode="lines",
                #             fill="tonexty",
                #             fillcolor="rgba(173, 216, 230, 0.2)",  # Semi-transparent fill color for shading
                #             line=dict(width=0),
                #             showlegend=False,
                #             name=f"{group_name} Lower Bound" if group_name else "Lower Bound"
                #         ))