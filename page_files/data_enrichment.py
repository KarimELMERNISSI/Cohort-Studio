import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from io import BytesIO
from multipage import load_dataframe, get_file_hash
import enrich.external_data as eed
from page_files.home import DataAnalyzer
from fuzzywuzzy import fuzz
import os
import re
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.manifold import TSNE
import umap
import enrich.data_imputation as edi
import prince
import logging


# class DataTransformationEngine: #in progress
#     def __init__(self, dataframe: pd.DataFrame):
#         """
#         Initialize the transformation engine with input dataframe
        
#         :param dataframe: Input pandas DataFrame
#         """
#         self.original_df = dataframe
#         self.transformed_df = None
#         self.transformation_history = []


#     def _validate_input(self, columns: List[str], transformation_type: str) -> bool:
#         """
#         Validate input columns and transformation type
        
#         :param columns: List of column names
#         :param transformation_type: Type of transformation
#         :return: Boolean indicating input validity
#         """
#         if not columns:
#             st.warning("No columns selected for transformation.")
#             return False
        
#         # Add specific validation logic based on transformation type
#         if transformation_type == "Dimensional Reduction-Based":
#             if len(columns) < 2:
#                 st.warning("Dimensional reduction requires at least 2 columns.")
#                 return False
        
#         return True


#     def statistical_transformations(self, 
#                                     columns: List[str], 
#                                     method: str) -> Dict[str, np.ndarray]:
#         """
#         Apply statistical transformations
        
#         :param columns: Columns to transform
#         :param method: Transformation method
#         :return: Dictionary of transformed columns
#         """
#         STATISTICAL_METHODS = {
#             "Mean": np.mean,
#             "Median": np.median,
#             "Summation": np.sum,
#             "Minimum Value": np.min,
#             "Maximum Value": np.max,
#             "Standard Deviation": np.std
#         }
        
#         transformed_columns = {}
        
#         for col in columns:
#             try:
#                 transform_func = STATISTICAL_METHODS.get(method)
#                 if transform_func:
#                     result = transform_func(self.original_df[col])
#                     new_col_name = f"{col}_{method.replace(' ', '_')}"
#                     transformed_columns[new_col_name] = [result] * len(self.original_df)
                    
#                     # Log transformation details
#             except Exception as e:
#                 st.error(f"Error transforming {col}: {e}")
        
#         return transformed_columns


#     def scaling_transformations(self, 
#                                 columns: List[str], 
#                                 method: str) -> Dict[str, np.ndarray]:
#         """
#         Apply scaling transformations
        
#         :param columns: Columns to transform
#         :param method: Scaling method
#         :return: Dictionary of transformed columns
#         """
#         from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
#         SCALING_METHODS = {
#             "Min-Max Normalization": lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten(),
#             "Z-Score Standardization": lambda x: (x - x.mean()) / x.std(),
#             "Natural Logarithm Transformation": lambda x: np.log(x + 1)
#         }
        
#         transformed_columns = {}
        
#         for col in columns:
#             try:
#                 transform_func = SCALING_METHODS.get(method)
#                 if transform_func:
#                     result = transform_func(self.original_df[col])
#                     new_col_name = f"{col}_{method.replace(' ', '_')}"
#                     transformed_columns[new_col_name] = result
#             except Exception as e:
#                 st.error(f"Error scaling {col}: {e}")

#         return transformed_columns


#     def _plot_elbow_curve(self, model, n_components, method='PCA'):
#         """
#         Plot elbow curve with red lines for selected components
#         """
#         # Calculate cumulative variance
#         cumulative_variance = np.cumsum(model.explained_variance_ratio_)

#         # Create figure
#         fig, ax = plt.subplots(figsize=(10, 6))

#         # Plot the full cumulative explained variance curve
#         ax.plot(range(1, len(cumulative_variance) + 1), 
#                 cumulative_variance, 
#                 marker='o', 
#                 linestyle='-', 
#                 color='b')

#         # Set the limits for the y-axis and x-axis
#         ax.set_ylim(0, max(cumulative_variance) * 1.1)
#         ax.set_xlim(1, len(cumulative_variance))

#         # Add vertical and horizontal lines at the selected n_components
#         ax.vlines(n_components, 0, cumulative_variance[n_components - 1], 
#                   color='red', linestyle='--', linewidth=2)
#         ax.hlines(cumulative_variance[n_components - 1], 0, n_components, 
#                   color='red', linestyle='--', linewidth=2)

#         # Display the value at n_components
#         ax.text(n_components, cumulative_variance[n_components - 1], 
#                 f"{cumulative_variance[n_components - 1]:.2f}", 
#                 ha='center', va='bottom', fontsize=12, color='red')

#         # Set labels and title
#         ax.set_title(f"{method} - Explained Variance by Components")
#         ax.set_xlabel("Number of Components")
#         ax.set_ylabel("Cumulative Explained Variance")
#         ax.grid(True)

#         st.pyplot(fig)


#     def dimensional_reduction(self, columns, method='PCA', n_components=3, show_elbow_curve=True):
#         """
#         Perform dimensional reduction with elbow curve visualization
#         """
#         if not self._validate_dimensional_reduction_input(columns):
#             return None, None

#         # Prepare data
#         scaler = StandardScaler()
#         scaled_data = scaler.fit_transform(self.original_df[columns])

#         # Dimensional Reduction Methods
#         try:
#             if method == 'PCA':
#                 reducer = PCA(n_components=n_components)
#                 transformed_data = reducer.fit_transform(scaled_data)
                
#                 # Elbow Curve
#                 if show_elbow_curve:
#                     full_pca = PCA()
#                     full_pca.fit(scaled_data)
#                     self._plot_elbow_curve(full_pca, n_components, 'PCA')

#                 # Create column names
#                 col_names = [f'PCA_Component_{i+1}' for i in range(n_components)]

#             elif method == 'Factor Analysis':
#                 reducer = FactorAnalysis(n_components=n_components, random_state=42)
#                 transformed_data = reducer.fit_transform(scaled_data)
#                 col_names = [f'Factor_{i+1}' for i in range(n_components)]

#             elif method == 't-SNE':
#                 reducer = TSNE(n_components=n_components, random_state=42)
#                 transformed_data = reducer.fit_transform(scaled_data)
#                 col_names = [f'tSNE_Component_{i+1}' for i in range(n_components)]

#             elif method == 'UMAP':
#                 reducer = umap.UMAP(n_components=n_components, random_state=42)
#                 transformed_data = reducer.fit_transform(scaled_data)
#                 col_names = [f'UMAP_Component_{i+1}' for i in range(n_components)]

#             # Create result DataFrame
#             result_df = pd.DataFrame(
#                 transformed_data, 
#                 columns=col_names, 
#                 index=self.original_df.index
#             )

#             return result_df, reducer

#         except Exception as e:
#             st.error(f"Dimensional Reduction Error: {e}")
#             return None, None
    

#     def categorical_encoding(self, columns, method='One-Hot Encoding'):
#         """
#         Encode categorical variables with improved handling
#         """
#         result_df = pd.DataFrame()

#         try:
#             for col in columns:
#                 if method == 'One-Hot Encoding':
#                     # Use pandas get_dummies for more reliable one-hot encoding
#                     encoded = pd.get_dummies(
#                         self.original_df[col], 
#                         prefix=col, 
#                         prefix_sep='_'
#                     )
#                     result_df = pd.concat([result_df, encoded], axis=1)

#                 elif method == 'Label Encoding':
#                     encoder = LabelEncoder()
#                     result_df[f'{col}_LabelEncoded'] = encoder.fit_transform(
#                         self.original_df[col]
#                     )

#                 elif method == 'Ordinal Encoding':
#                     encoder = OrdinalEncoder()
#                     result_df[f'{col}_OrdinalEncoded'] = encoder.fit_transform(
#                         self.original_df[[col]]
#                     )

#             return result_df

#         except Exception as e:
#             st.error(f"Categorical Encoding Error: {e}")
#             return None
        

#     def apply_transformation(self, 
#                              transformation_type: str, 
#                              method: str, 
#                              columns: List[str],
#                              **kwargs) -> Union[pd.DataFrame, None]:
#         """
#         Centralized method to apply transformations
        
#         :param transformation_type: Type of transformation
#         :param method: Specific method within transformation type
#         :param columns: Columns to transform
#         :param kwargs: Additional parameters
#         :return: Transformed DataFrame
#         """
#         # Validate input
#         if not self._validate_input(columns, transformation_type):
#             return None
        
#         # Select transformation based on type
#         if transformation_type == "Statistical-Based":
#             transformed_data = self.statistical_transformations(columns, method)
        
#         elif transformation_type == "Scaling-Based":
#             transformed_data = self.scaling_transformations(columns, method)
        
#         elif transformation_type == "Dimensional Reduction-Based":
#             n_components = kwargs.get('n_components', 3)
#             transformed_data, reducer = self.dimensional_reduction(columns, method, n_components)
            
#             # Optional: Visualize variance if available
#             if reducer is not None:
#                 self._plot_elbow_curve(model=reducer, n_components=n_components, method=method)
        
#         elif transformation_type == "Other":
#             transformed_data = self.categorical_encoding(columns, method)
        
#         else:
#             st.warning("Unsupported transformation type")
#             return None
        
#         # Create DataFrame from transformed columns
#         if transformed_data:
#             result_df = pd.DataFrame(transformed_data, index=self.original_df.index)
#             self.transformed_df = pd.concat([self.original_df, result_df], axis=1)
            
#             # Record transformation history
#             self.transformation_history.append({
#                 'type': transformation_type,
#                 'method': method,
#                 'columns': columns
#             })
            
#             return self.transformed_df
        
#         return None




def handle_main_data_upload() -> Optional[pd.DataFrame]:
    """Handle the upload or path input for the main dataset."""
    if 'data' in st.session_state and st.session_state.data is not None:
        st.success("✅ Main dataset loaded from session")
        return st.session_state.data

    upload_method = st.radio("Choose upload method for Main Dataset:", ["Upload File", "Enter File Path"], key='main_data_method')
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Drag and drop your main dataset here",
            type=['csv', 'xlsx'],
            key='main_data_uploader'
        )
        if uploaded_file is not None:
            try:
                st.write(f"file_name: {uploaded_file.name}")
                file_path = uploaded_file.name
                df = load_dataframe(uploaded_file)
                current_file_hash = get_file_hash(uploaded_file)
                
                if 'last_main_file_hash' not in st.session_state or \
                   current_file_hash != st.session_state['last_main_file_hash']:
                    st.session_state['last_main_file_hash'] = current_file_hash
                    st.success("✅ Main dataset successfully loaded from upload.")
                return df
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None
    else:
        file_path = st.text_input("Enter file path for the main dataset:", key='main_data_path')
        
        if file_path:
            try:
                df = load_dataframe(file_path)
                st.success("✅ Main dataset successfully loaded from provided path.")
                return df
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None
    return None

def handle_enrichment_data_upload() -> Optional[Union[pd.DataFrame, str]]:
    """Handle the upload or path input for the enrichment dataset."""
    upload_method = st.radio("Choose upload method for Enrichment Dataset:", ["Upload File", "Enter File Path"], key='enrichment_data_method')
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Drag and drop your enrichment dataset here",
            type=['csv', 'xlsx'],
            key='enrichment_data_uploader'
        )
        if uploaded_file is not None:
            try:
                st.write(f"file_name: {uploaded_file.name}")
                file_path = uploaded_file.name
                df = load_dataframe(uploaded_file)
                current_file_hash = get_file_hash(uploaded_file)
                
                if 'last_enrichment_file_hash' not in st.session_state or \
                   current_file_hash != st.session_state['last_enrichment_file_hash']:
                    st.session_state['last_enrichment_file_hash'] = current_file_hash
                    st.success("✅ Enrichment dataset successfully loaded from upload.")
                return df, file_path
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None, None
    else:
        # Get folder path and file name from user input
        folder_path = st.text_input("Enter folder path for the enrichment dataset:", key='enrichment_folder_path')
        file_name = st.text_input("Enter file name for the enrichment dataset:", key='enrichment_data_name')

        # Dynamically set file_path based on folder_path and file_name
        file_path = os.path.join(folder_path, file_name) if folder_path and file_name else ""

        # Display file path input field with computed value
        file_path_input = st.text_input("Enter file path for the enrichment dataset:", value=file_path, key='enrichment_data_path')
        if file_path_input:
            try:
                df = load_dataframe(file_path_input)
                st.success("✅ Enrichment dataset successfully loaded from provided path.")
                return df, file_name
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None, None
    return None, None


def configure_enrichment(main_df: pd.DataFrame, enrichment_df: pd.DataFrame, enrichment_file_path: str) -> Optional[pd.DataFrame]:
    """Configure and perform the data enrichment."""
    st.subheader("Enrichment Configuration")
    
    # Configuration form
    enrichment_name = st.text_input(
        "Enrichment Name", 
        "New Enrichment",
        help="Give a name to this enrichment configuration"
    )

    input_file_folder = st.text_input(
        "Input File Folder", 
        "./data/enrichment/",
        help="Give the path of the folder containing the data file"
    )

    input_file = st.text_input(
        "Input File Name", 
        enrichment_file_path,
        help="Give the name of the data file"
    )
    
    add_choice = st.radio(
        "What do you want to add?",
        ["Columns", "Rows"],
        help="Choose how to augment the datasets",
        horizontal=True
    )
    if add_choice == 'Columns':
        strategy = st.selectbox(
            "Join Strategy",
            ["left", "right", "outer", "inner", "cross"],
            help="Select how to join the datasets"
        )
    else:
        strategy = "rows"
    
    # Display strategy explanation
    strategy_explanations = {
        "left": ":grey_question: Keep all rows from main dataset, only matching rows from enrichment dataset",
        "right": ":grey_question: Keep all rows from enrichment dataset, only matching rows from main dataset",
        "outer": ":grey_question: Keep all rows from both datasets",
        "inner": ":grey_question: Keep only rows that match in both datasets",
        "cross": ":grey_question: Create all possible combinations of rows",
        "rows": ":grey_question: Stack the rows from both datasets"
    }
    st.info(strategy_explanations[strategy])

    # Select conflict resolution strategy
    conflict_resolution = st.selectbox(
        "Conflict Resolution",
        ['keep', 'replace', 'ignore'],
        index=0,
        help="Choose how to handle columns with the same name in the datasets: \
            'keep' to append suffixes to duplicates, 'replace' to overwrite with enrichment data, \
            'ignore' to keep original data columns."
    )
    
    row_id_approach = st.radio(
        "Row Identifier Approach",
        ["Common Row Identifier", "Distinct Row Identifiers"],
        help="Choose how to match rows between datasets",
        horizontal=True
    )
    
    enriched_df = None
    
    if row_id_approach == "Common Row Identifier":
        common_columns = list(set(main_df.columns) & set(enrichment_df.columns))
        if not common_columns:
            st.warning("No common columns found between datasets!")
            return None
        
        identifier = st.multiselect(
            "Select Common Row Identifier",
            options=common_columns,
            help="Choose the column that exists in both datasets"
        )
        
        if st.button("Perform Enrichment"):
            try:
                # Ensure right_identifier is a set for compatibility with the subtraction operation
                identifier_set = {identifier} if isinstance(identifier, str) else set(identifier)

                # Compute additional columns
                additional_columns = list(set(enrichment_df.columns) - identifier_set)

                enriched_df = eed.add_data(
                    df=main_df, additional_df=enrichment_df, 
                    left_id_names=identifier, right_id_names=identifier, 
                    additional_cols=additional_columns,
                    strategy=strategy, 
                    conflict_resolution=conflict_resolution)
                     
                save_enrichment_config(enrichment_name=enrichment_name, 
                                       strategy=strategy,
                                       input_file_folder=input_file_folder,
                                       input_file=input_file,
                                       identifier=identifier,
                                       additional_columns=additional_columns,
                                       conflict_resolution=conflict_resolution
                                       )
                st.success("✅ Enrichment completed successfully!")
            except Exception as e:
                st.error(f"Error performing enrichment: {str(e)}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            left_identifier = st.multiselect(
                "Select Main Dataset Row Identifier",
                options=main_df.columns,
                help="Choose the column from main dataset"
            )
        with col2:
            right_identifier = st.multiselect(
                "Select Enrichment Dataset Row Identifier",
                options=enrichment_df.columns,
                help="Choose the column from enrichment dataset"
            )
        
        if st.button("Perform Enrichment"):
            try:
                # Ensure right_identifier is a set for compatibility with the subtraction operation
                right_identifier_set = {right_identifier} if isinstance(right_identifier, str) else set(right_identifier)

                # Compute additional columns
                additional_columns = list(set(enrichment_df.columns) - right_identifier_set)

                enriched_df = eed.add_data(
                    df=main_df, additional_df=enrichment_df, 
                    left_id_names=left_identifier, right_id_names=right_identifier, 
                    additional_cols=additional_columns,
                    strategy=strategy, 
                    conflict_resolution=conflict_resolution)

                save_enrichment_config(enrichment_name=enrichment_name, 
                                       strategy=strategy,
                                       input_file_folder=input_file_folder,
                                       input_file=input_file,
                                       left_identifier=left_identifier,
                                       right_identifier=right_identifier,
                                       additional_columns=additional_columns,
                                       conflict_resolution=conflict_resolution
                                       )
                st.success("✅ Enrichment completed successfully!")
            except Exception as e:
                st.error(f"Error performing enrichment: {str(e)}")
    
    return enriched_df


def display_category_box(title, columns):
    if len(columns)>0:
        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <h4 style="margin: 0; padding: 0; color: #333;">{title} ({len(columns)} column(s))</h4>
                <p style="margin: 0; color: #666;">{', '.join(columns) if columns is not None else 'No columns found.'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_results(enriched_df: pd.DataFrame, title: str=None ,key_base='k'):
    """Display the results of the enrichment process."""

    # Preview the enriched dataset
    if title is not None:
        st.subheader(title)
    st.dataframe(enriched_df.head(), use_container_width=True)
    
    # Column information
    st.subheader("Columns Information")
    # Analyze the dataset using DataAnalyzer
    enr_analyzer = DataAnalyzer(enriched_df)
    column_categories = {
                "Quantitative: Columns with numerical quantivative data": sorted(enr_analyzer.numeric_cols),
                "Binary: Columns with 2 unique values (e.g., 0/1 or Yes/No)": sorted(enr_analyzer.binary_cols),
                "Non Binary Low-Cardinality Numeric: Numerical columns with few distinct values": sorted(enr_analyzer.non_binary_low_cardinality_numeric_cols),
                "Categorical: Columns with non-numerical data": sorted(enr_analyzer.categorical_cols),
                "High-Cardinality Categorical: Categorical columns with many unique values (e.g., id)": sorted(enr_analyzer.high_cardinality_cat_cols),
                "Date: Columns containing date or time data": sorted(enr_analyzer.date_cols),
                "Time Delta: Columns containing time interval data (e.g., age or period)": sorted(enr_analyzer.timedelta_cols)
            }
    for category, columns in column_categories.items():
        display_category_box(category, columns)

    col1, col2 = st.columns(2)
    with col1:
        # Download button
        st.download_button(
            label="Download Enriched Dataset",
            key= f"{key_base}_dl_bt",
            data=to_excel(enriched_df),
            file_name=f"{key_base}_dataset.xlsx",
            mime="text/xlsx",
            use_container_width=True
        )
    
    with col2:
        # Save to session state
        if st.button("Use This Dataset for Further Analysis", use_container_width=True, key=f"{key_base}_save_bt",):
            st.session_state.data = enriched_df
            st.session_state.working_df = enriched_df
            st.success("✅ Dataset saved and ready for further analysis!")


def display_external_data_results(enriched_df: pd.DataFrame, filtered_main_data: pd.DataFrame):
    """Display the results of the enrichment process."""
    st.subheader("Enrichment Results")
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Rows", value=len(enriched_df), delta=len(enriched_df)-len(filtered_main_data))
    with col2:
        st.metric("Total Columns", len(enriched_df.columns), delta=len(enriched_df.columns)-len(filtered_main_data.columns))
    # Preview the enriched dataset
    display_results(enriched_df=enriched_df, title="Enriched Dataset Preview", key_base="external_data")


def display_new_variables_results(enriched_df: pd.DataFrame, main_data: pd.DataFrame):
    """Display the results of the enrichment process."""
    st.subheader("Enrichment Results")
    
    # Display metrics
    col1, col2 = st.columns([1,3],vertical_alignment='center')
    with col1:
        st.metric(label="Total Columns", value=len(enriched_df.columns), delta=len(enriched_df.columns)-len(main_data.columns))
    with col2:
        st.write(list(set(enriched_df.columns) - set(main_data.columns)))
        
    # Preview the enriched dataset
    display_results(enriched_df=enriched_df, title="Enriched Dataset Preview", key_base="new_variables")


def display_imputation_results(imputed_data: pd.DataFrame, main_data: pd.DataFrame, cols_per_row: int = 8):
    """Display the results of the enrichment process."""
    st.subheader("Imputation Results")
    
    # List to hold column metrics
    columns = imputed_data.columns.tolist()

    # Iterate over columns in chunks based on the number of columns per row
    for i in range(0, len(columns), cols_per_row):
        # Create a row of columns
        cols = st.columns(cols_per_row)
        
        # Fill each column in the row with a metric
        for j, column in enumerate(columns[i:i + cols_per_row]):
            # Calculate non-null counts for the column
            total_non_null_imputed = int(imputed_data[column].notna().sum())
            total_non_null_main = int(main_data[column].notna().sum()) if column in main_data.columns else 0
            delta_non_null = total_non_null_imputed - total_non_null_main

            # Display metric
            cols[j].metric(
                label=f"{column} Non-Null",
                value=total_non_null_imputed,
                delta=delta_non_null  # Conversion handled by int()
            )
        
    # Preview the enriched dataset
    display_results(enriched_df=imputed_data, title="Imputed Dataset Preview", key_base="imputation")


def save_enrichment_config(enrichment_name: str, strategy: str, 
                         input_file_folder: str = "./data/enrichment/",
                         input_file: str = None,
                         identifier: str = None, 
                         left_identifier: str = None, 
                         right_identifier: str = None,
                         additional_columns: str = None,
                         conflict_resolution: str = None
                         ):
    """Save the enrichment configuration to session state."""
    if 'config' not in st.session_state:
        st.session_state.config = {'data_enrichments': {}}
    
    enrichment_config = {
        "STRATEGY": strategy,
        "INPUT_FILES_FOLDER": input_file_folder,
        "INPUT_FILE": input_file,
        "CONFLICT_RESOLUTION": conflict_resolution
    }

    if additional_columns is not None:
        enrichment_config["ADDITIONAL_COLUMNS"] = additional_columns
    
    if identifier:
        enrichment_config["COMMON_ROW_IDENTIFIER"] = identifier
    else:
        enrichment_config["LEFT_ROW_IDENTIFIER"] = left_identifier
        enrichment_config["RIGHT_ROW_IDENTIFIER"] = right_identifier
    
    st.session_state.config['data_enrichments'][enrichment_name] = enrichment_config


# Convert DataFrame to Excel for download using openpyxl
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
    return output.getvalue()


def calculate_prefix_score(input_token: str, column_name: str) -> float:
    """
    Calculate a score based on how well the input matches the beginning of the column name.
    
    Parameters:
    -----------
    input_token : str
        The user's input token
    column_name : str
        The column name to compare against
        
    Returns:
    --------
    float
        Prefix matching score between 0 and 1
    """
    input_lower = input_token.lower()
    col_lower = column_name.lower()
    
    # Perfect prefix match
    if col_lower.startswith(input_lower):
        return 1.0
    
    # Calculate longest common prefix
    common_prefix_length = 0
    for i in range(min(len(input_lower), len(col_lower))):
        if input_lower[i] == col_lower[i]:
            common_prefix_length += 1
        else:
            break
    
    # Return score based on prefix match quality
    if common_prefix_length == 0:
        return 0.0
    return common_prefix_length / len(input_lower)


def calculate_similarity_score(input_token: str, column_name: str) -> int:
    """
    Calculate a similarity score between input token and column name with
    prefix-weighted scoring and length-aware adjustments.
    
    Parameters:
    -----------
    input_token : str
        The user's input token
    column_name : str
        The column name to compare against
        
    Returns:
    --------
    int
        Adjusted similarity score
    """
    input_len = len(input_token)
    col_len = len(column_name)
    
    # Convert both to lowercase for comparison
    input_lower = input_token.lower()
    col_lower = column_name.lower()
    
    # Calculate base similarity scores
    ratio = fuzz.ratio(input_lower, col_lower)
    partial_ratio = fuzz.partial_ratio(input_lower, col_lower)
    token_sort_ratio = fuzz.token_sort_ratio(input_lower, col_lower)
    
    # Calculate prefix score (0 to 1)
    prefix_score = calculate_prefix_score(input_lower, col_lower)
    
    # Weight the scores (giving more weight to prefix matches)
    prefix_weight = 0.4  # 40% of the score comes from prefix matching
    fuzzy_weight = 0.6   # 60% from fuzzy matching
    
    # Calculate weighted fuzzy score
    fuzzy_score = max(ratio, partial_ratio, token_sort_ratio)
    
    # Combine scores with weights
    base_score = (prefix_score * 100 * prefix_weight) + (fuzzy_score * fuzzy_weight)
    
    # Apply length-based adjustments
    if col_len < input_len:
        # Penalize columns shorter than input unless it's an exact match
        if input_lower.startswith(col_lower):
            base_score = base_score * 0.9  # Small penalty for exact prefix match
        else:
            base_score = base_score * 0.5  # Heavy penalty for shorter non-matching columns
    
    # Penalize very short column names (less than 2 characters)
    if col_len <= 2:
        if not (input_lower == col_lower):  # Don't penalize exact matches
            base_score = base_score * 0.6
    
    # Additional penalty for single-character columns that don't exactly match
    if col_len == 1 and input_lower != col_lower:
        base_score = base_score * 0.3
    
    # Bonus for exact word beginnings in multi-word columns
    if "_" in col_lower or " " in col_lower:
        words = col_lower.replace("_", " ").split()
        if any(word.startswith(input_lower) for word in words):
            base_score = min(base_score * 1.2, 100)  # 20% bonus, capped at 100
    
    return int(base_score)


def tokenize_formula(formula: str) -> List[str]:
    """
    Tokenizes the formula, treating quoted variables as single tokens and splitting others by operators.
    
    Parameters:
    ----------
    formula : str
        The formula input by the user.
        
    Returns:
    -------
    List[str]
        The list of tokens extracted from the formula.
    """
    # Remove leading and trailing spaces from the formula
    formula = formula.strip()

    # Regex to capture anything within double quotes as a single token
    quoted_tokens = re.findall(r'\"\"[^\"]+\"\"', formula)
    
    # Replace quoted tokens in the formula with placeholders
    for idx, token in enumerate(quoted_tokens):
        placeholder = f"__quoted_{idx}__"
        formula = formula.replace(token, placeholder)
    
    # Now split the remaining formula by spaces and operators
    #tokens = re.findall(r'\S+', formula)
    tokens = (
        formula.replace(' + ', ' ; ').replace(' - ', ' ; ').replace(' * ', ' ; ').replace(' / ', ' ; ').replace(' ** ', ' ; ') # arithmetic operators
              .replace(' == ', ' ; ').replace(' >= ', ' ; ').replace(' <= ', ' ; ').replace(' < ', ' ; ').replace(' > ', ' ; ') # comparison operators
              .replace(' | ', ' ; ').replace(' & ', ' ; ') # logical operators
              .replace('( ', '').replace(' )', '') # priority operators
              .split(sep=" ; ")
              )
    tokens = [token.strip() for token in tokens]
    print(f"tokens:{tokens}")
    
    # Replace placeholders with the original quoted tokens
    for idx, token in enumerate(quoted_tokens):
        placeholder = f"__quoted_{idx}__"
        stripped_token = token[2:-2].strip()
        tokens = [stripped_token if t.strip() == placeholder else t.strip() for t in tokens]
    print(tokens)
    return tokens


def suggest_columns(computation_formula: str, column_names: List[str], threshold: int = 40, max_suggestions: int = 6) -> List[Tuple[str, int]]:
    """
    Suggest column names based on similarity to the input formula with enhanced
    prefix matching, including handling quoted variable names as unique tokens.
    
    Parameters:
    -----------
    computation_formula : str
        The formula input by the user.
    column_names : List[str]
        List of available column names in the dataset.
    threshold : int, optional
        Minimum similarity score to include in suggestions (default: 60).
    max_suggestions : int, optional
        Maximum number of suggestions to return (default: 5).
        
    Returns:
    --------
    List[Tuple[str, int]]
        List of tuples containing (column_name, similarity_score).
    """
    if not computation_formula or not column_names:
        return []
    
    # Tokenize the formula while treating quoted variables as single tokens
    tokens = tokenize_formula(computation_formula)
    
    last_token = tokens[-1] if tokens else ''
    
    # Skip suggestions if the last token is too short (unless it's the only token)
    if len(last_token) < 2 and len(tokens) > 1:
        return []
    
    # Calculate adjusted similarity scores for each column name
    suggestions = []
    for col in column_names:
        score = calculate_similarity_score(last_token, col)
        if score > threshold:
            suggestions.append((col, score))
    
    # Sort by score, prefix match, and length
    ranked_suggestions = sorted(
        suggestions,
        key=lambda x: (
            -x[1],  # Score (descending)
            -x[0].lower().startswith(last_token.lower()),  # Prefix match (True first)
            len(x[0])  # Length (ascending)
        )
    )
    
    # Return only the top N suggestions (max_suggestions)
    return ranked_suggestions[:max_suggestions]


def display_column_suggestions(suggestions):
    """
    Displays column suggestions as buttons and enables users to copy column names to the clipboard by clicking.

    Parameters:
    ----------
    suggestions : list of tuples
        List of (column name, score) tuples to display.
    """
    if suggestions:
        cols = st.columns(min(3, len(suggestions)))
        
        for idx, (col, score) in enumerate(suggestions):
            with cols[idx % 3]:
                # HTML code for the button with copy-to-clipboard functionality and temporary notification
                html_code = f"""
                <div style="margin: 5px 0; position: relative;">
                    <button 
                        onclick="
                            navigator.clipboard.writeText('{col}').then(() => {{
                                const notification = document.createElement('div');
                                notification.innerText = 'Copied: {col}';
                                notification.style = `
                                    position: absolute;
                                    top: -30px;
                                    left: 50%;
                                    transform: translateX(-50%);
                                    background-color: #4CAF50;
                                    color: white;
                                    padding: 5px 10px;
                                    border-radius: 4px;
                                    font-size: 12px;
                                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                                    z-index: 1000;
                                `;
                                this.parentElement.appendChild(notification);
                                
                                setTimeout(() => {{
                                    notification.remove();
                                }}, 800);
                            }});
                        "
                        style="
                            background-color: #f0f2f6;
                            border: 1px solid #e0e0e0;
                            border-radius: 4px;
                            padding: 8px;
                            cursor: pointer;
                            width: 100%;
                            text-align: left;
                            box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
                            font-size: 14px;"
                        onmouseover="this.style.backgroundColor='#e0e2e6'"
                        onmouseout="this.style.backgroundColor='#f0f2f6'">
                        <span style="color: #0066cc; font-weight: bold;">{col}</span>
                        <br/>
                        <small style="color: #666;">Score: {score}</small>
                    </button>
                </div>
                """
                st.components.v1.html(html_code, height=60)
    else:
        st.info("No matching columns found. Try typing part of a column name.")


def validate_variable_name(name, existing_columns):
    """
    Validates the variable name and returns (is_valid, error_message).
    """
    if not name:
        return False, "Variable name cannot be empty"
    
    if name in existing_columns:
        return False, f"Variable name '{name}' already exists"
    
    if not re.match(r'^[a-zA-Z_][ a-zA-Z0-9_\[\]\(\)]*$', name):
        return False, "Variable name must start with a letter or underscore and contain only letters, numbers, some special characters, spaces and underscores"
    
    return True, None


# Helper function to check if a string represents a number
def is_number(token):
    try:
        float(token)  # Try converting to float
        return True
    except ValueError:
        return False
    

def validate_and_extract_columns(formula: str, dataset_columns: List[str]) -> Tuple[bool, str, List[str], List[str]]:
    """
    Extract and validate column names from a formula string.
    
    Parameters:
    -----------
    formula : str
        The computation formula.
    dataset_columns : List[str] or pd.Index
        List of valid column names from the dataset.
        
    Returns:
    --------
    tuple:
        - is_valid (bool): True if the formula is valid, False otherwise.
        - error_message (str): Error message if validation fails, None if valid.
        - used_columns (List[str]): List of valid column names used in the formula.
    """

    # Ensure consistent formatting of dataset columns
    dataset_columns = sorted(dataset_columns, key=len, reverse=True)  # Sort by length to avoid partial matches
    valid_columns = []
    constant_tokens = []

    try:
        # Tokenize the formula while treating quoted variables as single tokens
        tokens = tokenize_formula(formula)

        # Check for column usage and extract valid columns from the formula
        for column in dataset_columns:
            if column in tokens:
                valid_columns.append(column)
                # Remove all occurrences of the column (.remove just act one the 1st occ)
                tokens = [token for token in tokens if token != column]
        
        constant_tokens = [token for token in tokens if (token.startswith('constant(') and token.endswith(')') )]
        # Handle constant values with simple string checks
        tokens = [token for token in tokens if not ((token.startswith('constant(') and token.endswith(')')) or is_number(token))]

        # Validate the formula syntax by trying to evaluate it
        if len(tokens)>0:
            raise ValueError(f"Invalid column reference detected in the formula: {tokens}")
        # Return success if valid, with used columns
        return True, None, valid_columns, constant_tokens
    except Exception as e:
        return False, str(e), valid_columns, constant_tokens


# Replace column names in the formula with their corresponding normalized names
def normalize_formula(formula: str, column_mapping: dict) -> str:
    """
    Replace column names in the formula with normalized names (e.g., var_1, var_2).
    
    Parameters:
    ----------
    formula : str
        The computation formula to be normalized.
    column_mapping : dict
        A mapping of original column names to normalized names.
    
    Returns:
    -------
    str
        The formula with normalized column names.
    """
    print("original formula:", formula)
    for normalized, original in column_mapping.items():
        print(f"normalized {normalized}, original {original.name}")
        # Replace each column name with its corresponding normalized variable name
        formula = formula.replace(f'""{original.name}""', f'{normalized}')
        formula = formula.replace(original.name, normalized)  # in case the column name is without quotes
        
    print("norm formula:", formula)
    return formula


def extract_constant_value(constant_token: str) -> str:
    """
    Extract the value from a constant token like constant(val1) -> val1
    """
    # Remove 'constant(' prefix and ')' suffix and return the inner value
    return constant_token[9:-1]  # len('constant(') == 9


def create_variables_dict(dataset: pd.DataFrame, identified_columns: List[str], constant_tokens: List[str]) -> Dict[str, pd.Series]:
    """
    Create a dictionary mapping variable names to their corresponding Series
    for both regular columns and constants.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        The input dataset
    identified_columns : List[str]
        List of valid column names used in the formula
    constant_tokens : List[str]
        List of constant tokens found in the formula
        
    Returns:
    --------
    Dict[str, pd.Series]:
        Dictionary mapping variable names to their Series
    """
    local_vars = {}
    
    # Handle regular columns
    for idx, col in enumerate(identified_columns):
        local_vars[f"var_{idx}"] = dataset[col]
    
    # Handle constants
    start_idx = len(identified_columns)
    for idx, constant_token in enumerate(constant_tokens, start=start_idx):
        constant_value = extract_constant_value(constant_token)
        # Create a Series with the constant value repeated for each row
        constant_series = pd.Series([constant_value] * len(dataset), index=dataset.index, name=constant_token)
        local_vars[f"var_{idx}"] = constant_series
    
    return local_vars


def evaluate_formula_safely(formula: str, variable_name: str, dataset: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    """
    Safely evaluate a computation formula to create a new variable in the dataset.

    Parameters:
    -----------
    formula : str
        The computation formula entered by the user.
    variable_name : str
        The name of the new variable to create.
    dataset : pd.DataFrame
        The dataset containing columns to use in the formula.

    Returns:
    --------
    tuple[bool, str, pd.DataFrame]
        A tuple containing:
        - A boolean indicating success or failure.
        - A success or error message.
        - The modified DataFrame (or the original DataFrame on failure).
    """
    # Extract dataset column names
    column_names = dataset.columns.tolist()
    
    # Validate formula and extract variables
    is_valid, error_message, identified_columns, constants = validate_and_extract_columns(formula, column_names)
    if not is_valid:
        return False, f"Invalid formula: {error_message}", dataset

    # Map column names to dataset references
    #local_vars = {col: dataset[col] for col in identified_columns}
    #local_vars = {f"var_{idx}": dataset[col] for idx, col in enumerate(identified_columns)}
    local_vars = create_variables_dict(dataset, identified_columns, constants)

    #normalize the formula
    print(f"local_vars dict: {local_vars} \nInitial formula: {formula}")
    normalized_formula = normalize_formula(formula, local_vars)
    try:
        # Safely evaluate the formula
        print(f"Normalized formula: {normalized_formula}")
        result = eval(normalized_formula, {"np":np, "pd":pd}, local_vars)
        print(result)
        # Check if result is a pandas Series with the correct length
        if isinstance(result, pd.Series) and result.shape[0] == dataset.shape[0]:
            # Add the new variable to the dataset
            if variable_name in dataset.columns:
                return False, f"Variable '{variable_name}' already exists.", dataset

            dataset[variable_name] = result
            return True, f"✨ Variable '{variable_name}' successfully added.", dataset

        # Handle cases where result is not a valid pandas Series
        return False, (
            f"Formula evaluation resulted in invalid output.\n"
            f"Expected a pandas Series with {dataset.shape[0]} rows.\n"
            f"Check for mismatched types or operations in your formula."
        ), dataset

    except Exception as e:
        return False, (
            f"Error evaluating formula: {str(e)}\n"
            f"Make sure your formula uses valid column names and operators (+, -, *, /, etc.)."
        ), dataset


# Create a statistics DataFrame based on the type of the column
def generate_stats(temp_main_data, variable_name):
    column_data = temp_main_data[variable_name]
    
    # Determine the column type
    if pd.api.types.is_numeric_dtype(column_data):
        # For numeric columns
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Std', 'Min', 'Max', 'Non-null Count'],
            'Value': [
                f"{column_data.mean():.2f}" if not column_data.isna().all() else "NaN",
                f"{column_data.std():.2f}" if not column_data.isna().all() else "NaN",
                f"{column_data.min():.2f}" if not column_data.isna().all() else "NaN",
                f"{column_data.max():.2f}" if not column_data.isna().all() else "NaN",
                f"{column_data.count()}"
            ]
        })
    elif pd.api.types.is_timedelta64_dtype(column_data):
        # For timedelta columns
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Std', 'Min', 'Max', 'Non-null Count'],
            'Value': [
                str(column_data.mean()) if not column_data.isna().all() else "NaT",
                str(column_data.std()) if not column_data.isna().all() else "NaT",
                str(column_data.min()) if not column_data.isna().all() else "NaT",
                str(column_data.max()) if not column_data.isna().all() else "NaT",
                f"{column_data.count()}"
            ]
        })
    elif pd.api.types.is_string_dtype(column_data):
        # For string columns, only non-null count is relevant
        stats_df = pd.DataFrame({
            'Statistic': ['Unique', 'Most Frequent', 'Non-null Count'],
            'Value': [
                column_data.nunique(),
                column_data.mode().iloc[0] if not column_data.mode().empty else "No Mode",
                f"{column_data.count()}"
            ]
        })
    else:
        # For other types, provide generic information
        stats_df = pd.DataFrame({
            'Statistic': ['Non-null Count'],
            'Value': [f"{column_data.count()}"]
        })
    
    return stats_df
  

def define_new_variables(main_data):
    """
    Enhanced version of define_new_variables with improved preview and validation.
    """
    if main_data is None:
        st.warning("Please load data first.")
        return main_data
    
    st.subheader("Define New Variable")
    
    # Get column names
    column_names = main_data.columns.tolist()
    
    # Create two columns for input fields
    col1, col2 = st.columns([1, 2])
    
    with col1:
        variable_name = st.text_input(
            "Enter new variable name",
            help="Variable name must start with a letter or underscore and contain only letters, numbers, and underscores"
        ).strip()
        
        # Real-time variable name validation
        if variable_name:
            is_valid, error_msg = validate_variable_name(variable_name, column_names)
            if not is_valid:
                st.error(error_msg)
    
    with col2:
        if st.checkbox("Show available columns"):
            st.write("Available columns:", ", ".join(f"`{col}`" for col in column_names))
    
    # Initialize session state
    if 'formula' not in st.session_state:
        st.session_state.formula = ""
    
    # Formula input with enhanced autocompletion
    computation_formula = st.text_area(
        label="Enter computation formula",
        value=st.session_state.formula,
        help=(
            "Type your formula using column names. For example: `column1 + column2 * 2`. "
            "If a column name contains special characters (e.g., 'Type 2 diabetes diagnosis - Year'), "
            "enclose the column name in double quotes like this: `\"\"Type 2 diabetes diagnosis - Year\"\"`."
            "\n\n"
            "You can use basic operators like `+`, `-`, `*`, `/` for arithmetic operations."
        ),
        key="formula_input",
        height=100
    )
    
    # Show real-time formula validation and preview
    if computation_formula:
        # Show suggestions
        st.markdown("### Column Suggestions")
        suggestions = suggest_columns(computation_formula, column_names, threshold=25)
        display_column_suggestions(suggestions)

        # Retrieve column names from the formula
        formula_valid, error_msg, used_columns, constants = validate_and_extract_columns(computation_formula, column_names)
        
        if not formula_valid:
            st.error(f"Formula error: {error_msg}")

        if used_columns or constants:
            st.markdown("### Source Variables and Constants Preview")
        if used_columns:
            preview_data = main_data[used_columns].head()
            styled_preview = (preview_data.style
                            .format(precision=2)
                            .highlight_null(props='color: red;')
                            .set_table_styles([
                                {'selector': 'th', 'props': [('background-color', '#f0f2f6')]},
                                {'selector': 'td', 'props': [('padding', '8px')]}
                            ]))
            st.dataframe(styled_preview, use_container_width=True, hide_index=True)
        if constants:
            st.write(f"Constants list: {[extract_constant_value(constant) for constant in constants]}")
        
        
    
    # Add new variable button with validation
    if st.button("Compute New Variable", disabled=not (variable_name and computation_formula)):
        # Final validation
        name_valid, name_error = validate_variable_name(variable_name, column_names)
        formula_valid, formula_error, used_columns, used_constants = validate_and_extract_columns(computation_formula, column_names)
        
        if not name_valid:
            st.error(f"Invalid variable name: {name_error}")
            return main_data
        
        if not formula_valid:
            st.error(f"Invalid formula: {formula_error}")
            return main_data
        
        try:
            # Add the new variable
            succ , e, temp_main_data = evaluate_formula_safely(computation_formula, variable_name, main_data)  #eval(computation_formula, {}, main_data)
            if succ:
                st.success(f"✨ Variable '{variable_name}' successfully computed!")
            
                # Show comprehensive preview
                st.write("### New Variable Preview")

                # Create preview with source columns and new variable
                preview_columns = used_columns + [variable_name]
                preview_df = temp_main_data[preview_columns].head()
                
                # Add statistics for the new variable
                stats_df = generate_stats(temp_main_data, variable_name)
            
                # Display preview with source columns
                st.write("First rows with source columns:")
                styled_preview = (preview_df.style
                                .format(precision=2)
                                .highlight_null(props='color: red;')
                                .set_table_styles([
                                    {'selector': 'th', 'props': [('background-color', '#f0f2f6')]},
                                    {'selector': 'td', 'props': [('padding', '8px')]}
                                ]))
                st.dataframe(styled_preview, use_container_width=True, hide_index=True)
                
                # Display statistics
                st.write(f"Statistics for new variable :blue[{variable_name}]:")
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                # Compute new variable button with validation
                if st.button("Add New Variable", disabled=not (variable_name and computation_formula)):
                    main_data = temp_main_data.copy()
                    st.success(f"✨ Variable '{variable_name}' successfully added!")
            else:
                st.write(e)
            
        except Exception as e:
            st.error(f"Error adding variable: {str(e)}")
            st.info("Make sure your formula uses valid column names and operators (+, -, *, /, etc.).")
    
    # Display the full dataset preview
    if st.checkbox("Show full dataset preview"):
        styled_df = (main_data.style
                    .format(precision=2)
                    .highlight_null(props='color: red;')
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#f0f2f6')]},
                        {'selector': 'td', 'props': [('padding', '8px')]}
                    ]))
        st.dataframe(styled_df, use_container_width=True)

    return main_data


# Safely merge transformed_data into main_data with replacement or ignoring duplicates
def merge_transformed_columns(main_data, transformed_data, handle_duplicates="replace"):
    """
    Merge transformed_data into main_data securely.

    Parameters:
    - main_data (pd.DataFrame): The original DataFrame.
    - transformed_data (pd.DataFrame): The DataFrame with new columns.
    - handle_duplicates (str): How to handle duplicate column names. 
                               Options: "replace" or "ignore".
    
    Returns:
    - pd.DataFrame: The merged DataFrame.
    """
    if handle_duplicates not in ["replace", "ignore"]:
        raise ValueError("handle_duplicates must be 'replace' or 'ignore'.")

    # Identify duplicate column names
    duplicate_columns = main_data.columns.intersection(transformed_data.columns)

    if not duplicate_columns.empty:
        if handle_duplicates == "replace":
            # Drop duplicate columns from main_data before merging
            main_data = main_data.drop(columns=duplicate_columns)
            st.info(f"Replaced duplicate columns: {list(duplicate_columns)}")
        elif handle_duplicates == "ignore":
            # Drop duplicate columns from transformed_data
            transformed_data = transformed_data.drop(columns=duplicate_columns)
            st.info(f"Ignored duplicate columns: {list(duplicate_columns)}")
    
    # Merge the DataFrames
    merged_data = pd.concat([main_data, transformed_data], axis=1)
    return merged_data



def compute_transformations(dataframe):
    st.subheader("Transformation and Aggregation")

    # Define available transformations
    TRANSFORMATIONS = {
        "Mean": lambda x: x.mean(),
        "Median": lambda x: x.median(),
        "Summation": lambda x: x.sum(),
        "Minimum Value": lambda x: x.min(),
        "Maximum Value": lambda x: x.max(),
        "Standard Deviation": lambda x: x.std(),
        "Natural Logarithm Transformation": lambda x: np.log(x + 1),
        "Min-Max Normalization": lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten(),
        "Z-Score Standardization": lambda x: (x - x.mean()) / x.std(),
    }

    # Step 1: Transformation Type Selection
    transformation_type = st.selectbox(
        "Choose a transformation type:",
        options=["Statistical-Based", "Dimensional Reduction-Based", "Scaling-Based", "Encoding"],
        help="Select the type of transformation."
    )

    # Step 2: Transformation Selection (based on transformation type)
    if transformation_type == "Statistical-Based":
        transformation = st.selectbox(
            "Choose a statistical transformation:",
            options=["Mean", "Median", "Summation", "Minimum Value", "Maximum Value", "Standard Deviation"],
            help="Select the statistical transformation to apply."
        )
    elif transformation_type == "Scaling-Based":
        transformation = st.selectbox(
            "Choose a scaling method:",
            options=["Min-Max Normalization", "Z-Score Standardization", "Natural Logarithm Transformation"],
            help="Select the scaling method."
        )
    elif transformation_type == "Dimensional Reduction-Based":
        # Define explanations for each dimensional reduction method
        transformation_explanations = {
            "PCA": ":grey_question: Principal Component Analysis (PCA) reduces dimensions by finding orthogonal directions (principal components) that explain the most variance in the dataset.",
            "t-SNE": ":grey_question: t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear technique that projects high-dimensional data into a lower-dimensional space, optimizing for local similarity.",
            "UMAP": ":grey_question: Uniform Manifold Approximation and Projection (UMAP) is a graph-based method that preserves both local and global structures of high-dimensional data.",
            "FAMD": ":grey_question: Factor Analysis of Mixed Data (FAMD) is designed for datasets containing both categorical and numerical variables, reducing dimensions while preserving mixed-data relationships."
        }
        
        transformation = st.selectbox(
            "Choose a dimensional reduction method:",
            options=["PCA", "t-SNE", "UMAP", "FAMD"],
            help="Select the dimensional reduction method."
        )
        
        # Display explanation for the selected transformation method
        st.info(transformation_explanations[transformation])


    else:
        # Define explanations for each numerization method
        numerization_explanations = {
            "One-Hot Encoding": ":grey_question: One-Hot Encoding creates binary columns for each category in a categorical variable, ensuring no ordinal relationship is implied. Useful for non-ordinal categorical data.",
            "Label Encoding": ":grey_question: Label Encoding assigns a unique integer to each category. Suitable for ordinal or small categorical datasets but may introduce ordinal bias for non-ordinal data.",
            "Ordinal Encoding": ":grey_question: Ordinal Encoding assigns integers to categories based on a meaningful order. Use this for ordinal variables where the order between categories is significant."
        }
        transformation = st.selectbox(
            "Choose a Numerization method:",
            options=["One-Hot Encoding", "Label Encoding", "Ordinal Encoding"],
            help="Select the encoding method for categorical variables."
        )
        # Display explanation for the selected numerization method
        st.info(numerization_explanations[transformation])

    # Step 3: Column Selection based on transformation type
    if transformation_type == "Encoding":
        columns = st.multiselect(
            "Choose categorical columns to encode:",
            options=dataframe.select_dtypes(include=['object', 'category']).columns,
            help="Select one or more categorical columns from the dataset."
        )
        if transformation == "Ordinal Encoding":
            category_orders = {}

            for col in columns:
                st.write(f"Specify the order for column: {col}")
                category_orders[col] = st.multiselect(
                            f"Select the order for {col}:",
                            options=dataframe[col].unique(),
                            default=dataframe[col].unique(),
                            key=f"order_{col}"
                        )
    elif transformation_type == "Dimensional Reduction-Based" and transformation == "FAMD":
        columns = st.multiselect(
            "Choose columns to apply the selected transformation:",
            options=dataframe.select_dtypes(include=[ np.number, 'object', 'category']).columns,
            help="Select one or more columns from the dataset."
        )
    else:
        columns = st.multiselect(
            "Choose columns to apply the selected transformation:",
            options=dataframe.select_dtypes(include=np.number).columns,
            help="Select one or more columns from the dataset."
        )

    # Step 4: Naming Pattern Customization
    if transformation_type == "Dimensional Reduction-Based":
        naming_pattern_val = "Name_{method_applied}_Component"
    elif transformation_type == "Encoding":
        naming_pattern_val = "{initial_variable}_{category}"
    else:
        naming_pattern_val = "{initial_variable}_{method_applied}"
    
    naming_pattern = st.text_input(
        "Enter a naming pattern for the new variables:",
        value=naming_pattern_val,
        help="Use placeholders: `{initial_variable}` for the original column name and `{method_applied}` for the selected transformation."
    )

    # Dimensional Reduction Settings
    if transformation_type == "Dimensional Reduction-Based" and len(columns) > 1:
        col1, col2, col3 = st.columns([1, 1, 3], vertical_alignment="center")

        with col1:
            st.markdown("### Reduction Settings")
            n_components = st.slider(
                "Select the number of components:",
                min_value=1,
                max_value=max(len(columns)-1, 2), 
                value=min(3, len(columns)-1),
                step=1,
                help="This determines how many dimensions the dataset will be reduced to. "
                    "The number of components should be less than or equal to the number of selected columns."
            )

            if transformation == "t-SNE":
                perplexity = st.slider(
                    "Perplexity:",
                    min_value=5,
                    max_value=50,
                    value=30,
                    help="Perplexity is a key parameter for t-SNE that defines the balance between local and global aspects of the data structure. "
                        "A higher perplexity considers more global patterns."
                )
                learning_rate = st.slider(
                    "Learning rate:",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    help="The learning rate determines how fast the t-SNE optimization algorithm converges."
                )

            elif transformation == "UMAP":
                n_neighbors = st.slider(
                    "Number of neighbors:",
                    min_value=2,
                    max_value=100,
                    value=15,
                    help="Defines the size of the local neighborhood UMAP uses for manifold approximation. "
                        "Larger values emphasize global structure."
                )
                min_dist = st.slider(
                    "Minimum distance:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.1,
                    help="Controls the spacing of points in the low-dimensional space. "
                        "Smaller values create tighter clusters."
                )

        with col2:
            show_explained_variance = None
            if transformation in ["PCA", "FAMD"]:
                show_explained_variance = st.checkbox(
                    "Show Explained Variance",
                    value=False,
                    help="Display the amount of variance explained by each component. "
                        "Useful for understanding how much information is retained after reduction."
                )

        with col3:
            if show_explained_variance and transformation in ["PCA", "FAMD"]:
                
                if transformation == "PCA":
                    #scaler = StandardScaler()
                    #scaled_data = scaler.fit_transform(dataframe[columns])
                    #model = PCA()
                    #result = model.fit_transform(scaled_data)
                    #variance_ratio = result.explained_variance_ratio_

                    # Initialize the PCA model
                    model = prince.PCA(
                        n_components=len(columns),  # Number of principal components to retain
                        rescale_with_mean=True,     # Center the data by subtracting the mean
                        rescale_with_std=True,      # Scale the data by dividing by standard deviation
                        copy=True,                  # Do not modify the original data
                        check_input=True,           # Ensure input is valid
                        engine='sklearn'            # Select sklearn computation engine
                    )

                    # Fit the PCA model to the selected columns & transform the data into the lower-dimensional space
                    result = model.fit_transform(dataframe[columns])

                    variance_ratio = model.percentage_of_variance_

                else:  # FAMD
                    model = prince.FAMD(n_components=len(columns), random_state=42)
                    #famd = model.fit(dataframe[columns])
                    result = model.fit_transform(dataframe[columns])

                    # Get eigenvalues (variance explained by each dimension)
                    eigenvalues = model.eigenvalues_

                    # Compute proportion of variance explained
                    #total_variance = sum(eigenvalues)
                    variance_ratio = model.percentage_of_variance_#[eig / total_variance for eig in eigenvalues]

                cumulative_variance = np.cumsum(variance_ratio)

                # Plot Explained Variance
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="-", color="b")
                ax.set_ylim(0, max(cumulative_variance) * 1.1)
                ax.set_xlim(1, len(cumulative_variance))
                ax.vlines(n_components, 0, cumulative_variance[n_components - 1], color="red", linestyle="--", linewidth=2)
                ax.hlines(cumulative_variance[n_components - 1], 0, n_components, color="red", linestyle="--", linewidth=2)
                ax.text(n_components, cumulative_variance[n_components - 1], f"{cumulative_variance[n_components - 1]:.2f}", ha="center", va="bottom", fontsize=12, color="red")
                ax.set_title("Explained Variance by Components")
                ax.set_xlabel("Number of Components")
                ax.set_ylabel("Cumulative Explained Variance")
                ax.grid(True)
                st.pyplot(fig)

        if transformation in ["PCA", "FAMD"] and show_explained_variance:  
            styled_df = model.column_contributions_.iloc[:, :n_components].style.format('{:.0%}').background_gradient(cmap='Blues', vmin=0, vmax=1)  # Gradient scale from 0 to 1
            st.subheader("Contributions to components")
            st.dataframe(styled_df, use_container_width=True)

    # Apply Transformation Button
    if st.button("Apply Transformation", disabled=not (naming_pattern and columns)):
        if not columns:
            st.warning("Please select at least one column to proceed.")
            return

        new_columns = {}

        try:
            if transformation_type == "Dimensional Reduction-Based":
                # Standardize data
                if transformation in ["t-SNE", "UMAP"]:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(dataframe[columns])

                if transformation == "PCA":
                    #model = PCA(n_components=n_components)
                    model = prince.PCA(
                        n_components=n_components,  # Number of principal components to retain
                        rescale_with_mean=True,     # Center the data by subtracting the mean
                        rescale_with_std=True,      # Scale the data by dividing by standard deviation
                        copy=True,                  # Do not modify the original data
                        check_input=True,           # Ensure input is valid
                        engine='sklearn'            # Select sklearn computation engine
                    )
                    transformed_data = model.fit_transform(dataframe[columns])

                elif transformation == "t-SNE":
                    model = TSNE(n_components=n_components, perplexity=perplexity,
                                learning_rate=learning_rate, random_state=42)
                    transformed_data = model.fit_transform(scaled_data)
                elif transformation == "UMAP":
                    model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                                    min_dist=min_dist, random_state=42)
                    transformed_data = model.fit_transform(scaled_data)
                else:  # FAMD
                    model = prince.FAMD(n_components=n_components, random_state=42)
                    transformed_data = model.fit_transform(dataframe[columns])

                # Create column names and DataFrame
                component_names = [naming_pattern.format(method_applied=transformation.replace(" ", "_")) + f"_{i + 1}" for i in range(n_components)]

                if transformation in ["t-SNE", "UMAP"]:
                    result_df = pd.DataFrame(transformed_data, columns=component_names, index=dataframe.index)
                else:
                    # Ensure it's treated as a DataFrame (since FAMD already returns one)
                    result_df = transformed_data.copy()
                    result_df.columns = component_names

                
                # Convert to dictionary and update new_columns
                new_columns.update(result_df.to_dict(orient="list"))


            elif transformation_type == "Encoding":
                if transformation == "One-Hot Encoding":
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    for col in columns:
                        encoded_data = encoder.fit_transform(dataframe[[col]])
                        categories = encoder.categories_[0]
                        for i, category in enumerate(categories):
                            
                            new_col_name = naming_pattern.format(
                                initial_variable=col,
                                category=category
                            )
                            print(f"{new_col_name}:{encoded_data[:, i]}")
                            new_columns[new_col_name]  = encoded_data[:, i]
                
                elif transformation == "Label Encoding":
                    encoder = LabelEncoder()
                    for col in columns:
                        new_col_name = naming_pattern.format(
                            initial_variable=col,
                            category="label_encoded"
                        )
                        new_columns[new_col_name] = encoder.fit_transform(dataframe[col])
                
                elif transformation == "Ordinal Encoding":

                    for col in columns:
                        # Initialize the encoder with the specified category order for the current column
                        encoder = OrdinalEncoder(categories=[category_orders[col]])
                        new_col_name = naming_pattern.format(
                            initial_variable=col,
                            category="ordinal"
                        )
                        new_columns[new_col_name] = encoder.fit_transform(dataframe[[col]]).flatten()

            else:
                # Handle statistical and scaling transformations
                method_function = TRANSFORMATIONS.get(transformation, None)
                if method_function:
                    for col in columns:
                        new_col_name = naming_pattern.format(
                            initial_variable=col,
                            method_applied=transformation.replace(" ", "_")
                        )
                        result = method_function(dataframe[col])
                        # Ensure result is array-like
                        if np.isscalar(result):
                            result = [result] * len(dataframe)
                        new_columns[new_col_name] = result

            # Create final result DataFrame
            if new_columns:
                result_df = pd.DataFrame(new_columns, index=dataframe.index)
                st.success("✨ Transformations applied successfully!")
                st.dataframe(result_df, use_container_width=True)

                # Update the session state
                st.session_state['enriched_df'] = pd.concat([dataframe, result_df], axis=1)
                return result_df
            else:
                st.warning("⚠️ No transformations were applied.")

        except Exception as e:
            st.error(f"Error applying transformation: {str(e)}")
            return None
        

# def compute_transformations_old3(dataframe):
#     st.subheader("Transformation and Aggregation")

#     # Define available transformations
#     TRANSFORMATIONS = {
#         "Mean": lambda x: x.mean(),
#         "Median": lambda x: x.median(),
#         "Summation": lambda x: x.sum(),
#         "Minimum Value": lambda x: x.min(),
#         "Maximum Value": lambda x: x.max(),
#         "Standard Deviation": lambda x: x.std(),
#         "Natural Logarithm Transformation": lambda x: np.log(x + 1),
#         "Min-Max Normalization": lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten(),
#         "Z-Score Standardization": lambda x: (x - x.mean()) / x.std(),
#     }

#     # Step 1: Transformation Type Selection
#     transformation_type = st.selectbox(
#         "Choose a transformation type:",
#         options=["Statistical-Based", "Dimensional Reduction-Based", "Scaling-Based", "Other"],
#         help="Select the type of transformation."
#     )

#     # Step 2: Transformation Selection (based on transformation type)
#     if transformation_type == "Statistical-Based":
#         transformation = st.selectbox(
#             "Choose a statistical transformation:",
#             options=["Mean", "Median", "Summation", "Minimum Value", "Maximum Value", "Standard Deviation"],
#             help="Select the statistical transformation to apply."
#         )
#     elif transformation_type == "Scaling-Based":
#         transformation = st.selectbox(
#             "Choose a scaling method:",
#             options=["Min-Max Normalization", "Z-Score Standardization", "Natural Logarithm Transformation"],
#             help="Select the scaling method."
#         )
#     elif transformation_type == "Dimensional Reduction-Based":
#         transformation = st.selectbox(
#             "Choose a dimensional reduction method:",
#             options=["PCA"],
#             help="Select the dimensional reduction method."
#         )
#     else:
#         transformation = st.selectbox(
#             "Choose a Numerization method:",
#             options=["dummy"],
#             help="Select the dumyfication method."
#         )

#     # Step 3: Column Selection
#     columns = st.multiselect(
#         "Choose columns to apply the selected transformation:",
#         options=dataframe.select_dtypes(include=np.number).columns,
#         help="Select one or more columns from the dataset."
#     )
#     if transformation_type == "Dimensional Reduction-Based":
#         naming_pattern_val = "Name_{method_applied}_Component"
#     else:
#         naming_pattern_val = "{initial_variable}_{method_applied}"
#     # Step 4: Naming Pattern
#     naming_pattern = st.text_input(
#         "Enter a naming pattern for the new variables:",
#         value=naming_pattern_val,
#         help="Use placeholders: `{initial_variable}` for the original column name and `{method_applied}` for the selected transformation."
#     )

#     # PCA Handling
#     if transformation == "PCA" and len(columns)>1:
#         col1, col2, col3 = st.columns([1, 1, 3], vertical_alignment='center')     

#         with col1:
#             st.markdown("### PCA Settings")
#             n_components = st.slider(
#                 "Select the number of principal components:",
#                 min_value=0, 
#                 max_value=max(len(columns)-1,1), 
#                 value=max(len(columns)-1,1), 
#                 step=1
#             )

#         with col2:
#             show_elbow_curve = st.checkbox("Show Elbow Curve", value=False)

#         with col3:
#             # Optional: Show elbow curve
#             if show_elbow_curve:
#                 # Standardize data
#                 scaler = StandardScaler()
#                 scaled_data = scaler.fit_transform(dataframe[columns])

#                 # Compute PCA
#                 pca = PCA()
#                 pca_result = pca.fit_transform(scaled_data)

#                 st.markdown("### PCA Elbow Curve")

#                 # Calculate cumulative variance for all components
#                 cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

#                 # Create figure
#                 fig, ax = plt.subplots(figsize=(8, 6))

#                 # Plot the full cumulative explained variance curve
#                 ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')

#                 # Set the limits for the y-axis so that the vertical lines go to the border
#                 ax.set_ylim(0, max(cumulative_variance) * 1.1)

#                 # Set the x-axis to start at 1
#                 ax.set_xlim(1, len(cumulative_variance))

#                 # Add vertical and horizontal lines at the selected n_components
#                 ax.vlines(n_components, 0, cumulative_variance[n_components - 1], color='red', linestyle='--', linewidth=2)  # Vertical line
#                 ax.hlines(cumulative_variance[n_components - 1], 0, n_components, color='red', linestyle='--', linewidth=2)  # Horizontal line

#                 # Display the value at n_components
#                 ax.text(n_components, cumulative_variance[n_components - 1], 
#                         f"{cumulative_variance[n_components - 1]:.2f}", 
#                         ha='center', va='bottom', fontsize=12, color='red')

#                 # Set labels and title
#                 ax.set_title("Explained Variance by Components")
#                 ax.set_xlabel("Number of Components")
#                 ax.set_ylabel("Cumulative Explained Variance")
#                 ax.grid(True)

#                 # Display plot
#                 st.pyplot(fig)

#     # General Transformation Handling
#     if st.button("Apply Transformation", disabled=not(naming_pattern and columns)):
#         if not columns:
#             st.warning("Please select at least one column to proceed.")
#             return

#         new_columns = {}

#         if transformation == "PCA":
#             # Special case for PCA
#             if len(columns) < 2:
#                 st.warning("PCA requires at least two numerical columns.")
#                 return

#             try:
#                 # Standardize data for PCA
#                 scaler = StandardScaler()
#                 scaled_data = scaler.fit_transform(dataframe[columns])

#                 # Apply PCA with the selected number of components
#                 pca_final = PCA(n_components=n_components)
#                 pca_data = pca_final.fit_transform(scaled_data)
#                 pca_nam = naming_pattern.format(method_applied=transformation.replace(" ", "_"))
#                 pca_df = pd.DataFrame(pca_data, columns=[f"{pca_nam}_{i+1}" for i in range(n_components)], index=dataframe.index)

#                 # Add PCA results to the new columns
#                 new_columns.update(pca_df.to_dict(orient='list'))

#                 #st.success("✨ PCA applied successfully!")
#                 #st.dataframe(pca_df, use_container_width=True)

#             except Exception as e:
#                 st.error(f"Error applying PCA: {e}")
#                 return
#         else:
#             # Handle other transformations
#             method_function = TRANSFORMATIONS.get(transformation, None)

#             if method_function:
#                 # Apply transformation to each selected column
#                 for col in columns:
#                     try:
#                         new_col_name = naming_pattern.format(initial_variable=col, method_applied=transformation.replace(" ", "_"))
#                         # Apply transformation
#                         result = method_function(dataframe[col])
#                         # Ensure result is array-like
#                         if np.isscalar(result):
#                             result = [result] * len(dataframe)  # Broadcast scalar to match DataFrame index
#                         new_columns[new_col_name] = result
#                     except Exception as e:
#                         st.error(f"Error applying transformation to column '{col}': {e}")
#                         continue

#         # Check if new_columns contains any transformations or PCA results
#         if new_columns:
#             # Create a DataFrame from the transformed columns
#             result_df = pd.DataFrame(new_columns, index=dataframe.index)
#             st.success("✨ Transformations applied successfully!")
#             st.dataframe(result_df, use_container_width=True)

#             # Update the session state with the enriched DataFrame
#             st.session_state['enriched_df'] = pd.concat([dataframe, result_df], axis=1)
#             return result_df
#         else:
#             st.warning("⚠️ No transformations were applied. Please select columns and transformations to proceed.")


# def compute_transformations_old2(dataframe):
#     st.subheader("Transformation and Aggregation")
    
#     # Define available transformations
#     TRANSFORMATIONS = {
#         "Mean": lambda x: x.mean(),
#         "Median": lambda x: x.median(),
#         "Summation": lambda x: x.sum(),
#         "Minimum Value": lambda x: x.min(),
#         "Maximum Value": lambda x: x.max(),
#         "Standard Deviation": lambda x: x.std(),
#         "Natural Logarithm Transformation": lambda x: np.log(x + 1),
#         "Min-Max Normalization": lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten(),
#         "Z-Score Standardization": lambda x: (x - x.mean()) / x.std(),
#     }

#     # Step 1: Naming Pattern
#     naming_pattern = st.text_input(
#         "Enter a naming pattern for the new variables:",
#         value="{initial_variable}_{method_applied}",
#         help="Use placeholders: `{initial_variable}` for the original column name and `{method_applied}` for the selected transformation."
#     )
    
#     # Step 2: Transformation Selection
#     transformation = st.selectbox(
#         "Choose a transformation method:",
#         options=list(TRANSFORMATIONS.keys()) + ["PCA"],
#         help="Select the mathematical, statistical, or PCA transformation to apply."
#     )
    
#     # Step 3: Column Selection
#     columns = st.multiselect(
#         "Choose columns to apply the selected transformation:",
#         options=dataframe.select_dtypes(include=np.number).columns,
#         help="Select one or more columns from the dataset."
#     )
    
#     # PCA-Specific Handling
#     if transformation == "PCA":
#         st.markdown("### PCA Settings")
#         n_components = st.slider(
#             "Select the number of principal components:",
#             min_value=1, max_value=min(len(columns), 10), value=2, step=1
#         )
        
#         show_elbow_curve = st.checkbox("Show Elbow Curve", value=True)
        
#         if st.button("Apply PCA"):
#             if len(columns) < 2:
#                 st.warning("PCA requires at least two numerical columns.")
#                 return
            
#             # Standardize data
#             scaler = StandardScaler()
#             scaled_data = scaler.fit_transform(dataframe[columns])

#             # Compute PCA
#             pca = PCA()
#             pca_result = pca.fit_transform(scaled_data)

#             # Optional: Show elbow curve
#             if show_elbow_curve:
#                 st.markdown("### PCA Elbow Curve")
#                 fig, ax = plt.subplots()
#                 ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
#                 ax.set_title("Explained Variance by Components")
#                 ax.set_xlabel("Number of Components")
#                 ax.set_ylabel("Cumulative Explained Variance")
#                 st.pyplot(fig)
            
#             # Apply PCA with the selected number of components
#             pca_final = PCA(n_components=n_components)
#             pca_data = pca_final.fit_transform(scaled_data)
#             pca_df = pd.DataFrame(pca_data, columns=[f"PCA_Component_{i+1}" for i in range(n_components)], index=dataframe.index)
            
#             st.success("✨ PCA applied successfully!")
#             st.dataframe(pca_df, use_container_width=True)
            
#             st.session_state['enriched_df'] = pd.concat([dataframe, pca_df], axis=1)
#             return pca_df

#     # General Transformation Handling
#     if st.button("Apply Transformation", disabled=not (naming_pattern and columns and transformation != "PCA")):
#         if not columns:
#             st.warning("Please select at least one column to proceed.")
#             return

#         method_function = TRANSFORMATIONS[transformation]
#         new_columns = {}

#         # Apply transformation to each selected column
#         for col in columns:
#             try:
#                 new_col_name = naming_pattern.format(initial_variable=col, method_applied=transformation.replace(" ", "_"))
#                 # Apply transformation
#                 result = method_function(dataframe[col])
#                 # Ensure result is array-like
#                 if np.isscalar(result):
#                     result = [result] * len(dataframe)  # Broadcast scalar to match DataFrame index
#                 new_columns[new_col_name] = result
#             except Exception as e:
#                 st.error(f"Error applying transformation to column '{col}': {e}")
#                 continue
        
#         # Check if new_columns contains any transformations
#         if new_columns:
#             # Create a DataFrame from the transformed columns
#             result_df = pd.DataFrame(new_columns, index=dataframe.index)
#             st.success("✨ Transformations applied successfully!")
#             st.dataframe(result_df, use_container_width=True)
#             st.session_state['enriched_df'] = pd.concat([dataframe, result_df], axis=1)
#             return result_df
#         else:
#             st.warning("⚠️ No transformations were applied. Please select columns and transformations to proceed.")

# def compute_transformations_old1(dataframe):
#     st.subheader("Transformation and Aggregation")
    
#     # Define available transformations
#     TRANSFORMATIONS = {
#         "Mean": lambda x: x.mean(),
#         "Median": lambda x: x.median(), # x.dropna().median() if pb with nan
#         "Summation": lambda x: x.sum(),
#         "Minimum Value": lambda x: x.min(),
#         "Maximum Value": lambda x: x.max(),
#         "Standard Deviation": lambda x: x.std(),
#         "Natural Logarithm Transformation": lambda x: np.log(x + 1),
#         "Min-Max Normalization": lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten(),
#         "Z-Score Standardization": lambda x: (x - x.mean()) / x.std(),
#     }

#     # Step 1: Naming Pattern
#     naming_pattern = st.text_input(
#         "Enter a naming pattern for the new variables:",
#         value="{initial_variable}_{method_applied}",
#         help="Use placeholders: `{initial_variable}` for the original column name and `{method_applied}` for the selected transformation."
#     )
    
#     # Step 2: Transformation Selection
#     transformation = st.selectbox(
#         "Choose a transformation method:",
#         options=list(TRANSFORMATIONS.keys()),
#         help="Select the mathematical or statistical transformation to apply."
#     )
    
#     # Step 3: Column Selection
#     columns = st.multiselect(
#         "Choose columns to apply the selected transformation:",
#         options=dataframe.select_dtypes(include=np.number).columns,
#         help="Select one or more columns from the dataset."
#     )
    
#     # Step 4: Submit Button
#     if st.button("Apply Transformation", disabled=not (naming_pattern and columns and transformation)):
#         if not columns:
#             st.warning("Please select at least one column to proceed.")
#             return

#         method_function = TRANSFORMATIONS[transformation]
#         new_columns = {}

#         # Apply transformation to each selected column
#         for col in columns:
#             try:
#                 new_col_name = naming_pattern.format(initial_variable=col, method_applied=transformation.replace(" ", "_"))
#                 # Apply transformation
#                 result = method_function(dataframe[col])
#                 # Ensure result is array-like
#                 if np.isscalar(result):
#                     result = [result] * len(dataframe)  # Broadcast scalar to match DataFrame index
#                 new_columns[new_col_name] = result
#             except Exception as e:
#                 st.error(f"Error applying transformation to column '{col}': {e}")
#                 continue
        
#         # Check if new_columns contains any transformations
#         if new_columns:
#             # Create a DataFrame from the transformed columns
#             result_df = pd.DataFrame(new_columns, index=dataframe.index)
            
#             # Success message
#             st.success("✨ Transformations applied successfully! ✨")
            
#             selected_columns = [col for col in new_columns]  # List of transformed column names
#             st.markdown(f"**Transformed Columns:** {', '.join(selected_columns)}")

#             # Display both original and transformed data side by side
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown("### 📊 Original Data:")
#                 st.dataframe(dataframe[columns], use_container_width=True, hide_index=True)

#             with col2:
#                 st.markdown("### 🔄 Transformed Data:")
#                 st.dataframe(result_df, use_container_width=True, hide_index=True)
           
#             st.session_state['enriched_df'] = merge_transformed_columns(dataframe, result_df, handle_duplicates="replace")
#             #st.session_state['enriched_df'] = pd.concat([dataframe, result_df], axis=1)
            
#             return result_df
#         else:
#             st.warning("⚠️ No transformations were applied. Please select columns and transformations to proceed.")


# Helper function for controlled debug printing
def debug_print(*args, debug=False):
    if debug:
        print(*args)


# def apply_imputer(data, numerical_imputation_method='mean', categorical_imputation_method='most_frequent',
#                   cat_encoder=False, num_scaler=False, remainder_columns=None, remainder_strategy='passthrough',
#                   remainder_threshold=0.8, progress_placeholder=None, debug=False):
#     """
#     Applies an imputer to a pandas DataFrame, with options for numerical and categorical imputations, 
#     and handling of remainder columns.
    
#     Parameters:
#         data (pd.DataFrame): The input data for imputation.
#         numerical_imputation_method (str): Method for imputing numerical columns ('mean', 'median', 'knn', etc.).
#         categorical_imputation_method (str): Method for imputing categorical columns ('most_frequent', 'missforest', etc.).
#         cat_encoder (bool): Whether to apply one-hot encoding to categorical columns.
#         num_scaler (bool): Whether to scale numerical columns.
#         remainder_columns (list or str): Columns to pass through without transformation ('auto' for automatic detection).
#         remainder_strategy (str): Strategy for handling remainder columns ('passthrough', 'drop', etc.).
#         remainder_threshold (float): Threshold for automatic remainder column detection.
#         debug (bool): Flag to enable debug print statements.

#     Returns:
#         pd.DataFrame: The imputed DataFrame with updated column names.
#     """
#     if not isinstance(data, pd.DataFrame):
#         raise ValueError("Input data must be a pandas DataFrame with named columns.")

#     debug_print("Starting imputation process...", debug=debug)

#     imputer, output_cols = edi.get_imputer(
#         numerical_imputation_method=numerical_imputation_method.lower(),
#         categorical_imputation_method=categorical_imputation_method.lower(),
#         cat_encoder=cat_encoder,
#         num_scaler=num_scaler,
#         data=data,
#         remainder_columns=remainder_columns,
#         remainder_strategy=remainder_strategy,
#         remainder_threshold=remainder_threshold,
#         progress_placeholder=progress_placeholder,
#         debug=debug
#     )

#     # Fit and transform the data
#     debug_print("Fitting and transforming data with the imputer...", debug=debug)
#     df_imputed = imputer.fit_transform(data)

#     # Reconstruct the DataFrame with proper column names
#     debug_print("Reconstructing the DataFrame with transformed columns...", debug=debug)
#     df_imputed = pd.DataFrame(df_imputed, columns=output_cols, index=data.index) #.apply(pd.to_numeric, errors='ignore')
    
#     for col in df_imputed.columns:
#         if df_imputed[col].dtype == 'object':
#             original_na_count = df_imputed[col].isna().sum()
#             print(f"\nProcessing column: '{col}'")
#             print(f"Original NA count: {original_na_count}")
            
#             try:
#                 # Attempt to convert to numeric, replacing ',' with '.' for floats
#                 converted_col = pd.to_numeric(df_imputed[col], errors='coerce')
                
#                 # Check if conversion introduces additional NaNs
#                 new_na_count = converted_col.isna().sum()
#                 if new_na_count > original_na_count:
#                     raise ValueError(f"Conversion introduced {new_na_count - original_na_count} additional NaN(s).")
                
#                 # Conversion successful
#                 df_imputed[col] = converted_col
#                 print(f"Column '{col}' successfully converted to numeric.")
            
#             except Exception as e:
#                 # Revert to original type
#                 df_imputed[col] = df_imputed[col].astype('object')
#                 print(f"Column '{col}' kept as object due to conversion issue: {e}")

#     # Output debug information
#     debug_print(f"Original DataFrame Columns: {data.columns.tolist()}", debug=debug)
#     debug_print(f"Columns After Imputer Processing: {output_cols}", debug=debug)

#     debug_print("Imputation process completed successfully.", debug=debug)
#     return df_imputed

# adapt mask size to transformation
def match_null_mask_to_encoded_columns(initial_null_mask, output_cols, original_cols):
    """
    Align initial null mask with output columns, including deducing one-hot encoded versions of categorical variables.
    Ensures the mask matches the order of output_cols.

    Parameters:
        initial_null_mask (pd.DataFrame): Original null mask for input data.
        output_cols (list): Ordered list of columns after imputation and encoding.
        original_cols (list): List of original columns from the input data.

    Returns:
        pd.DataFrame: Aligned null mask matching the column order in output_cols.
    """
    # Deduce categorical columns: they exist in original_cols but are split into new columns in output_cols
    cat_cols = [col for col in original_cols if col not in output_cols]

    # Initialize an empty mask with the same shape and order as output_cols
    aligned_mask = pd.DataFrame(False, index=initial_null_mask.index, columns=output_cols)
    
    for col in output_cols:
        # Check if the column corresponds to a one-hot encoded version of a categorical column
        matched_prefix = next((cat_col for cat_col in cat_cols if col.startswith(cat_col + "_")), None)
        
        if matched_prefix:
            # If matched, propagate the null mask of the original categorical column
            aligned_mask[col] = initial_null_mask[matched_prefix]
        elif col in initial_null_mask.columns:
            # If not encoded, directly use the original null mask
            aligned_mask[col] = initial_null_mask[col]
        else:
            # Default case: no match, keep as False (no nulls)
            aligned_mask[col] = False

    return aligned_mask


def apply_imputer(data, numerical_imputation_method='mean', categorical_imputation_method='most_frequent',
                  cat_encoder=False, num_scaler=False, remainder_columns=None, remainder_strategy='passthrough',
                  remainder_threshold=0.5, progress_placeholder=None, debug=False):
    """
    Applies an imputer to a pandas DataFrame, with options for numerical and categorical imputations, 
    and handling of remainder columns. Also tracks imputed values.
    
    Parameters:
        data (pd.DataFrame): The input data for imputation.
        numerical_imputation_method (str): Method for imputing numerical columns ('mean', 'median', 'knn', etc.).
        categorical_imputation_method (str): Method for imputing categorical columns ('most_frequent', 'missforest', etc.).
        cat_encoder (bool): Whether to apply one-hot encoding to categorical columns.
        num_scaler (bool): Whether to scale numerical columns.
        remainder_columns (list or str): Columns to pass through without transformation ('auto' for automatic detection).
        remainder_strategy (str): Strategy for handling remainder columns ('passthrough', 'drop', etc.).
        remainder_threshold (float): Threshold for automatic remainder column detection.
        progress_placeholder (object): Placeholder for showing progress in Streamlit.
        debug (bool): Flag to enable debug print statements.

    Returns:
        pd.DataFrame: The imputed DataFrame with updated column names.
        pd.DataFrame: A mask DataFrame indicating which values were imputed.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame with named columns.")

    debug_print("Starting imputation process...", debug=debug)

    # Record initial null values
    initial_null_mask = data.isna()
    if initial_null_mask.any().any():
        imputer, output_cols = edi.get_imputer(
            numerical_imputation_method=numerical_imputation_method.lower(),
            categorical_imputation_method=categorical_imputation_method.lower(),
            cat_encoder=cat_encoder,
            num_scaler=num_scaler,
            data=data,
            remainder_columns=remainder_columns,
            remainder_strategy=remainder_strategy,
            remainder_threshold=remainder_threshold,
            progress_placeholder=progress_placeholder,
            debug=debug
        )

        # Fit and transform the data
        debug_print("Fitting and transforming data with the imputer...", debug=debug)
        df_imputed = imputer.fit_transform(data)

        # Reconstruct the DataFrame with proper column names
        debug_print("Reconstructing the DataFrame with transformed columns...", debug=debug)
        df_imputed = pd.DataFrame(df_imputed, columns=output_cols, index=data.index)
        
        #adapt if a cat encoding output asked
        if cat_encoder:
            initial_null_mask = match_null_mask_to_encoded_columns(initial_null_mask, output_cols, data.columns.tolist())

        # Create an imputed values mask
        imputed_mask = initial_null_mask & df_imputed.notna()  # True for values that were imputed
        

        for col in df_imputed.columns:
            if df_imputed[col].dtype == 'object':
                original_na_count = df_imputed[col].isna().sum()
                debug_print(f"Processing column: '{col}'", debug=debug)
                debug_print(f"Original NA count: {original_na_count}", debug=debug)
                
                try:
                    # Attempt to convert to numeric, replacing ',' with '.' for floats
                    converted_col = pd.to_numeric(df_imputed[col], errors='coerce')
                    
                    # Check if conversion introduces additional NaNs
                    new_na_count = converted_col.isna().sum()
                    if new_na_count > original_na_count:
                        raise ValueError(f"Conversion introduced {new_na_count - original_na_count} additional NaN(s).")
                    
                    # Conversion successful
                    df_imputed[col] = converted_col
                    debug_print(f"Column '{col}' successfully converted to numeric.", debug=debug)
                
                except Exception as e:
                    # Revert to original type
                    df_imputed[col] = df_imputed[col].astype('object')
                    debug_print(f"Column '{col}' kept as object due to conversion issue: {e}", debug=debug)

        # Output debug information
        debug_print(f"Original DataFrame Columns: {data.columns.tolist()}", debug=debug)
        debug_print(f"Columns After Imputer Processing: {output_cols}", debug=debug)

        debug_print("Imputation process completed successfully.", debug=debug)
        
        return df_imputed, imputed_mask
    else:
        return data, initial_null_mask


# TBD: Rework the cat of missingforest based on set(binary + cat)
def app():
    """Improved page for external data and variable definition."""
    st.title("Data Enrichment and Variable Definition")

    # Main navigation tabs
    main_tab, impute_tab, define_var_tab = st.tabs([
    "📥 Import External Data",
    "🛠️ Handle Missing Data",
    "⚙️ Create New Variables"
    ])

    if 'data' in st.session_state and st.session_state.data is not None:
        if 'enriched_df' not in st.session_state:
            st.session_state['enriched_df'] = st.session_state['data'].copy()
    # 1. Add External Data
    with main_tab:
        with st.expander("📥 Upload External Data", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Main Dataset")
                main_data = handle_main_data_upload()
                if main_data is not None:
                    st.dataframe(main_data.head(), use_container_width=True)
                    st.info(f"Rows: {len(main_data)} | Columns: {len(main_data.columns)}")

                    selected_columns_main = st.multiselect(
                        "Select columns to retain in Main Dataset",
                        options=main_data.columns.tolist(),
                        default=main_data.columns.tolist()
                    )
                    filtered_main_data = main_data[selected_columns_main]

            with col2:
                st.subheader("Enrichment Dataset")
                enrichment_data, input_file = handle_enrichment_data_upload()
                if enrichment_data is not None:
                    st.dataframe(enrichment_data.head(), use_container_width=True)
                    st.info(f"Rows: {len(enrichment_data)} | Columns: {len(enrichment_data.columns)}")

                    selected_columns_enrichment = st.multiselect(
                        "Select columns to retain in Enrichment Dataset",
                        options=enrichment_data.columns.tolist(),
                        default=enrichment_data.columns.tolist()
                    )
                    filtered_enrichment_data = enrichment_data[selected_columns_enrichment]

        with st.expander("🔄 Enrichment Configuration", expanded=False):
            if main_data is not None and enrichment_data is not None:
                enriched_df = configure_enrichment(
                    main_df=filtered_main_data,
                    enrichment_df=filtered_enrichment_data,
                    enrichment_file_path=input_file
                )
                if enriched_df is not None:
                    st.success("Enrichment successfully configured!")
                    st.session_state['enriched_df'] = enriched_df
            else:
                st.warning("Please upload both datasets first.")


        with st.expander("📊 Results", expanded=False):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None and 'filtered_main_data' in locals():
                display_external_data_results(st.session_state['enriched_df'], filtered_main_data)
            else:
                st.warning("No enrichment results available. Please complete the enrichment process first.")

    # 2. Handle Missing Values
    with impute_tab:
        with st.expander("Configure and apply imputation to your dataset", expanded=True):
            # Parameter selection
            st.subheader("Select Imputation Parameters")
            main_data = st.session_state['enriched_df']
            numerical_imputation_method = st.selectbox(
                "Numerical Imputation Method",
                ["Mean", "Median", "KNN", "MissForest"]
            )
            categorical_imputation_method = st.selectbox(
                "Categorical Imputation Method",
                ["Most_Frequent", "MissForest"]
            )
            if categorical_imputation_method == numerical_imputation_method:
                st.info(f"{categorical_imputation_method} has been choose for both numerical and categorical columns, a unified approach will be used.")
            cat_encoder = st.checkbox("Apply One-Hot Encoding to Categorical Columns?", value=False)
            num_scaler = st.checkbox("Scale Numerical Columns?", value=False)

            remainder_option = st.radio(
                "Handle Remainder Columns",
                options=["Auto-detect", "Specify", "None"],
                index=0
            )

            remainder_columns = None
            if remainder_option == "Specify":
                remainder_columns = st.multiselect(
                    "Select Remainder Columns",
                    options=main_data.columns.tolist()
                )
            elif remainder_option == "Auto-detect":
                remainder_columns = "auto"

            remainder_strategy = st.selectbox(
                "Remainder Strategy",
                ["passthrough", "drop"]
            )
            remainder_threshold = st.slider(
                "Remainder Threshold for Auto-detection",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Filter out non-numeric columns having a ratio distinct modalities / non empty rows higher than the fixed threshold"
            )

            # Apply imputation
            if st.button("Apply Imputer"):
                try:
                    # Define a placeholder for the progress bar
                    progress_placeholder = st.empty()
                    # Call the imputer function
                    df_imputed, imputed_mask = apply_imputer(
                        data=main_data,
                        numerical_imputation_method=numerical_imputation_method,
                        categorical_imputation_method=categorical_imputation_method,
                        cat_encoder=cat_encoder,
                        num_scaler=num_scaler,
                        remainder_columns=remainder_columns,
                        remainder_strategy=remainder_strategy,
                        remainder_threshold=remainder_threshold,
                        progress_placeholder=progress_placeholder,
                        debug=True
                    )
                    if df_imputed is not None:
                        print(f"\n---------\n------\n --> Column dtypes after imputation:\n{df_imputed.dtypes}")

                        st.success("Imputation Applied Successfully!")
                        st.session_state['enriched_df'] = df_imputed

                        # Highlight imputed values
                        def highlight_imputed(data, mask):
                            """Apply highlighting to DataFrame based on imputed mask."""
                            styled = pd.DataFrame('', index=data.index, columns=data.columns)
                            styled[mask] = 'background-color: yellow'
                            return styled

                        styled_df = df_imputed.style.apply(highlight_imputed, mask=imputed_mask, axis=None)
                        st.write("Preview of Imputed Data:")
                        st.dataframe(styled_df)
                        #st.write(st.session_state['enriched_df'].info(verbose=True))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        with st.expander("📊 Results", expanded=False):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None:
                display_imputation_results(st.session_state['enriched_df'], main_data=st.session_state['working_df'])

    # 3. Define New Variables
    with define_var_tab:
        with st.expander(":abacus: Apply Common Transformations to Multiple Selected Variables", expanded=True):
            
            transformed_data = compute_transformations(st.session_state['enriched_df'])
            # If transformations were successfully applied, merge the data
            if transformed_data is not None:
                if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None:
                    # Add transformed columns to main_data
                    main_data = st.session_state['enriched_df']
                    st.success("Transformed data has been successfully added to the main dataset.")
                    st.subheader("Show Updated Dataset")
                    st.dataframe(main_data) #st.session_state['enriched_df'])
            else:
                st.warning("No transformation to apply.")
            

        with st.expander("⚙️ Variable Computation", expanded=True):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None:
                main_data = define_new_variables(main_data) #) st.session_state['enriched_df']
                #st.session_state['enriched_df'] = main_data
            else:
                st.warning("No enriched dataset")
            
        with st.expander("📊 Results", expanded=False):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None and main_data is not None:
                display_new_variables_results(main_data, main_data=st.session_state['working_df']) #st.session_state['enriched_df'] st.session_state['data']
                
                # st.dataframe(st.session_state['enriched_df'].head(), use_container_width=True)
                
                # col1, col2 = st.columns(2)
                
                # with col1:
                #     st.download_button(
                #         label="Download Dataset with New Variables",
                #         data=to_excel(st.session_state['enriched_df']),
                #         file_name="Dataset_with_New_Variables.xlsx",
                #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                #         use_container_width=True
                #     )
                # with col2:
                #     # Save to session state
                #     if st.button("Use This Dataset for Further Analysis ", use_container_width=True):
                #         st.session_state.data = st.session_state['enriched_df']
                #         st.session_state.working_df = st.session_state['enriched_df']
                #         st.success("✅ Dataset saved and ready for further analysis!")
            else:
                st.warning("No dataset available. Please upload data first.")