######################################## PACKAGES ########################################
import logging  # Added for logging
import os  # for path etc

import pandas as pd  # for Dataframes manipulation

import manage.file_handling as mf

######################################## FUNCTIONS FOR EXCEL SPECIFIC FUNCTIONALITIES ########################################

def highlight_modifications(row, dataset,common_id_only_in_df1, common_id_only_in_df2, common_id_modified, common_id=['visit_id']):
    """
    Highlight rows where 'modification' column is True, or visit_id is only in df1 or df2.
    """
    styles = [''] * len(row)  # Initialize styles list
    
    # Highlight rows where 'modification' column is True
    if row[common_id] in common_id_modified:
        styles = ['background-color: orange'] * len(row)
        
    # Highlight rows where visit_id is only in df1
    if common_id_only_in_df1 is not None and not dataset == 'df2':
        if row[common_id] in common_id_only_in_df1:
            styles = ['background-color: red'] * len(row)
    
    # Highlight rows where visit_id is only in df2
    if common_id_only_in_df2 is not None and not dataset == 'df1':        
        if row[common_id] in common_id_only_in_df2:
            styles = ['background-color: green'] * len(row)
    
    return styles


def highlight_modifications_tuples(row, dataset, common_id_only_in_df1, common_id_only_in_df2, common_id_modified, common_id=('subject_id', 'visit_id')):
    """
    Highlight rows where 'modification' column is True, or common_id is only in df1 or df2.
    """
    styles = [''] * len(row)  # Initialize styles list
    
    # Create a tuple of values from the columns in common_id
    row_id_tuple = tuple(row[col] for col in common_id)
    
    # Highlight rows where 'modification' column is True
    if row_id_tuple in common_id_modified:
        styles = ['background-color: orange'] * len(row)
        
    # Highlight rows where common_id is only in df1
    if common_id_only_in_df1 is not None and dataset != 'df2':
        if row_id_tuple in common_id_only_in_df1:
            styles = ['background-color: red'] * len(row)
    
    # Highlight rows where common_id is only in df2
    if common_id_only_in_df2 is not None and dataset != 'df1':        
        if row_id_tuple in common_id_only_in_df2:
            styles = ['background-color: green'] * len(row)
    
    return styles


# def keep_modifications(row, common_id_only_in_df1, common_id_only_in_df2, common_id='visit_id'):
#     """
#     Filter out rows where 'modification' column is False and where visit_id is not in visit_id_only_in_df1 or visit_id_only_in_df2.
#     """
#     keep_row = False
    
#     # Keep rows where 'modification' column is True
#     if row['modification']:
#         keep_row = True
    
#     # Keep rows where visit_id is only in df1
#     if common_id_only_in_df1 is not None:
#         if row[common_id] in common_id_only_in_df1:
#             keep_row = True
    
#     # Keep rows where visit_id is only in df2
#     if common_id_only_in_df2 is not None:        
#         if row[common_id] in common_id_only_in_df2:
#             keep_row = True
    
#     return keep_row


def keep_modifications(row, common_id_only_in_df1, common_id_only_in_df2, common_id=('subject_id', 'visit_id')):
    """
    Filter out rows where 'modification' column is False and where common_id (tuple) is not in 
    common_id_only_in_df1 or common_id_only_in_df2.
    """
    keep_row = False
    
    # Keep rows where 'modification' column is True
    if row['modification']:
        keep_row = True
    
    # Keep rows where common_id is only in df1
    if common_id_only_in_df1 is not None:
        if tuple(row[common_id]) in common_id_only_in_df1:
            keep_row = True
    
    # Keep rows where common_id is only in df2
    if common_id_only_in_df2 is not None:        
        if tuple(row[common_id]) in common_id_only_in_df2:
            keep_row = True
    
    return keep_row


######################################## COMPARE DATA FILES ########################################

def compare_dataframes(df1, df2, id_column, exception_list):
    """
    Compare two DataFrames based on a common row identifier and identify modifications.
    This is a part of the process of comparing two DataFrames.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame.
    - df2 (pd.DataFrame): Second DataFrame.
    - id_column (str): Column name containing the IDs for comparison.

    Returns:
    - pd.DataFrame: DataFrame with differing values between df1 and df2 identified.
    """
    try:
        # Merge df1 and df2 based on 'visit_id' using left join
        merged_df = pd.merge(df2, df1, on=id_column, suffixes=('_df2', '_df1'), how='outer', indicator=True)
        print("\n MERGED DF : \n",merged_df.head())
        # Initialize modification column with False
        merged_df['modification'] = False
        # Initialize column to store the names of differing columns
        merged_df['differing_columns'] = ''
        # Ensure id_column is a list
        if isinstance(id_column, str):
            id_column = [id_column]
        # Exclude id_column(s) from the column list
        cols_merge = [col for col in merged_df.columns if col not in id_column]
        print("\n cols_merge : \n",cols_merge) 
        #exception_list = []
        cols_to_compare = [col for col in df2.columns if col not in id_column and col in df1.columns and col not in exception_list]
        df1_not_matched_cols = [col for col in df1.columns if col not in id_column and col not in cols_to_compare and col not in exception_list]
        df2_not_matched_cols = [col for col in df2.columns if col not in id_column and col not in cols_to_compare and col not in exception_list]
        if df1_not_matched_cols or df2_not_matched_cols:
            logging.warning(f"Some unmatched columns won't be considered: IN FILE 1 ({df1_not_matched_cols}), IN FILE 2 ({df2_not_matched_cols})")
        print("\n-------------- cols_to_compare : \n",cols_to_compare)   
        print("____ DATAFRAME MERGED ____\n",merged_df.head(),"\n")
        # Iterate over rows and compare values
        for index, row in merged_df.iterrows():
            if row['_merge'] != 'both':
                continue
            # Check if any value in the row (excluding the ID column) is different between df1 and df2
            differing_columns = [
                col for col in cols_to_compare 
                if (pd.isna(row[f'{col}_df1']) != pd.isna(row[f'{col}_df2'])) or 
                   (pd.notna(row[f'{col}_df1']) and pd.notna(row[f'{col}_df2']) and row[f'{col}_df1'] != row[f'{col}_df2'])
            ]
            if differing_columns:
                print("\n ____ differing_columns : \n", differing_columns)
            differing_values = [(row[f'{col}_df1'],row[f'{col}_df2']) for col in differing_columns]
            if differing_values:
                print("\n ____ differing_values : \n", differing_values)
            if differing_columns:
                merged_df.at[index, 'modification'] = True
                merged_df.at[index, 'differing_columns'] = ','.join(differing_columns)

        return merged_df
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e!s}")
        return None


def process_comparison(comparison_name, comparison_info, config, exclude_cols=None): # config,
    """
    Process the comparison of two dataframes based on a common row identifier.
    This is a part of the process of comparing two DataFrames.

    Parameters:
        comparison_name (str): The name of the comparison.
        comparison_info (): Information about the comparison including input files, output file, folder paths, and common row identifier.

    Returns:
        A tuple containing the results of the comparison, including DataFrame 1, DataFrame 2, common row identifier name, filtered comparison result, 
        rows only in DataFrame 1, rows only in DataFrame 2, modified common row identifiers, and the output file path. Returns None if an error occurs.

    Raises:
        None
    """
    try:
        INPUT_FILES_FOLDER = comparison_info["INPUT_FILES_FOLDER"]
        input_file_1 = os.path.join(INPUT_FILES_FOLDER, comparison_info["INPUT_FILE_1"])
        input_file_2 = os.path.join(INPUT_FILES_FOLDER, comparison_info["INPUT_FILE_2"])
        comparison_folder = config.get("folder_names", {}).get("COMPARISONS_FOLDER")
        output_file = os.path.join(comparison_folder, comparison_info["OUTPUT_FILE"])
        common_id_name = comparison_info["COMMON_ROW_IDENTIFIER"]
        if not os.path.exists(comparison_folder):
            os.makedirs(comparison_folder)

        logging.info(f"Processing outliers file: {input_file_1}")
        df1 = mf.load_dataframe(input_file_1)
        logging.info(f"Processing outliers file: {input_file_2}")
        df2 = mf.load_dataframe(input_file_2)

        if exclude_cols is not None:
            cols_to_drop_df1 = [col for col in exclude_cols if col in df1.columns]
            cols_to_drop_df2 = [col for col in exclude_cols if col in df2.columns]
            df1.drop(columns=cols_to_drop_df1, inplace=True)
            df2.drop(columns=cols_to_drop_df2, inplace=True)

        common_id_df1 = set(df1[common_id_name])
        common_id_df2 = set(df2[common_id_name])

        common_id_only_in_df1 = list(common_id_df1 - common_id_df2)
        common_id_only_in_df2 = list(common_id_df2 - common_id_df1)

        comparison_result = compare_dataframes(df1, df2, common_id_name, exclude_cols)
        comparison_result_filtered = comparison_result[comparison_result.apply(lambda row: keep_modifications(row, common_id_only_in_df1, common_id_only_in_df2,common_id_name), axis=1)]
        common_id_modified = set(comparison_result_filtered.loc[comparison_result_filtered['modification'], common_id_name]) # deprecated form : set(comparison_result_filtered[comparison_result_filtered['modification'] == True] [common_id_name])

        return df1, df2, common_id_name, comparison_result_filtered, common_id_only_in_df1, common_id_only_in_df2, common_id_modified, output_file
    except Exception as e:
        logging.error(f"An unexpected error occurred for comparison: {comparison_name}. Error: {e!s}")
        return None

def process_comparisons_from_config(config, exclude_cols_stats):
    """
    Process requested comparisons from JSON config files and write results to Excel.

    Parameters:
    - config (dict): JSON configuration containing comparison details.
    - exclude_cols_stats (list): List of columns to exclude from statistics or processing.

    Returns:
    - None
    """
    comparisons = config.get("comparisons", {})
    
    if comparisons:
        for comparison, comparison_info in comparisons.items():
            logging.info(f"Processing comparison: {comparison}")
            
            # Call the method from your package to process the comparison
            df1, df2, common_id, comparison_result_filtered, common_id_only_in_df1, common_id_only_in_df2, common_id_modified, output_file = process_comparison(comparison, comparison_info, config, exclude_cols_stats)
            
            # Prepare Excel writer
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Write df1 to Excel
                styled_df1 = df1.style.apply(highlight_modifications, args=("df1", common_id_only_in_df1, common_id_only_in_df2, common_id_modified, common_id), axis=1)
                styled_df1.to_excel(writer, sheet_name=f"1_{comparison_info.get('INPUT_FILE_1')[:29]}", index=False)  # Limit sheet name length due to Excel constraints
                
                # Write df2 to Excel
                styled_df2 = df2.style.apply(highlight_modifications, args=("df2", common_id_only_in_df1, common_id_only_in_df2, common_id_modified, common_id), axis=1)
                styled_df2.to_excel(writer, sheet_name=f"2_{comparison_info.get('INPUT_FILE_2')[:29]}", index=False)  # Limit sheet name length due to Excel constraints
                
                # Write comparison_result_filtered to Excel
                styled_df = comparison_result_filtered.style.apply(highlight_modifications, args=("summary", common_id_only_in_df1, common_id_only_in_df2, common_id_modified, common_id), axis=1)
                styled_df.to_excel(writer, sheet_name='summary', index=False)
            
            logging.info(f"Comparison {comparison} has been successfully processed!")
    else:
        print("No comparison request")


def process_comparison_st(comparison_name, df1, df2, common_id=None, common_cols=None, output_file="comparison.xlsx"):
    """
    Compare two DataFrames based on a common row identifier and return comparison results.

    Parameters:
        comparison_name (str): Name of the comparison task.
        df1 (dataframe):
        df2 (dataframe):
        common_id (list): 
        common_cols (list, optional): Columns to exclude from both DataFrames. Defaults to None.

    Returns:
        tuple: (df1, df2, common_id, comparison_result_filtered, 
                rows_only_in_df1, rows_only_in_df2, modified_common_ids, output_file)

    Notes:
        - Handles missing folders and file paths.
        - Logs steps and handles errors gracefully.
    """
    try:
        # Ensure id_column is a list
        if isinstance(common_id, str):
            common_id = [common_id]
        
        # Identify common and unique row identifiers
        common_id_df1 = set(df1[common_id].dropna().apply(tuple, axis=1).unique())
        common_id_df2 = set(df2[common_id].dropna().apply(tuple, axis=1).unique())

        print(f"\n ------> common_id_df1 {common_id_df1}\n ------> common_id_df2 {common_id_df2}")

        exclude_cols = [col for col in df1.columns.union(df2.columns) if col not in common_cols + common_id]
        print(f"\n ------> exclude_cols {exclude_cols}")
        # Exclude specified columns (if provided)
        if exclude_cols:
            logging.info(f"Excluding columns: {exclude_cols}")
            df1.drop(columns=[col for col in exclude_cols if col in df1.columns], inplace=True)
            df2.drop(columns=[col for col in exclude_cols if col in df2.columns], inplace=True)

        rows_only_in_df1 = list(common_id_df1.difference(common_id_df2))
        rows_only_in_df2 = list(common_id_df2.difference(common_id_df1))


        print(f"\nRows only in DataFrame 1: {len(rows_only_in_df1)}")
        print(f"\nRows only in DataFrame 2: {len(rows_only_in_df2)}")

        # Perform comparison
        comparison_result = compare_dataframes(df1, df2, common_id, exclude_cols)
        logging.info("DataFrames comparison complete.")

        # Filter comparison results to keep modifications
        comparison_result_filtered = comparison_result[
            comparison_result.apply(
                lambda row: keep_modifications(row, rows_only_in_df1, rows_only_in_df2, common_id),
                axis=1
            )
        ]


        # Identify modified common row identifiers
        modified_common_ids = list(
            tuple(row) for _, row in 
            comparison_result_filtered.loc[
                comparison_result_filtered['modification'] == True, 
                common_id
            ].iterrows()
        )

        comparison_result_filtered['deletion'] = comparison_result_filtered[common_id].apply(lambda row: tuple(row) in rows_only_in_df1, axis=1)
        comparison_result_filtered['addition'] = comparison_result_filtered[common_id].apply(lambda row: tuple(row) in rows_only_in_df2, axis=1)

        # Return all relevant results
        print(f"Comparison for '{comparison_name}' successfully processed.")
        return df1, df2, common_id, comparison_result_filtered, rows_only_in_df1, rows_only_in_df2, modified_common_ids, output_file

    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
        return None
    except KeyError as key_error:
        logging.error(f"Missing key in comparison_info: {key_error}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred for '{comparison_name}': {e!s}")
        return None
