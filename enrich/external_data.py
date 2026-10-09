import logging  # Added for logging
import os

import pandas as pd  # for Dataframes manipulation

import manage.file_handling as mf

# def add_data(df, additional_df, left_id_names, right_id_names, additional_cols=None, strategy='left', conflict_resolution=None):
#     """
#     Enrich the dataset based on a common row identifier or multiple identifiers.
#     This is a part of the process of augmenting the main dataset with external data.

#     Parameters:
#         df (DataFrame): The dataset to augment.
#         additional_df (DataFrame): Information about the additional data including input files, folder paths, and common row identifier(s).
#         additional_cols (list, optional): List of columns to include from the additional data. If None, all columns are included.
#         strategy (str, optional): Merge strategy. Defaults to 'left'.
#         conflict_resolution (str, optional): Choice of how to handle column name conflicts. Can be 'replace', 'add', or 'ignore'. Defaults to 'add' if None.

#     Returns:
#         DataFrame: The dataset df enriched with data from the additional data file. Returns the original df if an error occurs.
#     """
#     try:
#         # Set default conflict_resolution if None and convert to lowercase for case-insensitive handling
#         if conflict_resolution is None:
#             conflict_resolution = 'add'
#         conflict_resolution = conflict_resolution.lower()

#         # Convert strategy to lowercase for case-insensitive handling
#         strategy = strategy.lower()
        
#         # Validate merge strategy
#         valid_strategies = {'left', 'right', 'outer', 'inner', 'cross'}
#         if strategy not in valid_strategies:
#             raise ValueError(f"Invalid merge strategy: {strategy}. Expected one of {valid_strategies}")

#         # Ensure additional_df is not empty
#         if additional_df.empty:
#             logging.warning("Additional DataFrame is empty. Returning the original DataFrame.")
#             return df

#         # Filter columns if additional_cols is provided
#         if additional_cols is not None:
#             cols_to_keep = set(additional_cols).union(set(right_id_names))
#             additional_df = additional_df[list(cols_to_keep)]

#         # Determine suffixes based on conflict_resolution
#         if conflict_resolution == 'replace':
#             suffixes = ('', '_drop')
#         elif conflict_resolution == 'ignore':
#             suffixes = ('_drop', '')
#         else:  # conflict_resolution == 'add'
#             suffixes = ('', '_add')
        
#         # Debug information
#         print("MAIN DF:\n", df.head())
#         print("ADD DF:\n", additional_df.head())
#         print(left_id_names, right_id_names)

#         # Merge the dataframes
#         augmented_df = pd.merge(left=df, right=additional_df, left_on=left_id_names, right_on=right_id_names, suffixes=suffixes, how=strategy)
        
#         # Drop unnecessary columns based on conflict_resolution
#         if conflict_resolution in {'replace', 'ignore'}:
#             augmented_df = augmented_df.drop([col for col in augmented_df.columns if col.endswith('_drop')], axis=1)

#         return augmented_df
    
#     except KeyError as e:
#         logging.error(f"Key error: {e}")
#         return df
#     except ValueError as e:
#         logging.error(f"Value error: {e}")
#         return df
#     except Exception as e:
#         logging.error(f"An unexpected error occurred during data enrichment: {str(e)}")
#         return df


def add_data(df, additional_df, left_id_names, right_id_names, additional_cols=None, strategy='left', conflict_resolution=None):
    """
    Enrich the dataset by either merging on common identifiers (columns) or appending rows.

    Parameters:
        df (DataFrame): The main dataset to augment.
        additional_df (DataFrame): The additional dataset to merge or append.
        left_id_names (list): List of identifier column(s) in the main dataset.
        right_id_names (list): List of identifier column(s) in the additional dataset.
        additional_cols (list, optional): List of columns to include from the additional dataset. If None, all columns are included.
        strategy (str, optional): Enrichment strategy. Options are 'left', 'right', 'outer', 'inner', 'cross' for column merges, or 'rows' for row append. Defaults to 'left'.
        conflict_resolution (str, optional): Handle row conflicts based on identifiers with 'replace', 'keep', or 'ignore'. Defaults to 'keep' if None.

    Returns:
        DataFrame: The enriched DataFrame or the original if an error occurs.
    """
    try:
        # Set default conflict_resolution if not provided
        if conflict_resolution is None:
            conflict_resolution = 'keep'
        conflict_resolution = conflict_resolution.lower()

        # Convert strategy to lowercase for consistency
        strategy = strategy.lower()
        valid_merge_strategies = {'left', 'right', 'outer', 'inner', 'cross'}
        
        # Verbose debug info
        print(f"Selected Strategy: {strategy}")
        print(f"Conflict Resolution: {conflict_resolution}")
        
        print("Checking column names in DataFrames...")
        print("\nMain DataFrame columns:", df.columns.tolist())
        print("\nAdditional DataFrame columns:", additional_df.columns.tolist())
        
        print("\nChecking for IDs...")
        print("Expected Left ID columns:", left_id_names)
        print("Expected Right ID columns:", right_id_names)
        if strategy == 'rows':
            # Filter to only the required columns in `additional_df`
            if additional_cols is not None:
                additional_df = additional_df[right_id_names+additional_cols]

            # Find rows in `df` with identifiers matching `additional_df`
            merged_identifiers_df = df[left_id_names].merge(
                additional_df[right_id_names],
                left_on=left_id_names,
                right_on=right_id_names,
                how='inner'
            )

            # Print matching identifiers for debugging
            print("Matching identifiers for conflict resolution:\n", merged_identifiers_df)

            

            if conflict_resolution == 'replace':
                # Preserve original index type for restoration if needed, but here we reset eventually.
                
                # prepare additional_df with matching column names
                rename_map = dict(zip(right_id_names, left_id_names))
                additional_df_aligned = additional_df.rename(columns=rename_map)
                
                # set index to identifiers for alignment
                # We need to ensure we don't lose data. 
                # df.update() matches on Index and Columns.
                
                # Create copies to avoid SettingWithCopy warnings and modify safely
                df_indexed = df.set_index(left_id_names)
                additional_indexed = additional_df_aligned.set_index(left_id_names)
                
                # 1. Update existing rows
                # update() modifies in-place and ignores NaNs in the source (additional_indexed)
                # It updates values in df_indexed where indices overlap.
                df_indexed.update(additional_indexed)
                
                # 2. Identify and append new rows (rows in additional not in df)
                # usage of difference depends on index type. MultiIndex vs Index.
                new_ids = additional_indexed.index.difference(df_indexed.index)
                new_rows = additional_indexed.loc[new_ids]
                
                # Concatenate updated existing data and new data
                combined_df = pd.concat([df_indexed, new_rows])
                
                # Reset index to restore identifier columns
                combined_df = combined_df.reset_index()
                
                print("Merged with update (upsert) strategy. Preserved columns.")
                
                
            elif conflict_resolution == 'ignore':
                # Step 1: Find unmatched identifiers in `df` as tuples
                unmatched_identifiers = additional_df[~additional_df[right_id_names].apply(tuple, axis=1).isin(merged_identifiers_df[left_id_names].apply(tuple, axis=1))][left_id_names]
                unmatched_identifiers = unmatched_identifiers.dropna(subset=right_id_names).apply(tuple, axis=1).tolist()

                # Step 2: Create a mask by checking each row against `unmatched_identifiers`
                unmatched_identifiers_mask = additional_df[right_id_names].apply(tuple, axis=1).isin(unmatched_identifiers)

                # Display unmatched identifiers for debugging
                print("Unmatched identifiers:", unmatched_identifiers)
                print("Mask for unmatched identifiers:\n", unmatched_identifiers_mask, unmatched_identifiers_mask.sum())
                additional_df_excluding_conflicts = additional_df[unmatched_identifiers_mask]
                print("Rows in additional DataFrame after excluding conflicts:\n", additional_df_excluding_conflicts)
                combined_df = pd.concat([df, additional_df_excluding_conflicts], ignore_index=True)
                
            else:  # Default or 'keep'
                # Directly append without removing duplicates
                combined_df = pd.concat([df, additional_df], ignore_index=True)
                print("Appending all rows, duplicates allowed.")

            # Debugging final result for rows strategy
            print("Final DataFrame after row appending:\n", combined_df)

        elif strategy in valid_merge_strategies:
            # Column-based merging strategy
            if additional_df.empty:
                logging.warning("Additional DataFrame is empty. Returning the original DataFrame.")
                return df

            # Filter columns if additional_cols is specified
            if additional_cols is not None:
                cols_to_keep = set(additional_cols).union(set(right_id_names))
                additional_df = additional_df[list(cols_to_keep)]

            # Determine suffixes based on conflict_resolution
            if conflict_resolution == 'replace':
                suffixes = ('', '_drop')
            elif conflict_resolution == 'ignore':
                suffixes = ('_drop', '')
            else:  # Default or 'keep'
                suffixes = ('', '_add')

            # Perform merge with column-based strategy
            combined_df = pd.merge(df, additional_df, left_on=left_id_names, right_on=right_id_names, suffixes=suffixes, how=strategy)
            print("DataFrame after column-based merging:\n", combined_df)

            # Remove unnecessary columns based on conflict_resolution
            if conflict_resolution in {'replace', 'ignore'}:
                combined_df = combined_df.drop([col for col in combined_df.columns if col.endswith('_drop')], axis=1)

        else:
            raise ValueError(f"Invalid strategy: {strategy}. Expected one of {valid_merge_strategies} or 'rows'.")

        return combined_df

    except KeyError as e:
        logging.error(f"Key error: {e}")
        return df
    except ValueError as e:
        logging.error(f"Value error: {e}")
        return df
    except Exception as e:
        logging.error(f"An unexpected error occurred during data enrichment: {e!s}")
        return df



def process_data_enrichment(df, additional_data_info):
    """
    Process requested comparisons from JSON config files and write results to Excel.

    Parameters:
    - df (DataFrame): The dataset to augment.
    - additional_data_info (dict): JSON configuration containing merging details.

    Returns:
    - DataFrame: The dataset df enriched with data from the additional data file. Returns the original df if an error occurs.
    """
    try:
        if additional_data_info:
            input_files_folder =  additional_data_info.get("INPUT_FILES_FOLDER")
            input_file = os.path.join(input_files_folder, additional_data_info.get("INPUT_FILE"))
            additional_cols = additional_data_info.get("ADDITIONAL_COLUMNS")
            common_id_names = additional_data_info.get("COMMON_ROW_IDENTIFIER")
            strategy = additional_data_info.get("STRATEGY")
            conflict_resolution = additional_data_info.get("CONFLICT_RESOLUTION")

            if common_id_names:
                # Ensure common_id_names is a list
                if isinstance(common_id_names, list):
                    left_id_names = common_id_names
                    right_id_names = common_id_names
                else:
                    left_id_names = [common_id_names]
                    right_id_names = [common_id_names]
            else:
                left_id_names = additional_data_info.get("LEFT_ROW_IDENTIFIER")
                right_id_names = additional_data_info.get("RIGHT_ROW_IDENTIFIER")
                if not isinstance(left_id_names, list):
                    left_id_names = [left_id_names]
                if not isinstance(right_id_names, list):
                    right_id_names = [right_id_names]

            logging.info(f"Processing additional data file: {input_file}")
            additional_df = mf.load_dataframe(input_file)

            if not additional_df.empty:
                enriched_df = add_data(df=df, additional_df=additional_df, left_id_names=left_id_names, right_id_names=right_id_names, additional_cols=additional_cols, strategy=strategy, conflict_resolution=conflict_resolution)
            else:
                enriched_df = df
            return enriched_df

        else:
            print("No enrichment request")
    except FileNotFoundError:
        logging.error(f"File not found: {input_file}")
        return df
    except KeyError as e:
        logging.error(f"Key error: {e}")
        return df
    except Exception as e:
        logging.error(f"An unexpected error occurred during data enrichment: {additional_data_info}. Error: {e!s}")
        return df


def process_data_enrichments_from_config(df, data_file_name, config):
    """
    Process requested data enrichments from JSON config files and write results to Excel.

    Parameters:
    - df (DataFrame): The dataset to augment.
    - config (dict): JSON configuration containing comparison details.

    Returns:
    - DataFrame: The dataset df enriched with data from all the additional data files. Returns the original df if an error occurs.
    """
    data_enrichments = config.get("data_enrichments", {})
    
    if data_enrichments:
        for data_enrichment, data_enrichment_info in data_enrichments.items():
            logging.info(f"Processing data enrichment: {data_enrichment}")
            df = process_data_enrichment(df, data_enrichment_info)
            
            logging.info(f"Data enrichment {data_enrichment} has been successfully processed!")
        directory = config.get("folder_names", {}).get("ENRICHED_FOLDER")
        filename = f"enriched_{data_file_name}.xlsx"
        df.to_excel(os.path.join(directory, filename))
        return df
    else:
        print("No enrichment request")
        return df