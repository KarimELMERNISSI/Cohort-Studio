import logging  # Added for logging
import os
from dataclasses import dataclass

import pandas as pd  # for Dataframes manipulation

from core.fingerprint import ROW_ID, extend_row_ids
from ingest.dates import smart_parse_dates
from ingest.loaders import read_path

logger = logging.getLogger(__name__)

# How many rows of the file a row of the dataset takes in a merge on keys, recorded with the step: "all" repeats the
# row once per matching row of the file (a long table), "first" and "last" keep one, in the file's order, "unique"
# expects one and refuses a file that repeats a key. A step recorded before had none: "all".
FILE_ROWS = ("all", "unique", "first", "last")
_KEYED = ("left", "right", "outer", "inner")


@dataclass(frozen=True)
class KeyCheck:
    """What a merge on keys would do to the rows of the dataset, before it is done."""

    file_keys_repeated: int  # keys that more than one row of the file holds
    rows_repeated: int  # rows of the dataset whose key more than one row of the file holds
    rows_added: int  # the rows the merge would add by repeating them, one per extra row of the file

    @property
    def repeats(self) -> bool:
        return self.rows_repeated > 0


def key_check(df, additional_df, left_id_names, right_id_names) -> KeyCheck | None:
    """
    How a merge on these keys would repeat the rows of the dataset: a row whose key more than one row of the file
    holds comes out once per such row. Missing keys match one another, as pandas merges them. None when the keys
    cannot be compared (a column missing, or of another type): the merge says why.
    """
    left = [left_id_names] if isinstance(left_id_names, str) else list(left_id_names or [])
    right = [right_id_names] if isinstance(right_id_names, str) else list(right_id_names or [])
    if not left or len(left) != len(right):
        return None
    try:
        per_key = additional_df.groupby(right, dropna=False, sort=False).size()
        repeated = per_key[per_key > 1].rename("__matches").reset_index()
        if repeated.empty:
            return KeyCheck(0, 0, 0)
        keys = df[left].set_axis(right, axis=1)
        matches = keys.merge(repeated, on=right, how="left")["__matches"].dropna()
    except (KeyError, TypeError, ValueError):
        return None
    return KeyCheck(len(repeated), int(len(matches)), int((matches - 1).sum()))


def align_date_keys(main_df, additional_df):
    """
    The additional table with its columns parsed as dates where the main table holds dates under the same name,
    so that date keys match in the merge. The Enrich page and the replay of the step both call it, on the same
    tables, so they merge the same rows.
    """
    aligned = additional_df.copy()
    aligned.attrs = dict(additional_df.attrs)
    for col in aligned.columns:
        if col in main_df.columns and pd.api.types.is_datetime64_any_dtype(main_df[col]):
            aligned[col], _ = smart_parse_dates(aligned[col])
    return aligned


def add_data(df, additional_df, left_id_names, right_id_names, additional_cols=None, strategy='left', conflict_resolution=None, file_rows="all"):
    """
    Enrich the dataset by either merging on common identifiers (columns) or appending rows.

    Rows of ``df`` keep their row id (``__row_id``, the index) through the merge; rows added by an outer or
    right join, appended rows and repeats of a row (one-to-many join) get fresh ids. The merged frame's
    ``attrs["join_counts"]`` says how many rows came from the main dataset, were added, dropped or repeated.
    ``file_rows`` (:data:`FILE_ROWS`) says how many rows of the file a row of the dataset takes in a merge on keys.
    Same parameters and result as :func:`_merge`.
    """
    if file_rows not in FILE_ROWS:
        raise ValueError(f"file_rows is one of {', '.join(FILE_ROWS)}, not {file_rows!r}")
    carries_ids = df.index.name == ROW_ID
    left = df.reset_index() if carries_ids else df
    right = additional_df.drop(columns=[ROW_ID], errors="ignore")  # ids of another table mean nothing here
    keyed = str(strategy).lower() in _KEYED and file_rows != "all"
    file_rows_left_out = 0
    if keyed and file_rows in ("first", "last"):
        right_ids = [right_id_names] if isinstance(right_id_names, str) else list(right_id_names)
        kept = right.drop_duplicates(subset=right_ids, keep=file_rows)
        file_rows_left_out, right = len(right) - len(kept), kept
    combined = _merge(left, right, left_id_names, right_id_names, additional_cols, strategy, conflict_resolution,
                      validate="many_to_one" if keyed else None)
    if combined is left:  # the merge failed: the main dataset is returned unchanged
        return df
    if not carries_ids or ROW_ID not in combined.columns:
        return combined
    ids = pd.to_numeric(combined[ROW_ID], errors="coerce")
    counts = {
        "rows_before": int(len(df)),
        "rows_after": int(len(combined)),
        "rows_from_main": int(ids.notna().sum()),
        "rows_added": int(ids.isna().sum()),
        "main_rows_dropped": int(len(df) - ids.dropna().nunique()),
        "main_rows_repeated": int(ids.dropna().duplicated().sum()),
        "file_rows": file_rows,
        "file_rows_left_out": int(file_rows_left_out),
    }
    combined = extend_row_ids(combined, used=df.index)
    combined.attrs["join_counts"] = counts
    logger.debug("Enrichment: %s", counts)
    return combined


def _merge(df, additional_df, left_id_names, right_id_names, additional_cols=None, strategy='left', conflict_resolution=None, validate=None):
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
        validate (str, optional): pandas' check of a column merge ("many_to_one": one row of the file per key); a file
            that breaks it is refused, and the main dataset returned unchanged.

    Returns:
        DataFrame: The enriched DataFrame or the original if an error occurs.
    """
    try:
        # Normalize single string identifiers to lists
        if isinstance(left_id_names, str):
            left_id_names = [left_id_names]
        if isinstance(right_id_names, str):
            right_id_names = [right_id_names]

        # Set default conflict_resolution if not provided
        if conflict_resolution is None:
            conflict_resolution = 'keep'
        conflict_resolution = conflict_resolution.lower()

        # Convert strategy to lowercase for consistency
        strategy = strategy.lower()
        valid_merge_strategies = {'left', 'right', 'outer', 'inner', 'cross'}
        
        # Names and sizes only: values are patient data and never go to stdout or the logs (audit E7)
        logger.debug(
            "Enrichment %s/%s: main %s, additional %s, ids %s / %s",
            strategy, conflict_resolution, df.shape, additional_df.shape, left_id_names, right_id_names,
        )
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
            logger.debug("Matching identifiers for conflict resolution: %d rows", len(merged_identifiers_df))

            

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
                

                
                
            elif conflict_resolution == 'ignore':
                # Step 1: Find unmatched identifiers in `df` as tuples
                unmatched_identifiers = additional_df[~additional_df[right_id_names].apply(tuple, axis=1).isin(merged_identifiers_df[left_id_names].apply(tuple, axis=1))][left_id_names]
                unmatched_identifiers = unmatched_identifiers.dropna(subset=right_id_names).apply(tuple, axis=1).tolist()

                # Step 2: Create a mask by checking each row against `unmatched_identifiers`
                unmatched_identifiers_mask = additional_df[right_id_names].apply(tuple, axis=1).isin(unmatched_identifiers)

                # Display unmatched identifiers for debugging
                logger.debug("Unmatched identifiers: %d", len(unmatched_identifiers))
                additional_df_excluding_conflicts = additional_df[unmatched_identifiers_mask]
                logger.debug("Rows kept from additional data after excluding conflicts: %d", len(additional_df_excluding_conflicts))
                combined_df = pd.concat([df, additional_df_excluding_conflicts], ignore_index=True)
                
            else:  # Default or 'keep'
                # Directly append without removing duplicates
                combined_df = pd.concat([df, additional_df], ignore_index=True)


            logger.debug("Rows appended: %s", combined_df.shape)

        elif strategy in valid_merge_strategies:
            # Column-based merging strategy
            if additional_df.empty:
                logging.warning("Additional DataFrame is empty. Returning the original DataFrame.")
                return df

            # Filter columns if additional_cols is specified, in the additional table's order: the order of a set
            # of names changes with Python's hash seed, so the merged columns would change order on replay
            if additional_cols is not None:
                cols_to_keep = set(additional_cols).union(set(right_id_names))
                additional_df = additional_df[[c for c in additional_df.columns if c in cols_to_keep]]

            # Determine suffixes based on conflict_resolution
            if conflict_resolution == 'replace':
                suffixes = ('', '_drop')
            elif conflict_resolution == 'ignore':
                suffixes = ('_drop', '')
            else:  # Default or 'keep'
                suffixes = ('', '_add')

            # Perform merge with column-based strategy
            combined_df = pd.merge(df, additional_df, left_on=left_id_names, right_on=right_id_names, suffixes=suffixes, how=strategy, validate=validate)
            logger.debug("Columns merged: %s", combined_df.shape)

            # Remove unnecessary columns based on conflict_resolution
            if conflict_resolution in {'replace', 'ignore'}:
                combined_df = combined_df.drop([col for col in combined_df.columns if col.endswith('_drop')], axis=1)

        else:
            raise ValueError(f"Invalid strategy: {strategy}. Expected one of {valid_merge_strategies} or 'rows'.")

        return combined_df

    except KeyError as e:
        logging.error(f"Key error: {e}")  # a column's name
        return df
    except Exception as e:
        # the type only: pandas' message quotes values (the keys a many-to-one merge finds repeated), patient data
        logging.error("The merge failed (%s)", type(e).__name__)
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
            additional_df = read_path(input_file).df

            if not additional_df.empty:
                enriched_df = add_data(df=df, additional_df=additional_df, left_id_names=left_id_names, right_id_names=right_id_names, additional_cols=additional_cols, strategy=strategy, conflict_resolution=conflict_resolution)
            else:
                enriched_df = df
            return enriched_df

        else:
            print("No enrichment request")
            return df  # unchanged: the caller goes on with it
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
