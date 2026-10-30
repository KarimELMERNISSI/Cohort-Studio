import logging  # Added for logging
import os
from dataclasses import dataclass

import pandas as pd  # for Dataframes manipulation

from core.fingerprint import ROW_ID, extend_row_ids
from ingest.dates import AS_DATES, CHOSEN, detect, read_dates, smart_parse_dates
from ingest.loaders import read_path

logger = logging.getLogger(__name__)

# How many rows of the file a row of the dataset takes in a merge on keys, recorded with the step: "all" repeats the
# row once per matching row of the file (a long table), "first" and "last" keep one, in the file's order, "unique"
# expects one and refuses a file that repeats a key. A step recorded before had none: "all".
FILE_ROWS = ("all", "unique", "first", "last")
# Whether a missing key matches another, recorded with the step: "never" (a row without an identifier takes nothing
# from the file, and a row of the file without one is no one's), or "match", as pandas merges them, which a step
# recorded before did.
MISSING_KEYS = ("never", "match")
_KEYED = ("left", "right", "outer", "inner")


@dataclass(frozen=True)
class KeyCheck:
    """What a merge on keys would do to the rows of the dataset, before it is done (missing keys never match)."""

    file_keys_repeated: int  # keys that more than one row of the file holds
    rows_repeated: int  # rows of the dataset whose key more than one row of the file holds
    rows_added: int  # the rows the merge would add by repeating them, one per extra row of the file
    rows_without_key: int = 0  # rows of the dataset with a missing key: they take nothing from the file

    @property
    def repeats(self) -> bool:
        return self.rows_repeated > 0


def key_check(df, additional_df, left_id_names, right_id_names) -> KeyCheck | None:
    """
    How a merge on these keys would repeat the rows of the dataset: a row whose key more than one row of the file
    holds comes out once per such row. A missing key matches nothing (``missing_keys="never"``, as Enrich merges
    now). None when the keys cannot be compared (a column missing, or of another type): the merge says why.
    """
    left = [left_id_names] if isinstance(left_id_names, str) else list(left_id_names or [])
    right = [right_id_names] if isinstance(right_id_names, str) else list(right_id_names or [])
    if not left or len(left) != len(right):
        return None
    try:
        complete = df[left].notna().all(axis=1)
        per_key = additional_df.dropna(subset=right).groupby(right, sort=False).size()
        repeated = per_key[per_key > 1].rename("__matches").reset_index()
        if repeated.empty:
            return KeyCheck(0, 0, 0, int((~complete).sum()))
        keys = df.loc[complete, left].set_axis(right, axis=1)
        matches = keys.merge(repeated, on=right, how="left")["__matches"].dropna()
    except (KeyError, TypeError, ValueError):
        return None
    return KeyCheck(len(repeated), int(len(matches)), int((matches - 1).sum()), int((~complete).sum()))


def _unmatchable(values: pd.Series, others: pd.Series):
    """A value of ``values``' kind that ``others`` and ``values`` never hold, for a missing key to match nothing."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        known = pd.concat([pd.to_numeric(values, errors="coerce"), pd.to_numeric(others, errors="coerce")])
        top = known.max()
        return float(top) + 1.0 if pd.notna(top) else 0.0
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.Timestamp("1677-09-22")  # before any date a cohort holds
    taken = set(values.dropna().astype(str)) | set(others.dropna().astype(str))
    marker = "\x00no identifier"
    while marker in taken:
        marker += "\x00"
    return marker


# the order of day and month each recorded format reads; the others (ISO, Excel serials) read none
_DAY_FIRST = {"day_first": True, "dmy_compact": True, "month_first": False, "ymd_compact": False}


def date_key_formats(main_df, additional_df, recorded, *, dayfirst=False):
    """
    How the Enrich page reads the file's columns named as a date column of the main table, so that date keys match:
    the file's date cells as they are ("dates"); a column of text with the format the shape of its own values gives
    (ingest.dates.detect: an ISO export reads as ISO whatever the dataset's), the order of day and month, where its
    values do not show it, that of the main column's recorded format (``recorded``, from
    :func:`ingest.dates.recorded_formats`), else ``dayfirst`` (the import's setting). A column whose values have no
    date shape, Excel serial numbers, and two-digit years without the main column's pivot are left out: read with
    smart_parse_dates, as before. Recorded with the step: the replay reads them, never detects again.
    """
    found = {}
    for col in additional_df.columns:
        if col not in main_df.columns or not pd.api.types.is_datetime64_any_dtype(main_df[col]):
            continue
        values = additional_df[col]
        if pd.api.types.is_datetime64_any_dtype(values):
            if getattr(values.dt, "tz", None) is None:
                found[str(col)] = {"format": AS_DATES}
            continue
        mine = (recorded or {}).get(str(col)) or {}
        how = detect(values, dayfirst=_DAY_FIRST.get(mine.get("format"), dayfirst), name=str(col))
        if how.format in (None, "excel_serial") or how.shape < CHOSEN * how.values:
            continue
        read = {"format": how.format}
        if how.two_digit_years:
            if mine.get("pivot") is None:
                continue
            read["pivot"] = int(mine["pivot"])
        found[str(col)] = read
    return found


def align_date_keys(main_df, additional_df, formats=None):
    """
    The additional table with its columns parsed as dates where the main table holds dates under the same name,
    so that date keys match in the merge: each with its format in ``formats`` (from :func:`date_key_formats` on the
    page, from the step on replay); a column without one, and every column of a step recorded before (``formats``
    None), with smart_parse_dates, day first by position, as before. The formats applied are in
    ``attrs["date_formats"]``, recorded with the step. The Enrich page and the replay of the step both call it, on the
    same tables, so they merge the same rows.
    """
    aligned = additional_df.copy()
    aligned.attrs = dict(additional_df.attrs)
    applied = {}
    for col in aligned.columns:
        if col in main_df.columns and pd.api.types.is_datetime64_any_dtype(main_df[col]):
            how = (formats or {}).get(str(col))
            if how is None:
                aligned[col], _ = smart_parse_dates(aligned[col])
                continue
            applied[str(col)] = dict(how)
            aligned[col] = read_dates(aligned[col], str(how["format"]), pivot=how.get("pivot"))
    aligned.attrs["date_formats"] = applied
    return aligned


def _order_dates(rows, order_by) -> pd.Series:
    """The dates that order a file's rows for "first" or "last": its column, read with the format recorded for it."""
    column = order_by.get("column")
    if column not in rows.columns:
        raise ValueError(f"order_by names no column of the file: {column!r}")
    if order_by.get("format"):
        return read_dates(rows[column], str(order_by["format"]), pivot=order_by.get("pivot"))
    if not pd.api.types.is_datetime64_any_dtype(rows[column]):
        raise ValueError(f"{column!r} holds no dates: the format it is written in is needed")
    return rows[column]


def _unmatched(left, right, left_ids, right_ids) -> int:
    """Rows of the dataset whose identifier no row of the file holds, the rows without one included."""
    if not left_ids or len(left_ids) != len(right_ids):
        return 0
    keys = right[right_ids].drop_duplicates().set_axis(left_ids, axis=1)
    flag = "_found"
    while flag in left_ids:  # an identifier of that name: the indicator takes another
        flag += "_"
    try:
        found = left[left_ids].merge(keys, how="left", on=left_ids, indicator=flag)[flag].eq("both")
    except (ValueError, TypeError):  # identifiers pandas cannot compare: the merge itself said so
        return 0
    return int((~found).sum())


def add_data(df, additional_df, left_id_names, right_id_names, additional_cols=None, strategy='left', conflict_resolution=None, file_rows="all", missing_keys="match", order_by=None):
    """
    Enrich the dataset by either merging on common identifiers (columns) or appending rows.

    Rows of ``df`` keep their row id (``__row_id``, the index) through the merge; rows added by an outer or
    right join, appended rows and repeats of a row (one-to-many join) get fresh ids. The merged frame's
    ``attrs["join_counts"]`` says how many rows came from the main dataset, were added, dropped or repeated.
    ``file_rows`` (:data:`FILE_ROWS`) says how many rows of the file a row of the dataset takes in a merge on keys;
    ``missing_keys`` (:data:`MISSING_KEYS`) whether a missing key matches another ("match", pandas' way, kept for
    the steps recorded before; Enrich merges with "never"). ``order_by`` ({"column", "format", "pivot"}): with "first"
    or "last", a participant's rows of the file are taken in the order of that column's dates, the earliest or the
    latest (rows on the same date in the file's order; a row without a date only when none has one), not in the file's
    order; "format" reads a text column (ingest.dates), None a column of dates. Recorded with the step; a step
    recorded before (None) took the file's order. Same parameters and result as :func:`_merge`.
    """
    if file_rows not in FILE_ROWS:
        raise ValueError(f"file_rows is one of {', '.join(FILE_ROWS)}, not {file_rows!r}")
    if missing_keys not in MISSING_KEYS:
        raise ValueError(f"missing_keys is one of {', '.join(MISSING_KEYS)}, not {missing_keys!r}")
    carries_ids = df.index.name == ROW_ID
    left = df.reset_index() if carries_ids else df
    right = additional_df.drop(columns=[ROW_ID], errors="ignore")  # ids of another table mean nothing here
    on_keys = str(strategy).lower() in _KEYED
    keyed = on_keys and file_rows != "all"
    left_ids = [left_id_names] if isinstance(left_id_names, str) else list(left_id_names or [])
    right_ids = [right_id_names] if isinstance(right_id_names, str) else list(right_id_names or [])
    never = on_keys and missing_keys == "never" and bool(left_ids) and len(left_ids) == len(right_ids)
    identified = right[right_ids].notna().all(axis=1) if never else pd.Series(True, index=right.index)
    file_rows_left_out = 0
    if keyed and file_rows in ("first", "last"):  # one row per identifier; rows without one are no one's
        candidates = right[identified]
        if order_by:  # the earliest or the latest date first or last; undated rows behind the dated ones
            when = _order_dates(candidates, order_by).reset_index(drop=True)
            undated = "last" if file_rows == "first" else "first"
            candidates = candidates.iloc[when.sort_values(kind="stable", na_position=undated).index]
        kept = pd.concat([candidates.drop_duplicates(subset=right_ids, keep=file_rows), right[~identified]])
        kept = kept.loc[right.index.intersection(kept.index)]  # in the file's order
        file_rows_left_out, right = len(right) - len(kept), kept
        identified = identified.loc[kept.index]
    validate = "many_to_one" if keyed else None
    without_key = 0
    restored: dict[str, tuple[object, object]] = {}  # key column -> (its marker, its dtype)
    if never:
        if keyed and file_rows == "unique" and right.loc[identified, right_ids].duplicated().any():
            logger.error("The merge failed (an identifier repeated in the file)")  # the keys are patient data
            return df
        validate = None  # checked above on the identified rows: rows without an identifier match no one
        missing = left[left_ids].isna().any(axis=1)
        without_key = int(missing.sum())
        if without_key:
            left = left.copy()
            for lcol, rcol in zip(left_ids, right_ids, strict=True):
                dtype = left[lcol].dtype
                if isinstance(dtype, pd.CategoricalDtype):  # a category cannot take the marker: merged as text
                    left[lcol] = left[lcol].astype(object)
                marker = _unmatchable(left[lcol], right[rcol])
                restored[lcol] = (marker, dtype)
                left.loc[missing & left[lcol].isna(), lcol] = marker
    combined = _merge(left, right, left_id_names, right_id_names, additional_cols, strategy, conflict_resolution,
                      validate=validate)
    if combined is left:  # the merge failed: the main dataset is returned unchanged
        return df
    # rows of the dataset no row of the file matched (an inner or right merge leaves them out: dropped as well)
    unmatched = _unmatched(left, right, left_ids, right_ids) if on_keys else 0
    for col, (marker, dtype) in restored.items():  # the missing keys back, in their own kind
        if col in combined.columns:
            combined[col] = combined[col].mask(combined[col] == marker)
            if isinstance(dtype, pd.CategoricalDtype):
                combined[col] = combined[col].astype(dtype)
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
        "main_rows_without_key": without_key,
        "main_rows_unmatched": max(unmatched - without_key, 0),
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
