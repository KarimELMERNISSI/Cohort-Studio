"""
Single imputation of the session data (moved from the Enrich page, no Streamlit).

``apply_imputer`` fills every missing value of a table (mean, median, KNN, chained equations, missForest);
``impute_rows`` fills only the rows of the analysis population with an imputer fitted on those rows, so
excluded patients neither inform nor receive imputed values; ``fill_with_formula`` is the targeted
imputation of one column. The replay engine calls the same functions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import enrich.data_imputation as edi
from enrich.formulas import evaluate_formula_safely

logger = logging.getLogger(__name__)


# Helper for controlled debug output (column names and counts only, never values: audit E7)
def debug_print(*args, debug=False):
    if debug:
        logger.debug(" ".join(str(a) for a in args))

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


@dataclass(frozen=True)
class ImputationOutcome:
    frame: pd.DataFrame  # the whole table: imputed rows filled, the others untouched
    params: dict[str, Any]
    columns: tuple[str, ...]  # columns in which at least one value was imputed
    mask: pd.DataFrame = field(repr=False)  # True where a value was imputed (rows imputed only)


def impute_rows(
    frame: pd.DataFrame,
    rows: Any = None,
    *,
    numerical: str = "Median",
    categorical: str = "Most_Frequent",
    cat_encoder: bool = False,
    num_scaler: bool = False,
    remainder_columns: Any = "auto",
    remainder_strategy: str = "passthrough",
    remainder_threshold: float = 0.3,
    progress_placeholder: Any = None,
) -> ImputationOutcome:
    """Impute the missing values of ``rows`` (row ids; every row when None) with an imputer fitted on them.

    Rows outside ``rows`` are returned untouched. One-hot encoding, scaling and dropping remainder columns
    change the columns of the table; they are refused when only part of the rows is imputed.
    """
    index = frame.index if rows is None else pd.Index(rows)
    partial = len(index) != len(frame)
    # with only part of the rows imputed, the other rows would be left in another encoding or scale
    if partial and (cat_encoder or num_scaler or remainder_strategy == "drop"):
        raise ValueError(
            "One-hot encoding, scaling and dropping remainder columns change the columns of the table: "
            "they can only be used when every row is imputed."
        )
    subset = frame.loc[index]
    imputed, mask = apply_imputer(
        data=subset,
        numerical_imputation_method=numerical,
        categorical_imputation_method=categorical,
        cat_encoder=cat_encoder,
        num_scaler=num_scaler,
        remainder_columns=remainder_columns,
        remainder_strategy=remainder_strategy,
        remainder_threshold=remainder_threshold,
        progress_placeholder=progress_placeholder,
    )
    if not partial:
        out = imputed
    else:
        out = frame.copy()
        for col in imputed.columns:
            if col not in out.columns or col not in mask.columns or not bool(mask[col].any()):
                continue
            try:
                out.loc[index, col] = imputed[col]
            except (TypeError, ValueError):  # e.g. a mean imputed into a nullable integer column
                out[col] = out[col].astype(imputed[col].dtype)
                out.loc[index, col] = imputed[col]
    columns = tuple(str(c) for c in mask.columns if bool(mask[c].any()))
    params = {
        "numerical_imputation_method": numerical,
        "categorical_imputation_method": categorical,
        "cat_encoder": cat_encoder,
        "num_scaler": num_scaler,
        "remainder_columns": remainder_columns,
        "remainder_strategy": remainder_strategy,
        "remainder_threshold": remainder_threshold,
        "rows": "population" if partial else "all",
        "n_rows_fitted": int(len(index)),
        "imputed_values": {c: int(mask[c].sum()) for c in columns},
    }
    return ImputationOutcome(frame=out, params=params, columns=columns, mask=mask)


def fill_with_formula(
    frame: pd.DataFrame, target: str, formula: str, rows: pd.Index | None = None
) -> tuple[pd.DataFrame, int]:
    """
    Fill the missing values of ``target`` with a formula of the other columns; returns (table, filled). With ``rows``
    (the analysis population), the formula is computed on those rows only, so that an aggregate such as a mean sees no
    excluded patient, and fills them only: the excluded rows keep their missing values, as with the global imputation.
    The page and the replay both fill through here.
    """
    scope = frame if rows is None else frame.loc[frame.index.intersection(rows)]
    ok, result, _ = evaluate_formula_safely(formula, target, scope, return_series=True)
    if not ok:
        raise ValueError(f"formula for {target}: {result}")
    out = frame.copy()
    before = int(out[target].isna().sum())
    # missing outside the scope: fillna leaves those rows as they are (the whole column, so its type rules apply)
    out[target] = out[target].fillna(pd.Series(result, index=scope.index).reindex(out.index))
    return out, before - int(out[target].isna().sum())


def describe_imputation(numerical: str, categorical: str, n_rows: int | None = None) -> str:
    """The trace's description of a single imputation; ``n_rows``: the analysis population it was fitted on."""
    text = f"Applied data imputation ({numerical} / {categorical}"
    return f"{text}, fitted on the analysis population of {n_rows:,} rows)" if n_rows is not None else f"{text})"


def describe_formula_fill(target: str, filled: int, *, exact: bool, n_rows: int | None = None) -> str:
    """The trace's description of a targeted imputation; ``n_rows``: the analysis population it was computed on."""
    values = "value" if filled == 1 else "values"
    text = f"Imputed {filled} {values} in {target} using {'an exact formula' if exact else 'a formula'}"
    return f"{text}, computed on the analysis population of {n_rows:,} rows" if n_rows is not None else text


__all__ = [
    "ImputationOutcome",
    "apply_imputer",
    "describe_formula_fill",
    "describe_imputation",
    "fill_with_formula",
    "impute_rows",
]
