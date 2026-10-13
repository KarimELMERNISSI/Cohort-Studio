"""
Statistics Utilities.

Functions for statistical analysis including normality tests.
"""
import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats


def sanitize_dataframe_for_arrow(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Sanitize a pandas DataFrame so it can be converted to PyArrow without ArrowInvalid errors.

    Ensures columns containing mixed types (e.g. numeric values mixed with Timestamp or Timedelta objects)
    are safely converted to uniform types (such as strings), while preserving purely numeric and boolean dtypes.

    Args:
        df: Input pandas DataFrame to sanitize.

    Returns:
        pd.DataFrame | None: Sanitized DataFrame guaranteed to be compatible with PyArrow and Streamlit.
    """
    if df is None or getattr(df, "empty", False):
        return df

    clean_df = df.copy()

    for col in clean_df.columns:
        series = clean_df[col]
        # Only object dtype columns can hold problematic heterogeneous Python objects
        if series.dtype == "object":
            # Check for temporal or timedelta objects
            has_temporal = series.apply(
                lambda x: isinstance(x, (pd.Timestamp, pd.Timedelta, datetime.datetime, datetime.date, np.datetime64))
            ).any()
            if has_temporal:
                clean_df[col] = series.apply(lambda x: str(x) if pd.notna(x) else None).astype(str)
            else:
                # Attempt to convert to numeric if all non-null values are numeric
                try:
                    clean_df[col] = pd.to_numeric(series, errors="raise")
                except (ValueError, TypeError):
                    pass

    # Fail-safe validation against pyarrow
    try:
        pa.Table.from_pandas(clean_df)
    except Exception:
        for col in clean_df.columns:
            try:
                pa.array(clean_df[col], from_pandas=True)
            except Exception:
                clean_df[col] = clean_df[col].astype(str)

    return clean_df


def normality_test(column, method='dagostino'):
    """
    Perform specified normality test on a column and return the p-value.
    
    Args:
        column: pandas Series with numeric data
        method: One of 'shapiro', 'dagostino', 'ks', or 'anderson'
        
    Returns:
        float: p-value or np.nan if test fails
    """
    if column.isnull().all():
        return np.nan
    
    try:
        if method == 'shapiro':
            stat, p_value = stats.shapiro(column.dropna())
        
        elif method == 'dagostino':
            stat, p_value = stats.normaltest(column.dropna())
        
        elif method in ('ks', 'lilliefors'):
            # Lilliefors: KS with estimated mean/SD uses its own null distribution (audit S12)
            from biostats.normality import lilliefors_p

            p_value = lilliefors_p(column.dropna())
            if np.isnan(p_value):
                return np.nan

        elif method == 'anderson':
            from biostats.normality import anderson_p

            p_value = anderson_p(column.dropna())
            if np.isnan(p_value):
                return np.nan

        else:
            raise ValueError("Unknown method. Choose 'shapiro', 'dagostino', 'ks', or 'anderson'.")
        
        return round(p_value, 4)
    
    except Exception:
        return np.nan


def show_test_guidelines():
    """Compatibility shim: the guide widgets live in :mod:`app_pages.statistics_guides`."""
    from app_pages.statistics_guides import show_test_guidelines as _show

    return _show()
