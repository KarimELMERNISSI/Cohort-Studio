"""
Statistics Utilities.

Functions for statistical analysis including normality tests.
"""
import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
import streamlit as st
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
        
        elif method == 'ks':
            clean_col = column.dropna()
            std_val = float(np.std(clean_col))
            if std_val == 0 or len(clean_col) == 0:
                return np.nan
            standardized = (clean_col - float(np.mean(clean_col))) / std_val
            stat, p_value = stats.kstest(standardized, 'norm')
        
        elif method == 'anderson':
            result = stats.anderson(column.dropna(), dist='norm')
            for i, critical_value in enumerate(result.critical_values):
                if result.statistic > critical_value:
                    return result.significance_level[i] / 100.0
            return result.significance_level[-1] / 100.0
        
        else:
            raise ValueError("Unknown method. Choose 'shapiro', 'dagostino', 'ks', or 'anderson'.")
        
        return round(p_value, 4)
    
    except Exception:
        return np.nan


def show_test_guidelines():
    """
    Displays a detailed explanation of the normality tests,
    including algorithms, formulas, and recommendations.
    """
    show_help = st.checkbox("Show Normality Test Selection Guide and Detailed Algorithm Explanations")

    if show_help:
        st.write("### Help Menu")
        st.write("Select an option to learn more about normality tests.")

        display_option = st.radio(
            "Choose what to display:", 
            ["Normality Test Overview and Selection Guide", 
             "Detailed Explanation of a Specific Algorithm"]
        )

        if display_option == "Normality Test Overview and Selection Guide":
            _show_overview_guide()
        else:
            _show_algorithm_details()


def _show_overview_guide():
    """Display normality test overview and selection guide."""
    st.write("### Normality Test Overview and Selection Guide")
    st.write("""
    Select the appropriate normality test based on your dataset's sample size 
    and its distribution properties.
    """)

    st.write("### Normality Test Selection Table")
    data = {
        "Test Name": ["Shapiro-Wilk", "D'Agostino-Pearson", "Anderson-Darling", "Kolmogorov-Smirnov"],
        "Sample Size": ["3 < n < 50", "50 <= n <= 1000", "n > 5", "n > 50"],
        "Characteristics": [
            "Most powerful for small datasets, detects deviations from normality",
            "Effective for moderate datasets with skewness and kurtosis",
            "Sensitive to tail deviations, good for large datasets",
            "Compares sample distribution with a known distribution"
        ],
    }
    st.table(data)

    st.write("### Recommendations")
    st.write("""
    - **Small Samples (3 < n < 50)**: Use **Shapiro-Wilk** test.
    - **Moderate Samples (50 <= n <= 1000)**: Use **D'Agostino-Pearson** or **Anderson-Darling**.
    - **Large Samples (n > 1000)**: Use **Kolmogorov-Smirnov** or **Anderson-Darling**.
    """)


def _show_algorithm_details():
    """Display detailed algorithm explanations."""
    st.write("### Detailed Algorithm Explanation")
    
    test_choice = st.selectbox(
        "Choose a normality test:", 
        ["Shapiro-Wilk", "D'Agostino-Pearson", "Anderson-Darling", "Kolmogorov-Smirnov"]
    )
    
    detail_level = st.radio("Select Level of Detail:", ["Brief", "Detailed"])

    if test_choice == "Shapiro-Wilk":
        st.write("#### Shapiro-Wilk Test")
        if detail_level == "Brief":
            st.write("""
            - **Best for**: Small to moderate datasets (3 < n < 5000).
            - **Characteristics**: Most powerful test, highly sensitive to deviations.
            """)
        else:
            st.latex(r"W = \frac{\left( \sum_{i=1}^n a_i x_i \right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
            st.write("""
            - **W**: Test statistic measuring deviation from normality.
            - **a_i**: Constants derived from sample size.
            - Reject H₀ if p-value < 0.05.
            """)

    elif test_choice == "D'Agostino-Pearson":
        st.write("#### D'Agostino-Pearson Test")
        if detail_level == "Brief":
            st.write("""
            - **Best for**: Moderate-sized datasets (50 <= n <= 1000).
            - **Characteristics**: Detects skewness and kurtosis.
            """)
        else:
            st.latex(r"Z = \frac{\gamma_1}{\sigma_{\gamma_1}} + \frac{\gamma_2}{\sigma_{\gamma_2}}")
            st.write("""
            - Combines skewness (γ₁) and kurtosis (γ₂) measures.
            - Best when skewness or kurtosis is noticeable.
            """)

    elif test_choice == "Anderson-Darling":
        st.write("#### Anderson-Darling Test")
        if detail_level == "Brief":
            st.write("""
            - **Best for**: Large datasets (n > 5).
            - **Characteristics**: Sensitive to distribution tails.
            """)
        else:
            st.latex(r"A^2 = -n - S_n")
            st.write("""
            - Compares EDF with normal CDF.
            - Effective for tail deviation detection.
            """)

    elif test_choice == "Kolmogorov-Smirnov":
        st.write("#### Kolmogorov-Smirnov Test")
        if detail_level == "Brief":
            st.write("""
            - **Best for**: Large datasets (n > 50).
            - **Characteristics**: Compares sample to normal distribution.
            """)
        else:
            st.latex(r"D = \sup_x |F_n(x) - F(x)|")
            st.write("""
            - D measures max difference between EDF and normal CDF.
            - Ideal for comparing against known distributions.
            """)
