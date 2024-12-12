# pages/home.py
import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np
import explore.corr_matrix as ecm
from scipy import stats


class RenameColumnsComponent:
    """A Streamlit component for renaming DataFrame columns with sorting, searching, type filtering, and top values display."""

    def __init__(self, df, analyzer):
        self.original_df = df  # Preserve original DataFrame for reference
        self.analyzer = analyzer

        # Initialize rename_dict and working_df in session state if they don't exist
        if "rename_dict" not in st.session_state:
            st.session_state["rename_dict"] = {col: col for col in df.columns}
        if "working_df" not in st.session_state:
            st.session_state["working_df"] = df.copy()

    def display(self):
        """Displays the column renaming UI with sorting, searching, type filtering, and a compact layout."""
        st.subheader("Rename Columns")

        # Layout for search, sorting, and type filter
        col1, col2, col3 = st.columns([1, 2, 2])

        with col1:
            sort_order = st.checkbox("Sort alphabetically", value=True, key="sort_order")
        with col2:
            type_filter = st.selectbox(
                "Filter by type",
                options=["All", "Numeric", "Categorical", "Date", "Binary", "Non-Binary Low-Cardinality Numeric", "High-Cardinality Categorical"],
                key="type_filter"
            )
        with col3:
            search_term = st.text_input("Search columns", "", key="search_term")


        # Refresh the analyzer with the updated DataFrame
        self.analyzer.refresh(st.session_state["working_df"])
        # Map the filter to the appropriate columns
        type_map = {
            "All": st.session_state["working_df"].columns, #self.original_df.columns,
            "Numeric": self.analyzer.numeric_cols,
            "Categorical": self.analyzer.categorical_cols,
            "Date": self.analyzer.date_cols,
            "Binary": self.analyzer.binary_cols,
            "Non-Binary Low-Cardinality Numeric": self.analyzer.non_binary_low_cardinality_numeric_cols,
            "High-Cardinality Categorical": self.analyzer.high_cardinality_cat_cols
        }
        columns = type_map[type_filter]

        # Sort and filter columns
        if sort_order:
            columns = sorted(columns)
        if search_term:
            columns = [col for col in columns if search_term.lower() in col.lower()]

        # Display renaming interface
        st.write("Rename columns by entering new names. Top 5 values are shown where applicable:")
        for col in columns:
            with st.container():
                col_current, col_rename, col_info = st.columns([1, 2, 3])

                with col_current:
                    st.markdown(f"**{col}**")  # Display original column name

                with col_rename:
                    new_name = st.text_input(f"Rename '{col}' to:", value=st.session_state["rename_dict"].get(col, col), key=f"rename_{col}")
                    st.session_state["rename_dict"][col] = new_name

                with col_info:
                    if col in st.session_state["working_df"].columns: 
                        if col in self.analyzer.categorical_cols or col in self.analyzer.low_cardinality_numeric_cols:
                            top_values = st.session_state["working_df"][col].value_counts().head(5)
                            top_values_str = ", ".join([f"{index} ({count})" for index, count in top_values.items()])
                            st.markdown(f"<div style='padding:5px; color: #4A4A4A;'>{top_values_str}</div>", unsafe_allow_html=True)
                        elif col in self.analyzer.numeric_cols:
                            # Compute statistics for numerical columns
                            col_data = st.session_state["working_df"][col].dropna()  # Exclude NaN values
                            median = col_data.median()
                            q1 = col_data.quantile(0.25)
                            q3 = col_data.quantile(0.75)
                            col_min = col_data.min()
                            col_max = col_data.max()

                            # Format all statistics in a single line
                            stats_str = (
                                f"<div style='padding:5px; color: #4A4A4A;'>"
                                f"<strong>Median[25%; 75%]:</strong> {median:.2f}[{q1:.2f}; {q3:.2f}]<br>"
                                f"<strong>Min:</strong> {col_min:.2f}, "
                                f"<strong>Max:</strong> {col_max:.2f}"
                                f"</div>"
                            )
                            st.markdown(stats_str, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='padding:5px; color: #A9A9A9;'>N/A</div>", unsafe_allow_html=True)
                    else:
                        st.warning(f"Column '{col}' is missing in the DataFrame.")

                # Styling for separation
                st.markdown("<hr style='margin: 5px 0; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

        # Apply renaming and refresh analyzer
        if st.button("Apply Renaming"):
            rename_dict = st.session_state["rename_dict"]
            if rename_dict != {col: col for col in self.original_df.columns}:
                # Apply renaming to working_df
                st.session_state["working_df"] = st.session_state["working_df"].rename(columns=rename_dict)
                st.success("Columns renamed successfully!")
                self.analyzer.refresh(st.session_state["working_df"])
            else:
                st.info("No changes made to column names.")

        st.write("Updated DataFrame:")
        st.dataframe(st.session_state["working_df"])
        

        return st.session_state["working_df"]

class DataAnalyzer:
    """Helper class to analyze and validate data columns"""
    def __init__(self, df):
        #self.df = df
        # Coerce object columns to numeric if possible
        self.df = df #.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtypes == 'object' else col)
        self.analyze_columns()
    
    def refresh(self, df):
        """Refresh the column metadata based on the latest DataFrame state."""
        self.df = df #.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtypes == 'object' else col)
        self.analyze_columns()
    
    def analyze_columns(self):
        """Analyze and categorize columns by data type"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = self._detect_date_columns()
        self.binary_cols, self.num_binary_cols, self.non_num_binary_cols = self._detect_binary_columns()
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
        
        # Print column states for debugging
        print(f"\n############################################################\nNumeric columns: {self.numeric_cols}")
        print(f"\nCategorical columns: {self.categorical_cols}")
        print(f"\nDate columns: {self.date_cols}")
        print(f"\nBinary columns: {self.binary_cols}")
        print(f"\nLow-cardinality numeric columns: {self.low_cardinality_numeric_cols}")
        print(f"\nHigh-cardinality categorical columns: {self.high_cardinality_cat_cols}\n############################################################")
    
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
        return [col for col in self.df.columns if self.df[col].nunique() == 2],[col for col in self.df.select_dtypes(include=[np.number]).columns if self.df[col].nunique() == 2],[col for col in self.df.select_dtypes(exclude=[np.number]).columns if self.df[col].nunique() == 2]
    
    def _detect_low_cardinality_numeric_columns(self):
        """Detect numeric columns with fewer or eq to 5 unique values to treat as categorical"""
        return [col for col in self.numeric_cols if self.df[col].nunique() <= 10]
    
    def _detect_high_cardinality_cat_columns(self):
        """Detect categorical columns with many unique values"""
        return [col for col in self.categorical_cols 
                if self.df[col].nunique() > 0.5 * len(self.df)]
    
    
# Convert DataFrame to Excel for download using openpyxl
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
    return output.getvalue()

def to_excel_sheets(dataframes_dict):
    """
    Converts multiple DataFrames into an Excel file with separate sheets.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            print(f"\n-----> sheet_name {sheet_name}")
            if type(sheet_name) != 'str':
                sheet_name = str(sheet_name)
            df.to_excel(writer, sheet_name=sheet_name[:27], index=True)  # Excel sheet names max length: 31
    return output.getvalue()


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


def move_columns_to_front(df, columns_to_move):
    """
    Move specified columns to the beginning of the DataFrame.

    Parameters:
    df (pd.DataFrame): The original DataFrame.
    columns_to_move (list): List of column names to move to the front.

    Returns:
    pd.DataFrame: DataFrame with specified columns moved to the front.
    """
    # Ensure only existing columns are included
    columns_to_move = [col for col in columns_to_move if col in df.columns]
    
    # Get the remaining columns
    remaining_columns = [col for col in df.columns if col not in columns_to_move]
    
    # Reorder the DataFrame
    return df[columns_to_move + remaining_columns]


def show_test_guidelines():
    """
    Displays a detailed explanation of the normality tests, including algorithms, formulas, and recommendations.
    """
    # Help menu checkbox with informative label
    show_help = st.checkbox("Show Normality Test Selection Guide and Detailed Algorithm Explanations")

    if show_help:
        st.write("### Help Menu")
        st.write("Select an option to learn more about normality tests.")

        # Select what to display
        display_option = st.radio("Choose what to display:", 
                                  ["Normality Test Overview and Selection Guide", 
                                   "Detailed Explanation of a Specific Algorithm"])

        if display_option == "Normality Test Overview and Selection Guide":
            st.write("### Normality Test Overview and Selection Guide")
            st.write("""
            Select the appropriate normality test based on your dataset's sample size and its distribution properties. 
            Each test has specific advantages depending on the data size and the characteristics of the distribution.
            """)

            # Table for selecting method based on sample size
            st.write("### Normality Test Selection Table")
            data = {
                "Test Name": ["Shapiro-Wilk", "D'Agostino-Pearson", "Anderson-Darling", "Kolmogorov-Smirnov"],
                "Sample Size": ["3 < n < 50", "50 <= n <= 1000", "n > 5", "n > 50"],
                "Characteristics": [
                    "Most powerful for small datasets, detects deviations from normality",
                    "Effective for moderate datasets with skewness and kurtosis",
                    "Sensitive to tail deviations, good for large datasets",
                    "Compares sample distribution with a known distribution, works well with large datasets"
                ],
                "Notes": [
                    "Most powerful for small datasets, sensitive to all deviations from normality",
                    "Useful for detecting skewness and kurtosis, requires moderate sample sizes",
                    "Effective for datasets with heavy tails, highly sensitive to the distribution's tails",
                    "Compares sample with a reference normal distribution, best for large datasets"
                ]
            }
            st.table(data)

            st.write("### Recommendations Based on Sample Size")
            st.write("""
            - **Small Samples (3 < n < 50)**: The **Shapiro-Wilk** test is the most powerful and recommended for small datasets.
            - **Moderate Samples (50 <= n <= 1000)**: The **D'Agostino-Pearson** test detects skewness and kurtosis. The **Anderson-Darling** test is also a strong choice, especially for its sensitivity to tail deviations.
            - **Large Samples (n > 1000)**: The **Kolmogorov-Smirnov** test is suitable for comparing the sample distribution to a normal distribution. The **Anderson-Darling** test remains valuable for evaluating tail behavior in large datasets.
            """)

            st.write("### Conclusion")
            st.write("""
            Selecting the appropriate normality test is crucial to understanding your dataset. Use the table and the recommendations above to choose the most suitable test based on sample size and distribution characteristics.
            """)

        elif display_option == "Detailed Explanation of a Specific Algorithm":
            st.write("### Detailed Explanation of a Specific Algorithm")
            st.write("Select a normality test to learn more about its methodology, when to use it, and the detailed algorithm.")

            # Select normality test
            test_choice = st.selectbox("Choose a normality test:", 
                                       ["Shapiro-Wilk", "D'Agostino-Pearson", "Anderson-Darling", "Kolmogorov-Smirnov"])

            # Select level of detail
            detail_level = st.radio("Select Level of Detail:", ["Brief", "Detailed"])

            if test_choice == "Shapiro-Wilk":
                st.write("#### Shapiro-Wilk Test")
                if detail_level == "Brief":
                    st.write("""
                    - **Best for**: Small to moderate datasets (3 < n < 5000).
                    - **Characteristics**: Most powerful test for small datasets, highly sensitive to deviations from normality.
                    - **When to Use**: Best for small to moderate datasets (3 < n < 5000).
                    """)
                else:
                    st.latex(r"""
                    W = \frac{\left( \sum_{i=1}^n a_i x_i \right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
                    """)
                    st.write("""
                    - **Variables**:
                        - \( W \): Test statistic measuring the deviation of the sample from normality.
                        - \( a_i \): Constants derived from the sample size, used to weight the ordered sample values.
                        - \( x_i \): Ordered sample values.
                        - \( \bar{x} \): Sample mean.
                    - **Algorithm**:
                        1. Calculate the test statistic \( W \).
                        2. Compare \( W \) with a known distribution to determine if the data is normally distributed.
                        3. Reject \( H_0 \) if the p-value is below the threshold (typically 0.05).
                    - **When to Use**: The Shapiro-Wilk test is best for small to moderate datasets (3 < n < 5000), as it is the most powerful for detecting normality.
                    """)

            elif test_choice == "D'Agostino-Pearson":
                st.write("#### D'Agostino-Pearson Test")
                if detail_level == "Brief":
                    st.write("""
                    - **Best for**: Moderate-sized datasets (50 <= n <= 1000).
                    - **Characteristics**: Detects skewness and kurtosis.
                    - **When to Use**: Best for moderate sample sizes when there is significant skewness or kurtosis.
                    """)
                else:
                    st.latex(r"""
                    Z = \frac{\gamma_1}{\sigma_{\gamma_1}} + \frac{\gamma_2}{\sigma_{\gamma_2}}
                    """)
                    st.write("""
                    - **Variables**:
                        - \( Z \): Combined statistic for assessing deviations from normality.
                        - \( \gamma_1 \): Skewness measure.
                        - \( \gamma_2 \): Kurtosis measure.
                        - \( \sigma_{\gamma_1} \): Standard error of skewness.
                        - \( \sigma_{\gamma_2} \): Standard error of kurtosis.
                    - **Algorithm**:
                        1. Calculate skewness (\( \gamma_1 \)) and kurtosis (\( \gamma_2 \)).
                        2. Compute the combined statistic \( Z \).
                        3. Use the p-value to assess normality.
                    - **When to Use**: Best for moderate-sized datasets (50 <= n <= 1000), particularly when skewness or kurtosis is noticeable.
                    """)

            elif test_choice == "Anderson-Darling":
                st.write("#### Anderson-Darling Test")
                if detail_level == "Brief":
                    st.latex("""
                    - **Best for**: Large datasets (n > 5).
                    - **Characteristics**: Sensitive to the tails of the distribution.
                    - **When to Use**: Particularly useful for large datasets where tail behavior is important.
                    """)
                else:
                    st.latex(r"""
                    A^2 = -n - S_n
                    """)
                    st.latex("""
                    - **Variables**:
                        - \( A^2 \): Test statistic for measuring deviation from normality.
                        - \( n \): Sample size.
                        - \( S_n \): Weighted sum of squared differences between the empirical distribution function (EDF) and the CDF of the normal distribution.
                    - **Algorithm**:
                        1. Compare the EDF with the normal distribution's CDF.
                        2. Compute the statistic \( A^2 \).
                        3. Use the p-value to assess normality.
                    - **When to Use**: Effective for large datasets and datasets where tail deviations are important.
                    """)

            elif test_choice == "Kolmogorov-Smirnov":
                st.write("#### Kolmogorov-Smirnov Test")
                if detail_level == "Brief":
                    st.latex("""
                    - **Best for**: Large datasets (n > 50).
                    - **Characteristics**: Compares sample distribution to normal distribution.
                    - **When to Use**: Suitable for large datasets and when comparing the sample with a known distribution.
                    """)
                else:
                    st.latex(r"""
                    D = \sup_x |F_n(x) - F(x)|
                    """)
                    st.latex("""
                    - **Variables**:
                        - \( D \): Test statistic measuring the maximum difference between the empirical and normal CDFs.
                        - \( F_n(x) \): Empirical distribution function (EDF).
                        - \( F(x) \): Cumulative distribution function (CDF) of the normal distribution.
                    - **Algorithm**:
                        1. Compare the EDF with the CDF of the normal distribution.
                        2. Compute the supremum (maximum difference).
                        3. Use the p-value to assess normality.
                    - **When to Use**: Ideal for large datasets (n > 50) and when comparing data against a known distribution.
                    """)



def normality_test(column, method='dagostino'):
    """Perform specified normality test on a column and return the p-value."""
    if column.isnull().all():  # Handle empty or all-NaN columns
        return np.nan
    
    try:
        # Shapiro-Wilk test for small sample sizes
        if method == 'shapiro':
            stat, p_value = stats.shapiro(column.dropna())
        
        # D'Agostino's K-squared test for moderate sample sizes
        elif method == 'dagostino':
            stat, p_value = stats.normaltest(column.dropna())
        
        # Kolmogorov-Smirnov test for large sample sizes
        elif method == 'ks':
            stat, p_value = stats.kstest(column.dropna(), 'norm', args=(np.mean(column.dropna()), np.std(column.dropna())))
        
        # Anderson-Darling test for normality
        elif method == 'anderson':
            result = stats.anderson(column.dropna(), dist='norm')
            
            # Determine the most significant level where the statistic exceeds the critical value
            for i, critical_value in enumerate(result.critical_values):
                if result.statistic > critical_value:
                    # Return the corresponding significance level
                    return result.significance_level[i] / 100.0
            
            # If statistic is smaller than all critical values, return the smallest significance level
            return result.significance_level[-1] / 100.0
        
        else:
            raise ValueError("Unknown method. Please choose 'shapiro', 'dagostino', 'ks', or 'anderson'.")
        
        # Return the rounded p-value
        return round(p_value, 4)
    
    except Exception as e:
        print(f"Error in performing {method} test: {e}")
        return np.nan  # Handle errors (e.g., insufficient data)





def get_statistics_dataframe(df, analyzer, nb_top_categories=4, exclude_columns=None, qual_var_threshold=50):
    """
    Creates new DataFrames with statistics for numerical and non-numerical columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame for which to calculate statistics.
    - analyzer (DataAnalyzer): An instance of the DataAnalyzer class with analyzed columns.
    - nb_top_categories (int): Number of top categories to consider for categorical columns.
    - exclude_columns (list): List of column names to exclude from the statistics.
    - qual_var_threshold (int): Threshold for distinguishing qualitative variables based on unique values.

    Returns:
    - tuple of pd.DataFrames: Two DataFrames with statistics, one for numerical and one for non-numerical columns.
    """
    # Exclude specified columns
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore')
    
    # Handle date columns by converting to datetime if not already in that format
    for col in analyzer.date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Round Low-Cardinality Numercial Values
    for col in analyzer.low_cardinality_numeric_cols:
        df[col] = df[col].round(2)
    # Initialize statistics DataFrames
    numerical_df = df[analyzer.numeric_cols + analyzer.date_cols + analyzer.timedelta_cols]
    categorical_df = df[analyzer.categorical_cols + analyzer.low_cardinality_numeric_cols]
    
    # Numerical statistics (including date columns)
    numerical_stats = numerical_df.describe().round(2).transpose()
    numerical_stats['std'] = numerical_df.select_dtypes(include=[np.number]).std().round(2)

    #Normality tests
    # Add columns for p-values of different normality tests
    numerical_stats["Shapiro-Wilk p-value"] = numerical_df.select_dtypes(include=[np.number]).apply(normality_test, axis=0, method='shapiro')
    numerical_stats["D'Agostino's K² p-value"] = numerical_df.select_dtypes(include=[np.number]).apply(normality_test, axis=0, method='dagostino')
    numerical_stats["Kolmogorov-Smirnov p-value"] = numerical_df.select_dtypes(include=[np.number]).apply(normality_test, axis=0, method='ks')
    #numerical_stats["Anderson-Darling significance level"] = numerical_df.select_dtypes(include=[np.number]).apply(normality_test, axis=0, method='anderson')
    
    # Add columns to indicate whether the data is NOT normal (p < 0.05) for each test
    numerical_stats["Not Normal (Shapiro-Wilk)"] = numerical_stats["Shapiro-Wilk p-value"] < 0.05
    numerical_stats["Not Normal (D'Agostino K²)"] = numerical_stats["D'Agostino's K² p-value"] < 0.05
    numerical_stats["Not Normal (KS)"] = numerical_stats["Kolmogorov-Smirnov p-value"] < 0.05
    #numerical_stats["Not Normal (Anderson-Darling)"] = numerical_stats["Anderson-Darling significance level"] < 0.05

    # Set the variable type as 'Numeric' or 'Date' based on column type
    numerical_stats['variable_type'] = [
        'Date' if col in analyzer.date_cols else
        'Time Interval' if col in analyzer.timedelta_cols else
        'Numeric'
        for col in numerical_stats.index
        ]
    numerical_stats['nb_modalities'] = numerical_df.apply(lambda x: x.nunique())
    numerical_stats['fill_percentage'] = (1 - numerical_df.isnull().mean()) * 100

    
    # Remove low-cardinality numeric columns from numerical stats (added to categorical stats below)
    for col in analyzer.low_cardinality_numeric_cols:
        if col in numerical_stats.index:
            numerical_stats.drop(index=col, inplace=True)

    # Categorical statistics
    categorical_stats = categorical_df.describe(include='all').round(2).transpose()
    # categorical_stats.drop(columns=['min','std'], inplace=True)
    columns_to_drop = ['min','std']
    categorical_stats.drop(columns=[col for col in columns_to_drop if col in categorical_stats.columns], inplace=True)
    categorical_stats['nb_modalities'] = categorical_df.apply(lambda x: x.nunique())
    categorical_stats['fill_percentage'] = (1 - categorical_df.isnull().mean()) * 100
    categorical_stats['variable_type'] = 'Categorical'

    # Add high-cardinality categorical columns
    for col in analyzer.high_cardinality_cat_cols:
        if col in categorical_stats.index:
            categorical_stats.at[col, 'variable_type'] = 'High Cardinality Categorical'

    # Add low-cardinality numeric columns
    for col in analyzer.low_cardinality_numeric_cols:
        if col in categorical_stats.index:
            categorical_stats.at[col, 'variable_type'] = 'Low Cardinality Numeric'

    # Add binary and high-cardinality categorical column labels in variable type
    for col in analyzer.binary_cols:
        if col in categorical_stats.index:
            categorical_stats.at[col, 'variable_type'] = 'Binary'

    # Drop irrelevant columns in categorical statistics
    categorical_stats = categorical_stats.drop(['top', 'freq', 'mean', 'unique', '25%', '50%', '75%', 'max'], axis=1, errors='ignore')

    # Add top categories and percentages for categorical data with cardinality below threshold
    for col in analyzer.categorical_cols + analyzer.low_cardinality_numeric_cols:
        if col in df and df[col].nunique() < qual_var_threshold:
            value_counts = df[col].value_counts()
            percentage_counts = (value_counts / len(df)) * 100
            top_categories = percentage_counts.head(nb_top_categories)

            for i, (value, representation) in enumerate(top_categories.items(), start=1):
                categorical_stats.at[col, f'top_{i}_value'] = str(value)
                categorical_stats.at[col, f'top_{i}_representation'] = representation

    # Format float columns to 2 decimals for cleaner display
    float_columns = numerical_stats.select_dtypes(include='float').columns
    numerical_stats[float_columns] = numerical_stats[float_columns].round(2)
    float_columns = categorical_stats.select_dtypes(include='float').columns
    categorical_stats[float_columns] = categorical_stats[float_columns].round(2)

    columns_to_move = ['fill_percentage', 'variable_type', 'nb_modalities', 'count' ]
    categorical_stats = move_columns_to_front(categorical_stats, columns_to_move)
    numerical_stats = move_columns_to_front(numerical_stats, columns_to_move)

    return numerical_stats, categorical_stats


def app():
    # Load or select dataset (assuming df is already loaded into session state) 
    df = st.session_state.get('data', None)
    if df is not None:
        analyzer = DataAnalyzer(st.session_state.get('data', None))
    # Create tabs for different visualization aspects
    tabs = st.tabs(["Dataset Statistics", "Columns Renaming"])

    with tabs[1]:
        if 'working_df' in st.session_state:
            analyzer.refresh(st.session_state["working_df"])
        st.title("Columns Renaming")
        #df = st.session_state.get('data', None)
        if df is not None:
            # Instantiate DataAnalyzer and RenameColumnsComponent
            
            #analyzer = DataAnalyzer(st.session_state.get('data', None))
            rename_component = RenameColumnsComponent(st.session_state.get('data', None), analyzer)

            # Display the component and get the updated DataFrame
            df = rename_component.display()
            st.session_state["data"] = df
            #st.write("test: ")
            #st.dataframe(st.session_state["data"])
            
        else:
            st.warning("Please upload data to view statistics.")

    with tabs[0]:
        if 'working_df' in st.session_state:
            analyzer.refresh(st.session_state["working_df"])
        st.title("Dataset Statistics")
        #df = st.session_state.get('data', None)
        if "max_top_modalities" not in locals():
            max_top_modalities = 3
        if df is not None:
            
            st.subheader("Dataset Preview")
            with st.expander("Preview the first lines of your dataset", expanded=True):
                col_inc, col_ano = st.columns(2)

                # Checkbox for filtering out anomalies
                with col_ano:
                    filter_out_anomaly = st.checkbox(
                        label="Exclude Rows with Anomalies",
                        value=False,
                        help="Check this box to exclude rows flagged as anomalies from the dataset."
                    )

                # Checkbox for limiting rows based on inclusion criteria
                with col_inc:
                    filter_on_inclusion = st.checkbox(
                        label="Restrict to Inclusion Criteria",
                        value=False,
                        help="Check this box to limit the displayed rows to those meeting the inclusion criteria."
                    )


                # Base DataFrame
                filtered_df = df.copy()

                # Apply filters based on checkbox states
                if filter_on_inclusion:
                    if "study_inclusion_all" not in filtered_df.columns:
                        st.warning(f"The inclusion criteria are missing and need to be defined first.")
                    else:
                        filtered_df = filtered_df[filtered_df['study_inclusion_all'] == True]

                if filter_out_anomaly:
                    if "clinical_anomalies_any" not in filtered_df.columns:
                        st.warning(f"The anomaly criteria are missing and need to be defined first.")
                    else:
                        filtered_df = filtered_df[filtered_df['clinical_anomalies_any'] == False]

                st.dataframe(filtered_df.head(), use_container_width=True, hide_index=True)
                st.write(f"📊 **Filtered Data Overview:** {filtered_df.shape[0]:,} rows and {filtered_df.shape[1]:,} columns selected.")

                #st.subheader("Data session")
                #st.dataframe(st.session_state["data"].head(), use_container_width=True, hide_index=True)

                # Column information
                st.subheader("Columns Information")
                # Analyze the dataset using DataAnalyzer
                column_categories = {
                            "Quantitative: Columns with numerical quantivative data": sorted(analyzer.numeric_cols),
                            "Binary: Columns with 2 unique values (e.g., 0/1 or Yes/No)": sorted(analyzer.binary_cols),
                            "Non Binary Low-Cardinality Numeric: Numerical columns with few distinct values": sorted(analyzer.non_binary_low_cardinality_numeric_cols),
                            "Categorical: Columns with non-numerical data": sorted(analyzer.categorical_cols),
                            "High-Cardinality Categorical: Categorical columns with many unique values (e.g., id)": sorted(analyzer.high_cardinality_cat_cols),
                            "Date: Columns containing date or time data": sorted(analyzer.date_cols),
                            "Time Delta: Columns containing time interval data (e.g., age or period)": sorted(analyzer.timedelta_cols)
                        }
                for category, columns in column_categories.items():
                    display_category_box(category, columns)
                
                # Generate statistics
                #numerical_stats, non_numerical_stats = get_statistics_dataframe(df, analyzer, nb_top_categories=max_top_modalities, qual_var_threshold=50) #get_statistics_dataframe(df)
            
            group_select = sorted(list(set(analyzer.categorical_cols + analyzer.low_cardinality_numeric_cols + analyzer.binary_cols) - set(analyzer.high_cardinality_cat_cols)))

            #st.write(group_select, analyzer.high_cardinality_cat_cols)

            st.subheader("Quantitative Data Statistics")
            with st.expander("Get a descriptive view of your quantitative variables", expanded=True): 
                # User selects a categorical variable
                col_quant_1, col_quant_2 = st.columns(2)

                with col_quant_1:
                    selected_group_quant = st.selectbox(label="Select grouping column (optional):", options=["None"] + group_select, index=0, key='select_group_quant')
                
                if selected_group_quant == "None":
                    # Display numerical stats
                    numerical_stats, _ = get_statistics_dataframe(filtered_df, analyzer, nb_top_categories=max_top_modalities, qual_var_threshold=50)
                    # Show available columns and let user select columns to retain
                    selected_quant_columns = st.multiselect(
                            "Select columns to retain in descriptive statistics of quantitative variables",
                            options=numerical_stats.columns.tolist(),
                            default=numerical_stats.columns.tolist()
                        )
                    st.dataframe(numerical_stats[selected_quant_columns], 
                                use_container_width=True, 
                                column_config={
                                    "fill_percentage": st.column_config.ProgressColumn(
                                        "Fill Percentage",
                                        help="Shows the percentage of non-missing values in this column.",
                                        format="%d%%",  # Display as a percentage
                                        min_value=0,
                                        max_value=100,
                                    )}
                                )
                    
                    st.download_button(
                        label="Download as Excel",
                        data=to_excel(numerical_stats[selected_quant_columns]),
                        file_name="Quantitative Data Statistics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    # Group data by the selected categorical variable
                    quant_category_groups = filtered_df.groupby(selected_group_quant)
                    # Generate descriptive statistics for each group using the custom function
                    quant_dict = {}
                    for category, group in quant_category_groups:
                        #st.write(f"---------------\ncategory : {category}, \ngroup : {group}\n------------")
                        numerical_stats, _ = get_statistics_dataframe(
                            df=group, 
                            analyzer=analyzer, 
                            nb_top_categories=1
                        )
                        quant_dict[category] = numerical_stats
                    
                    with col_quant_2:
                        selected_category = st.selectbox(label="Select grouping value:", options=quant_dict.keys(), index=0)
                    
                    
                    selected_quant_columns = st.multiselect(
                            "Select columns to retain in descriptive statistics of quantitative variables",
                            options=numerical_stats.columns.tolist(),
                            default=numerical_stats.columns.tolist()
                        )
                    
                    # Filter columns for all groups
                    for category in quant_dict.keys():
                        quant_dict[category] = quant_dict[category][selected_quant_columns]

                    # Display the first table as an example
                    st.dataframe(quant_dict[selected_category],
                                 use_container_width=True,
                                 column_config={
                                    "fill_percentage": st.column_config.ProgressColumn(
                                        "Fill Percentage",
                                        help="Shows the percentage of non-missing values in this column.",
                                        format="%d%%",  # Display as a percentage
                                        min_value=0,
                                        max_value=100,
                                    )}
                                 )

                    # Allow user to download all data as a multi-sheet Excel file
                    st.download_button(
                        label="Download as Excel",
                        data=to_excel_sheets(quant_dict),
                        file_name=f"Descriptive_Statistics_by_{selected_group_quant}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                # Call the function to display the guide
                #show_test_guidelines()


            st.subheader("Qualitative Data Statistics")
            with st.expander("Get a descriptive view of your qualitative variables", expanded=True):
                # User selects a categorical variable
                col_qual_1, col_qual_2 = st.columns(2)

                with col_qual_1:
                    selected_group_qual = st.selectbox(label="Select grouping column (optional):", options=["None"] + group_select, index=0, key='select_group_qual')

                # Display non-numerical stats
                col1, col2 = st.columns([1, 5])
                with col1:
                    max_top_modalities = st.number_input(
                        "Select the number of top modalities to display",
                        min_value=1, 
                        max_value=50, 
                        value=3, 
                        step=1,
                        help="This controls the number of top categories displayed for qualitative data."
                    )

                if selected_group_qual == "None":
                    _, non_numerical_stats = get_statistics_dataframe(filtered_df, analyzer, nb_top_categories=max_top_modalities, qual_var_threshold=50)
                    
                    with col2:
                        # Show available columns and let user select columns to retain
                        selected_qual_columns = st.multiselect(
                                "Select columns to retain in descriptive statistics of quantitative variables",
                                options=non_numerical_stats.columns.tolist(),
                                default=non_numerical_stats.columns.tolist()
                            )
                        
                    st.dataframe(non_numerical_stats[selected_qual_columns], 
                                use_container_width=True, 
                                column_config={
                                    "fill_percentage": st.column_config.ProgressColumn(
                                        "Fill Percentage",
                                        help="Shows the percentage of non-missing values in this column.",
                                        format="%d%%",  # Display as a percentage
                                        min_value=0,
                                        max_value=100,
                                    )}
                                )
                    
                    st.download_button(
                        label="Download as Excel",
                        data=to_excel(non_numerical_stats[selected_qual_columns]),
                        file_name="Qualitative Data Statistics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                else:
                    # Group data by the selected categorical variable
                    qual_category_groups = filtered_df.groupby(selected_group_qual)
                    # Generate descriptive statistics for each group using the custom function
                        
                    qual_dict = {}
                    for category, group in qual_category_groups:
                        #st.write(f"---------------\ncategory : {category}, \ngroup : {group}\n------------")
                        _, non_numerical_stats = get_statistics_dataframe(
                            df=group, 
                            analyzer=analyzer, 
                            nb_top_categories=max_top_modalities
                        )
                        qual_dict[category] = non_numerical_stats

                    with col2:
                        # Show available columns and let user select columns to retain
                        selected_qual_columns = st.multiselect(
                                "Select columns to retain in descriptive statistics of quantitative variables",
                                options=non_numerical_stats.columns.tolist(),
                                default=non_numerical_stats.columns.tolist()
                            )
                        
                    with col_qual_2:
                        selected_category = st.selectbox(label="Select grouping value:", options=qual_dict.keys(), index=0)

                    # Filter columns for all groups
                    for category in qual_dict.keys():
                        qual_dict[category] = qual_dict[category][selected_qual_columns]

                    # Display the first table as an example
                    st.dataframe(qual_dict[selected_category],
                                 use_container_width=True,
                                 column_config={
                                    "fill_percentage": st.column_config.ProgressColumn(
                                        "Fill Percentage",
                                        help="Shows the percentage of non-missing values in this column.",
                                        format="%d%%",  # Display as a percentage
                                        min_value=0,
                                        max_value=100,
                                    )}
                                 )

                    # Allow user to download all data as a multi-sheet Excel file
                    st.download_button(
                        label="Download as Excel",
                        data=to_excel_sheets(qual_dict),
                        file_name=f"Descriptive_Statistics_by_{selected_group_qual}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            st.subheader("Correlation Matrix Analysis")
            with st.expander("Get a view of variables interactions", expanded=True):
                
                
                #group_select = list(set(analyzer.binary_cols + analyzer.categorical_cols + analyzer.non_binary_low_cardinality_numeric_cols)) # Grouping column
                colcor1, colcor2, colcor3 = st.columns(3)
                
                with colcor1:
                    # Columns selection
                    sym = st.checkbox("symetrical", True)
                    
                with colcor2:
                    # Correlation method
                    corr_method = st.selectbox("Select correlation method", ["kendall", "spearman", "pearson"])
                    if corr_method == "pearson":
                        cols_select = analyzer.numeric_cols # numerical columns
                    else:
                        cols_select = list(set(analyzer.num_binary_cols + analyzer.low_cardinality_numeric_cols + analyzer.numeric_cols)) # numerical columns
                with colcor3:
                    
                    group_column = st.selectbox("Select grouping column (optional)", ["None"] + group_select) # sorted(group_select)
                    group_column = None if group_column == "None" else group_column
                
                if sym:
                    cols = st.multiselect("Select columns", options=cols_select, default=cols_select)
                    targets = cols
                    predictors = cols
                    #triangle = 'lower'
                else:
                    targets = st.multiselect("Select target columns", cols_select)
                    predictors = st.multiselect("Select predictor columns", cols_select)
                    #triangle = 'full'

                # Generate correlation matrix
                try:
                    # # Generate the heatmap figure
                    # fig = ecm.plotly_corr_mat(
                    #     data=df,
                    #     targets=targets,
                    #     predictors=predictors,
                    #     method="pearson",
                    #     selected_color='Plasma',
                    #     triangle=triangle,  # Show only the lower triangle
                    # )
                    # st.plotly_chart(fig, use_container_width=True)
                    
                    # Define output file name
                    file_name = "correlation_matrix"
                    # Generate Excel file
                    
                    st.download_button(
                        on_click=ecm.custom_corr_mat_to_excel(
                        data=filtered_df,
                        targets=targets,
                        predictors=predictors,
                        group_column=group_column,
                        file_name=file_name,
                        method=corr_method,
                        threshold="Negligible"
                        ),
                        label="Download Correlation Matrix",
                        data=open(f"{file_name}.xlsx", "rb"),
                        file_name=f"{file_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        else:
            st.warning("Please upload data to view statistics.")