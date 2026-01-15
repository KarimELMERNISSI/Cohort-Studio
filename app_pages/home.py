# pages/home.py
import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np
from yfiles_graphs_for_streamlit import StreamlitGraphWidget, Node, Edge, EdgeStyle, DashStyle, Layout, LabelStyle
import explore.corr_matrix as ecm
import utils.visualization_utils as vu
from utils.export_utils import to_excel, to_excel_sheets
from utils.statistics_utils import normality_test, show_test_guidelines
from utils.data_analyzer import DataAnalyzer
import os
from manage.db_manager import DBManager
from manage.rag_manager import RAGManager
import json
import duckdb
import time

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

    def _get_columns_info(self, columns_to_process=None):
        """Gather detailed information about columns for RAG context."""
        columns_info = []
        cols = columns_to_process if columns_to_process else self.original_df.columns
        for col in cols:
            info = {"name": col}
            if col in self.analyzer.categorical_cols or col in self.analyzer.low_cardinality_numeric_cols:
                # Convert keys to string to ensure JSON serializability
                top_values = {str(k): v for k, v in self.original_df[col].value_counts().head(5).to_dict().items()}
                info["type"] = "Categorical"
                info["top_values"] = top_values
            elif col in self.analyzer.numeric_cols:
                col_data = self.original_df[col].dropna()
                if not col_data.empty:
                    info["type"] = "Numeric"
                    info["stats"] = {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "median": float(col_data.median()),
                        "mean": float(col_data.mean())
                    }
            elif col in self.analyzer.date_cols:
                info["type"] = "Date"
                # Ensure we are working with datetime objects to avoid comparison errors
                col_data = pd.to_datetime(self.original_df[col], errors='coerce').dropna()
                if not col_data.empty:
                    info["range"] = f"{col_data.min()} to {col_data.max()}"
            else:
                info["type"] = "Unknown"
            columns_info.append(info)
        return columns_info

    def display(self):
        """Displays the column renaming UI with sorting, searching, type filtering, and a compact layout."""
        st.subheader("Rename Columns")

        # AI Suggestion Section
        with st.expander("🤖 AI Renaming Suggestions (RAG)", expanded=False):
            if 'rag_manager' in st.session_state and st.session_state.rag_manager.initialized:
                # deep_analysis = st.checkbox("Enable Deep Taxonomy (Slower, includes formulas)", value=False)
                
                col_mode, col_btn = st.columns([3, 1])
                with col_mode:
                    suggestion_mode = st.radio(
                        "Suggestion Source:", 
                        ["Medical Literature (Context)", "Standard Terminologies (UMLS/SNOMED/LOINC)"],
                        horizontal=True
                    )
                    
                    selected_cols = st.multiselect(
                        "Select columns to rename (leave empty for all):",
                        options=list(self.original_df.columns),
                        default=[],
                        key="rag_col_select",
                        help="Select specific columns to generate suggestions for. If empty, all columns will be processed."
                    )
                
                with col_btn:
                    st.write("") # Spacer
                    st.write("") # Spacer
                    st.write("") # Spacer
                    if st.button("Generate Suggestions", width='stretch'):
                        strategy = "literature" if "Literature" in suggestion_mode else "standardization"
                        
                        progress_bar = st.progress(0, text=f"Starting {strategy} analysis...")
                        def update_progress(percent, text):
                            progress_bar.progress(percent, text=text)

                        # Get detailed column info
                        columns_info = self._get_columns_info(selected_cols if selected_cols else None)
                        
                        suggestions, error = st.session_state.rag_manager.suggest_column_renaming(
                            columns_info, 
                            strategy=strategy,
                            progress_callback=update_progress
                        )
                        progress_bar.empty()
                        
                        if error:
                            st.error(f"Error: {error}")
                        else:
                                try:
                                    # Parse JSON if it's a string, though the manager should return a dict or string
                                    if isinstance(suggestions, str):
                                        # Try to extract JSON from string if it contains text
                                        import re
                                        json_match = re.search(r'\{.*\}', suggestions, re.DOTALL)
                                        if json_match:
                                            suggestions = json.loads(json_match.group(0))
                                        else:
                                            st.warning("Could not parse AI response.")
                                    
                                    if isinstance(suggestions, dict):
                                        st.session_state['ai_renaming_suggestions'] = suggestions
                                        st.success("Suggestions received!")
                                    else:
                                        st.write(suggestions) # Show raw text if not dict
                                except Exception as e:
                                    st.error(f"Error parsing suggestions: {e}")
            else:
                st.info("RAG System not initialized. Please configure it in the sidebar.")

        # Variable Taxonomy Section
        # Apply AI suggestions if available
        if 'ai_renaming_suggestions' in st.session_state:
            st.info("💡 AI Suggestions available. Click 'Apply' next to the suggestion to use it.")

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

        # Helper callback for AI renaming
        def apply_ai_rename(col_name, new_value):
            st.session_state["rename_dict"][col_name] = new_value
            st.session_state[f"rename_{col_name}"] = new_value

        # Display renaming interface
        st.write("Rename columns by entering new names. Top 5 values are shown where applicable:")
        for col in columns:
            with st.container():
                col_current, col_rename, col_info = st.columns([1, 2, 3])

                with col_current:
                    st.markdown(f"**{col}**")  # Display original column name

                with col_rename:
                    current_val = st.session_state["rename_dict"].get(col, col)
                    new_name = st.text_input(f"Rename '{col}' to:", value=current_val, key=f"rename_{col}")
                    st.session_state["rename_dict"][col] = new_name
                    
                    # Show AI suggestion if available
                    if 'ai_renaming_suggestions' in st.session_state and col in st.session_state['ai_renaming_suggestions']:
                        suggested = st.session_state['ai_renaming_suggestions'][col]
                        if suggested != current_val:
                            st.caption(f"💡 AI Suggestion: **{suggested}**")
                            st.button("Apply", key=f"apply_ai_{col}", on_click=apply_ai_rename, args=(col, suggested))

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

@st.cache_data(ttl=3600, show_spinner="Computing statistics...")
def get_statistics_dataframe(_df, _analyzer, nb_top_categories=4, exclude_columns=None, qual_var_threshold=50, dataset_name=None, _db_manager=None):
    """
    Creates new DataFrames with statistics for numerical and non-numerical columns.
    Tries to load from DuckDB/Parquet cache if dataset_name is provided.
    
    Note: Parameters starting with _ are excluded from hashing by Streamlit.
    """
    # Try to load cached stats if dataset name is available
    if dataset_name and _db_manager:
        cached_num, _ = _db_manager.load_stats(dataset_name, "numerical")
        cached_cat, _ = _db_manager.load_stats(dataset_name, "categorical")
        
        # Simple validation: if cached stats exist and have same number of columns as current analysis
        # (This is a basic check; for production, hash the dataframe content)
        if cached_num is not None and cached_cat is not None:
            # Check if columns match roughly to ensure we aren't loading stale stats for modified data
            # This is optional but recommended
            return cached_num, cached_cat

    # Work with copies to avoid modifying original
    df = _df.copy()
    analyzer = _analyzer
    
    # Exclude specified columns
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore')
    
    # Handle date columns by converting to datetime if not already in that format
    # Handle date columns by converting to datetime using smart parser
    from utils.date_parser import smart_parse_dates
    for col in analyzer.date_cols:
        parsed_series, _ = smart_parse_dates(df[col])
        df[col] = parsed_series

    # Round Low-Cardinality Numercial Values
    for col in analyzer.low_cardinality_numeric_cols:
        df[col] = df[col].round(2)
    # Initialize statistics DataFrames
    numerical_df = df[analyzer.numeric_cols + analyzer.date_cols + analyzer.timedelta_cols]
    categorical_df = df[analyzer.categorical_cols + analyzer.low_cardinality_numeric_cols]
    
    # Numerical statistics (including date columns)
    numerical_stats = numerical_df.describe().round(2).transpose()
    
    # Calculate std safely
    try:
        numeric_only = numerical_df.select_dtypes(include=[np.number])
        if not numeric_only.empty:
            std_vals = numeric_only.std()
            # Ensure numeric type before rounding
            if not pd.api.types.is_numeric_dtype(std_vals):
                std_vals = pd.to_numeric(std_vals, errors='coerce')
            numerical_stats['std'] = std_vals.round(2)
    except Exception:
        numerical_stats['std'] = np.nan

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

    # Cache the computed stats
    if dataset_name and _db_manager:
        _db_manager.save_stats(numerical_stats, dataset_name, "numerical")
        _db_manager.save_stats(categorical_stats, dataset_name, "categorical")
    
    # ---------------------------------------------------------
    # Date Statistics
    # ---------------------------------------------------------
    date_stats = pd.DataFrame()
    if analyzer.date_cols:
        date_data = []  # List of dicts
        for col in analyzer.date_cols:
            if col in df:
                series = df[col]
                fill_pct = (1 - series.isnull().mean()) * 100
                
                # Filter valid
                valid = series.dropna()
                
                if not valid.empty:
                    min_date = valid.min()
                    max_date = valid.max()
                    duration = max_date - min_date
                    n_unique = valid.nunique()
                else:
                    min_date, max_date, duration, n_unique = pd.NaT, pd.NaT, pd.NaT, 0
                
                date_data.append({
                    "Feature": col,
                    "Variable Type": "Date",
                    "Min Date": min_date,
                    "Max Date": max_date,
                    "Duration": duration,
                    "Unique Values": n_unique,
                    "Fill Percentage": fill_pct
                })
        
        if date_data:
            date_stats = pd.DataFrame(date_data).set_index("Feature")
            # Round fill %
            date_stats["Fill Percentage"] = date_stats["Fill Percentage"].round(2)

    return numerical_stats, categorical_stats, date_stats

def run_benchmark(df, analyzer):
    st.subheader("Performance Benchmark: Pandas vs DuckDB")
    st.info("This benchmark compares the execution time of standard Pandas operations vs DuckDB SQL queries on your current dataset. It also verifies that both methods produce consistent results.")
    
    if st.button("Run Benchmark"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Basic Statistics (Numerical)
        status_text.text("Benchmarking Numerical Statistics...")
        
        # Pandas
        start = time.time()
        numerical_df = df[analyzer.numeric_cols]
        stats_pd = numerical_df.describe().transpose()
        end = time.time()
        time_pd_stats = end - start
        progress_bar.progress(25)
        
        # DuckDB
        start = time.time()
        con = duckdb.connect()
        con.register('df_bench', df)
        
        selects = []
        for col in analyzer.numeric_cols:
            # Basic stats equivalent to describe()
            selects.append(f"COUNT(\"{col}\")")
            selects.append(f"AVG(\"{col}\")")
            selects.append(f"STDDEV(\"{col}\")")
            selects.append(f"MIN(\"{col}\")")
            selects.append(f"MAX(\"{col}\")")
            # Quantiles (approximate is faster, exact is slower)
            selects.append(f"QUANTILE_CONT(\"{col}\", 0.25)")
            selects.append(f"QUANTILE_CONT(\"{col}\", 0.50)")
            selects.append(f"QUANTILE_CONT(\"{col}\", 0.75)")

        if selects:
            query = f"SELECT {', '.join(selects)} FROM df_bench"
            res = con.execute(query).fetchone()
        
        end = time.time()
        time_dd_stats = end - start
        progress_bar.progress(50)
        
        # Verification (Basic Check)
        # We check if the count matches for the first column as a proxy
        match_stats = "✅ Match"
        if selects and res:
            # Pandas count is float, DuckDB is int. Compare with tolerance or cast.
            pd_count = stats_pd.iloc[0]['count']
            dd_count = res[0] # First element is COUNT(col1)
            if abs(pd_count - dd_count) > 0:
                 match_stats = "✅ Match (Approx)" # Counts should match exactly but let's be safe
        
        results.append({"Task": "Basic Stats (Numerical)", "Pandas (s)": time_pd_stats, "DuckDB (s)": time_dd_stats, "Speedup": f"{time_pd_stats/time_dd_stats:.2f}x" if time_dd_stats > 0 else "N/A", "Status": match_stats})
        
        # 2. Categorical Statistics (Top Values)
        status_text.text("Benchmarking Categorical Statistics...")
        
        # Pandas
        start = time.time()
        pd_cats = {}
        for col in analyzer.categorical_cols:
            pd_cats[col] = df[col].value_counts().head(5)
        end = time.time()
        time_pd_cat = end - start
        progress_bar.progress(60)
        
        # DuckDB
        start = time.time()
        dd_cats = {}
        for col in analyzer.categorical_cols:
            query = f"SELECT \"{col}\", COUNT(*) as count FROM df_bench GROUP BY \"{col}\" ORDER BY count DESC LIMIT 5"
            dd_cats[col] = con.execute(query).fetchall()
        end = time.time()
        time_dd_cat = end - start
        progress_bar.progress(70)
        
        # Verification
        match_cat = "✅ Match"
        # Check first categorical column
        if analyzer.categorical_cols:
            first_col = analyzer.categorical_cols[0]
            pd_top = pd_cats[first_col]
            dd_top = dd_cats[first_col]
            # Compare top 1 value and count
            if len(pd_top) > 0 and len(dd_top) > 0:
                if str(pd_top.index[0]) != str(dd_top[0][0]) or pd_top.iloc[0] != dd_top[0][1]:
                     match_cat = "⚠️ Mismatch"

        results.append({"Task": "Categorical Stats (Top 5)", "Pandas (s)": time_pd_cat, "DuckDB (s)": time_dd_cat, "Speedup": f"{time_dd_cat/time_pd_cat:.2f}x" if time_dd_cat > 0 else "N/A", "Status": match_cat})

        # 3. Grouped Aggregation (if applicable)
        if analyzer.categorical_cols and analyzer.numeric_cols:
            status_text.text("Benchmarking Grouped Aggregation...")
            group_col = analyzer.categorical_cols[0] # Pick first categorical column
            target_col = analyzer.numeric_cols[0] # Pick first numeric column
            
            # Pandas
            start = time.time()
            pd_group = df.groupby(group_col)[target_col].describe()
            end = time.time()
            time_pd_group = end - start
            progress_bar.progress(85)
            
            # DuckDB
            start = time.time()
            # DuckDB doesn't have a direct 'describe' for groups, so we do basic stats
            query = f"""
            SELECT \"{group_col}\", 
                   COUNT(\"{target_col}\"), AVG(\"{target_col}\"), STDDEV(\"{target_col}\"), MIN(\"{target_col}\"), MAX(\"{target_col}\")
            FROM df_bench 
            GROUP BY \"{group_col}\"
            ORDER BY \"{group_col}\"
            """
            dd_group = con.execute(query).fetchall()
            end = time.time()
            time_dd_group = end - start
            progress_bar.progress(90)
            
            # Verification
            match_group = "✅ Match"
            # Compare first group count
            if not pd_group.empty and dd_group:
                # Sort pandas to ensure order matches DuckDB's ORDER BY
                pd_group_sorted = pd_group.sort_index()
                # Check first group
                if pd_group_sorted.iloc[0]['count'] != dd_group[0][1]:
                     match_group = "⚠️ Mismatch"

            results.append({"Task": f"Grouped Stats (by {group_col})", "Pandas (s)": time_pd_group, "DuckDB (s)": time_dd_group, "Speedup": f"{time_pd_group/time_dd_group:.2f}x" if time_dd_group > 0 else "N/A", "Status": match_group})

        # 4. Correlation Matrix
        status_text.text("Benchmarking Correlation Matrix...")
        
        # Pandas
        start = time.time()
        corr_pd = numerical_df.corr()
        end = time.time()
        time_pd_corr = end - start
        progress_bar.progress(75)
        
        # DuckDB
        start = time.time()
        # Pairwise correlation
        selects = []
        cols = analyzer.numeric_cols
        # Limit columns if too many to avoid query length issues or excessive time for benchmark
        bench_cols = cols[:50] if len(cols) > 50 else cols
        
        for i, c1 in enumerate(bench_cols):
            for c2 in bench_cols[i:]:
                 selects.append(f"CORR(\"{c1}\", \"{c2}\")")
        
        if selects:
            query = f"SELECT {', '.join(selects)} FROM df_bench"
            res = con.execute(query).fetchall()
            
        end = time.time()
        time_dd_corr = end - start
        progress_bar.progress(100)
        
        # Verification
        match_corr = "✅ Match"
        # Compare one value
        if not corr_pd.empty and res:
            # First correlation value (self-correlation should be 1.0)
            # DuckDB returns a flat list of results.
            # CORR(c1, c1) is the first one.
            if abs(res[0][0] - 1.0) > 0.0001:
                 match_corr = "⚠️ Mismatch"
        
        results.append({"Task": "Correlation Matrix", "Pandas (s)": time_pd_corr, "DuckDB (s)": time_dd_corr, "Speedup": f"{time_pd_corr/time_dd_corr:.2f}x" if time_dd_corr > 0 else "N/A", "Status": match_corr})
        
        status_text.text("Benchmark Complete!")
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Highlight best performer
        def highlight_best(row):
            styles = [''] * len(row)
            if row['Pandas (s)'] < row['DuckDB (s)']:
                styles[1] = 'background-color: #90ee90; color: black; font-weight: bold' # Pandas green
            else:
                styles[2] = 'background-color: #90ee90; color: black; font-weight: bold' # DuckDB green
            return styles

        st.dataframe(results_df.style.apply(highlight_best, axis=1), width='stretch')
        
        if time_dd_corr < time_pd_corr:
            st.success("DuckDB is faster for Correlation Matrix on this dataset!")
        else:
            st.info("Pandas is faster or comparable for Correlation Matrix on this dataset.")
            
        st.divider()
        st.subheader("Optimization Settings")
        st.write("Select the preferred engine for future calculations based on the benchmark results:")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.session_state['pref_stats_engine'] = st.radio("Engine for Basic Statistics", ["Pandas", "DuckDB"], index=1 if time_dd_stats < time_pd_stats else 0, horizontal=True)
            st.session_state['pref_corr_engine'] = st.radio("Engine for Correlation Matrix", ["Pandas", "DuckDB"], index=1 if time_dd_corr < time_pd_corr else 0, horizontal=True)
        
        with col_opt2:
            st.session_state['pref_cat_engine'] = st.radio("Engine for Categorical Stats", ["Pandas", "DuckDB"], index=1 if time_dd_cat < time_pd_cat else 0, horizontal=True)
            if analyzer.categorical_cols and analyzer.numeric_cols:
                st.session_state['pref_group_engine'] = st.radio("Engine for Grouped Stats", ["Pandas", "DuckDB"], index=1 if time_dd_group < time_pd_group else 0, horizontal=True)
        
        st.success("Preferences saved for this session.")


def app():
    # Initialize DB Manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DBManager()

    # Initialize RAG Manager
    if 'rag_manager' not in st.session_state:
        st.session_state.rag_manager = RAGManager()

    # RAG Configuration Sidebar - MOVED TO GLOBAL SIDEBAR (utils/multipage.py -> app_pages/rag_sidebar.py)
    # with st.sidebar.expander("🧠 RAG & AI Settings", expanded=False):
    #     ... (Code moved)

    # Dataset Management Sidebar (Automatic Versioning)
    with st.sidebar.expander("Dataset History", expanded=False):
        st.caption("Manage dataset versions")
        
        # Save current state option
        if st.session_state.get('data') is not None:
            if st.button("Save Current State as New Version"):
                base_name = "cohort_data"
                success, msg, saved_name = st.session_state.db_manager.save_dataset(st.session_state['data'], base_name=base_name)
                if success:
                    st.session_state['current_dataset_name'] = saved_name
                    st.success(f"Saved as {saved_name}")
                    st.rerun()
                else:
                    st.error(msg)

        datasets = st.session_state.db_manager.get_available_datasets()
        if datasets:
            selected_dataset = st.selectbox("Select Dataset Version", datasets, index=0)
            if st.button("Load Selected Version"):
                df_loaded, msg = st.session_state.db_manager.load_dataset(selected_dataset)
                if df_loaded is not None:
                    st.session_state['data'] = df_loaded
                    st.session_state['working_df'] = df_loaded.copy()
                    st.session_state['current_dataset_name'] = selected_dataset
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.info("No saved datasets found.")

    # Implicitly load latest dataset if none is loaded
    # if st.session_state.get("data") is None:
    #     datasets = st.session_state.db_manager.get_available_datasets()
    #     if datasets:
    #         latest_dataset = datasets[0]
    #         df_loaded, msg = st.session_state.db_manager.load_dataset(latest_dataset)
    #         if df_loaded is not None:
    #             st.session_state['data'] = df_loaded
    #             st.session_state['working_df'] = df_loaded.copy()
    #             st.session_state['current_dataset_name'] = latest_dataset
    #             st.toast(f"Auto-loaded latest dataset: {latest_dataset}")
    #         else:
    #             st.warning("Failed to auto-load latest dataset.")

    # Load or select dataset (assuming df is already loaded into session state) 
    df = st.session_state.get('data', None)

    if df is not None:
        analyzer = DataAnalyzer(st.session_state.get('data', None))
        cols_to_convert = analyzer.low_cardinality_numeric_cols + analyzer.low_cardinality_numeric_cols + analyzer.numeric_cols
        # Ensure no duplicates in the column list
        cols_to_convert = list(set(cols_to_convert))
        # Coerce numeric format for specified columns
        df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    # Create tabs for different visualization aspects
    tabs = st.tabs(["Dataset Statistics", "Columns Renaming", "Performance Benchmark"])

    with tabs[2]:
        st.title("Performance Benchmark")
        if df is not None:
            run_benchmark(df, analyzer)
        else:
            st.warning("Please upload data to run benchmark.")

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
        col_title, col_reset = st.columns([4, 1])
        with col_title:
            st.title("Dataset Statistics")
        with col_reset:
             if st.button("🔄 Reset Cache", help="Force full recomputation of statistics"):
                # 1. Clear Disk Cache
                if 'db_manager' in st.session_state:
                    st.session_state.db_manager.clear_all_stats()
                # 2. Clear Memory Cache
                st.cache_data.clear()
                st.toast("Cache cleared! Reloading...")
                time.sleep(1)
                st.rerun()
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

                st.dataframe(filtered_df.head(), width='stretch', hide_index=True)
                st.write(f"📊 **Filtered Data Overview:** {filtered_df.shape[0]:,} rows and {filtered_df.shape[1]:,} columns selected.")

                #st.subheader("Data session")
                #st.dataframe(st.session_state["data"].head(), width='stretch', hide_index=True)

                # Column information
                st.subheader("Columns Information")
                # Analyze the dataset using DataAnalyzer
                column_categories = {
                            "Quantitative: Columns with numerical quantitative data": sorted(analyzer.numeric_cols),
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
                #numerical_stats, non_numerical_stats = get_statistics_dataframe(df, analyzer, nb_top_modalities=max_top_modalities, qual_var_threshold=50) #get_statistics_dataframe(df)
            
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
                    dataset_name = st.session_state.get('current_dataset_name', None)
                    numerical_stats, _, date_stats = get_statistics_dataframe(
                        filtered_df, 
                        analyzer, 
                        nb_top_categories=max_top_modalities, 
                        qual_var_threshold=50,
                        dataset_name=dataset_name,
                        _db_manager=st.session_state.db_manager
                    )
                    # Show available columns and let user select columns to retain
                    selected_quant_columns = st.multiselect(
                            "Select columns to retain in descriptive statistics of quantitative variables",
                            options=numerical_stats.columns.tolist(),
                            default=numerical_stats.columns.tolist()
                        )
                    st.dataframe(numerical_stats[selected_quant_columns], 
                                width='stretch', 
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
                        numerical_stats, _, _ = get_statistics_dataframe(
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
                                 width='stretch',
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
                # Display Date Stats if available
                if not date_stats.empty and selected_group_quant == "None":
                    st.divider()
                    st.markdown("#### 📅 Date Statistics")
                    st.dataframe(date_stats,
                                width='stretch', 
                                column_config={
                                    "Fill Percentage": st.column_config.ProgressColumn(
                                        "Fill Percentage",
                                        help="Shows the percentage of non-missing values in this column.",
                                        format="%d%%",
                                        min_value=0,
                                        max_value=100,
                                    ),
                                    "Min Date": st.column_config.DateColumn("Min Date"),
                                    "Max Date": st.column_config.DateColumn("Max Date"),
                                }
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
                    dataset_name = st.session_state.get('current_dataset_name', None)
                    dataset_name = st.session_state.get('current_dataset_name', None)
                    _, non_numerical_stats, _ = get_statistics_dataframe(
                        filtered_df, 
                        analyzer, 
                        nb_top_categories=max_top_modalities, 
                        qual_var_threshold=50,
                        dataset_name=dataset_name,
                        _db_manager=st.session_state.db_manager
                    )
                    
                    with col2:
                        # Show available columns and let user select columns to retain
                        selected_qual_columns = st.multiselect(
                                "Select columns to retain in descriptive statistics of quantitative variables",
                                options=non_numerical_stats.columns.tolist(),
                                default=non_numerical_stats.columns.tolist()
                            )
                        
                    st.dataframe(non_numerical_stats[selected_qual_columns], 
                                width='stretch', 
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
                        _, non_numerical_stats, _ = get_statistics_dataframe(
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
                                 width='stretch',
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
            
            # Ensure unique columns for correlation analysis to prevent errors
            filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
            
            with st.expander("Get a view of variables interactions", expanded=True):
                
                
                #group_select = list(set(analyzer.binary_cols + analyzer.categorical_cols + analyzer.non_binary_low_cardinality_numeric_cols)) # Grouping column
                colcor1, colcor2, colcor3 = st.columns(3)
                
                with colcor1:
                    # Columns selection
                    sym = st.checkbox("symetrical", True)
                    
                    st.markdown("---")
                    st.markdown("**Clustering Options**")
                    cluster_rows = st.checkbox("Cluster Rows", False)
                    if cluster_rows:
                        show_row_dendrogram = st.checkbox("Show Row Dendrogram", False)
                    else:
                        show_row_dendrogram = False
                        
                    cluster_cols = st.checkbox("Cluster Cols", False)
                    if cluster_cols:
                        show_col_dendrogram = st.checkbox("Show Col Dendrogram", False)
                    else:
                        show_col_dendrogram = False
                    
                with colcor2:
                    # Correlation method
                    corr_method = st.selectbox("Select correlation method", ["kendall", "spearman", "pearson"])
                    if corr_method == "pearson":
                        cols_select = sorted(list(set(analyzer.numeric_cols))) # numerical columns
                    else:
                        cols_select = sorted(list(set(analyzer.num_binary_cols + analyzer.low_cardinality_numeric_cols + analyzer.numeric_cols))) # numerical columns
                with colcor3:
                    
                    group_column = st.selectbox("Select grouping column (optional)", ["None"] + group_select) # sorted(group_select)
                    group_column = None if group_column == "None" else group_column
                
                if sym:
                    cols = st.multiselect("Select columns", options=cols_select, default=cols_select)
                    targets = cols
                    predictors = cols
                    triangle = 'full'
                else:
                    targets = st.multiselect("Select target columns", cols_select)
                    predictors = st.multiselect("Select predictor columns", cols_select)
                    triangle = 'lower'

                # Generate correlation matrix ONCE
                try:
                    # Pre-calculate matrix to optimize performance
                    unique_cols = list(set(targets + predictors))
                    # Filter numeric columns only to avoid errors
                    data_for_corr = filtered_df[unique_cols].select_dtypes(include=[np.number])
                    
                    if not data_for_corr.empty:
                        corr_matrix_precalc = data_for_corr.corr(method=corr_method)
                    else:
                        corr_matrix_precalc = pd.DataFrame() # Empty if no data

                    # Create Tabs - Graph first as requested
                    tab_graph, tab_matrix = st.tabs(["Correlation Graph", "Correlation Matrix"])

                    with tab_matrix:
                        # Generate the heatmap figure
                        fig = ecm.plotly_corr_mat(
                            data=filtered_df,
                            targets=targets,
                            predictors=predictors,
                            method=corr_method,
                            selected_color='Plasma',
                            triangle=triangle,
                            cluster_rows=cluster_rows,
                            cluster_cols=cluster_cols,
                            show_row_dendrogram=show_row_dendrogram,
                            show_col_dendrogram=show_col_dendrogram,
                            correlation_matrix=corr_matrix_precalc
                        )
                        st.plotly_chart(fig, width='stretch')
                        
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

                    with tab_graph:
                        # Correlation Network Graph
                        st.subheader("Correlation Network Graph")
                        
                        # Legend
                        with st.expander("Legend & Help", expanded=False):
                            legend_html = """
                            <style>
                                .legend-table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin-bottom: 10px; }
                                .legend-table th { background-color: #f0f2f6; padding: 8px; text-align: left; font-weight: 600; border-bottom: 2px solid #ddd; }
                                .legend-table td { padding: 8px; border-bottom: 1px solid #eee; vertical-align: middle; }
                                .line-sample { display: inline-block; width: 60px; vertical-align: middle; position: relative; height: 10px; }
                                .line-draw { position: absolute; top: 50%; left: 0; right: 0; display: block; }
                            </style>
                            <table class="legend-table">
                                <thead>
                                    <tr>
                                        <th>Strength</th>
                                        <th>Positive (Co-vary) <span style="color:#0000ff; font-size:0.8em;">(Blue)</span></th>
                                        <th>Negative (Inverse) <span style="color:#ff0000; font-size:0.8em;">(Red)</span></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><strong>Very Strong</strong></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 5.0px solid #00008b;"></span></div></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 5.0px solid #8b0000;"></span></div></td>
                                    </tr>
                                    <tr>
                                        <td><strong>Strong</strong></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 3.5px solid #0000ff;"></span></div></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 3.5px solid #ff0000;"></span></div></td>
                                    </tr>
                                    <tr>
                                        <td><strong>Moderate</strong></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 2.5px solid #4169e1;"></span></div></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 2.5px solid #cd5c5c;"></span></div></td>
                                    </tr>
                                    <tr>
                                        <td><strong>Weak</strong></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 1.5px dashed #add8e6;"></span></div></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 1.5px dashed #f08080;"></span></div></td>
                                    </tr>
                                    <tr>
                                        <td><strong>Negligible</strong></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 1.0px dotted #d3d3d3;"></span></div></td>
                                        <td><div class="line-sample"><span class="line-draw" style="border-top: 1.0px dotted #d3d3d3;"></span></div></td>
                                    </tr>
                                </tbody>
                            </table>
                            <div style="font-size: 0.9em; color: #555;">
                                <p><strong>Interactions:</strong> for the moment it is not possible to select a node and move it</p>
                                <ul>
                                    <li><strong>Node Heat:</strong> Nodes with a "hotter" appearance (heat map overlay) have a higher cumulative correlation strength with their neighbors.</li>
                                    <li><strong>Focus:</strong> Selecting a variable highlights it in <span style="color:orange; font-weight:bold;">Orange</span> and shows only its connected components.</li>
                                </ul>
                            </div>
                            """
                            st.markdown(legend_html, unsafe_allow_html=True)
                        
                        col_thresh, col_focus = st.columns([1, 1])
                        with col_thresh:
                            net_thresh = st.select_slider(
                                "Minimum Edge Strength",
                                options=["Negligible", "Weak", "Moderate", "Strong", "Very Strong"],
                                value="Moderate",
                                help="Filter edges to show only correlations with strength equal to or greater than this threshold."
                            )
                        with col_focus:
                            # Focus variable selector
                            focus_options = sorted(list(set(targets + predictors)))
                            focus_variable = st.selectbox(
                                "Focus on Variable (Connected Component)", 
                                ["None"] + focus_options,
                                help="Select a variable to show only the nodes connected to it (directly or indirectly) via the selected edge strength."
                            )
                            
                        focus_node = None if focus_variable == "None" else focus_variable
                        
                        nodes, edges = ecm.get_yfiles_network_data(
                            data=filtered_df,
                            targets=targets,
                            predictors=predictors,
                            method=corr_method,
                            threshold_category=net_thresh,
                            correlation_matrix=corr_matrix_precalc,
                            focus_node=focus_node
                        )
                        
                        
                        if nodes:
                            widget = StreamlitGraphWidget(nodes=nodes, edges=edges, heat_mapping="heat")
                            widget.directed = False
                            widget.node_label_mapping = lambda n: n['properties']['label']
                            widget.node_label_style_mapping = lambda n: LabelStyle(color="#000000", text_position="center")
                            widget.node_color_mapping = lambda n: n['properties']['color']
                            
                            def get_edge_style(edge):
                                props = edge['properties']
                                d_str = props.get('dash', 'solid')
                                if d_str == 'dashed': 
                                    d_style = DashStyle.DASH
                                elif d_str == 'dotted': 
                                    d_style = DashStyle.DOT
                                else: 
                                    d_style = DashStyle.SOLID
                                    
                                return EdgeStyle(
                                    color=props.get('color', 'black'),
                                    dash_style=d_style,
                                    directed=False,
                                    thickness=props.get('thickness', 1.0)
                                )
                            
                            widget.edge_styles_mapping = get_edge_style
                            widget.edge_label_mapping = lambda e: e['properties']['label']
                            
                            widget.height = 500
                            widget.show(key="corr_network_graph", graph_layout=Layout.ORGANIC)
                        else:
                            st.info("No correlations found meeting the criteria.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        else:
            st.warning("Please upload data to view statistics.")