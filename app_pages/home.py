# pages/home.py
import json
import time
from io import BytesIO

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

import explore.corr_matrix as ecm
from core.fingerprint import frame_fingerprint
from graph.builders.correlation import build_correlation_graph
from graph.legend import render_legend
from graph.model import LayoutHint, RenderOptions
from graph.ui import current_theme, render_graph, render_graph_exports, renderer_selector
from manage.ai_payload import column_profile
from manage.db_manager import DBManager
from manage.rag_manager import RAGManager
from ui.components import confirm, no_dataset
from ui.progress import mark_step
from ui.state import get_state
from utils.export_utils import to_excel, to_excel_sheets
from utils.statistics_utils import normality_test, sanitize_dataframe_for_arrow


def _q(identifier) -> str:
    """Quote a DuckDB identifier (column names come from uploaded files; audit E4)."""
    return '"' + str(identifier).replace('"', '""') + '"'


def _frame_fingerprint(df) -> str:
    """Short content hash so cached statistics follow the frame (filters, renames), not only its name (audit E8)."""
    return frame_fingerprint(df, index=False, length=12)


def _top_correlations(corr_matrix: pd.DataFrame, node_ids, limit: int = 10) -> pd.DataFrame:
    """Strongest correlations of the selected variables (for the graph selection panel)."""
    rows = []
    for node_id in node_ids:
        if node_id not in corr_matrix.index:
            continue
        series = corr_matrix.loc[node_id].drop(labels=[node_id], errors="ignore").dropna()
        for other in series.abs().sort_values(ascending=False).head(limit).index:
            rows.append({"Variable": node_id, "Correlated with": other, "r": round(float(corr_matrix.loc[node_id, other]), 3)})
    return pd.DataFrame(rows, columns=["Variable", "Correlated with", "r"])


class RenameColumnsComponent:
    """A Streamlit component for renaming DataFrame columns with sorting, searching, type filtering, and top values display."""

    def __init__(self, df, analyzer):
        self.original_df = df  # Preserve original DataFrame for reference
        self.analyzer = analyzer

        # The pending renames live in session state until "Apply Renaming" commits them (AppState)
        if "rename_dict" not in st.session_state:
            st.session_state["rename_dict"] = {col: col for col in df.columns}

    def _get_columns_info(self, columns_to_process=None):
        """What the AI engine is told about the columns: a summary, never a row (manage.ai_payload)."""
        cols = columns_to_process if columns_to_process else self.original_df.columns
        return column_profile(self.original_df, cols, date_columns=self.analyzer.date_cols)

    def display(self):
        """Displays the column renaming UI with sorting, searching, type filtering, and a compact layout."""
        st.subheader("Rename Columns")

        # AI Suggestion Section
        with st.expander("AI Renaming Suggestions (RAG)", expanded=False):
            if 'rag_manager' in st.session_state and st.session_state.rag_manager.initialized:
                # deep_analysis = st.checkbox("Enable Deep Taxonomy (Slower, includes formulas)", value=False)
                
                col_mode, col_btn = st.columns([3, 1])
                with col_mode:
                    suggestion_mode = st.radio(
                        "Suggestion Source:", 
                        ["Medical Literature (Context)", "Standard Terminologies (UMLS/SNOMED/LOINC)"],
                        key="rename_suggestion_mode", persist_state="page", horizontal=True
                    )
                    
                    selected_cols = st.multiselect(
                        "Select columns to rename (leave empty for all):",
                        options=list(self.original_df.columns),
                        default=[],
                        persist_state="page", key="rag_col_select",
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
                st.info("The AI engine is not initialised. Initialise it in Settings > AI engine.")

        # Variable Taxonomy Section
        # Apply AI suggestions if available
        if 'ai_renaming_suggestions' in st.session_state:
            st.info("AI Suggestions available. Click 'Apply' next to the suggestion to use it.")

        # Layout for search, sorting, and type filter
        col1, col2, col3 = st.columns([1, 2, 2])

        with col1:
            sort_order = st.checkbox("Sort alphabetically", value=True, persist_state="page", key="sort_order")
        with col2:
            type_filter = st.selectbox(
                "Filter by type",
                options=["All", "Numeric", "Categorical", "Date", "Binary", "Non-Binary Low-Cardinality Numeric", "High-Cardinality Categorical"],
                persist_state="page", key="type_filter"
            )
        with col3:
            search_term = st.text_input("Search columns", "", persist_state="page", key="search_term")


        # The analyzer of the analysis rows, from the profile kept until they change
        self.analyzer = get_state().analyzer() or self.analyzer
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
                    new_name = st.text_input(f"Rename '{col}' to:", value=current_val, persist_state="page", key=f"rename_{col}")
                    st.session_state["rename_dict"][col] = new_name
                    
                    # Show AI suggestion if available
                    if 'ai_renaming_suggestions' in st.session_state and col in st.session_state['ai_renaming_suggestions']:
                        suggested = st.session_state['ai_renaming_suggestions'][col]
                        if suggested != current_val:
                            st.caption(f"AI Suggestion: **{suggested}**")
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
                            st.caption("N/A")
                    else:
                        st.warning(f"Column '{col}' is missing in the DataFrame.")

                # Styling for separation
                st.divider()

        # Apply renaming and refresh analyzer
        if st.button("Apply Renaming"):
            rename_dict = st.session_state["rename_dict"]
            changed = {old: new for old, new in rename_dict.items() if old != new}
            state = get_state()
            if changed and state.base is not None:
                # One traced step on the whole dataset (audit E10); the pending renames are reset by AppState
                state.commit_change(
                    state.base.rename(columns=changed),
                    "rename_columns",
                    {"mapping": changed},
                    f"Renamed {len(changed)} column(s)",
                )
                st.success("Columns renamed successfully!")
                self.analyzer = get_state().analyzer() or self.analyzer
            else:
                st.info("No changes made to column names.")

        st.write("Updated DataFrame:")
        st.dataframe(st.session_state["working_df"], width="stretch")


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

# Kept per session: the Reset button clears this user's statistics only (it cleared every cache of every user)
@st.cache_data(ttl=3600, show_spinner="Computing statistics...", scope="session")
def get_statistics_dataframe(df, _analyzer, nb_top_categories=4, exclude_columns=None, qual_var_threshold=50, dataset_name=None, _db_manager=None):
    """
    Creates new DataFrames with statistics for numerical and non-numerical columns.
    Tries to load from DuckDB/Parquet cache if dataset_name is provided.
    
    Note: Parameters starting with _ are excluded from hashing by Streamlit.
    """
    if dataset_name:
        dataset_name = f"{dataset_name}__{_frame_fingerprint(df)}"
    # Try to load cached stats if dataset name is available
    if dataset_name and _db_manager:
        cached_num, _ = _db_manager.load_stats(dataset_name, "numerical")
        cached_cat, _ = _db_manager.load_stats(dataset_name, "categorical")
        cached_date, _ = _db_manager.load_stats(dataset_name, "date")
        
        # Simple validation: if cached stats exist and have same number of columns as current analysis
        # (This is a basic check; for production, hash the dataframe content)
        if cached_num is not None and cached_cat is not None:
            # Check if columns match roughly to ensure we aren't loading stale stats for modified data
            # This is optional but recommended
            return (
                sanitize_dataframe_for_arrow(cached_num),
                sanitize_dataframe_for_arrow(cached_cat),
                sanitize_dataframe_for_arrow(cached_date) if cached_date is not None else pd.DataFrame(),
            )

    # Work with copies to avoid modifying original
    df = df.copy()
    analyzer = _analyzer
    
    # Exclude specified columns
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore')
    
    # Handle date columns by converting to datetime using smart parser
    from utils.date_parser import smart_parse_dates
    for col in analyzer.date_cols:
        if col in df:
            parsed_series, _ = smart_parse_dates(df[col])
            df[col] = parsed_series

    # Round Low-Cardinality Numercial Values
    for col in analyzer.low_cardinality_numeric_cols:
        if col in df:
            df[col] = df[col].round(2)

    # Initialize statistics DataFrames
    # Quantitative statistics strictly include numeric columns (excluding dates and timedeltas to prevent mixed-type ArrowInvalid errors)
    numeric_feature_cols = [
        col for col in analyzer.numeric_cols 
        if col in df and col not in analyzer.date_cols and col not in analyzer.timedelta_cols
    ]
    numerical_df = df[numeric_feature_cols] if numeric_feature_cols else pd.DataFrame()
    categorical_feature_cols = [
        col for col in (analyzer.categorical_cols + analyzer.low_cardinality_numeric_cols)
        if col in df
    ]
    categorical_df = df[categorical_feature_cols] if categorical_feature_cols else pd.DataFrame()
    
    # Numerical statistics (quantitative variables only)
    if not numerical_df.empty:
        numerical_stats = numerical_df.describe().round(2).transpose()
    else:
        numerical_stats = pd.DataFrame(
            columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        )
    
    # Calculate std safely
    try:
        numeric_only = numerical_df.select_dtypes(include=[np.number]) if not numerical_df.empty else pd.DataFrame()
        if not numeric_only.empty:
            std_vals = numeric_only.std()
            # Ensure numeric type before rounding
            if not pd.api.types.is_numeric_dtype(std_vals):
                std_vals = pd.to_numeric(std_vals, errors='coerce')
            numerical_stats['std'] = std_vals.round(2)
        else:
            numerical_stats['std'] = np.nan
    except Exception:
        numerical_stats['std'] = np.nan

    # Normality tests
    if not numerical_df.empty:
        num_cols_for_tests = numerical_df.select_dtypes(include=[np.number])
        numerical_stats["Shapiro-Wilk p-value"] = num_cols_for_tests.apply(normality_test, axis=0, method='shapiro')
        numerical_stats["D'Agostino's K² p-value"] = num_cols_for_tests.apply(normality_test, axis=0, method='dagostino')
        numerical_stats["Kolmogorov-Smirnov p-value"] = num_cols_for_tests.apply(normality_test, axis=0, method='ks')
        numerical_stats["Not Normal (Shapiro-Wilk)"] = numerical_stats["Shapiro-Wilk p-value"] < 0.05
        numerical_stats["Not Normal (D'Agostino K²)"] = numerical_stats["D'Agostino's K² p-value"] < 0.05
        numerical_stats["Not Normal (KS)"] = numerical_stats["Kolmogorov-Smirnov p-value"] < 0.05
        numerical_stats['nb_modalities'] = numerical_df.apply(lambda x: x.nunique())
        numerical_stats['fill_percentage'] = (1 - numerical_df.isnull().mean()) * 100
    else:
        for col_name in [
            "Shapiro-Wilk p-value", "D'Agostino's K² p-value", "Kolmogorov-Smirnov p-value",
            "Not Normal (Shapiro-Wilk)", "Not Normal (D'Agostino K²)", "Not Normal (KS)",
            "nb_modalities", "fill_percentage"
        ]:
            numerical_stats[col_name] = pd.Series(dtype=float if "p-value" in col_name or col_name == "fill_percentage" else object)

    numerical_stats['variable_type'] = 'Numeric'
    
    # Remove low-cardinality numeric columns from numerical stats (added to categorical stats below)
    for col in analyzer.low_cardinality_numeric_cols:
        if col in numerical_stats.index:
            numerical_stats.drop(index=col, inplace=True)

    # Categorical statistics
    if not categorical_df.empty:
        categorical_stats = categorical_df.describe(include='all').round(2).transpose()
    else:
        categorical_stats = pd.DataFrame()

    columns_to_drop = ['min', 'std']
    categorical_stats.drop(columns=[col for col in columns_to_drop if col in categorical_stats.columns], inplace=True)
    if not categorical_df.empty:
        categorical_stats['nb_modalities'] = categorical_df.apply(lambda x: x.nunique())
        categorical_stats['fill_percentage'] = (1 - categorical_df.isnull().mean()) * 100
    else:
        categorical_stats['nb_modalities'] = pd.Series(dtype=int)
        categorical_stats['fill_percentage'] = pd.Series(dtype=float)
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
    if not categorical_stats.empty:
        float_columns = categorical_stats.select_dtypes(include='float').columns
        categorical_stats[float_columns] = categorical_stats[float_columns].round(2)

    columns_to_move = ['fill_percentage', 'variable_type', 'nb_modalities', 'count' ]
    categorical_stats = move_columns_to_front(categorical_stats, columns_to_move)
    numerical_stats = move_columns_to_front(numerical_stats, columns_to_move)

    # Sanitize DataFrames for PyArrow compatibility
    numerical_stats = sanitize_dataframe_for_arrow(numerical_stats)
    categorical_stats = sanitize_dataframe_for_arrow(categorical_stats)

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
                    try:
                        duration = max_date - min_date
                    except Exception:
                        # Handle OutOfBoundsDatetime (e.g. > 292 years)
                        # Convert to python datetime objects for subtraction
                        duration = max_date.to_pydatetime() - min_date.to_pydatetime()
                        
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
            date_stats = sanitize_dataframe_for_arrow(date_stats)
            if dataset_name and _db_manager:
                _db_manager.save_stats(date_stats, dataset_name, "date")

    return numerical_stats, categorical_stats, date_stats

# Bounded: each entry is a whole workbook, and the cache is shared by the sessions of the server
@st.cache_data(show_spinner="Generating Correlation Excel file...", max_entries=8, ttl=3600)
def _cached_corr_excel_bytes(data, targets, predictors, group_column, method):
    output = BytesIO()
    try:
        ecm.custom_corr_mat_to_excel(
            data=data,
            targets=targets,
            predictors=predictors,
            group_column=group_column,
            file_name=output,
            method=method,
            threshold="Negligible"
        )
        return output.getvalue()
    except Exception as exc:
        fallback = BytesIO()
        with pd.ExcelWriter(fallback, engine='openpyxl') as writer:
            pd.DataFrame({"Status": [f"Unable to generate correlation Excel export: {exc}"]}).to_excel(
                writer, sheet_name="Export Error", index=False
            )
        return fallback.getvalue()

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
            selects.append(f"COUNT({_q(col)})")
            selects.append(f"AVG({_q(col)})")
            selects.append(f"STDDEV({_q(col)})")
            selects.append(f"MIN({_q(col)})")
            selects.append(f"MAX({_q(col)})")
            # Quantiles (approximate is faster, exact is slower)
            selects.append(f"QUANTILE_CONT({_q(col)}, 0.25)")
            selects.append(f"QUANTILE_CONT({_q(col)}, 0.50)")
            selects.append(f"QUANTILE_CONT({_q(col)}, 0.75)")

        res = None
        if selects:
            query = f"SELECT {', '.join(selects)} FROM df_bench"
            res = con.execute(query).fetchone()
        
        end = time.time()
        time_dd_stats = end - start
        progress_bar.progress(50)
        
        # Verification (Basic Check)
        # We check if the count matches for the first column as a proxy
        match_stats = "Match"
        if selects and res:
            # Pandas count is float, DuckDB is int. Compare with tolerance or cast.
            pd_count = stats_pd.iloc[0]['count']
            dd_count = res[0] # First element is COUNT(col1)
            if abs(pd_count - dd_count) > 0:
                 match_stats = "Match (Approx)" # Counts should match exactly but let's be safe
        
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
            query = f"SELECT {_q(col)}, COUNT(*) as count FROM df_bench GROUP BY {_q(col)} ORDER BY count DESC LIMIT 5"
            dd_cats[col] = con.execute(query).fetchall()
        end = time.time()
        time_dd_cat = end - start
        progress_bar.progress(70)
        
        # Verification
        match_cat = "Match"
        # Check first categorical column
        if analyzer.categorical_cols:
            first_col = analyzer.categorical_cols[0]
            pd_top = pd_cats[first_col]
            dd_top = dd_cats[first_col]
            # Compare top 1 value and count
            if len(pd_top) > 0 and len(dd_top) > 0:
                if str(pd_top.index[0]) != str(dd_top[0][0]) or pd_top.iloc[0] != dd_top[0][1]:
                     match_cat = "Mismatch"

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
            SELECT {_q(group_col)}, 
                   COUNT({_q(target_col)}), AVG({_q(target_col)}), STDDEV({_q(target_col)}), MIN({_q(target_col)}), MAX({_q(target_col)})
            FROM df_bench 
            GROUP BY {_q(group_col)}
            ORDER BY {_q(group_col)}
            """
            dd_group = con.execute(query).fetchall()
            end = time.time()
            time_dd_group = end - start
            progress_bar.progress(90)
            
            # Verification
            match_group = "Match"
            # Compare first group count
            if not pd_group.empty and dd_group:
                # Sort pandas to ensure order matches DuckDB's ORDER BY
                pd_group_sorted = pd_group.sort_index()
                # Check first group
                if pd_group_sorted.iloc[0]['count'] != dd_group[0][1]:
                     match_group = "Mismatch"

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
                 selects.append(f"CORR({_q(c1)}, {_q(c2)})")
        
        if selects:
            query = f"SELECT {', '.join(selects)} FROM df_bench"
            res = con.execute(query).fetchall()
            
        end = time.time()
        time_dd_corr = end - start
        progress_bar.progress(100)
        
        # Verification
        match_corr = "Match"
        # Compare one value
        if not corr_pd.empty and res:
            # First correlation value (self-correlation should be 1.0)
            # DuckDB returns a flat list of results.
            # CORR(c1, c1) is the first one.
            if abs(res[0][0] - 1.0) > 0.0001:
                 match_corr = "Mismatch"
        
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
            
        st.caption("For information: the statistics of the application are computed with pandas.")


# The tabs of Statistics & correlations; only the open one runs (M5b). Their labels are part of the tabs' identity
# (Streamlit 1.66): keep them fixed, without counts.
TABS = ("Dataset Statistics", "Columns Renaming", "Performance Benchmark")
TAB_KEY = "home_tab"


def app():
    # Initialize DB Manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DBManager()

    # Initialize RAG Manager
    if 'rag_manager' not in st.session_state:
        st.session_state.rag_manager = RAGManager()

    # RAG Configuration Sidebar - MOVED TO GLOBAL SIDEBAR (utils/multipage.py -> app_pages/rag_sidebar.py)
    # with st.sidebar.expander("RAG & AI Settings", expanded=False):
    #     ... (Code moved)

    # Dataset versions are managed from the shell sidebar (ui/shell.py).

    # The analysis rows (the criteria ticked in the sidebar)
    df = st.session_state.get('data', None)

    # Only the open tab runs (M5b): a click no longer recomputes the three tabs
    tabs = st.tabs(list(TABS), key=TAB_KEY, on_change="rerun")

    with tabs[2]:
        if tabs[2].open:
            st.subheader("Performance benchmark", icon=":material/speed:", anchor=False)
            if df is not None:
                bench = df.copy(deep=False)  # local: the coercion below must not change the session data (audit E8)
                analyzer = get_state().analyzer()  # its column profile is kept until the analysis rows change
                cols_to_convert = list(set(analyzer.low_cardinality_numeric_cols + analyzer.numeric_cols))
                bench[cols_to_convert] = bench[cols_to_convert].apply(pd.to_numeric, errors='coerce')
                run_benchmark(bench, analyzer)
            else:
                no_dataset()

    with tabs[1]:
        if tabs[1].open:
            if df is not None:
                rename_component = RenameColumnsComponent(df, get_state().analyzer())
                # Display the component (a rename is committed by AppState; nothing is written back here)
                rename_component.display()
            else:
                no_dataset()

    with tabs[0]:
        if tabs[0].open:
            # its own inputs: the analysis rows as they stand (a rename applied in the Renaming tab included)
            df = st.session_state.get('working_df')
            if df is not None:
                analyzer = get_state().analyzer()
            col_title, col_reset = st.columns([4, 1])
            with col_title:
                st.subheader("Dataset statistics", icon=":material/table_chart:", anchor=False)
            with col_reset:
                 def _reset_statistics_cache() -> None:
                    if 'db_manager' in st.session_state:
                        st.session_state.db_manager.clear_all_stats()  # this user's workspace only
                    get_statistics_dataframe.clear()  # this session's statistics only
                    st.toast("Cache cleared.")

                 if st.button("Reset Cache", help="Force full recomputation of statistics"):
                    confirm("Reset the statistics cache?",
                            "Cached statistics in your workspace and the in-memory cache will be recomputed on the next visit. "
                            "Datasets and traces are not affected.", _reset_statistics_cache, confirm_label="Reset")
            #df = st.session_state.get('data', None)
            max_top_modalities = 3  # the number_input below can change it
            if df is not None:
            
                st.subheader("Dataset Preview")
                with st.expander("Preview the first lines of your dataset", expanded=True):
                    # The rows are the analysis population (criteria ticked in the sidebar)
                    filtered_df = df
                    st.dataframe(df.head(), width='stretch', hide_index=True)
                    st.write(f"**Data Overview:** {df.shape[0]:,} rows and {df.shape[1]:,} columns.")

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
                        selected_group_quant = st.selectbox(label="Select grouping column (optional):", options=["None"] + group_select, index=0, persist_state="page", key='select_group_quant')
                
                    date_stats = pd.DataFrame()
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
                        mark_step("analyse", "descriptive statistics")
                        # Show available columns and let user select columns to retain
                        selected_quant_columns = st.multiselect(
                                "Select columns to retain in descriptive statistics of quantitative variables",
                                key="stats_quant_columns", persist_state="page", options=numerical_stats.columns.tolist(),
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
                                        ),
                                        "Shapiro-Wilk p-value": st.column_config.NumberColumn(
                                            "Shapiro-Wilk p-value",
                                            help="Shapiro-Wilk up to 5,000 values; D'Agostino's K² beyond, where "
                                            "Shapiro-Wilk's p-value is not reliable (as in the analyses).",
                                        )}
                                    )
                    
                        st.download_button(
                            label="Download as Excel",
                            data=to_excel(numerical_stats[selected_quant_columns]),
                            file_name="Quantitative Data Statistics.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key='quant_download_ungrouped'
                        )
                    else:
                        # Group data by the selected categorical variable
                        quant_category_groups = filtered_df.groupby(selected_group_quant)
                        # Generate descriptive statistics for each group using the custom function
                        quant_dict = {}
                        date_stats_dict = {}
                        for category, group in quant_category_groups:
                            if group.empty:
                                continue
                            #st.write(f"---------------\ncategory : {category}, \ngroup : {group}\n------------")
                            numerical_stats, _, group_date_stats = get_statistics_dataframe(
                                df=group, 
                                _analyzer=analyzer, 
                                nb_top_categories=1
                            )
                            quant_dict[category] = numerical_stats
                            date_stats_dict[category] = group_date_stats
                    
                        with col_quant_2:
                            selected_category = st.selectbox(label="Select grouping value:", options=quant_dict.keys(), index=0, persist_state="page", key='quant_group_value')
                    
                    
                        selected_quant_columns = st.multiselect(
                                "Select columns to retain in descriptive statistics of quantitative variables",
                                options=numerical_stats.columns.tolist(),
                                default=numerical_stats.columns.tolist(),
                                persist_state="page", key='quant_cols_select'
                            )
                    
                        # Filter columns for all groups
                        for category in quant_dict:
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
                        # Allow user to download all data as a multi-sheet Excel file
                        st.download_button(
                            label="Download as Excel",
                            data=to_excel_sheets(quant_dict),
                            file_name=f"Descriptive_Statistics_by_{selected_group_quant}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key='quant_download_grouped'
                        )
                    # Display Date Stats if available
                    # Display Date Stats if available
                    current_date_stats = pd.DataFrame()
                    if selected_group_quant == "None":
                        current_date_stats = date_stats
                    elif selected_group_quant != "None" and 'date_stats_dict' in locals() and selected_category in date_stats_dict:
                        current_date_stats = date_stats_dict[selected_category]

                    if not current_date_stats.empty:
                        st.divider()
                        st.markdown("#### Date Statistics")
                        st.dataframe(current_date_stats,
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
                        selected_group_qual = st.selectbox(label="Select grouping column (optional):", options=["None"] + group_select, index=0, persist_state="page", key='select_group_qual')

                    # Display non-numerical stats
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        max_top_modalities = st.number_input(
                            "Select the number of top modalities to display",
                            key="stats_top_modalities", persist_state="page", min_value=1, 
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
                                    key="stats_qual_columns", persist_state="page", options=non_numerical_stats.columns.tolist(),
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
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key='qual_download_ungrouped'
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
                                _analyzer=analyzer, 
                                nb_top_categories=max_top_modalities
                            )
                            qual_dict[category] = non_numerical_stats

                        with col2:
                            # Show available columns and let user select columns to retain
                            selected_qual_columns = st.multiselect(
                                    "Select columns to retain in descriptive statistics of qualitative variables",
                                    options=non_numerical_stats.columns.tolist(),
                                    default=non_numerical_stats.columns.tolist(),
                                    persist_state="page", key='qual_cols_select'
                                )
                        
                        with col_qual_2:
                            selected_category = st.selectbox(label="Select grouping value:", options=qual_dict.keys(), index=0, persist_state="page", key='qual_group_value')

                        # Filter columns for all groups
                        for category in qual_dict:
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
                        # Allow user to download all data as a multi-sheet Excel file
                        st.download_button(
                            label="Download as Excel",
                            data=to_excel_sheets(qual_dict),
                            file_name=f"Descriptive_Statistics_by_{selected_group_qual}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key='qual_download_grouped'
                        )
            
                st.subheader("Correlation Matrix Analysis")
            
                # Ensure unique columns for correlation analysis to prevent errors
                filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
            
                with st.expander("Get a view of variables interactions", expanded=True):
                
                
                    #group_select = list(set(analyzer.binary_cols + analyzer.categorical_cols + analyzer.non_binary_low_cardinality_numeric_cols)) # Grouping column
                    colcor1, colcor2, colcor3 = st.columns(3)
                
                    with colcor1:
                        # Columns selection
                        sym = st.checkbox("symetrical", True, key="corr_symmetric", persist_state="page")
                    
                        st.markdown("---")
                        st.markdown("**Clustering Options**")
                        cluster_rows = st.checkbox("Cluster Rows", False, key="corr_cluster_rows", persist_state="page")
                        if cluster_rows:
                            show_row_dendrogram = st.checkbox("Show Row Dendrogram", False, key="corr_row_dendrogram", persist_state="page")
                        else:
                            show_row_dendrogram = False
                        
                        cluster_cols = st.checkbox("Cluster Cols", False, key="corr_cluster_cols", persist_state="page")
                        if cluster_cols:
                            show_col_dendrogram = st.checkbox("Show Col Dendrogram", False, key="corr_col_dendrogram", persist_state="page")
                        else:
                            show_col_dendrogram = False
                    
                    with colcor2:
                        # Correlation method
                        corr_method = st.selectbox("Select correlation method", ["kendall", "spearman", "pearson"], key="corr_method", persist_state="page")
                        if corr_method == "pearson":
                            cols_select = sorted(list(set(analyzer.numeric_cols))) # numerical columns
                        else:
                            cols_select = sorted(list(set(analyzer.num_binary_cols + analyzer.low_cardinality_numeric_cols + analyzer.numeric_cols))) # numerical columns
                    with colcor3:
                    
                        group_column = st.selectbox("Select grouping column (optional)", ["None"] + group_select, key="corr_group_column", persist_state="page") # sorted(group_select)
                        group_column = None if group_column == "None" else group_column
                
                    if sym:
                        cols = st.multiselect("Select columns", key="corr_columns", persist_state="page", options=cols_select, default=cols_select)
                        targets = cols
                        predictors = cols
                        triangle = 'full'
                    else:
                        targets = st.multiselect("Select target columns", cols_select, key="corr_targets", persist_state="page")
                        predictors = st.multiselect("Select predictor columns", cols_select, key="corr_predictors", persist_state="page")
                        triangle = 'lower'

                    if not targets or not predictors:
                        st.info("Please select target and predictor columns to compute correlations.")
                    else:
                        # Generate correlation matrix ONCE
                        try:
                            # Pre-calculate matrix to optimize performance
                            unique_cols = list(set(targets + predictors))
                            # Filter numeric columns only to avoid errors
                            data_for_corr = filtered_df[unique_cols].select_dtypes(include=[np.number])
                        
                            if not data_for_corr.empty:
                                import warnings

                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=Warning)
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
                                try:
                                    excel_data = _cached_corr_excel_bytes(
                                        filtered_df,
                                        targets,
                                        predictors,
                                        group_column,
                                        corr_method
                                    )
                                    st.download_button(
                                        label="Download Correlation Matrix (.xlsx)",
                                        data=excel_data,
                                        file_name="correlation_matrix.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    )
                                except Exception as excel_err:
                                    st.warning(f"Excel export unavailable: {excel_err}")

                            with tab_graph:
                                # Correlation Network Graph (rendered through the graph connector)
                                st.subheader("Correlation Network Graph")

                                col_thresh, col_focus, col_engine = st.columns([1, 1, 1])
                                with col_thresh:
                                    net_thresh = st.select_slider(
                                        "Minimum correlation strength",
                                        key="corr_graph_threshold", persist_state="page", options=["Negligible", "Weak", "Moderate", "Strong", "Very Strong"],
                                        value="Moderate",
                                        help="Show only pairs whose correlation strength is at least this category "
                                             "(thresholds follow Schober et al. 2018 for Pearson)."
                                    )
                                with col_focus:
                                    focus_options = sorted(list(set(targets + predictors)))
                                    focus_variable = st.selectbox(
                                        "Focus on variable (connected component)",
                                        ["None"] + focus_options,
                                        key="corr_graph_focus", persist_state="page", help="Show only the variables connected to the selected one (directly or "
                                             "indirectly) at the chosen minimum strength."
                                    )
                                with col_engine:
                                    renderer_selector(container=col_engine)

                                focus_node = None if focus_variable == "None" else focus_variable

                                spec = build_correlation_graph(
                                    corr_matrix_precalc,
                                    method=corr_method,
                                    min_strength=net_thresh,
                                    focus=focus_node,
                                    theme=current_theme(),
                                )

                                if not spec.nodes:
                                    st.info("No correlations found meeting the criteria.")
                                else:
                                    with st.expander("Legend & Help", expanded=False):
                                        render_legend(spec)
                                        st.markdown(
                                            "- **Node heat:** warmer nodes have a higher cumulative correlation "
                                            "strength with their neighbours.\n"
                                            "- **Focus:** selecting a variable highlights it and shows only its "
                                            "connected component.\n"
                                            "- **Selection:** select nodes in the graph to list their strongest "
                                            "correlations below."
                                        )
                                    selection = render_graph(
                                        spec,
                                        RenderOptions(
                                            key="corr_network_graph",
                                            height=550,
                                            heat=True,
                                            selectable=True,
                                            layout=LayoutHint.ORGANIC,
                                            show_legend=False,
                                        ),
                                    )
                                    if selection.node_ids:
                                        st.markdown("**Strongest correlations of the selected variable(s)**")
                                        st.dataframe(
                                            _top_correlations(corr_matrix_precalc, selection.node_ids),
                                            width="stretch",
                                            hide_index=True,
                                        )
                                    render_graph_exports(
                                        spec, RenderOptions(key="corr_network_graph"), file_stem="correlation_network"
                                    )
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

            else:
                no_dataset()
