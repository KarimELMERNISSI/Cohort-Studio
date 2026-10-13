"""Analysis configuration widgets of the Analyze page (moved out of utils so the engine side stays UI-free)."""

import streamlit as st

from biostats.multiplicity import UI_OPTIONS as CORRECTION_OPTIONS


def render_analysis_configuration(df, numeric_cols, categorical_cols, binary_cols, group_col_options=None, default_group_col="None", target_col_options=None, key_prefix=""):
    """
    Renders the Analysis Configuration expander.
    Returns a dictionary with the selected configuration.
    """
    with st.expander("Analysis Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Select Grouping Variable
            if group_col_options is None:
                potential_groupers = [col for col in categorical_cols + binary_cols + numeric_cols 
                                      if df[col].nunique() < 20]
                group_col_options = ["None"] + sorted(potential_groupers)
            
            group_col = st.selectbox(
                "Select Grouping Variable (Factor)", 
                options=group_col_options,
                index=group_col_options.index(default_group_col) if default_group_col in group_col_options else 0,
                help="Select a categorical variable to define groups (e.g., Treatment vs Control).",
                key=f"{key_prefix}group_col"
            )
            
            # Group Filtering
            selected_groups = []
            if group_col != "None" and group_col in df.columns:
                unique_groups = sorted(df[group_col].dropna().unique())
                selected_groups = st.multiselect(
                    "Groups to Include",
                    options=unique_groups,
                    default=unique_groups,
                    help="Deselect groups with small sample sizes (e.g., < 5) to exclude them from analysis.",
                    key=f"{key_prefix}selected_groups"
                )
            elif group_col != "None":
                 # Handle case where group_col is not in df (e.g. "Missingness Status" created on the fly)
                 # In that case, we might not be able to filter groups easily here without the derived df.
                 # For MAR check, we usually just have 2 groups (Missing, Observed).
                 pass

        with col2:
            # Select Target Variables
            if target_col_options is None:
                target_col_options = [c for c in df.columns if c != group_col]
            
            # Default selection logic
            default_targets = []
            if numeric_cols:
                default_targets = [c for c in numeric_cols if c != group_col][:5]
            
            target_cols = st.multiselect(
                "Select Variables to Test",
                options=target_col_options,
                default=default_targets,
                help="Select variables to compare across groups.",
                key=f"{key_prefix}target_cols"
            )
            
        # Global Test Preference
        col3, col4 = st.columns(2)
        with col3:
            test_preference = st.selectbox(
                "Test Selection Strategy",
                ["Auto-Detect (Recommended)", "Force Parametric (T-test/ANOVA)", "Force Non-Parametric (Mann-Whitney/Kruskal)"],
                help="Auto-Detect checks normality and variance assumptions. You can force specific test families if needed.",
                key=f"{key_prefix}test_pref"
            )
        with col4:
            correction_method = st.selectbox(
                "Multiple Testing Correction",
                list(CORRECTION_OPTIONS),
                help="The family is the set of variables tested in this run. Holm controls the family-wise error rate and is "
                     "more powerful than Bonferroni; Benjamini-Hochberg controls the false discovery rate (exploratory screens).",
                key=f"{key_prefix}correction"
            )
            
    return {
        "group_col": group_col,
        "selected_groups": selected_groups,
        "target_cols": target_cols,
        "test_preference": test_preference,
        "correction_method": correction_method
    }


__all__ = ["render_analysis_configuration"]
