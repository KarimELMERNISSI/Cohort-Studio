# pages/statistical_tests.py
import streamlit as st
import streamlit as st
import json
import re
from typing import Dict, Any, Optional

def add_statistical_test(config: Dict[str, Any], st_container):
    """
    Add a new statistical test with flexible group/target configuration.
    
    Args:
        config: Dictionary containing the configuration
        st_container: Streamlit container for rendering UI elements
    """
    st_container.subheader("Add New Statistical Test")
    
    # Basic test configuration
    test_name = st_container.text_input("Test Name")
    test_type = st_container.selectbox(
        "Test Type",
        ["t-test", "anova", "kruskal-wallis", "wilcoxon", "chi2", "fisher"]
    )
    
    # Variables selection
    variables = st_container.text_area("Variables (one per line)")
    variables = [v.strip() for v in variables.split('\n') if v.strip()]
    
    # Group configuration approach selection
    config_approach = st_container.radio(
        "Configuration Approach",
        ["Use Target Variable", "Define Groups"]
    )
    
    test_config = {
        "TEST_TYPE": test_type,
        "VARIABLES": variables,
        "CONFIG_APPROACH": config_approach
    }
    
    if config_approach == "Use Target Variable":
        target_var = st_container.text_input("Target Variable")
        test_config["TARGET_VARIABLE"] = target_var
        
    else:  # Define Groups
        groups = []
        # For ANOVA and Kruskal-Wallis, allow more groups
        max_groups = 10 if test_type in ["anova", "kruskal-wallis"] else 2
        num_groups = st_container.number_input(
            "Number of Groups",
            min_value=2,
            max_value=max_groups,
            value=2
        )
        
        st_container.write("Define each group:")
        for i in range(num_groups):
            with st_container.expander(f"Group {i+1} Configuration"):
                col1, col2 = st_container.columns(2)
                with col1:
                    group_name = st_container.text_input(f"Group {i+1} Name", key=f"name_{i}")
                    mask = st_container.text_input(f"Group {i+1} Mask Family", key=f"mask_{i}")
                with col2:
                    operator = st_container.selectbox(
                        f"Group {i+1} Operator",
                        ["UNION", "INTER"],
                        key=f"op_{i}"
                    )
                    condition = st_container.text_input(
                        f"Group {i+1} Condition",
                        key=f"condition_{i}",
                        help="Optional filtering condition for this group"
                    )
                
                groups.append({
                    "name": group_name,
                    "family_mask": mask,
                    "operator": operator,
                    "condition": condition
                })
        
        test_config["GROUPS"] = groups
    
    # Common configuration options
    col1, col2 = st_container.columns(2)
    with col1:
        test_config["MISSING_STRATEGY"] = st_container.selectbox(
            "Missing Strategy",
            ["mean", "median", "knn"]
        )
    with col2:
        test_config["DESCRIBE_GROUPS"] = st_container.checkbox("Include Descriptive Statistics")
    
    # Additional test-specific parameters
    # if test_type in ["t-test", "wilcoxon"]:
    #     test_config["PAIRED"] = st_container.checkbox("Paired Test")
    #     test_config["ALTERNATIVE"] = st_container.selectbox(
    #         "Alternative Hypothesis",
    #         ["two-sided", "greater", "less"]
    #     )
    # elif test_type in ["anova", "kruskal-wallis"]:
    #     test_config["POST_HOC"] = st_container.checkbox("Perform Post-hoc Analysis")
    #     if test_config["POST_HOC"]:
    #         test_config["POST_HOC_METHOD"] = st_container.selectbox(
    #             "Post-hoc Method",
    #             ["tukey", "bonferroni", "scheffe"]
    #         )
    
    # Save configuration button
    if st_container.button("Add Statistical Test"):
        if not test_name:
            st_container.error("Please provide a test name")
            return
        
        if not variables:
            st_container.error("Please provide at least one variable")
            return
        
        if config_approach == "Use Target Variable" and not test_config.get("TARGET_VARIABLE"):
            st_container.error("Please provide a target variable")
            return
        
        config["statistical_tests"][test_name] = test_config
        st_container.success(f"Test '{test_name}' added successfully!")


def app():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    st.subheader("Statistical Tests")
    
    with st.expander("Add New Test"):
        add_statistical_test(st.session_state.config, st)
    
    if st.session_state.config["statistical_tests"]:
        st.subheader("Existing Tests")
        for test_name, test_config in st.session_state.config["statistical_tests"].items():
            with st.expander(f"Test: {test_name}"):
                st.json(test_config)
                if st.button(f"Run {test_name}"):
                    st.write("Running test... (implement test execution)")