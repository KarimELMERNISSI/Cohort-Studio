import streamlit as st
import os
# Disable ChromaDB telemetry to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "false"

import json
import re
from typing import Dict, Any, Optional

def create_empty_config() -> Dict[str, Any]:
    """Create an empty configuration structure."""
    return {
        "mask_families": {},
        "transformations": [],
        "computed_columns": {},
        "thresholds": {
            "ZSCORE_THRESHOLD": 3,
            "IQR_TOLERANCE": 1.5,
            "CATEGORICAL_OUTLIER_THRESHOLD": 1.0
        },
        "statistical_tests": {},
        "comparisons": {},
        "output_format": {
            "CSV": False,
            "XLSX": True,
            "PDF": False,
            "PNG": False
        },
        "folder_names": {
            "DATA_FOLDER": "./data/",
            "ENRICHMENT_FOLDER": "./data/enrichment/",
            "ENRICHED_FOLDER": "./data/enriched/",
            "DESCRIPTIVE_FOLDER": "./data/descriptive/",
            "HYPOTHESIS_TESTING_FOLDER": "./data/descriptive/hypothesis_testing/",
            "OUTLIERS_FOLDER": "./data/outliers/",
            "COMPARISONS_FOLDER": "./data/comparisons/"
        },
        "data_enrichments": {},
        "unicode_latex_mapping": {}
    }


def transform_expression(expression, df_name='df'):
    # Pattern to match:
    # 1. Quoted variable names allowing spaces and special characters within quotes.
    # 2. Unquoted variable names (including single letters) that start with a letter or underscore, 
    #    and may include spaces or digits, forming multi-word names.
    variable_pattern = r"(['\"])(.*?)\1|\b([a-zA-Z_]\w*(?:\s+\w+)*)\b"
    
    def replace_variable(match):
        # Handle quoted variables (group 2)
        if match.group(1):  # Quoted match
            return f"{df_name}[\"{match.group(2)}\"]"
        # Handle unquoted variables (group 3)
        elif match.group(3):  # Unquoted variable (can be a single letter or multi-word)
            return f"{df_name}['{match.group(3).strip()}']"
        return match.group(0)  # Return the match unchanged if no replacement is needed

    # Substitute the matches in the expression for variables
    transformed_expression = re.sub(variable_pattern, replace_variable, expression)
    
    # Insert multiplication symbol * between a number and df[] if no operator exists
    transformed_expression = re.sub(r"(\d)\s*(df\['[^\]]+'\])", r"\1 * \2", transformed_expression)
    
    return transformed_expression


def add_mask_family(config: Dict[str, Any], st_container):
    """Add a new mask family."""
    st_container.subheader("Add New Mask Family")
    family_name = st_container.text_input("Family Name")
    
    if family_name:
        if family_name not in config["mask_families"]:
            config["mask_families"][family_name] = {}
        
        mask_name = st_container.text_input("Mask Name")
        if mask_name:
            is_numeric = st_container.checkbox("Is Numeric")
            
            if is_numeric:
                is_lower_bound = st_container.checkbox("Need lower bound ?")
                is_upper_bound = st_container.checkbox("Need upper bound ?")
                if is_lower_bound:
                    lower_bound = st_container.number_input("Lower Bound", value=0.0)
                if is_upper_bound:
                    upper_bound = st_container.number_input("Upper Bound", value=0.0)
                if is_lower_bound and is_upper_bound:
                    strategy = st_container.selectbox("Strategy", ["exclude", "include"])
                
                if st_container.button("Add Mask"):
                    if is_lower_bound and is_upper_bound:
                        config["mask_families"][family_name][mask_name] = {
                            "numeric": True,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                            "strategy": strategy
                        }
                    elif is_lower_bound:
                        config["mask_families"][family_name][mask_name] = {
                            "numeric": True,
                            "lower_bound": lower_bound
                        }
                    elif is_upper_bound:
                        config["mask_families"][family_name][mask_name] = {
                            "numeric": True,
                            "upper_bound": upper_bound
                        }
                        
            else:
                expression = st_container.text_area("Expression")
                use_simplified_mask_expression = st_container.checkbox("Use Simplified Mask Expression")
                if st_container.button("Add Mask"):
                    if use_simplified_mask_expression:
                        expression = transform_expression(expression)
                    config["mask_families"][family_name][mask_name] = {
                        "numeric": False,
                        "expression": expression
                    }


def add_feature(config: Dict[str, Any], st_container):
    """Add a new mask family."""
    st_container.subheader("Add New Feature")
    feature_name = st_container.text_input("Feature Name")
    expression = st_container.text_area("Feature Expression", help="Formula to compute the new feature from initial dataset columns")
    use_simplified_expression = st_container.checkbox("Use Simplified Expression")
    if st_container.button("Add Feature"):
        if feature_name and expression:
            if use_simplified_expression:
                expression = transform_expression(expression)
            config["computed_columns"][feature_name] = {"expression": expression}


def add_conversion(config: Dict[str, Any], st_container):
    """Add a new transformation (ie. unit of measure conversion)."""
    st_container.subheader("Add New Measurement Unit Conversion")
    
    col1, col2 = st_container.columns(2)
    
    with col1:
        condition_col = st_container.text_input("Measurement Unit Column")
        source_col = st_container.text_input("Source Column")
        target_col = st_container.text_input("Target Column")
    
    with col2:
        condition_source = st_container.text_input("Source Measurement Unit")
        condition_target = st_container.text_input("Target Measurement Unit")
        conversion_rate = st_container.number_input("Conversion Rate", value=1.0,format="%0.4f")
    
    if st_container.button("Add Measurement Unit Conversion"):
        transform = {
            "condition_column": condition_col,
            "condition_source_expr": f"df[{condition_col}] == '{condition_source}'",
            "condition_target_expr": f"df[{condition_col}] == '{condition_target}'",
            "source_column": source_col,
            "target_column": target_col,
            "conversion_rate": conversion_rate,
            "transformation_source": f"df.loc[condition, parameter_column] * {conversion_rate}",
            "transformation_target": f"df.loc[condition, parameter_column] / {conversion_rate}"
        }
        config["transformations"].append(transform)


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


def add_comparison(config: Dict[str, Any], st_container):
    """Add a new comparison."""
    st_container.subheader("Add New Comparison")
    
    comparison_name = st_container.text_input("Comparison Name")
    folder = st_container.text_input("Input Comparison Folder Path", value="./data/outliers/")
    identifier = st_container.text_input("Common Row Identifier")
    file1 = st_container.text_input("Input File 1")
    file2 = st_container.text_input("Input File 2")
    output = st_container.text_input("Output File")
    
    if st_container.button("Add Comparison"):
        config["comparisons"][comparison_name] = {
            "INPUT_FILES_FOLDER": folder,
            "COMMON_ROW_IDENTIFIER": identifier,
            "INPUT_FILE_1": file1,
            "INPUT_FILE_2": file2,
            "OUTPUT_FILE": output
        }


def add_enrichment(config: Dict[str, Any], st_container):
    """Add a new data enrichment."""
    st_container.subheader("Add New Data Enrichment")
        
    enrichment_name = st_container.text_input("Enrichment Name")
    if enrichment_name:
        folder = st_container.text_input("Input Enrichment Folder Path", value="./data/enrichment/")
        input_file = st_container.text_input("Input File")
        strategy = st_container.selectbox("Strategy", ["left", "right", "outer", "inner", "cross"])

        # Group configuration approach selection
        row_id_approach = st_container.radio(
            "Row Identifier",
            ["Common Row Identifier", "Distinct Row Identifiers"]
        )

        if row_id_approach == "Common Row Identifier":
            identifier = st_container.text_input("Row Identifier")

            if st_container.button("Add Enrichment"):
                config["data_enrichments"][enrichment_name] = {
                    "INPUT_FILES_FOLDER": folder,
                    "INPUT_FILE": input_file,
                    "COMMON_ROW_IDENTIFIER": identifier,
                    "STRATEGY": strategy
                }
        else:
            left_identifier = st_container.text_input("Left Row Identifier")
            right_identifier = st_container.text_input("Right Row Identifier")

            if st_container.button("Add Enrichment"):
                config["data_enrichments"][enrichment_name] = {
                    "INPUT_FILES_FOLDER": folder,
                    "INPUT_FILE": input_file,
                    "LEFT_ROW_IDENTIFIER": left_identifier,
                    "RIGHT_ROW_IDENTIFIER": right_identifier,
                    "STRATEGY": strategy
                }


def main():
    st.set_page_config(page_title="Config Editor", layout="wide")
    st.title("Configuration File Editor")
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = create_empty_config()
    
    # Sidebar options
    action = st.sidebar.radio(
        "Action",
        ["Load Existing Config", "Create New Config"]
    )
    
    if action == "Load Existing Config":
        uploaded_file = st.sidebar.file_uploader("Upload configuration file", type=['json'])
        if uploaded_file:
            st.session_state.config = json.load(uploaded_file)
    
    # Main editor tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Mask Families",
        "Computed Features",
        "Measurement Unit Conversions",
        "Statistical Tests",
        "Comparisons",
        "Enrichments",
        "Settings"
    ])
    
    with tab0:
        add_mask_family(st.session_state.config, st)
        
        st.subheader("Current Mask Families")
        st.json(st.session_state.config["mask_families"])
    
    with tab1:
        add_feature(st.session_state.config, st)
        
        st.subheader("Current Features")
        st.json(st.session_state.config["computed_columns"])

    with tab2:
        add_conversion(st.session_state.config, st)
        
        st.subheader("Current Measurement Unit Conversions")
        st.json(st.session_state.config["transformations"])
    
    with tab3:
        add_statistical_test(st.session_state.config, st)
        
        st.subheader("Current Statistical Tests")
        st.json(st.session_state.config["statistical_tests"])
    
    with tab4:
        add_comparison(st.session_state.config, st)
        st.subheader("Current Comparisons")
        st.json(st.session_state.config["comparisons"])
        
    with tab5:
        add_enrichment(st.session_state.config, st)
        st.subheader("Current Enrichments")
        st.json(st.session_state.config["data_enrichments"])

    with tab6:
        st.subheader("Thresholds")
        thresholds = st.session_state.config["thresholds"]
        for key in thresholds:
            thresholds[key] = st.number_input(key, value=float(thresholds[key]))
        
        st.subheader("Output Formats")
        formats = st.session_state.config["output_format"]
        for key in formats:
            formats[key] = st.checkbox(key, value=formats[key])
    
    # Save configuration
    if st.button("Save Configuration"):
        st.download_button(
            "Download Configuration File",
            json.dumps(st.session_state.config, indent=4),
            "config.json",
            "application/json"
        )

if __name__ == "__main__":
    main()