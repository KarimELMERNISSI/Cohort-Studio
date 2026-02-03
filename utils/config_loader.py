
import re
from typing import Dict, Any

def create_empty_config() -> Dict[str, Any]:
    """
    Create an empty configuration structure.
    
    Returns:
    --------
    dict
        A dictionary containing the default configuration structure for the application,
        including sections for masks, transformations, thresholds, and folder paths.
    """
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
    """
    Transform a user-friendly expression into a valid Python expression for DataFrame filtering.
    
    Parameters:
    -----------
    expression : str
        The input expression (e.g., "Age > 18").
    df_name : str, optional
        The name of the dataframe variable to use in the expression (default is 'df').
        
    Returns:
    --------
    str
        The transformed expression executable in Python (e.g., "df['Age'] > 18").
    """
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
    transformed_expression = re.sub(variable_pattern, replace_variable, expression.strip())
    
    # Insert multiplication symbol * between a number and df[] if no operator exists
    transformed_expression = re.sub(r"(\d)\s*(df\['[^\]]+'\])", r"\1 * \2", transformed_expression)
    
    return transformed_expression
