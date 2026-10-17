"""
Formulas for derived variables and targeted imputation (moved from the Enrich page, no Streamlit).

``evaluate_formula_safely`` validates a formula against the dataset's columns, rewrites date subtractions to
give days, and evaluates it through the whitelisted evaluator (``core.expressions``). The replay engine uses
it as well, so a derived variable is recomputed exactly as it was created.
"""

import ast
import datetime as dt
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from core.expressions import evaluate_tree
from utils.data_analyzer import DataAnalyzer
from utils.date_parser import smart_parse_dates
from utils.logging_config import single_line

logger = logging.getLogger(__name__)


def validate_variable_name(name, existing_columns):
    """
    Validates the variable name and returns (is_valid, error_message).
    """
    if not name:
        return False, "Variable name cannot be empty"
    
    if name in existing_columns:
        return False, f"Variable name '{name}' already exists"
    
    if not re.match(r'^[a-zA-Z_][ a-zA-Z0-9_\[\]\(\)]*$', name):
        return False, "Variable name must start with a letter or underscore and contain only letters, numbers, some special characters, spaces and underscores"
    
    return True, None


# Helper function to check if a string represents a number
def is_number(token):
    """
    Check if a string token represents a valid number.

    Parameters:
    -----------
    token : str
        The token to check.

    Returns:
    --------
    bool
        True if the token can be converted to a float, False otherwise.
    """
    try:
        float(token)  # Try converting to float
        return True
    except ValueError:
        return False
    


def validate_and_extract_columns_ast(formula: str, dataset_columns: list) -> tuple[bool, str, list[str], list[str]]:
    """
    Parses formula using Python AST to safely extract variables and constants 
    regardless of spacing.
    """
    formula = formula.strip()
    used_columns = []

    try:
        # Handle quoted variables for AST parsing
        temp_formula = formula
        quoted_vars = {}
        # Find ""Variable Name"" pattern
        quoted_matches = re.findall(r'""([^"]+)""', temp_formula)
        for i, match in enumerate(quoted_matches):
            placeholder = f"__quoted_var_{i}__"
            quoted_vars[placeholder] = match
            temp_formula = temp_formula.replace(f'""{match}""', placeholder)

        # Parse the string into an AST
        tree = ast.parse(temp_formula, mode='eval')
        
        # Traverse the tree to find variable names (nodes of type ast.Name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                # Resolve placeholder if it exists
                var_name = quoted_vars.get(node.id, node.id)
                
                # Check if it's a known library alias or function (np, pd, log, etc.)
                if var_name in ['np', 'pd', 'log', 'sqrt', 'exp', 'abs', 'min', 'max', 'constant']: 
                    continue
                
                # Check if it matches a dataset column
                if var_name in dataset_columns:
                    used_columns.append(var_name)
                else:
                    # Logic for "quoted" columns if you keep your specific syntax
                    # or handle mismatched names here
                    return False, f"Unknown variable: '{var_name}'", [], []

        return True, None, list(set(used_columns)), []

    except SyntaxError as e:
        return False, f"Syntax Error: {e}", [], []


# Replace column names in the formula with their corresponding normalized names
def normalize_formula(formula: str, column_mapping: dict) -> str:
    """
    Replace column names in the formula with normalized names (e.g., var_1, var_2).
    
    Parameters:
    ----------
    formula : str
        The computation formula to be normalized.
    column_mapping : dict
        A mapping of original column names to normalized names.
    
    Returns:
    -------
    str
        The formula with normalized column names.
    """
    logger.debug("original formula: %s", single_line(formula))
    
    # Sort mapping by length of column name descending to prevent substring replacement issues
    # column_mapping values are Series, so we access .name
    sorted_mapping = sorted(column_mapping.items(), key=lambda x: len(str(x[1].name)), reverse=True)
    
    for normalized, original in sorted_mapping:
        col_name = str(original.name)
        logger.debug("normalized column name %s -> %s", col_name, normalized)
        # Replace each column name with its corresponding normalized variable name
        formula = formula.replace(f'""{col_name}""', f'{normalized}')
        formula = formula.replace(col_name, normalized)  # in case the column name is without quotes
        
    logger.debug("normalized formula: %s", single_line(formula))
    return formula


def extract_constant_value(constant_token: str) -> str:
    """
    Extract the value from a constant token like constant(val1) -> val1
    """
    # Remove 'constant(' prefix and ')' suffix and return the inner value
    return constant_token[9:-1]  # len('constant(') == 9


def create_variables_dict(dataset: pd.DataFrame, identified_columns: list[str], constant_tokens: list[str]) -> dict[str, pd.Series]:
    """
    Create a dictionary mapping variable names to their corresponding Series
    for both regular columns and constants.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        The input dataset
    identified_columns : List[str]
        List of valid column names used in the formula
    constant_tokens : List[str]
        List of constant tokens found in the formula
        
    Returns:
    --------
    Dict[str, pd.Series]:
        Dictionary mapping variable names to their Series
    """
    local_vars = {}
    
    # Handle regular columns
    for idx, col in enumerate(identified_columns):
        local_vars[f"var_{idx}"] = dataset[col]
    
    # Handle constants
    start_idx = len(identified_columns)
    for idx, constant_token in enumerate(constant_tokens, start=start_idx):
        constant_value = extract_constant_value(constant_token)
        # Create a Series with the constant value repeated for each row
        constant_series = pd.Series([constant_value] * len(dataset), index=dataset.index, name=constant_token)
        local_vars[f"var_{idx}"] = constant_series
    
    return local_vars


def evaluate_formula_safely(formula: str, variable_name: str, dataset: pd.DataFrame, return_series: bool = False) -> tuple[bool, Any, pd.DataFrame]:
    """
    Safely evaluate a computation formula to create a new variable in the dataset.

    Parameters:
    -----------
    formula : str
        The computation formula entered by the user.
    variable_name : str
        The name of the new variable to create.
    dataset : pd.DataFrame
        The dataset containing columns to use in the formula.
    return_series : bool, optional
        If True, returns the computed Series instead of a success message, and does not modify the dataset.

    Returns:
    --------
    tuple[bool, Any, pd.DataFrame]
        A tuple containing:
        - A boolean indicating success or failure.
        - A success message OR the computed Series (if return_series=True), OR an error message.
        - The modified DataFrame (or the original DataFrame on failure).
    """
    dataset = dataset.copy(deep=False)  # dates are parsed and the variable added on a copy (audit E8)
    column_names = dataset.columns.tolist()
    formula = formula.strip()
    
    is_valid, error_message, identified_columns, constants = validate_and_extract_columns_ast(formula, column_names)
    if not is_valid:
        return False, f"Invalid formula: {error_message}", dataset

    # Ensure date columns are properly converted
    try:
        analyzer = DataAnalyzer(dataset)
        for col in identified_columns:
            if col in analyzer.date_cols:
                # Use smart_parse_dates for robust conversion
                parsed_series, _ = smart_parse_dates(dataset[col])
                dataset[col] = parsed_series
    except Exception as e:
        logger.debug("Date conversion failed: %s", e)

    # Helper for robust subtraction (handles Date-Date -> Days automatically)
    def safe_sub(a, b):
        try:
            res = a - b
            # If result is a Series with timedelta dtype
            if hasattr(res, 'dtype') and pd.api.types.is_timedelta64_dtype(res.dtype):
                return res.dt.days
            # If result is a scalar Timedelta
            elif isinstance(res, (pd.Timedelta, dt.timedelta)):
                return res.days
            return res
        except Exception:
            return a - b

    # AST Transformer to rewrite Subtraction: A - B  -->  safe_sub(A, B)
    class DateSubRewriter(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.Sub):
                return ast.Call(
                    func=ast.Name(id='__safe_sub__', ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[]
                )
            return node

    local_vars = create_variables_dict(dataset, identified_columns, constants)
    
    logger.debug("evaluate_formula: %d variables available; formula=%s", len(local_vars), single_line(formula))
    normalized_formula = normalize_formula(formula, local_vars)
    
    local_vars['__safe_sub__'] = safe_sub # Inject helper AFTER normalization to avoid attribute error
    
    try:
        # Apply AST Transformation for safe subtraction
        # We need to parse the normalized formula again to rewrite the ops
        tree = ast.parse(normalized_formula, mode='eval')
        tree = DateSubRewriter().visit(tree)
        ast.fix_missing_locations(tree)
        
        # Whitelisted evaluation (no builtins, restricted np/pd facades): see core.expressions
        logger.debug("Normalized formula: %s", single_line(normalized_formula))
        result = evaluate_tree(tree, dataset, extra_locals=local_vars)
        
        # Convert numpy array to Series if needed
        if isinstance(result, np.ndarray):
            result = pd.Series(result, index=dataset.index)
        
        # Convert scalar to Series if needed
        if np.isscalar(result):
            result = pd.Series(result, index=dataset.index)

        # Check if result is a pandas Series with the correct length
        if isinstance(result, pd.Series) and result.shape[0] == dataset.shape[0]:
            
            # If requesting series return, skip validation and modification
            if return_series:
                 return True, result, dataset

            # Add the new variable to the dataset
            if variable_name in dataset.columns:
                return False, f"Variable '{variable_name}' already exists.", dataset

            dataset[variable_name] = result
            return True, f"Variable '{variable_name}' successfully added.", dataset

        # Handle cases where result is not a valid pandas Series
        return False, (
            f"Formula evaluation resulted in invalid output.\n"
            f"Expected a pandas Series with {dataset.shape[0]} rows.\n"
            f"Check for mismatched types or operations in your formula."
        ), dataset

    except Exception as e:
        return False, (
            f"Error evaluating formula: {e!s}\n"
            f"Make sure your formula uses valid column names and operators (+, -, *, /, etc.)."
        ), dataset


def numeric_summary(values: pd.Series) -> pd.DataFrame:
    """
    Classical and robust statistics of a numeric variable: N, missing, mean, SD, median, quartiles, min and max.

    The median and the quartiles are not moved by a few extreme values (entry errors), the mean and the SD are;
    infinite values (a division by zero) are counted apart and left out of the other statistics.
    """
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    infinite = np.isinf(numbers)
    finite = numbers[~infinite].dropna()
    rows: list[tuple[str, str]] = [
        ("N (values)", f"{len(finite):,}"),
        ("Missing", f"{int(numbers.isna().sum()):,}"),
    ]
    if infinite.any():
        rows.append(("Infinite (division by zero)", f"{int(infinite.sum()):,}"))
    if len(finite):
        q1, median, q3 = finite.quantile([0.25, 0.5, 0.75])
        rows += [
            ("Mean", f"{finite.mean():.4g}"),
            ("SD", f"{finite.std():.4g}" if len(finite) > 1 else "n/a"),
            ("Median", f"{median:.4g}"),
            ("Quartiles (Q1 to Q3)", f"{q1:.4g} to {q3:.4g}"),
            ("Min", f"{finite.min():.4g}"),
            ("Max", f"{finite.max():.4g}"),
        ]
    return pd.DataFrame(rows, columns=["Statistic", "Value"])


def far_outside(values: pd.Series, k: float = 3.0) -> tuple[pd.Index, float | None, float | None]:
    """
    Rows whose value lies beyond Tukey's far-out fences, Q1 - k IQR and Q3 + k IQR (k = 3), with the fences:
    values that are probably entry errors (a height in metres among centimetres, a weight of 0) rather than
    the tail of the distribution. Infinite values are always included.
    """
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numbers[np.isfinite(numbers)]
    if len(finite) < 4:
        return pd.Index([]), None, None
    q1, q3 = finite.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    outside = np.isinf(numbers) | (numbers < low) | (numbers > high)
    return numbers.index[outside.to_numpy()], float(low), float(high)
