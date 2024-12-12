######################################## PACKAGES ########################################
import logging  # Added for logging
import ast # dynamic python script interpretation
import pandas as pd # for Dataframes manipulation
import numpy as np # extend some specific Dataframes manipulation

######################################## TRANSFORMATION FUNCTIONS DEFINITION ######################################## 

def global_column_mapping(unit_columns, gl_columns, mmoll_columns, unit_column_suffix, gl_column_suffix, mmoll_column_suffix):
    """
    Creates a global column mapping for different units based on the provided column lists and suffixes.
    This function is a part of a process for handling multiple columns with same names but different content.
    The aim of this process is to remap column names based on column contents before using them for filling columns empty values with the transformation function (units or g/L or mmol/L).

    Parameters:
    - unit_columns (list): List of column names for units.
    - gl_columns (list): List of column names for g/L units.
    - mmoll_columns (list): List of column names for mmol/L units.
    - unit_column_suffix (str): Suffix for columns related to units.
    - gl_column_suffix (str): Suffix for columns related to g/L units.
    - mmoll_column_suffix (str): Suffix for columns related to mmol/L units.

    Returns:
    - dict: A dictionary mapping original column names to their corresponding global column names.
    """
    column_tuples = [(unit_columns, unit_column_suffix), (gl_columns, gl_column_suffix), (mmoll_columns, mmoll_column_suffix)]
    final_mapping = create_column_mapping(column_tuples)
    return final_mapping


def create_column_mapping(column_tuples):
    """
    Creates a mapping between original column names and their corresponding global column names.
    This function is a part of a process for handling multiple columns with same names but different content.
    The aim of this process is to remap column names based on column contents before using them for filling columns empty values with the transformation function (units or g/L or mmol/L).

    Parameters:
        column_tuples (List[Tuple[List[str], str]]): A list of tuples, where each tuple contains a list of
            column names and a column suffix.

    Returns:
        dict: A dictionary mapping original column names to their corresponding global column names.
    """
    final_mapping = {}

    for columns, column_suffix in column_tuples:
        for column in columns:
            base_name_parts = column.split('.')
            base_name = base_name_parts[0] if len(base_name_parts) > 1 else column
            if column_suffix is None:
                new_column_name = base_name
            else:
                new_column_name = base_name + '.' + column_suffix
            final_mapping[column] = new_column_name

    return final_mapping


def count_non_null_values(df, unit_column, unit_value='mmol/L'):
    """
    Counts non-null values in specific columns of a DataFrame that match a given unit value in a specified unit column.

    Parameters:
        df (pd.DataFrame): The DataFrame to search for non-null values.
        unit_column (str): The name of the unit column.
        unit_value (str, optional): The unit value to filter rows (default: 'mmol/L').

    Returns:
        Union[pd.Series, None]: A Series containing counts of non-null values for each base column in the filtered rows,
            or None if no base columns are found.
    """
    base_columns = find_measures_with_unit_column(df, unit_column)
    if not base_columns:
        return None
    # Apply mask to filter rows where unit column value is unit_value(default: 'mmol/L')
    unit_mask = df[unit_column] == unit_value
    unit_rows = df[unit_mask]
    # Count non-null values for each base column in the filtered rows
    non_null_counts = unit_rows[base_columns].count()  
    return non_null_counts


def find_measures_with_unit_column(columns, unit_column):
    """
    Finds base columns associated with a given unit column in a list of column names.

    Parameters:
        columns (List[str]): The list of column names to search.
        unit_column (str): The name of the unit column.

    Returns:
        List[str]: A list of base column names associated with the unit column.
    """
    base_columns = [col for col in columns if col.startswith(unit_column) and col != unit_column]
    
    if not base_columns:
        # Attempt to extract base name without suffix
        base_name_parts = unit_column.split('.')
        if len(base_name_parts) > 1:
            base_name = base_name_parts[0]
            base_columns = [col for col in columns if col.startswith(base_name) and col != unit_column]
    
    return base_columns


def find_unit_columns(df, units, columns):
    """
    Finds columns containing unit information in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to search.
        units (List[str]): A list of unit strings to search for.
        columns (List[str]): A list of column names to search.

    Returns:
        List[str]: A list of column names that contain unit information.
    """
    unit_columns = []
    for column in columns:
        unique_values = df[column].unique()
        if any(unit in unique_values for unit in units):
            unit_columns.append(column)
    return unit_columns


def retrieve_transform_column_names(config):
    """
    Retrieves column names involved in transformations from the configuration json data.

    Parameters:
        config : The configuration json data containing transformation information.

    Returns:
        List[str]: The list of column names involved in transformations.
    """
    try:
        tr_col_names = []
        for transformation in config.get("transformations", []):
            tr_col_names.append(transformation["condition_column"])
            tr_col_names.append(transformation["source_column"])
            tr_col_names.append(transformation["target_column"])
        return tr_col_names
    except Exception as e:
        logging.error(f"Error in listing transformation columns: {str(e)}")
        return None
    

def transform_column(df, config):
    """
    Apply column transformations to a DataFrame based on the provided configuration.

    Parameters:
        df (pd.DataFrame): The DataFrame to be transformed.
        config (Dict[str, Any]): The configuration json data containing transformation information.

    Returns:
        None
    """
    for transformation in config.get("transformations", []):
        # retrieve configuration information
        condition_source = create_condition_mask(df, transformation["condition_source_expr"])
        condition_target = create_condition_mask(df, transformation["condition_target_expr"])
        source_column = transformation["source_column"]
        target_column = transformation["target_column"]
        conversion_rate = transformation["conversion_rate"]
        # apply transformations
        apply_transformation(df=df, condition=condition_target, parameter_column=target_column, transformed_column=source_column, conversion_rate=conversion_rate, transformation_expr=transformation["transformation_source"])
        apply_transformation(df=df, condition=condition_source, transformed_column=target_column, parameter_column=source_column, conversion_rate=conversion_rate, transformation_expr=transformation["transformation_target"])



def create_condition_mask(df, expression):
    """
    Parse and evaluate a given expression within the context of a DataFrame,
    generating a boolean mask based on the evaluation result.
    This is a part of the process of applying transformation, it'll drive where the transformation is applied.

    Parameters:
        df (pd.DataFrame): The DataFrame used as the context for the condition mask.
        expression (str): The expression to be parsed and evaluated.

    Returns:
        pd.Series: A boolean mask indicating the evaluation result for each row in the DataFrame.
    """
    try:
        parsed_expr = ast.parse(expression, mode='eval')
        transformed_expr = ast.Expression(
            body=ast.fix_missing_locations(parsed_expr.body)
        )
        compiled_expr = compile(transformed_expr, filename="<string>", mode='eval')
        result = eval(compiled_expr, globals(), {'df': df})
        return pd.Series(result, index=df.index, dtype=bool)
    except Exception as e:
        raise ValueError(f"Error in evaluating expression: {str(e)}")


def create_transformation(df, expression, condition_column, parameter_column, condition, conversion_rate):
    """
    Create a transformation to apply to a DataFrame based on an expression, with optional conditional application and conversion rates.
    This is a part of the process of applying transformation.

    Parameters:
        df (pd.DataFrame): The DataFrame to be transformed.
        expression (str): The expression defining the transformation.
        condition_column (str): The name of the column used for conditional application of the transformation.
        parameter_column (str): The name of the column containing parameters for the transformation.
        condition (pd.Series): A boolean Series representing the condition for applying the transformation.
        conversion_rate (float): The conversion rate used in the transformation.

    Returns:
        pd.Series: The transformed column as a Series.
    """
    try:
        print("\nin transformation --------------", expression)
        parsed_expr = ast.parse(expression, mode='eval')
        print("\nparse",parsed_expr)
        transformed_expr = ast.Expression(
            body=ast.fix_missing_locations(parsed_expr.body)
        )
        compiled_expr = compile(transformed_expr, filename="<string>", mode='eval')
        result = eval(compiled_expr, globals(), {'df': df, 'condition_column':condition_column, 'parameter_column':parameter_column,'condition':condition,'conversion_rate':conversion_rate})
        return result
    except Exception as e:
        raise ValueError(f"Error in evaluating expression: {str(e)}")


def apply_transformation(df, condition, parameter_column, transformed_column, conversion_rate, transformation_expr):
    """
    Apply the created transformation to a DataFrame based on specified conditions and expressions.
    This is a part of the process of applying transformation.

    Parameters:
        df (pd.DataFrame): The DataFrame to be transformed.
        condition (pd.Series): A boolean Series representing the condition for applying the transformation.
        parameter_column (str): The name of the column containing parameters for the transformation.
        transformed_column (str): The name of the column to be transformed.
        conversion_rate (float): The conversion rate used in the transformation.
        transformation_expr (str): The expression defining the transformation.

    Raises:
        ValueError: If an error occurs during the transformation.

    Returns:
        None
    """
    try:
        
        print("\n----------------------------------------------------------------------------")
        print(df[[parameter_column,transformed_column,condition.name]].head(15))
        print(condition.name)
        print(parameter_column)
        print(transformed_column)
        print(conversion_rate)
        print(transformation_expr)
        print("\n----------------------------------------------------------------------------\n")
        # Convert '[]' to NaN (assuming the target column is numeric)
        df[transformed_column] = df[transformed_column].replace('[]', np.nan)
        df.loc[condition, transformed_column] = create_transformation(df=df, expression=transformation_expr,condition_column=condition.name,parameter_column=parameter_column, condition=condition,conversion_rate=conversion_rate)
        print("\ntransf: ____ cond:",condition,"\n_____param:", parameter_column, "\n____rate:", conversion_rate)
    except Exception as e:
        raise ValueError(f"Error in applying transformation: {str(e)}")

