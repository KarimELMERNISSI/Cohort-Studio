######################################## PACKAGES ########################################
import ast  # dynamic python script interpretation
import logging  # Added for logging

import pandas as pd  # for Dataframes manipulation

######################################## CUSTOM CONDITIONAL MASKS FUNCTIONS DEFINITION ######################################## 

def create_mask(df, condition, mask_name):
    """
    Create a mask based on conditions specified in the config file.

    Parameters:
        df (pd.DataFrame): The DataFrame to create the mask from.
        condition (dict): A dictionary containing the conditions for creating the mask.
        mask_name (str): The name of the mask.

    Returns:
        pd.Series or None: The mask as a boolean Series or None if an error occurs.
    """
    try:
        print("In create_mask:---",mask_name,":",condition)
        if "expression" in condition:
            print("create_mask:expression:",condition)
            return create_expression_mask(df, condition["expression"])
        elif condition.get("numeric"):
            print("numeric create_mask:numeric:",condition)
            lower_bound = condition.get("lower_bound")
            upper_bound = condition.get("upper_bound")
            strategy= condition.get("strategy")
            return create_numeric_mask(df[mask_name], lower_bound=lower_bound, upper_bound=upper_bound, strategy=strategy)
        else:
            logging.warning(f"Fail '{condition}' due to condition type errors. It should be expression or numeric")
            return None
    except Exception as e:
        logging.warning(f"An error occurred while creating mask {mask_name}: {e} - As a consequence, it won't be applied to the outliers detection")
        return None


def create_expression_mask(df, expression):
    """
    Create a boolean mask based on the evaluation of the given expression using the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to evaluate the expression on.
        expression (str): The expression to evaluate.

    Returns:
        pd.Series: A boolean Series representing the result of the expression evaluation.
    
    Raises:
        ValueError: If an error occurs during expression evaluation.
    """
    try:
        # Parse the expression and convert it into a valid Python expression
        parsed_expr = ast.parse(expression, mode='eval')
        transformed_expr = ast.Expression(body=ast.fix_missing_locations(parsed_expr.body))
        compiled_expr = compile(transformed_expr, filename="<string>", mode='eval')

        # Evaluate the expression using the DataFrame
        result = eval(compiled_expr, globals(), {'df': df})

        # Return the result as a boolean Series
        return pd.Series(result, index=df.index, dtype=bool)
    except Exception as e:
        raise ValueError(f"Error in evaluating expression: {e!s}")


def create_numeric_mask(series, lower_bound=None, upper_bound=None, strategy=None, exclude_na = True): 
    """
    Create a boolean mask based on numeric conditions applied to a Series.

    Parameters:
        series (pd.Series): The Series to create the mask from.
        lower_bound (float, optional): The lower bound for the numeric condition. Defaults to None.
        upper_bound (float, optional): The upper bound for the numeric condition. Defaults to None.
        strategy (str, optional): The strategy to apply when creating the mask. Can be 'include' if you want to keep the values inside the boundaries as outliers, 'exclude' if you want to keep the values outside the boundaries as outliers, or None if you use only one boundary. Defaults to None.
        exclude_na (bool, optional): Whether to exclude NaN values from the mask. Defaults to True.

    Returns:
        pd.Series: A boolean Series representing the numeric mask.

    Raises:
        ValueError: If an invalid strategy is provided.
    """
    print("create_numeric_mask:in numeric function!!")
    try:
        if strategy not in ['include', 'exclude', None]:
            raise ValueError("Invalid strategy. Use 'include', 'exclude', or None.")
        if lower_bound is None and upper_bound is None:
            return pd.Series(True, index=series.index, dtype=bool)
        
        mask = pd.Series(True, index=series.index, dtype=bool)

        if exclude_na and series.isna().any():
            mask[series.isna()] = False
        if lower_bound is not None and upper_bound is None:
            print("create_numeric_mask: lower_bound is not None")
            mask &= series >= lower_bound
        if upper_bound is not None and lower_bound is None:
            print("create_numeric_mask: upper_bound is not None")
            mask &= series <= upper_bound
        if strategy == 'include':
            print(f"create_numeric_mask: including values between {lower_bound} and {upper_bound}")
            mask &= (series >= lower_bound) & (series <= upper_bound)
        elif strategy == 'exclude':
            print(f"create_numeric_mask: excluding values between {lower_bound} and {upper_bound}")
            mask &= ~((series >= lower_bound) & (series <= upper_bound))

        return mask
    
    except Exception as e:
        # Handle errors gracefully
        logging.error(f"An error occurred while creating numeric mask: {e}")
        # Return a mask with all values set to False if an error occurs
        return pd.Series(False, index=series.index, dtype=bool)
    

def zip_masks(df, config, mask_type=None): 
    """
    Zip masks created from the configuration with their respective names.

    Parameters:
        df (pd.DataFrame): The DataFrame to create masks from.
        config (dict from json data file): The configuration containing mask definitions.
        mask_type (str, optional): The type of masks to create from the configuration. If None, it will take all the masks of the config data given in parameter.

    Returns:
        List[Tuple[str, pd.Series]]: A list of tuples containing the mask names and their corresponding boolean Series.

    Raises:
        ValueError: If an error occurs while creating a mask.
    """
    mask_names = []
    masks = []

    if mask_type:
        config = config.get(mask_type, {})
    for mask_name, mask_config in config.items():
        if mask_name!= "operator":
            mask_names.append(mask_name)
            print("apply_masks:",mask_name)
            print("apply_masks:",mask_config)
            masks.append(create_mask(df=df, mask_name=mask_name, condition=mask_config))
            print("append:",mask_name,mask_config)
    print("m: ",mask_names)
    return zip(mask_names, masks)

######################################## CUSTOM CONDITIONAL MASK TAGGING FUNCTION DEFINITION ######################################## 

def tag_masks(row):
    """
    Create a comma-separated list of expert tests columns based on specified conditions in a row.

    Parameters:
    - row (pd.Series): A row in the DataFrame.

    Returns:
    - str: Comma-separated list of columns with expert test outliers.
    """
    cond_activ_columns = []
    for col in row.index:
        if row[col]:
            cond_activ_columns.append(col)
    return ', '.join(map(str, cond_activ_columns))

######################################## FUNCTIONS DEFINITION FOR COMPUTING DERIVED METRICS ######################################## 

def generate_computed_column(df, config):
    """
    Generate and append computed columns to a DataFrame based on the provided configuration and the DataFrame itself.
    This is a part of the process of derivating new computed features.
    Parameters:
        df (pd.DataFrame): The DataFrame to which computed columns will be added.
        config (Dict[str, Any]): The configuration dictionary containing computed column information.

    Returns:
        None
    """
    #print("list computed col:", config.get("computed_columns", {}))
    for computation_name, computation_config in config.get("computed_columns", {}).items():
        column_expr = computation_config.get("expression")
        if column_expr is not None:
            if computation_name not in df.columns:
                print("created column:", computation_name,":", column_expr)
                apply_computation(df=df, column_name=computation_name, computation_expr=column_expr)
        else:
            logging.warning(f"No expression found for computed column {computation_name}. Skipping...")


def apply_computation(df, column_name, computation_expr):
    """
    Apply a computation expression to a DataFrame and create a new column based on the result.
    This is a part of the process of derivating new computed features.

    Parameters:
        df (pd.DataFrame): The DataFrame to which the computation will be applied.
        column_name (str): The name of the new column to be created.
        computation_expr (str): The computation expression to be applied.

    Returns:
        None
    """
    try:
        print("\n----------------------------------------------------------------------------")
        print("column_name:", column_name)
        print("computation_expr:", computation_expr)
        print("\n----------------------------------------------------------------------------\n")
        computed_col = create_computation(df=df, expression=computation_expr)
        if computed_col is not None:
            df[column_name] = computed_col
    except Exception as e:
        logging.warning(f"An error occurred while applying new column computation: {e!s} - As a consequence, the column won't be added")


def create_computation(df, expression):
    """
    Parse and evaluate a given expression within the context of a DataFrame.
    This is a part of the process of derivating new computed features.

    Parameters:
        df (pd.DataFrame): The DataFrame used as the context for the computation.
        expression (str): The expression to be parsed and evaluated.

    Returns:
        Any: The result of the evaluated expression.
    """
    try:
        print("\nin computation --------------", expression)
        parsed_expr = ast.parse(expression, mode='eval')
        print("\nparse",parsed_expr)
        computed_expr = ast.Expression(
            body=ast.fix_missing_locations(parsed_expr.body)
        )
        compiled_expr = compile(computed_expr, filename="<string>", mode='eval')
        result = eval(compiled_expr, globals(), {'df': df})
        return result
    except Exception as e:
        raise ValueError(f"Error in evaluating expression: {e!s}")