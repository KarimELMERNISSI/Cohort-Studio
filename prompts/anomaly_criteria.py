"""Prompt for generating anomaly and inclusion criteria."""

def anomaly_criteria_prompt(
    description: str,
    columns_info: str,
    sample_data: str = "",
    mode: str = "anomaly"
) -> str:
    """
    Generates a prompt for converting natural language to boolean expressions.
    
    Args:
        description: User's natural language description of the criteria.
        columns_info: List of available columns in the dataframe.
        mode: 'anomaly' or 'inclusion'.
        
    Returns:
        Formatted prompt string.
    """
    
    task_desc = "identify DATA ANOMALIES (rows to flag as suspicious)" if mode == "anomaly" else "define INCLUSION CRITERIA (rows to KEEP in the study)"
    
    return f"""Role: Data Science Expert.

Task: You are helping a user {task_desc}. converting their natural language description into valid Python Pandas boolean expressions.

Context:
- The dataframe is named `df`.
- **CRITICAL:** You must ONLY use the provided columns. Do NOT invent new column names.
- If a user asks for a variable not in the list, try to find the closest match from the Available Columns.
- Return a JSON object with a list of criteria.

Available Columns:
{columns_info}

Sample Data (Use this to infer date formats, value ranges, and categorical values):
{sample_data}

User Description:
"{description}"

Instructions:
1. Parse the user's description to identify specific conditions.
2. If the user asks for multiple criteria, split them into separate items in the list.
3. Write a VALID Python boolean expression for each condition (e.g., `(df['age'] > 18) & (df['status'] == 'Active')`).
    - **Date Logic:** Check the Sample Data. If dates look like 'DD/MM/YYYY', use `format='%d/%m/%Y'`. If ambiguous, use `dayfirst=True` and `errors='coerce'`.
    - **Example:** `(pd.to_datetime(df['visit'], dayfirst=True) - pd.to_datetime(df['dob'], dayfirst=True)) > pd.Timedelta(days=365*18)`
4. Choose a short, descriptive name for the criteria (snake_case).
5. Provide a brief explanation.

Output Format (JSON):
{{
    "criteria": [
        {{
            "name": "high_blood_pressure",
            "expression": "(df['systolic'] > 140) | (df['diastolic'] > 90)",
            "explanation": "Flags rows where systolic BP > 140 or diastolic BP > 90."
        }},
        {{
            "name": "invalid_age_gap",
            "expression": "(pd.to_datetime(df['visit_date']) - pd.to_datetime(df['birth_date'])) < pd.Timedelta(days=0)",
            "explanation": "Flags rows where visit date is before birth date."
        }}
    ]
}}
"""
