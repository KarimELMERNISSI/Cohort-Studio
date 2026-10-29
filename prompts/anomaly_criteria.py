"""Prompt for generating plausibility rules and inclusion criteria."""

import json

# the examples the model sees, by the kind of rule asked for: a plausibility rule flags what cannot be or is very
# unlikely, an inclusion criterion selects whom the study is about
_EXAMPLES = {
    "anomaly": [
        {
            "name": "implausible_heart_rate",
            "expression": "(df['heart_rate'] < 20) | (df['heart_rate'] > 250)",
            "explanation": "Flags heart rates out of the range a living patient can have.",
        },
        {
            "name": "discharge_before_admission",
            "expression": "pd.to_datetime(df['discharge_date'], dayfirst=True) < pd.to_datetime(df['admission_date'], dayfirst=True)",
            "explanation": "Flags rows where the discharge is recorded before the admission.",
        },
    ],
    "inclusion": [
        {
            "name": "adults_up_to_85",
            "expression": "(df['age'] >= 18) & (df['age'] <= 85)",
            "explanation": "Keeps patients aged 18 to 85.",
        },
        {
            "name": "written_consent",
            "expression": "df['consent'] == 'Yes'",
            "explanation": "Keeps patients who gave written consent.",
        },
    ],
}

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

    task_desc = "write PLAUSIBILITY RULES (rows whose values, alone or together, domain knowledge says cannot be right)" if mode == "anomaly" else "define INCLUSION CRITERIA (rows to KEEP in the study)"
    examples = json.dumps({"criteria": _EXAMPLES["anomaly" if mode == "anomaly" else "inclusion"]}, indent=4)

    return f"""Role: Data Science Expert.

Task: You are helping a user {task_desc}. converting their natural language description into valid Python Pandas boolean expressions.

Context:
- The dataframe is named `df`.
- **CRITICAL:** You must ONLY use the provided columns. Do NOT invent new column names.
- If a user asks for a variable not in the list, try to find the closest match from the Available Columns.
- Return a JSON object with a list of criteria.

Available Columns:
{columns_info}

Data Summary (column types, percentiles, frequent values and date-format hints; no row of the data is shown):
{sample_data}

User Description:
"{description}"

Instructions:
1. Parse the user's description to identify specific conditions.
2. If the user asks for multiple criteria, split them into separate items in the list.
3. Write a VALID Python boolean expression for each condition (e.g., `(df['age'] > 18) & (df['status'] == 'Active')`).
    - **Date Logic:** Check the Date Parsing Hints of the Data Summary. If dates are 'DD/MM/YYYY', use `format='%d/%m/%Y'`. If ambiguous, use `dayfirst=True` and `errors='coerce'`.
    - **Example:** `(pd.to_datetime(df['visit'], dayfirst=True) - pd.to_datetime(df['dob'], dayfirst=True)) > pd.Timedelta(days=365*18)`
4. Choose a short, descriptive name for the criteria (snake_case).
5. Provide a brief explanation.

Output Format (JSON):
{examples}
"""
