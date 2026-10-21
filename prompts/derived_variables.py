"""Prompt for finding the derived variables of a dataset: the columns computed from other columns of the same row."""


def derived_variables(columns_text: str, max_candidates: int = 15) -> str:
    """
    Asks for the definitions that compute a column from other columns, row by row, with their inverses.

    Args:
        columns_text: one line per column: its name, how it is stored, its unit when the variable map has one, and
            its typical values (5th to 95th percentiles) when the user lets them be sent
        max_candidates: the most candidates to return

    Returns:
        The prompt; the answer is checked against the data before anything is offered.
    """
    return f"""Role: Clinical data manager.
Task: Among the columns of a dataset, find those that are COMPUTED from other columns of the SAME ROW by a
definition: a unit conversion, a ratio or product (a BMI from the weight and the height), a total of items, an index
or score, a published clinical equation (an eGFR), a class of a variable (a BMI class).

Columns (how each is stored; its unit and typical values when known):
{columns_text}

Rules:
1. Each candidate computes one listed column, its "target", from other listed columns of the same row. Never an
   aggregate over rows (a mean, a sum over rows, a rank, a shifted value), a regression, a prediction model, a proxy
   or an estimate: only a definition that gives the exact recorded value.
2. Use only the columns listed, with their names exactly as written.
3. Give every published variant and unit version as its own candidate (CKD-EPI 2009 and 2021; a creatinine in mg/dL
   or in µmol/L): the data will tell which one holds. Use the units and typical values to choose the factors.
4. For each candidate, give its inverses: for each column of its formula that the target and the other columns
   determine, the formula that recovers it (a weight from the BMI and the height). Leave out a column that cannot be
   recovered (a code, a class).
5. Syntax, as the application computes it:
   - column names directly; a name with spaces or symbols between double double quotes (""My Var"");
   - np.where(condition, a, b), np.sqrt, np.log, np.exp, np.maximum, np.minimum;
   - ** for a power, never ^;
   - a (text) column compared only with == or .isin([...]) against text in quotes (sex == 'F'), never with a number;
   - a (date) minus a (date) gives days.
6. At most {max_candidates} candidates; none when no column is computed from others.

Return JSON:
{{"candidates": [{{"target": "column", "formula": "...", "kind": "conversion|ratio|total|index|equation|class",
"inverses": {{"column": "formula"}}, "units": "the units the formula assumes", "source": "its reference"}}]}}"""
