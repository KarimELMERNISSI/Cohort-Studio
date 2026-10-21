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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.expressions import DEFAULT_POLICY, evaluate_tree
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
    


# Names that are not columns: the whitelisted evaluator's numpy and pandas facades, and its functions (core.expressions)
FORMULA_MODULES = frozenset({"np", "pd"})
FORMULA_FUNCTIONS = frozenset(DEFAULT_POLICY.builtins)  # abs, bool, float, int, len, max, min, round, str
_QUOTED = re.compile(r'""([^"]+)""')  # ""a column with spaces or symbols""
_BACKTICKED = re.compile(r"`([^`]+)`")
# a quoted column, a text literal (its ^ is a character) or a ^
_QUOTED_OR_CARET = re.compile(r"\"\"[^\"]*\"\"|'[^'\n]*'|\"[^\"\n]*\"|\^")
_ASSIGNED = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)\s*")  # "bmi = weight / ...": the formula is the right side
MAX_FORMULA_LENGTH = 3_000  # characters, as the whitelisted evaluator allows (core.expressions)
_EXAMPLE_ROWS = 5  # a value shown in a message is held by this many rows at least (as manage.ai_payload.MIN_COUNT)
_EXAMPLE_LENGTH = 24
_LONG_QUOTE = re.compile(r"\"[^\"\n]{13,}\"|'[^'\n]{13,}'")  # a cell a library quotes in its message


@dataclass(frozen=True)
class FormulaReading:
    """What a formula uses, read as the evaluator reads it (:func:`read_formula`)."""

    names: tuple[str, ...] = ()  # the variables it uses, in the order they first appear
    problem: str | None = None  # why it cannot be read (then it names no variable)
    power_caret: bool = False  # it uses ^, which is a bitwise exclusive or here, not a power (write **)


def _parsed(formula: str) -> tuple[ast.Expression | None, dict[str, str], str | None]:
    """The formula's tree, its ``""quoted""`` names by the placeholders that stand for them in the tree, or why it
    cannot be read."""
    text = (formula or "").strip()
    if len(text) > MAX_FORMULA_LENGTH:
        return None, {}, f"a formula longer than {MAX_FORMULA_LENGTH:,} characters"
    quoted: dict[str, str] = {}
    for i, match in enumerate(dict.fromkeys(_QUOTED.findall(text))):
        placeholder = f"__quoted_var_{i}__"
        quoted[placeholder] = match
        text = text.replace(f'""{match}""', placeholder)
    try:
        return ast.parse(text, mode="eval"), quoted, None
    except SyntaxError as exc:
        return None, quoted, f"Syntax Error: {exc}"
    except (ValueError, RecursionError, MemoryError) as exc:  # a null byte, a nesting too deep to read
        return None, quoted, f"the formula cannot be read ({type(exc).__name__})"


def _functions(tree: ast.Expression) -> set[int]:
    """The ids of the Name nodes a formula calls (``round(x)``), as opposed to the variables it uses."""
    return {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


def read_formula(formula: str) -> FormulaReading:
    """
    The variables a formula uses: a name, or ``""a name""`` for a column with spaces or symbols. ``np`` and ``pd``,
    and the functions of the evaluator (``round``, ``abs``, ``min``... when called) are not variables; another function
    called by its bare name (``sqrt``) is refused with the hint ``np.sqrt``. The one reading of a formula: the
    evaluator, the AI engine's suggestions and their diagrams use it, so a diagram shows what computing it uses.
    """
    tree, quoted, problem = _parsed(formula)
    if tree is None:
        return FormulaReading(problem=problem)
    called = _functions(tree)
    found = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Name)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    names: list[str] = []
    for node in found:
        name = quoted.get(node.id, node.id)
        if name in FORMULA_MODULES or (id(node) in called and name in FORMULA_FUNCTIONS):
            continue
        if id(node) in called:  # a function the evaluator does not have
            hint = f": write np.{name}(...)" if hasattr(np, name) else ""
            return FormulaReading(problem=f"{name}() is not a function of formulas{hint}")
        names.append(name)
    return FormulaReading(
        names=tuple(dict.fromkeys(names)),
        power_caret=any(isinstance(node, ast.BitXor) for node in ast.walk(tree)),
    )


def formula_from_model(formula: str, *, targets: Iterable[str] = ()) -> str:
    """
    A formula written by a language model, in this application's notation: ``name =`` dropped when it starts the
    formula and names one of ``targets`` (the variable it computes: ``sex = 'F'`` stays refused, it is no formula),
    `` `a name` `` written ``""a name""``, and ``^`` (a power, as in mathematics) written ``**`` outside text literals
    (here ``^`` is a bitwise exclusive or, which gives wrong numbers silently).
    """
    text = (formula or "").strip()
    if len(text) > MAX_FORMULA_LENGTH:
        return text  # read_formula refuses it
    assigned = _ASSIGNED.match(text)
    if assigned and assigned.group(1) in set(targets):
        text = text[assigned.end() :]
    text = _BACKTICKED.sub(lambda match: f'""{match.group(1)}""', text)
    return _QUOTED_OR_CARET.sub(lambda match: "**" if match.group(0) == "^" else match.group(0), text)


CARET_IS_NOT_A_POWER = (
    "^ is not a power in a formula: it is a bitwise exclusive or, which gives wrong numbers without an error. "
    "Write ** for a power, as in weight_kg / (height_cm / 100) ** 2. For one of two yes/no conditions but not both, "
    "write (a) != (b), with the parentheses."
)

# The functions and methods of the evaluator (core.expressions) whose values depend on other rows than the row computed:
# aggregates, quantiles, ranks, running and shifted values. Every function the evaluator offers is classified (the test
# lists the others): one added there is added here when it reads other rows.
OF_OTHER_ROWS = frozenset(
    {
        "all",
        "any",
        "count",
        "cummax",
        "cummin",
        "cumsum",
        "diff",
        "max",
        "mean",
        "median",
        "min",
        "nanmax",
        "nanmean",
        "nanmedian",
        "nanmin",
        "nanpercentile",
        "nanstd",
        "nansum",
        "nunique",
        "pct_change",
        "percentile",
        "qcut",
        "quantile",
        "rank",
        "shift",
        "std",
        "sum",
        "var",
    }
)
_OF_OTHER_ROWS_CALLED = frozenset({"len", "max", "min"})  # len(x), max(x): of the whole column, called by their name

NOT_ROW_WISE = (
    "An exact formula gives each row its value from that row's own variables. This one also uses other rows "
    "({reason}), so its values are estimates: untick Exact formula to impute them as such."
)


def _written_out(node: ast.AST) -> bool:
    return isinstance(node, ast.List | ast.Tuple | ast.Set)


def _argument(call: ast.Call, position: int, keyword: str) -> ast.AST | None:
    if len(call.args) > position:
        return call.args[position]
    return next((k.value for k in call.keywords if k.arg == keyword), None)


def _across_columns(call: ast.Call) -> bool:
    """``np.nanmean([a, b], axis=0)``: across the columns written in the list, row by row."""
    axis = _argument(call, 1, "axis")
    return bool(call.args) and _written_out(call.args[0]) and isinstance(axis, ast.Constant) and axis.value == 0


def other_rows(formula: str, data: pd.DataFrame | None = None) -> str | None:
    """
    What makes ``formula`` use other rows than the row it computes, in words, or None when it gives each row a value of
    that row's own variables only, as an exact formula does (a weight from the BMI and the height). Other rows are used
    by an aggregate (a mean, a maximum, a quantile), a rank, a running or shifted value, a value taken from another row
    (``x[0]``) or a test against a column (``a.isin(b)``): the values are then estimates, and change with the rows the
    formula is computed on. Given the ``data``, so are dates kept as text, whose order of day and month the evaluator
    reads from the rows. A formula that cannot be read is not of each row's own values.
    """
    tree, _, problem = _parsed(formula)
    if tree is None:
        return problem or "it cannot be read"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                numpy = isinstance(func.value, ast.Name) and func.value.id == "np"
                if func.attr in OF_OTHER_ROWS and not (numpy and _across_columns(node)):
                    return f"{func.attr}()"
                if func.attr == "cut" and not _written_out(_argument(node, 1, "bins") or ast.Constant(None)):
                    return "cut() with a number of bins"
                if func.attr == "isin" and not (node.args and _written_out(node.args[0])):
                    return "isin() with a column"
            elif isinstance(func, ast.Name) and func.id in _OF_OTHER_ROWS_CALLED:
                return f"{func.id}()"
        elif isinstance(node, ast.Subscript):
            value = node.value
            column = isinstance(value, ast.Name) and value.id in DEFAULT_POLICY.frame_names  # df["a column"]
            character = isinstance(value, ast.Attribute) and value.attr == "str"  # x.str[0]
            if not (column or character):
                return "the value of another row"
    if data is not None:
        text = [name for name in read_formula(formula).names if name in data.columns and is_text_column(data[name])]
        dates = DataAnalyzer(data[text]).date_cols if text else []
        kept = [name for name in text if name in dates]
        if kept:
            return (
                f"the dates of {kept[0]} are kept as text, and the order of their day and month is read from the rows: "
                "convert them to dates on Validate first"
            )
    return None


def row_wise(formula: str, data: pd.DataFrame | None = None) -> bool:
    """Whether ``formula`` gives each row a value of that row's own variables only (:func:`other_rows`)."""
    return other_rows(formula, data) is None


def is_text_column(series: pd.Series) -> bool:
    """A column of text, neither numbers, yes/no values nor dates: it compares with text only."""
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    )


def usable_values(values: pd.Series) -> pd.Series:
    """The rows with a usable value: not missing, and finite for numbers (``fillna`` would put an infinity in)."""
    present = values.notna()
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        present &= np.isfinite(values.astype(float))
    return present


@dataclass(frozen=True)
class FormulaTrial:
    """A formula tried on the data as computing it would (an AI engine's suggestion, before it is offered)."""

    values: pd.Series | None = None  # one value per row
    problem: str | None = None  # why it cannot be computed, in words
    inputs: tuple[str, ...] = ()  # the columns it uses
    warnings: tuple[str, ...] = ()  # computed, but read it twice (a text it compares with that never occurs)

    @property
    def ok(self) -> bool:
        return self.problem is None

    def given(self) -> pd.Series:
        """The rows that get a usable value (none when it cannot be computed)."""
        return usable_values(self.values) if self.values is not None else pd.Series(dtype=bool)

    def from_missing(self, data: pd.DataFrame, within: pd.Series | None = None) -> list[tuple[str, int]]:
        """The inputs whose value is missing in rows that still get a value (a comparison or ``np.where`` sends a
        missing value to its "false" branch), with how many such rows, among ``within`` when it is given."""
        given = self.given()
        if within is not None:
            given = given & within
        counts = [(name, int((given & data[name].isna()).sum())) for name in self.inputs if name in data.columns]
        return [(name, n) for name, n in counts if n]


def _absent_literals(formula: str, data: pd.DataFrame) -> list[str]:
    """The texts a formula compares a text column with (``==``, ``!=``, ``.isin``) that occur in no row of it: the
    comparison then gives the same answer everywhere, which no error reveals."""
    tree, quoted, _ = _parsed(formula)
    if tree is None:
        return []

    def column(node: ast.AST) -> str | None:
        name = quoted.get(node.id, node.id) if isinstance(node, ast.Name) else None
        return name if name in data.columns and is_text_column(data[name]) else None

    pairs: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            for left, right in zip(sides, sides[1:], strict=False):
                for col, other in ((column(left), right), (column(right), left)):
                    if col:
                        pairs.append((col, other))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "isin":
            col = column(node.func.value)
            if col and node.args and isinstance(node.args[0], ast.List | ast.Tuple | ast.Set):
                pairs.extend((col, element) for element in node.args[0].elts)
    absent: list[str] = []
    for col, literal in pairs:
        if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
            present = set(data[col].dropna().astype(str))
            if literal.value not in present:
                absent.append(f"{literal.value[:_EXAMPLE_LENGTH]!r} occurs in no row of {col}")
    return list(dict.fromkeys(absent))


def try_formula(formula: str, data: pd.DataFrame, *, target: str | None = None) -> FormulaTrial:
    """
    ``formula`` computed on ``data`` by the evaluator of "Compute" and "Apply", or why it cannot be: a syntax the
    application does not read, ``^``, a name that is not a column, or a computation the data refuse (a column that
    holds text compared with a number, say: the text columns it uses are named, with values held by several rows).
    With a ``target`` that holds numbers (an imputation), a result that is not numbers is refused too: filling the
    column with yes/no values or text would change its type. Nothing is sent anywhere: the trial runs here.
    """
    reading = read_formula(formula)
    if reading.problem:
        return FormulaTrial(problem=reading.problem)
    if reading.power_caret:
        return FormulaTrial(problem=CARET_IS_NOT_A_POWER)
    unknown = [name for name in reading.names if name not in data.columns]
    if unknown:
        return FormulaTrial(problem=f"not in the data: {', '.join(unknown)}")
    ok, result, _ = evaluate_formula_safely(formula, "__trial__", data, return_series=True)
    if ok:
        if (
            target in data.columns
            and pd.api.types.is_numeric_dtype(data[target])
            and not pd.api.types.is_bool_dtype(data[target])
            and (pd.api.types.is_bool_dtype(result) or not pd.api.types.is_numeric_dtype(result))
        ):
            kind = "yes/no values" if pd.api.types.is_bool_dtype(result) else "values that are not numbers"
            return FormulaTrial(problem=f"it gives {kind}, while {target} holds numbers", inputs=reading.names)
        return FormulaTrial(
            values=result, inputs=reading.names, warnings=tuple(_absent_literals(formula, data))
        )
    problem = str(result).split("\n")[0]
    for prefix in ("Error evaluating formula: ", "Invalid formula: "):
        problem = problem.removeprefix(prefix)
    problem = _LONG_QUOTE.sub('"…"', problem)[:200]  # a library can quote a cell: not a long one
    for name in (name for name in reading.names if is_text_column(data[name])):
        counts = data[name].dropna().astype(str).value_counts()
        frequent = counts[counts >= _EXAMPLE_ROWS].index.tolist()[:3]  # never a value that identifies a row
        examples = ", ".join(repr(str(value)[:_EXAMPLE_LENGTH]) for value in frequent)
        problem += f"; {name} holds text" + (f" (such as {examples})" if examples else "")
    return FormulaTrial(problem=problem, inputs=reading.names)


def validate_and_extract_columns_ast(formula: str, dataset_columns: list) -> tuple[bool, str, list[str], list[str]]:
    """
    Reads the formula (:func:`read_formula`) and checks that every variable it uses is a column of the data: the
    variables it uses, or why it cannot be computed.
    """
    reading = read_formula(formula)
    if reading.problem:
        return False, reading.problem, [], []
    for name in reading.names:
        if name not in dataset_columns:
            return False, f"Unknown variable: '{name}'", [], []
    return True, None, list(reading.names), []


def extract_constant_value(constant_token: str) -> str:
    """
    Extract the value from a constant token like constant(val1) -> val1
    """
    # Remove 'constant(' prefix and ')' suffix and return the inner value
    return constant_token[9:-1]  # len('constant(') == 9


def _names_in_literals(tree: ast.Expression, columns: list[str]) -> bool:
    """Whether a text literal of the formula holds one of its columns' names (the old renaming changed that text)."""
    texts = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    return any(column in text for column in columns for text in texts)


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

    # The columns are renamed in the formula's tree, never in its text: a text literal that holds a column's name
    # ('Type 2 diabetes' for a column diabetes) stays as written, and no column can be taken for another
    tree, quoted, _ = _parsed(formula)
    columns = {name: f"__column_{i}__" for i, name in enumerate(identified_columns)}
    if tree is not None and _names_in_literals(tree, identified_columns):
        logger.warning(
            "A formula holds a column's name inside a text literal: earlier versions renamed that text too, so a "
            "variable recorded with them may have been computed differently"
        )

    class ColumnRenamer(ast.NodeTransformer):
        def visit_Name(self, node):
            name = columns.get(quoted.get(node.id, node.id))
            return ast.copy_location(ast.Name(id=name, ctx=node.ctx), node) if name else node

    local_vars: dict[str, Any] = {columns[name]: dataset[name] for name in identified_columns}
    local_vars['__safe_sub__'] = safe_sub
    logger.debug("evaluate_formula: %d variables available; formula=%s", len(columns), single_line(formula))

    try:
        # date subtractions give days, then the whitelisted evaluation (no builtins, restricted np/pd: core.expressions)
        tree = DateSubRewriter().visit(ColumnRenamer().visit(tree))
        ast.fix_missing_locations(tree)
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
