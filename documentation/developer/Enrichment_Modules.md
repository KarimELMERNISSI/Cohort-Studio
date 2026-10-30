# Enrichment Modules Documentation

## data_imputation.py

_No module description._

**Imports**:
`pandas`, `sklearn.base.TransformerMixin`, `sklearn.compose.ColumnTransformer`, `sklearn.experimental.enable_iterative_imputer`, `sklearn.impute.IterativeImputer`, `sklearn.impute.KNNImputer`, `sklearn.impute.SimpleImputer`, `sklearn.pipeline.Pipeline`, `sklearn.preprocessing.OneHotEncoder`, `sklearn.preprocessing.StandardScaler`, `biostats._types.RANDOM_STATE`, `ingest.analyzer.DataAnalyzer`, `utils.miss_forest.missforest.MissForest`

### class `MissForestTransformer` (data_imputation.py)

No description available.

**Methods:**

- **__init__**(`self, categorical_cols, debug, progress_bar`) -> `None`
  > A transformer wrapper for MissForest imputation for both numerical and categorical data.
  >
  >
  > **Parameters**:
  >
  > categorical_cols : list, default=None
  >     List of categorical column names for imputation. If None, assumes no categorical columns.
  > debug : bool, default=False
  >     If True, enables debug information output.

- **debug_print**(`self`) -> `None`
  > Helper function for controlled debug printing.

- **fit**(`self, X, y`) -> `None`
  > Fit the MissForest imputer on X.
  >
  >
  > **Parameters**:
  >
  > X : pd.DataFrame
  >     The input dataframe to fit the imputer.
  > y : Ignored
  >
  >
  > **Returns**:
  >
  > self : object
  >     Returns self.

- **transform**(`self, X`) -> `None`
  > Impute missing values in X based on the fitted MissForest imputer.
  >
  >
  > **Parameters**:
  >
  > X : pd.DataFrame
  >     The input dataframe to impute.
  >
  >
  > **Returns**:
  >
  > X_imputed : pd.DataFrame
  >     Dataframe with imputed values.

- **fit_transform**(`self, X, y`) -> `None`
  > Fit the MissForest imputer and transform X.
  >
  >
  > **Parameters**:
  >
  > X : pd.DataFrame
  >     The input dataframe to fit and transform.
  > progress_placeholder : st.empty, optional
  >     Streamlit placeholder for the progress bar. If None, a new progress bar is created.
  > y : Ignored
  >
  >
  > **Returns**:
  >
  > X_imputed : pd.DataFrame
  >     Dataframe with imputed values.

### def `_is_text` (data_imputation.py)

- **Arguments**: `series: pd.Series`

- **Returns**: `bool`

Text-like column: object, pandas string ('str' in pandas 3) or categorical dtype.

### def `detect_remainder_columns` (data_imputation.py)

- **Arguments**: `data, threshold, date_like_columns`

- **Returns**: `None`

Columns that should not be imputed or encoded: dates (true date/time dtypes and text columns
recognised as dates) and high-cardinality text columns such as identifiers.

Parameters:
- data: pd.DataFrame
    The input DataFrame.

- threshold: float, default=0.8
    The proportion of distinct values to non-null values to qualify as a remainder column.
    If the value is greater than 1, it defines the exact number of distinct values used as the threshold.

- date_like_columns: iterable of str, optional
    Text columns already recognised as dates (e.g. ``DataAnalyzer.date_cols``).

Returns:
- list: Names of columns to be included in the remainder, preserving the order in the input DataFrame.

### def `debug_print` (data_imputation.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `get_feature_names` (data_imputation.py)

- **Arguments**: `preprocessor, column_names, num_scaler, cat_encoder, debug`

- **Returns**: `None`

Retrieve the new feature names after applying transformations in the preprocessor.

Parameters:
- preprocessor: The fitted ColumnTransformer or Pipeline that has been applied to the data.

- column_names (list): List of original column names.

- num_scaler (bool): Whether numerical columns are scaled.

- cat_encoder (bool): Whether categorical columns are encoded.

- debug (bool): Whether to print debug information.

Returns:
- new_cols (list): List of transformed column names.

### def `reorder_data_columns` (data_imputation.py)

- **Arguments**: `data, numerical_columns, categorical_columns`

- **Returns**: `None`

Numerical columns first, then categorical ones, then every other column in its original order.
No column is ever dropped (true/false, date and text columns used to disappear here).

### def `partition_columns` (data_imputation.py)

- **Arguments**: `data, remainder_columns, analyzer`

- **Returns**: `None`

Assign every column to exactly one group: (numerical, categorical, passthrough).

- categorical: text and categorical columns, true/false columns, binary and low-cardinality numeric columns;

- numerical: the other numeric columns;

- passthrough: everything else (requested remainder columns, dates, identifiers, any other dtype), kept in
  the input order, which is the order in which a ColumnTransformer appends its remainder.

### def `get_imputer` (data_imputation.py)

- **Arguments**: `numerical_imputation_method, categorical_imputation_method, cat_encoder, num_scaler, data, remainder_columns, remainder_strategy, remainder_threshold, total_steps, progress_placeholder, debug`

- **Returns**: `None`

Returns the appropriate imputer based on the specified imputation methods for numerical and categorical columns.

Parameters:
- numerical_imputation_method (str): Imputation method for numerical columns ('mean', 'median', 'knn', 'missforest').

- categorical_imputation_method (str): Imputation method for categorical columns ('most_frequent', 'missforest').

- cat_encoder (bool): Whether to apply one-hot encoding to categorical columns.

- num_scaler (bool): Whether to scale numerical columns.

- data (pandas.DataFrame): Input DataFrame used to identify categorical and numerical columns.

- remainder_columns: list, default=None: Columns to pass through without transformation (e.g., ID columns), will be excluded from imputation, applying remainder strategy.
    NB : If it is set to 'auto', it will automatically guess the remainder columns based on your input data.

- remainder_threshold (float): Default (0.8), from 0 to 1, filter out non-numeric columns having a ratio distinct modalities / non empty rows higher than the fixed threshold if ramainder columns is set to 'auto'

- remainder_strategy (str): 'passthrough' (keep as it is) or 'drop' (remove from the final dataset)

- progress_placeholder: Streamlit empty container for showing the progress bar, default=None.

- debug (bool): Whether to print debug information.

Returns:
- A ColumnTransformer object or a MissForestTransformer if using MissForest for both.

- A list of all output columns in the final transformed DataFrame.

---

## derived.py

_The check of a declared derived variable against the data (core.derived declares it). On the rows where the variable
and every variable of its formula have a value, the formula is computed here, by the evaluator of every formula, and
compared with the recorded value. Nothing is estimated and nothing leaves the machine; the disagreeing rows are kept
for the page to show, never written anywhere.

A definition the data disagree with, or one that cannot be checked, is never used to fill or impute unless the user
accepts it; agreeing with the data says nothing of the published reference, which a definition's note names._

**Imports**:
`__future__.annotations`, `collections.abc.Callable`, `collections.abc.Mapping`, `collections.abc.Sequence`, `dataclasses.dataclass`, `dataclasses.field`, `dataclasses.replace`, `typing.Any`, `numpy`, `pandas`, `biostats.mi.spec.DERIVED`, `core.derived.DerivedVariable`, `core.derived.fill_order`, `enrich.formulas.CARET_IS_NOT_A_POWER`, `enrich.formulas.evaluate_formula_safely`, `enrich.formulas.is_text_column`, `enrich.formulas.other_rows`, `enrich.formulas.read_formula`, `enrich.formulas.try_formula`, `enrich.formulas.usable_values`, `ingest.analyzer.DataAnalyzer`

### class `DerivedCheck` (derived.py)

What the data say of a definition (:func:`check_derived`): counts, never values, except the rows kept for the
page to show (``disagreeing``) and to fill (``fillable_rows``).

**Methods:**

- **label**(`self`) -> `str`
  > No description available.

- **summary**(`self`) -> `dict[ComplexType]`
  > The check in numbers, as a step that used the definition records it: never its words (a reason can name a
  > value of the data, "where sex = F").

- **fillable**(`self`) -> `int`
  > No description available.

- **usable**(`self`) -> `bool`
  > It may fill and impute: the data agree with it, or the user accepted it although they mostly agree or
  > cannot check it.

- **evidence**(`self`) -> `str`
  > The check in words: 'gives the recorded value in 368 of 371 rows where bmi, weight_kg and height_cm are
  > recorded (±0.05)'.

- **describe**(`self, name: Callable[ComplexType]`) -> `str`
  > :attr:`evidence`, each variable written by ``name`` (core.registry.as_code where markdown renders it).

### class `ExactFill` (derived.py)

The missing values of one variable computed by a declared definition (its formula, or an inverse of it): one
targeted imputation of the Enrich draft. ``target`` is the variable filled, the definition's own or a variable of
its formula; ``decimals`` rounds the values as the variable is recorded (None: a text, or no precision found);
``levels`` limits the fill to the values a text or code variable took in the rows checked ({}: no limit).

**Methods:**

- **inverse**(`self`) -> `bool`
  > No description available.

### class `FillPlan` (derived.py)

What filling exactly would do: the fills in order, why a declared definition fills nothing (by variable), the
values each variable would still miss in the rows filled (by variable), and why nothing can be filled.

**Methods:**

- **filled**(`self`) -> `int`
  > No description available.

### def `resolution` (derived.py)

- **Arguments**: `values: pd.Series`

- **Returns**: `float`

The precision a column is recorded with: the coarsest power of ten that divides at least 99% of its values (0.1
for a weight recorded to 100 g, 1 for an age in years). 0 when none does (a value computed to full precision), and
for scores and counts (at most twenty distinct whole numbers), which carry no rounding.

### def `_levels` (derived.py)

- **Arguments**: `series: pd.Series`

- **Returns**: `bool`

A variable checked level by level: text, yes/no, or codes (at most ten distinct whole numbers).

### def `_is_date` (derived.py)

- **Arguments**: `series: pd.Series`

- **Returns**: `bool`

No description available.

### def `_computed` (derived.py)

- **Arguments**: `formula: str, frame: pd.DataFrame`

- **Returns**: `ComplexType`

No description available.

### def `_floor` (derived.py)

- **Arguments**: `values: np.ndarray`

- **Returns**: `np.ndarray`

No description available.

### def `_propagated` (derived.py)

- **Arguments**: `definition: DerivedVariable, rows: pd.DataFrame, inputs: list[str], values: np.ndarray`

- **Returns**: `np.ndarray`

Per row: half the variable's recording unit, plus what rounding each input to its own unit moves the formula, up
or down, whichever moves it more (a BMI to 0.1 from a weight to 0.1 kg and a height to 1 cm: about 0.21; a curved
formula moves more one way).

### def `_unit_hint` (derived.py)

- **Arguments**: `recorded: np.ndarray, values: np.ndarray`

- **Returns**: `ComplexType`

A constant factor between the recorded values and the formula's in most disagreeing rows: another unit.

### def `_agree` (derived.py)

- **Arguments**: `recorded: np.ndarray, values: np.ndarray, tau: ComplexType`

- **Returns**: `np.ndarray`

No description available.

### def `_cannot` (derived.py)

- **Arguments**: `definition: DerivedVariable, reason: str`

- **Returns**: `DerivedCheck`

No description available.

### def `_multiple_of_one` (derived.py)

- **Arguments**: `definition: DerivedVariable, rows: pd.DataFrame, inputs: list[str], got: np.ndarray`

- **Returns**: `bool`

The formula is a constant times its one variable: a copy or a unit conversion, which agrees by construction.

### def `_level_disagreeing` (derived.py)

- **Arguments**: `rows: pd.DataFrame, inputs: list[str], agree: np.ndarray`

- **Returns**: `ComplexType`

The first level of a text or code variable whose rows give the recorded value less often than 90%: the formula
is wrong for it (a factor for women only), however well the others agree. The variable, and the reason, which names
the level.

### def `_imputed_rows` (derived.py)

- **Arguments**: `index: pd.Index, imputed: ComplexType, names: list[str]`

- **Returns**: `np.ndarray`

The rows where an imputation filled one of ``names`` (row ids per column, as the session keeps them).

### def `check_derived` (derived.py)

- **Arguments**: `definition: DerivedVariable, data: pd.DataFrame`

- **Returns**: `DerivedCheck`

The definition checked against ``data`` (the Enrich data, every row: nothing is filled here). ``imputed``: the
cells an imputation filled (row ids per column), which are not compared: a value an exact formula filled agrees by
construction, and would add to the evidence.

It cannot be used when its formula cannot be read or computed, uses the variable it computes, uses other rows (a
mean, a rank: its values are no row's own) or names a variable the data do not have. It cannot be checked on fewer
than 60 rows with every variable recorded, or when the variable takes one value nearly everywhere (a fit by chance).
Otherwise the formula gives the recorded value within the precision the variable is recorded with, or, failing
that, within what rounding the formula's variables moves it: it agrees (99% of the rows at least, beyond chance),
mostly agrees (90%), or disagrees, also when one level of a text or code variable disagrees (a factor wrong for
women only).

### def `inverse_definition` (derived.py)

- **Arguments**: `definition: DerivedVariable, name: str`

- **Returns**: `DerivedVariable`

The inverse of ``definition`` that recovers ``name``, as a definition of its own (checked like one).

### def `check_inverses` (derived.py)

- **Arguments**: `definition: DerivedVariable, data: pd.DataFrame`

- **Returns**: `dict[ComplexType]`

Each inverse of the definition checked as a definition of its own (its variable, its formula): the tolerance
declared for the definition is in its variable's unit, not theirs.

### def `_decimals` (derived.py)

- **Arguments**: `recorded: pd.Series, unit: float`

- **Returns**: `ComplexType`

The decimals a variable is written with: those of its recording unit (1 for 0.1, 0 for 1, -1 for 10); 0 for
whole numbers (a score: its unit of 0 means no rounding slack, not full precision); None for a text or a variable
recorded to full precision.

### def `_exact` (derived.py)

- **Arguments**: `check: DerivedCheck, recorded: pd.Series, unit: float`

- **Returns**: `bool`

The values count as observed when the data agree with the formula (an accepted definition only mostly agrees, or
could not be checked: its values are estimates) and either the variable was computed from the recorded values (it
agrees within its own rounding) or rounding its formula's variables moves it, in 95% of the rows, by at most the
variable's own precision or 5% of its standard deviation: a height from a BMI to 0.1 and a weight to 0.1 kg moves
by about 0.2 cm. A declared tolerance is never the variable's own rounding.

### def `_unused` (derived.py)

- **Arguments**: `check: DerivedCheck`

- **Returns**: `ComplexType`

No description available.

### def `_fill` (derived.py)

- **Arguments**: `frame: pd.DataFrame, fill: ExactFill, rows: ComplexType`

- **Returns**: `tuple[ComplexType]`

One fill, as its step records it and the replay computes it (manage.reproduction_manager).

### def `plan_exact_fills` (derived.py)

- **Arguments**: `definitions: Sequence[DerivedVariable], data: pd.DataFrame, rows: ComplexType`

- **Returns**: `FillPlan`

The missing values the declared definitions compute in ``data``, within ``rows`` (the analysis population; every
row when None): each formula, then its inverses, in the order the definitions depend on each other (a variable's
exact formulas before its others), and again while a fill gives another formula a value. A definition fills when
the data agree with it, or the user accepted it; an inverse when the data agree with it; a date never. A value is
filled where every variable of its formula has a finite value, and a text or code variable takes a value the check
saw in at least LEVEL_ROWS rows; it is rounded as the variable is written. It counts as observed when
:func:`_exact` holds for its own formula and no variable of it was single-imputed in its row, as the session sorts
imputed cells (core.session.sort_filled). ``imputed``: every imputed cell (left out of the checks); ``single``: the
single-imputed ones (row ids per column). ``check`` and ``inverse_check`` give the checks (the page's, kept); by
default they are computed here.

### def `contradicted` (derived.py)

- **Arguments**: `definition: DerivedVariable, check: DerivedCheck, frame: pd.DataFrame, imputed: Mapping[ComplexType]`

- **Returns**: `int`

The rows where the variable is recorded but a variable of its formula was imputed (``imputed``: row ids per
column), and the formula of the imputed values no longer gives the recorded value within the check's tolerance: a
recorded BMI beside an imputed height. An inverse of the definition would have recovered that variable exactly.

### def `recomputed` (derived.py)

- **Arguments**: `definitions: Sequence[DerivedVariable], check: Callable[ComplexType]`

- **Returns**: `list[tuple[ComplexType]]`

The definitions whose variable an imputation keeps out and computes afterwards from the imputed values: those
:func:`plan_exact_fills` fills by their formula (usable, not a date, no text of too many values), in the order it
fills them; none when definitions compute each other. One rule for setting aside and recomputing, so that no
variable is kept out that nothing computes again.

### def `derivation_steps` (derived.py)

- **Arguments**: `definitions: Sequence[DerivedVariable], check: Callable[ComplexType], data: pd.DataFrame`

- **Returns**: `list[dict[ComplexType]]`

What a multiple imputation computes instead of imputing, as its step records it and the replay repeats it: each
declared derived variable :func:`recomputed` keeps out that has missing values in ``data``, in fill order, with its
formula and the variables it reads, the rounding of its recorded values, the values of a text or code variable its
check saw when the data hold another, and its record (core.derived.record).

### def `derive` (derived.py)

- **Arguments**: `mi: Any, steps: Sequence[Mapping[ComplexType]]`

- **Returns**: `tuple[ComplexType]`

``mi`` (biostats.mi.MIData) with the missing values of each derived variable of ``steps`` (:func:`derivation_steps`,
as recorded) computed in every completed dataset by its formula from the imputed values, in their order (a variable
computed earlier is there for a later formula), rounded and limited as recorded; method "derived". A row whose
formula gives no value in every imputation stays missing: they are returned, by variable.

### def `_mismatched` (derived.py)

- **Arguments**: `mi: Any, records: Sequence[Mapping[ComplexType]]`

- **Returns**: `dict[ComplexType]`

For each definition of ``records`` (core.derived.record), the rows where its variable is recorded but a variable of
its formula was imputed, and the formula of the imputed values gives another value in at least one imputed dataset
(beyond the check's tolerance): a recorded BMI beside an imputed height. An inverse filled exactly first avoids it.

### def `impute_and_derive` (derived.py)

- **Arguments**: `frame: pd.DataFrame, spec: Any, steps: Sequence[Mapping[ComplexType]]`

- **Returns**: `Any`

A multiple imputation (biostats.mi.impute, ``spec`` an ImputationSpec) with declared derived variables
(IMPUTE_THEN_DERIVE): each is left out of the chains, so neither imputed nor a predictor of any variable, then
computed in every completed dataset from the imputed values (:func:`derive`). The imputation keeps how
(``info["derived"]``: the approach, the steps, the values left missing and, for the definitions of ``watch``, the
recorded values their imputed variables no longer match; or the records of ``declined`` definitions the user chose
to impute instead), which the trace, a saved version, the diagnostics and the methods paragraph read. The page and
the replay both run through here, so that they compute the same cells.

### def `fill_steps` (derived.py)

- **Arguments**: `plan: FillPlan, data: pd.DataFrame, rows: ComplexType, run: str`

- **Returns**: `list[tuple[ComplexType]]`

The plan as steps of the Enrich draft, one targeted imputation per fill: the table once filled (computed again
here by the call the replay makes), the step's parameters and its description. The parameters hold what the replay
repeats rather than decides again (the formula computed, whether the values count as observed, their rounding,
the levels a fill is limited to) and, as a record only, the definition used (core.derived.record): the replay never
reads the declared list.

---

## external_data.py

_No module description._

**Imports**:
`logging`, `os`, `dataclasses.dataclass`, `pandas`, `core.fingerprint.ROW_ID`, `core.fingerprint.extend_row_ids`, `ingest.dates.AS_DATES`, `ingest.dates.CHOSEN`, `ingest.dates.detect`, `ingest.dates.read_dates`, `ingest.dates.smart_parse_dates`, `ingest.loaders.read_path`

### class `KeyCheck` (external_data.py)

What a merge on keys would do to the rows of the dataset, before it is done (missing keys never match).

**Methods:**

- **repeats**(`self`) -> `bool`
  > No description available.

### def `key_check` (external_data.py)

- **Arguments**: `df, additional_df, left_id_names, right_id_names`

- **Returns**: `ComplexType`

How a merge on these keys would repeat the rows of the dataset: a row whose key more than one row of the file
holds comes out once per such row. A missing key matches nothing (``missing_keys="never"``, as Enrich merges
now). None when the keys cannot be compared (a column missing, or of another type): the merge says why.

### def `_unmatchable` (external_data.py)

- **Arguments**: `values: pd.Series, others: pd.Series`

- **Returns**: `None`

A value of ``values``' kind that ``others`` and ``values`` never hold, for a missing key to match nothing.

### def `date_key_formats` (external_data.py)

- **Arguments**: `main_df, additional_df, recorded`

- **Returns**: `None`

How the Enrich page reads the file's columns named as a date column of the main table, so that date keys match:
the file's date cells as they are ("dates"); a column of text with the format the shape of its own values gives
(ingest.dates.detect: an ISO export reads as ISO whatever the dataset's), the order of day and month, where its
values do not show it, that of the main column's recorded format (``recorded``, from
:func:`ingest.dates.recorded_formats`), else ``dayfirst`` (the import's setting). A column whose values have no
date shape, Excel serial numbers, and two-digit years without the main column's pivot are left out: read with
smart_parse_dates, as before. Recorded with the step: the replay reads them, never detects again.

### def `align_date_keys` (external_data.py)

- **Arguments**: `main_df, additional_df, formats`

- **Returns**: `None`

The additional table with its columns parsed as dates where the main table holds dates under the same name,
so that date keys match in the merge: each with its format in ``formats`` (from :func:`date_key_formats` on the
page, from the step on replay); a column without one, and every column of a step recorded before (``formats``
None), with smart_parse_dates, day first by position, as before. The formats applied are in
``attrs["date_formats"]``, recorded with the step. The Enrich page and the replay of the step both call it, on the
same tables, so they merge the same rows.

### def `_order_dates` (external_data.py)

- **Arguments**: `rows, order_by`

- **Returns**: `pd.Series`

The dates that order a file's rows for "first" or "last": its column, read with the format recorded for it.

### def `_unmatched` (external_data.py)

- **Arguments**: `left, right, left_ids, right_ids`

- **Returns**: `int`

Rows of the dataset whose identifier no row of the file holds, the rows without one included.

### def `add_data` (external_data.py)

- **Arguments**: `df, additional_df, left_id_names, right_id_names, additional_cols, strategy, conflict_resolution, file_rows, missing_keys, order_by`

- **Returns**: `None`

Enrich the dataset by either merging on common identifiers (columns) or appending rows.

Rows of ``df`` keep their row id (``__row_id``, the index) through the merge; rows added by an outer or
right join, appended rows and repeats of a row (one-to-many join) get fresh ids. The merged frame's
``attrs["join_counts"]`` says how many rows came from the main dataset, were added, dropped or repeated.
``file_rows`` (:data:`FILE_ROWS`) says how many rows of the file a row of the dataset takes in a merge on keys;
``missing_keys`` (:data:`MISSING_KEYS`) whether a missing key matches another ("match", pandas' way, kept for
the steps recorded before; Enrich merges with "never"). ``order_by`` ({"column", "format", "pivot"}): with "first"
or "last", a participant's rows of the file are taken in the order of that column's dates, the earliest or the
latest (rows on the same date in the file's order; a row without a date only when none has one), not in the file's
order; "format" reads a text column (ingest.dates), None a column of dates. Recorded with the step; a step
recorded before (None) took the file's order. Same parameters and result as :func:`_merge`.

### def `_merge` (external_data.py)

- **Arguments**: `df, additional_df, left_id_names, right_id_names, additional_cols, strategy, conflict_resolution, validate`

- **Returns**: `None`

Enrich the dataset by either merging on common identifiers (columns) or appending rows.

Parameters:
    df (DataFrame): The main dataset to augment.
    additional_df (DataFrame): The additional dataset to merge or append.
    left_id_names (list): List of identifier column(s) in the main dataset.
    right_id_names (list): List of identifier column(s) in the additional dataset.
    additional_cols (list, optional): List of columns to include from the additional dataset. If None, all columns are included.
    strategy (str, optional): Enrichment strategy. Options are 'left', 'right', 'outer', 'inner', 'cross' for column merges, or 'rows' for row append. Defaults to 'left'.
    conflict_resolution (str, optional): Handle row conflicts based on identifiers with 'replace', 'keep', or 'ignore'. Defaults to 'keep' if None.
    validate (str, optional): pandas' check of a column merge ("many_to_one": one row of the file per key); a file
        that breaks it is refused, and the main dataset returned unchanged.

Returns:
    DataFrame: The enriched DataFrame or the original if an error occurs.

### def `process_data_enrichment` (external_data.py)

- **Arguments**: `df, additional_data_info`

- **Returns**: `None`

Process requested comparisons from JSON config files and write results to Excel.

Parameters:
- df (DataFrame): The dataset to augment.

- additional_data_info (dict): JSON configuration containing merging details.

Returns:
- DataFrame: The dataset df enriched with data from the additional data file. Returns the original df if an error occurs.

### def `process_data_enrichments_from_config` (external_data.py)

- **Arguments**: `df, data_file_name, config`

- **Returns**: `None`

Process requested data enrichments from JSON config files and write results to Excel.

Parameters:
- df (DataFrame): The dataset to augment.

- config (dict): JSON configuration containing comparison details.

Returns:
- DataFrame: The dataset df enriched with data from all the additional data files. Returns the original df if an error occurs.

---

## formulas.py

_Formulas for derived variables and targeted imputation (moved from the Enrich page, no Streamlit).

``evaluate_formula_safely`` validates a formula against the dataset's columns, rewrites date subtractions to
give days, and evaluates it through the whitelisted evaluator (``core.expressions``). The replay engine uses
it as well, so a derived variable is recomputed exactly as it was created._

**Imports**:
`ast`, `datetime`, `logging`, `re`, `collections.abc.Iterable`, `dataclasses.dataclass`, `typing.Any`, `numpy`, `pandas`, `core.expressions.DEFAULT_POLICY`, `core.expressions.evaluate_tree`, `ingest.analyzer.DataAnalyzer`, `ingest.dates.smart_parse_dates`, `utils.logging_config.single_line`

### class `FormulaReading` (formulas.py)

What a formula uses, read as the evaluator reads it (:func:`read_formula`).

### class `FormulaTrial` (formulas.py)

A formula tried on the data as computing it would (an AI engine's suggestion, before it is offered).

**Methods:**

- **ok**(`self`) -> `bool`
  > No description available.

- **given**(`self`) -> `pd.Series`
  > The rows that get a usable value (none when it cannot be computed).

- **from_missing**(`self, data: pd.DataFrame, within: ComplexType`) -> `list[tuple[ComplexType]]`
  > The inputs whose value is missing in rows that still get a value (a comparison or ``np.where`` sends a
  > missing value to its "false" branch), with how many such rows, among ``within`` when it is given.

### def `validate_variable_name` (formulas.py)

- **Arguments**: `name, existing_columns`

- **Returns**: `None`

Validates the variable name and returns (is_valid, error_message).

### def `is_number` (formulas.py)

- **Arguments**: `token`

- **Returns**: `None`

Check if a string token represents a valid number.

**Parameters**:

token : str
    The token to check.

**Returns**:

bool
    True if the token can be converted to a float, False otherwise.

### def `_parsed` (formulas.py)

- **Arguments**: `formula: str`

- **Returns**: `tuple[ComplexType]`

The formula's tree, its ``""quoted""`` names by the placeholders that stand for them in the tree, or why it
cannot be read.

### def `_functions` (formulas.py)

- **Arguments**: `tree: ast.Expression`

- **Returns**: `set[int]`

The ids of the Name nodes a formula calls (``round(x)``), as opposed to the variables it uses.

### def `read_formula` (formulas.py)

- **Arguments**: `formula: str`

- **Returns**: `FormulaReading`

The variables a formula uses: a name, or ``""a name""`` for a column with spaces or symbols. ``np`` and ``pd``,
and the functions of the evaluator (``round``, ``abs``, ``min``... when called) are not variables; another function
called by its bare name (``sqrt``) is refused with the hint ``np.sqrt``. The one reading of a formula: the
evaluator, the AI engine's suggestions and their diagrams use it, so a diagram shows what computing it uses.

### def `formula_from_model` (formulas.py)

- **Arguments**: `formula: str`

- **Returns**: `str`

A formula written by a language model, in this application's notation: ``name =`` dropped when it starts the
formula and names one of ``targets`` (the variable it computes: ``sex = 'F'`` stays refused, it is no formula),
`` `a name` `` written ``""a name""``, and ``^`` (a power, as in mathematics) written ``**`` outside text literals
(here ``^`` is a bitwise exclusive or, which gives wrong numbers silently).

### def `_written_out` (formulas.py)

- **Arguments**: `node: ast.AST`

- **Returns**: `bool`

No description available.

### def `_argument` (formulas.py)

- **Arguments**: `call: ast.Call, position: int, keyword: str`

- **Returns**: `ComplexType`

No description available.

### def `_across_columns` (formulas.py)

- **Arguments**: `call: ast.Call`

- **Returns**: `bool`

``np.nanmean([a, b], axis=0)``: across the columns written in the list, row by row.

### def `other_rows` (formulas.py)

- **Arguments**: `formula: str, data: ComplexType`

- **Returns**: `ComplexType`

What makes ``formula`` use other rows than the row it computes, in words, or None when it gives each row a value of
that row's own variables only, as an exact formula does (a weight from the BMI and the height). Other rows are used
by an aggregate (a mean, a maximum, a quantile), a rank, a running or shifted value, a value taken from another row
(``x[0]``) or a test against a column (``a.isin(b)``): the values are then estimates, and change with the rows the
formula is computed on. Given the ``data``, so are dates kept as text, whose order of day and month the evaluator
reads from the rows. A formula that cannot be read is not of each row's own values.

### def `row_wise` (formulas.py)

- **Arguments**: `formula: str, data: ComplexType`

- **Returns**: `bool`

Whether ``formula`` gives each row a value of that row's own variables only (:func:`other_rows`).

### def `is_text_column` (formulas.py)

- **Arguments**: `series: pd.Series`

- **Returns**: `bool`

A column of text, neither numbers, yes/no values nor dates: it compares with text only.

### def `usable_values` (formulas.py)

- **Arguments**: `values: pd.Series`

- **Returns**: `pd.Series`

The rows with a usable value: not missing, and finite for numbers (``fillna`` would put an infinity in).

### def `_absent_literals` (formulas.py)

- **Arguments**: `formula: str, data: pd.DataFrame`

- **Returns**: `list[str]`

The texts a formula compares a text column with (``==``, ``!=``, ``.isin``) that occur in no row of it: the
comparison then gives the same answer everywhere, which no error reveals.

### def `try_formula` (formulas.py)

- **Arguments**: `formula: str, data: pd.DataFrame`

- **Returns**: `FormulaTrial`

``formula`` computed on ``data`` by the evaluator of "Compute" and "Apply", or why it cannot be: a syntax the
application does not read, ``^``, a name that is not a column, or a computation the data refuse (a column that
holds text compared with a number, say: the text columns it uses are named, with values held by several rows).
With a ``target`` that holds numbers (an imputation), a result that is not numbers is refused too: filling the
column with yes/no values or text would change its type. Nothing is sent anywhere: the trial runs here.

### def `validate_and_extract_columns_ast` (formulas.py)

- **Arguments**: `formula: str, dataset_columns: list`

- **Returns**: `tuple[ComplexType]`

Reads the formula (:func:`read_formula`) and checks that every variable it uses is a column of the data: the
variables it uses, or why it cannot be computed.

### def `extract_constant_value` (formulas.py)

- **Arguments**: `constant_token: str`

- **Returns**: `str`

Extract the value from a constant token like constant(val1) -> val1

### def `_names_in_literals` (formulas.py)

- **Arguments**: `tree: ast.Expression, columns: list[str]`

- **Returns**: `bool`

Whether a text literal of the formula holds one of its columns' names (the old renaming changed that text).

### def `evaluate_formula_safely` (formulas.py)

- **Arguments**: `formula: str, variable_name: str, dataset: pd.DataFrame, return_series: bool`

- **Returns**: `tuple[ComplexType]`

Safely evaluate a computation formula to create a new variable in the dataset.

**Parameters**:

formula : str
    The computation formula entered by the user.
variable_name : str
    The name of the new variable to create.
dataset : pd.DataFrame
    The dataset containing columns to use in the formula.
return_series : bool, optional
    If True, returns the computed Series instead of a success message, and does not modify the dataset.

**Returns**:

tuple[bool, Any, pd.DataFrame]
    A tuple containing:
    - A boolean indicating success or failure.

    - A success message OR the computed Series (if return_series=True), OR an error message.

    - The modified DataFrame (or the original DataFrame on failure).

### def `numeric_summary` (formulas.py)

- **Arguments**: `values: pd.Series`

- **Returns**: `pd.DataFrame`

Classical and robust statistics of a numeric variable: N, missing, mean, SD, median, quartiles, min and max.

The median and the quartiles are not moved by a few extreme values (entry errors), the mean and the SD are;
infinite values (a division by zero) are counted apart and left out of the other statistics.

### def `far_outside` (formulas.py)

- **Arguments**: `values: pd.Series, k: float`

- **Returns**: `tuple[ComplexType]`

Rows whose value lies beyond Tukey's far-out fences, Q1 - k IQR and Q3 + k IQR (k = 3), with the fences:
values that are probably entry errors (a height in metres among centimetres, a weight of 0) rather than
the tail of the distribution. Infinite values are always included.

---

## imputation_service.py

_Single imputation of the session data (moved from the Enrich page, no Streamlit).

``apply_imputer`` fills every missing value of a table (mean, median, KNN, chained equations, missForest);
``impute_rows`` fills only the rows of the analysis population with an imputer fitted on those rows, so
excluded patients neither inform nor receive imputed values; ``fill_with_formula`` is the targeted
imputation of one column. The replay engine calls the same functions._

**Imports**:
`logging`, `collections.abc.Collection`, `collections.abc.Mapping`, `collections.abc.Sequence`, `dataclasses.dataclass`, `dataclasses.field`, `typing.Any`, `numpy`, `pandas`, `enrich.data_imputation`, `enrich.formulas.evaluate_formula_safely`, `enrich.formulas.read_formula`, `enrich.formulas.usable_values`

### class `ImputationOutcome` (imputation_service.py)

No description available.

### def `debug_print` (imputation_service.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `match_null_mask_to_encoded_columns` (imputation_service.py)

- **Arguments**: `initial_null_mask, output_cols, original_cols`

- **Returns**: `None`

Align initial null mask with output columns, including deducing one-hot encoded versions of categorical variables.
Ensures the mask matches the order of output_cols.

Parameters:
    initial_null_mask (pd.DataFrame): Original null mask for input data.
    output_cols (list): Ordered list of columns after imputation and encoding.
    original_cols (list): List of original columns from the input data.

Returns:
    pd.DataFrame: Aligned null mask matching the column order in output_cols.

### def `apply_imputer` (imputation_service.py)

- **Arguments**: `data, numerical_imputation_method, categorical_imputation_method, cat_encoder, num_scaler, remainder_columns, remainder_strategy, remainder_threshold, progress_placeholder, debug`

- **Returns**: `None`

Applies an imputer to a pandas DataFrame, with options for numerical and categorical imputations,
and handling of remainder columns. Also tracks imputed values.

Parameters:
    data (pd.DataFrame): The input data for imputation.
    numerical_imputation_method (str): Method for imputing numerical columns ('mean', 'median', 'knn', etc.).
    categorical_imputation_method (str): Method for imputing categorical columns ('most_frequent', 'missforest', etc.).
    cat_encoder (bool): Whether to apply one-hot encoding to categorical columns.
    num_scaler (bool): Whether to scale numerical columns.
    remainder_columns (list or str): Columns to pass through without transformation ('auto' for automatic detection).
    remainder_strategy (str): Strategy for handling remainder columns ('passthrough', 'drop', etc.).
    remainder_threshold (float): Threshold for automatic remainder column detection.
    progress_placeholder (object): Placeholder for showing progress in Streamlit.
    debug (bool): Flag to enable debug print statements.

Returns:
    pd.DataFrame: The imputed DataFrame with updated column names.
    pd.DataFrame: A mask DataFrame indicating which values were imputed.

### def `impute_rows` (imputation_service.py)

- **Arguments**: `frame: pd.DataFrame, rows: Any`

- **Returns**: `ImputationOutcome`

Impute the missing values of ``rows`` (row ids; every row when None) with an imputer fitted on them.

Rows outside ``rows`` are returned untouched. One-hot encoding, scaling and dropping remainder columns
change the columns of the table; they are refused when only part of the rows is imputed. The ``set_aside``
columns (declared derived variables, computed afterwards from the imputed values) are neither imputed nor used to
impute the others: the imputer never sees them, and they come back as they were.

### def `round_half_away` (imputation_service.py)

- **Arguments**: `values: pd.Series, decimals: int`

- **Returns**: `pd.Series`

Rounded as records are (Excel, SAS, SPSS, Stata), half away from zero, not to the even neighbour as ``round`` does:
2.5 gives 3. A value stored a hair below its written half (2.675 as 2.67499...) rounds as written.

### def `fill_with_formula` (imputation_service.py)

- **Arguments**: `frame: pd.DataFrame, target: str, formula: str, rows: ComplexType`

- **Returns**: `tuple[ComplexType]`

Fill the missing values of ``target`` with a formula of the other columns; returns (table, filled). With ``rows``
(the analysis population), the formula is computed on those rows only, so that an aggregate such as a mean sees no
excluded patient, and fills them only: the excluded rows keep their missing values, as with the global imputation.
With ``complete_inputs`` (an exact formula), only the rows where every variable of the formula has a value are
filled: a comparison or ``np.where`` answers a missing value with its "false" branch, which is no exact value. With
``finite_only``, an infinity (a division by zero) is no value either; a step recorded without it fills one, as it
did. ``decimals`` rounds the filled values as the variable is recorded; ``levels`` fills only the rows whose text or
code variables take one of these values (those a derived variable's check saw). The page and the replay both fill
through here.

### def `formula_inputs` (imputation_service.py)

- **Arguments**: `formula: str, target: str, frame: pd.DataFrame`

- **Returns**: `list[str]`

The variables of ``formula`` other than ``target`` (``weight.fillna(...)`` names the variable it fills).

### def `left_missing` (imputation_service.py)

- **Arguments**: `frame: pd.DataFrame, target: str, formula: str, rows: ComplexType`

- **Returns**: `tuple[ComplexType]`

The missing values of ``target`` an exact formula leaves, because a variable of it has no value in the row, and
those variables (counts and names only: the trace's description says them).

### def `describe_imputation` (imputation_service.py)

- **Arguments**: `numerical: str, categorical: str, n_rows: ComplexType, set_aside: Sequence[str]`

- **Returns**: `str`

The trace's description of a single imputation; ``n_rows``: the analysis population it was fitted on;
``set_aside``: the derived variables kept out of it, to be computed from the imputed values.

### def `describe_formula_fill` (imputation_service.py)

- **Arguments**: `target: str, filled: int`

- **Returns**: `str`

The trace's description of a targeted imputation; ``n_rows``: the analysis population it was computed on;
``left``: the missing values an exact formula left, and the variables without a value there (:func:`left_missing`);
``derived_of``: the declared derived variable whose definition computed it (its own, or an inverse); ``single``:
the values an exact formula computed from imputed ones, single-imputed as they are.

---

## recode.py

_The values of a categorical variable as written, recoded (Data > Clean > Categories): each value given a clean value
or set to missing, and written into a new text column beside the original, which is kept. One traced step
(``recode_levels``) that the replay redoes.

What the step records, and what the replay recomputes:

- the fold rule (:data:`FOLD`), never the values it folds. Values that differ by spaces take their group's spelling,
  the one most rows hold. So do values that differ by capitals in words of 4 letters or more (Male, MALE, male).
  In shorter codes, case can carry meaning (genotypes AA, Aa, aa; Rh haplotypes Cde, cde; pH): a variant there is
  folded only when it holds under :data:`CASE_SHARE` of its group's rows and the group writes none of its spellings
  in mixed case (M and m, NR and nr); otherwise it is offered. The replay folds the values it reads with this rule;

- the user's choices (``mapping``): a ticked offer or a renamed value applies to its spelling's whole group, a value
  given its own clean value to itself only, and ``kept`` lists the folded values put back as written;

- what each choice is (``kinds``), what each value set to missing means (``missing_meaning``), and why two values
  that each hold :data:`GROUPING_SHARE` of the rows were merged (``groupings``);

- counts, never a count per value.

The offers, never applied:

- a one-letter code and the only longer value it begins (the rarer joins the more frequent);

- a rare near spelling of a frequent value (a one-row "Luxembourgg" beside 397 "Luxembourg");

- a case variant the fold leaves apart;

- the codes that may mean missing. Some are read as unknown; for others the user must say what they mean ("n/a" can
  mean not applicable, "none" an absence, "NR" non-reactive).

One edit apart can be two categories, so a near spelling is never offered in these cases:

- across a token holding a digit, a short token or an acronym (HFmrEF and HFrEF, Stage IIIB and IIIA, 3a and 3b);

- when the words differ in their first four letters (Unstable and Stable, Abnormal and Normal, Hypotension and
  Hypertension);

- between two values that are not rare against each other.

Nothing here reads an outcome: the values and their rows only._

**Imports**:
`__future__.annotations`, `difflib`, `re`, `unicodedata`, `collections.defaultdict`, `collections.abc.Collection`, `collections.abc.Iterable`, `collections.abc.Mapping`, `dataclasses.dataclass`, `typing.Any`, `pandas`

### class `Offer` (recode.py)

A spelling that could join another, or become missing (``clean`` None): offered, never applied.

### class `Suggestions` (recode.py)

No description available.

### class `Candidate` (recode.py)

A text column the Categories tab lists.

### def `_spaced` (recode.py)

- **Arguments**: `value: str`

- **Returns**: `str`

No description available.

### def `_folded` (recode.py)

- **Arguments**: `value: str`

- **Returns**: `str`

No description available.

### def `_plain` (recode.py)

- **Arguments**: `value: str`

- **Returns**: `str`

No description available.

### def `_words` (recode.py)

- **Arguments**: `text: str`

- **Returns**: `list[str]`

No description available.

### def `_why` (recode.py)

- **Arguments**: `value: str, spelling: str`

- **Returns**: `str`

No description available.

### def `_case_rank` (recode.py)

- **Arguments**: `text: str`

- **Returns**: `int`

Capitalised first (Female), then lower case (male), then the rest (MALE).

### def `_plain_case` (recode.py)

- **Arguments**: `word: str`

- **Returns**: `bool`

No description available.

### def `_case_free` (recode.py)

- **Arguments**: `variant: str, spelling: str`

- **Returns**: `bool`

Whether two spellings differ only in words whose capitals carry no meaning: 4 letters or more, each in lower
case, capitals or capitalised.

### def `level_counts` (recode.py)

- **Arguments**: `values: pd.Series`

- **Returns**: `dict[ComplexType]`

The values as written (their text) and their rows, the most frequent first (ties by value); missing values
left out.

### def `_spelling` (recode.py)

- **Arguments**: `rows: Mapping[ComplexType]`

- **Returns**: `str`

A group's spelling: the most rows; on a tie, capitalised, then lower case, then the first in order.

### def `_groups` (recode.py)

- **Arguments**: `counts: Mapping[ComplexType]`

- **Returns**: `list[dict[ComplexType]]`

The spellings (spaces collapsed) that differ by capitals only, and their rows: one dict per group.

### def `fold` (recode.py)

- **Arguments**: `counts: Mapping[ComplexType]`

- **Returns**: `dict[ComplexType]`

The rule :data:`FOLD` on the values as written (``level_counts``): each value -> the spelling it takes, where
they differ.

### def `_marked` (recode.py)

- **Arguments**: `text: str`

- **Returns**: `bool`

Holds a digit, a sign or a Roman numeral: what names a category of its own.

### def `_strict` (recode.py)

- **Arguments**: `token: str`

- **Returns**: `bool`

A token that names a category by itself: a digit, four characters or fewer, or capitals inside (HFmrEF).

### def `_apart` (recode.py)

- **Arguments**: `a: str, b: str`

- **Returns**: `bool`

Whether two spellings differ by what names a category: digits, Roman numerals, signs, a strict token.

### def `_near` (recode.py)

- **Arguments**: `rare: str, common: str`

- **Returns**: `bool`

A near spelling: no category apart, a first difference past a word's first four letters, difflib >= 0.85.

### def `suggest` (recode.py)

- **Arguments**: `counts: Mapping[ComplexType]`

- **Returns**: `Suggestions`

What each value as written could become (``level_counts``): the fold, pre-filled, and the offers (near spellings
only with ``near``: they compare every rare value with every frequent one).

### def `clean_value` (recode.py)

- **Arguments**: `value: str, spellings: Mapping[ComplexType], mapping: Mapping[ComplexType], kept: Collection[str]`

- **Returns**: `ComplexType`

What one value as written becomes: as written when ``kept``; its own choice when the fold gave it a spelling and
the mapping names it; else the choice for its spelling (its group's), or that spelling.

### def `_spellings` (recode.py)

- **Arguments**: `counts: Mapping[ComplexType], fold_rule: ComplexType`

- **Returns**: `dict[ComplexType]`

No description available.

### def `plan` (recode.py)

- **Arguments**: `counts: Mapping[ComplexType], mapping: Mapping[ComplexType]`

- **Returns**: `dict[ComplexType]`

Each value as written that a recode changes -> its clean value (None: missing). ``fold_rule`` None: no fold
(the mapping lists every change).

### def `recode` (recode.py)

- **Arguments**: `values: pd.Series, full: Mapping[ComplexType]`

- **Returns**: `pd.Series`

The text of ``values`` with each value ``full`` lists replaced by its clean value (None: missing), as a text
column: 2 and "2" are one value, as ``level_counts`` counts them.

### def `recode_column` (recode.py)

- **Arguments**: `df: pd.DataFrame, source: str, target: str, mapping: Mapping[ComplexType]`

- **Returns**: `tuple[ComplexType]`

The table with ``target``, the recoded copy of ``source``, right after it (``replaces``: an earlier copy of that
name is written anew). Returns the table and how many distinct values were kept as written.

### def `groupings` (recode.py)

- **Arguments**: `counts: Mapping[ComplexType], full: Mapping[ComplexType]`

- **Returns**: `dict[ComplexType]`

The clean values that merge two spellings (after the fold) each holding :data:`GROUPING_SHARE` of the rows or
more (NYHA III and IV): a grouping of categories, not a correction. Clean value -> those spellings.

### def `changes` (recode.py)

- **Arguments**: `counts: Mapping[ComplexType], mapping: Mapping[ComplexType], kinds: Mapping[ComplexType]`

- **Returns**: `dict[ComplexType]`

The values a recode changes, counted, never named: ``{"by_kind": {kind: [values, rows]}, "set_to_missing":
{meaning: [values, rows]}}``. A value takes the kind of the choice it follows, or the fold's reason.

### def `report_params` (recode.py)

- **Arguments**: `params: Mapping[ComplexType]`

- **Returns**: `dict[ComplexType]`

The step's parameters as the trace report prints them: the choices counted, never named (the trace keeps them
for the replay).

### def `recoded_columns` (recode.py)

- **Arguments**: `steps: Iterable[Mapping[ComplexType]]`

- **Returns**: `dict[ComplexType]`

Each column recoded in a trace's steps -> the copy its last recode wrote.

### def `last_record` (recode.py)

- **Arguments**: `steps: Iterable[Mapping[ComplexType]], source: str, target: str`

- **Returns**: `ComplexType`

The parameters of the last recode of ``source`` into ``target`` in a trace's steps: cleaning again starts from
its choices.

### def `candidates` (recode.py)

- **Arguments**: `df: pd.DataFrame, recoded: ComplexType`

- **Returns**: `list[Candidate]`

The text and categorical columns with 2 to :data:`MAX_VALUES` values as written, the copies ``recoded`` names
(source -> copy) left out: those with values to check first, then the cleaned ones, then the others, each in the
dataset's order. Near spellings are not looked for here (they compare every pair of values): only in the open
variable.

---

## transformation_logic.py

_New variables derived from existing ones: statistics, scaling, dimensionality reduction, clustering and encoding.

``apply_variable_transformation`` is the only implementation: the Enrich page calls it, and the replay of a trace
calls it again with the parameters the page recorded, so both produce the same columns with the same names._

**Imports**:
`numpy`, `pandas`, `sklearn.preprocessing.LabelEncoder`, `sklearn.preprocessing.MinMaxScaler`, `sklearn.preprocessing.OneHotEncoder`, `sklearn.preprocessing.OrdinalEncoder`, `sklearn.preprocessing.StandardScaler`, `biostats.reduction.fit_famd`, `biostats.reduction.fit_pca`, `core.expressions.safe_format`

### def `_unique_names` (transformation_logic.py)

- **Arguments**: `new_columns: dict, existing`

- **Returns**: `dict`

A name already in the table, or already given to another new column, gets the first free suffix _1, _2...

### def `apply_variable_transformation` (transformation_logic.py)

- **Arguments**: `dataframe, params`

- **Returns**: `None`

The new columns of one transformation, on ``dataframe``'s index (None when the transformation makes none).

Raises ValueError when the input cannot be transformed: missing values for a dimensionality reduction or a
clustering (they need complete data), fewer than two columns to reduce, fewer than two rows to cluster.

---
