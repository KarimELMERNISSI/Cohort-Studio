"""
The check of a declared derived variable against the data (core.derived declares it). On the rows where the variable
and every variable of its formula have a value, the formula is computed here, by the evaluator of every formula, and
compared with the recorded value. Nothing is estimated and nothing leaves the machine; the disagreeing rows are kept
for the page to show, never written anywhere.

A definition the data disagree with, or one that cannot be checked, is never used to fill or impute unless the user
accepts it; agreeing with the data says nothing of the published reference, which a definition's note names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from core.derived import DerivedVariable, fill_order
from enrich.formulas import (
    CARET_IS_NOT_A_POWER,
    evaluate_formula_safely,
    is_text_column,
    other_rows,
    read_formula,
    try_formula,
    usable_values,
)
from utils.data_analyzer import DataAnalyzer

# the verdicts: ids kept with a check, and their words
AGREES, MOSTLY, DISAGREES, TOO_FEW, CANNOT = "agrees", "mostly", "disagrees", "too_few", "cannot"
LABELS = {
    AGREES: "agrees",
    MOSTLY: "mostly agrees",
    DISAGREES: "disagrees",
    TOO_FEW: "too few rows to check",
    CANNOT: "cannot be used",
}
MIN_ROWS = 60  # no disagreement in 60 rows bounds the disagreement rate below 5% (95% confidence)
AGREE_SHARE, MOSTLY_SHARE = 0.99, 0.90
# agreement beyond chance: Cohen's kappa, chance estimated with the recorded values permuted
AGREE_KAPPA, MOSTLY_KAPPA = 0.95, 0.80
LEVEL_SHARE, LEVEL_ROWS = (
    0.90,
    10,
)  # a level of a text or code variable agreeing less than this disagrees
_CONSTANT_SHARE = 0.95  # a variable holding one value in more rows than this tells little
_PERMUTATIONS, _SEED = 20, 20261021
# the precision a value is recorded with: 1000 down to 0.000001
_UNITS = tuple(10.0**k for k in range(3, -7, -1))
_CODES = (
    10  # whole numbers with at most this many distinct values are codes, checked level by level
)
_EXACT = (
    20  # whole numbers with at most this many distinct values (scores, counts) carry no rounding
)
_HINT_SHARE, _HINT_SPREAD = 0.8, 0.02
# the factors between two units of a clinical variable, beside the powers of ten
_FACTORS = {
    2.2046: "kg and lb",
    2.54: "inch and cm",
    8.84: "mg/L and µmol/L",
    18.0: "mmol/L and mg/dL, glucose",
    38.67: "mmol/L and mg/dL, cholesterol",
    88.4: "mg/dL and µmol/L, creatinine",
}


@dataclass(frozen=True)
class DerivedCheck:
    """What the data say of a definition (:func:`check_derived`): counts, never values, except the rows kept for the
    page to show (``disagreeing``) and to fill (``fillable_rows``)."""

    definition: DerivedVariable
    verdict: str
    inputs: tuple[str, ...] = ()
    compared: int = 0  # rows with the variable and every variable of its formula recorded
    agreeing: int = 0
    # the typical (median) difference counted as agreement, in the variable's unit; None for text
    tolerance: float | None = None
    # the tolerance of the variable's own rounding; False: with its formula's variables' rounding, or declared
    strict: bool = True
    tolerance_high: float | None = None  # the 95th percentile of the rows' tolerances
    reason: str | None = None  # why it disagrees, or cannot be used or checked
    unit_hint: str | None = None  # the disagreeing rows are another unit by a constant factor
    warnings: tuple[str, ...] = ()  # what the check could not test (a variable taking one value)
    fills: bool = True  # False for a date: dates are never imputed
    kappa: float | None = None
    disagreeing: pd.Index = field(default_factory=lambda: pd.Index([]), repr=False, compare=False)
    # missing values of the variable whose formula has every variable recorded, in a checked level
    fillable_rows: pd.Index = field(default_factory=lambda: pd.Index([]), repr=False, compare=False)
    # the values each text or code variable of the formula takes in the rows compared: no other was checked
    checked_levels: Mapping[str, frozenset[str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def label(self) -> str:
        return LABELS[self.verdict]

    def summary(self) -> dict[str, Any]:
        """The check in numbers, as a step that used the definition records it: never its words (a reason can name a
        value of the data, "where sex = F")."""
        return {
            "verdict": self.verdict,
            "compared": self.compared,
            "agreeing": self.agreeing,
            "tolerance": self.tolerance,
            "strict": self.strict,
            "kappa": None if self.kappa is None else round(self.kappa, 4),
        }

    @property
    def fillable(self) -> int:
        return len(self.fillable_rows) if self.fills else 0

    @property
    def usable(self) -> bool:
        """It may fill and impute: the data agree with it, or the user accepted it although they mostly agree or
        cannot check it."""
        return self.verdict == AGREES or (
            self.definition.accepted and self.verdict in (MOSTLY, TOO_FEW)
        )

    @property
    def evidence(self) -> str:
        """The check in words: 'gives the recorded value in 368 of 371 rows where bmi, weight_kg and height_cm are
        recorded (±0.05)'."""
        if self.verdict == CANNOT or not self.compared:
            return self.reason or self.label
        names = [self.definition.target, *self.inputs]
        where = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
        if self.tolerance is None:
            within = ""
        elif self.strict and self.tolerance < 1e-5:
            within = " (exactly)"
        else:
            within = f" (±{self.tolerance:.2g})"
        return (
            f"gives the recorded value in {self.agreeing:,} of {self.compared:,} rows where {where} are recorded"
            f"{within}"
        )


def resolution(values: pd.Series) -> float:
    """
    The precision a column is recorded with: the coarsest power of ten that divides at least 99% of its values (0.1
    for a weight recorded to 100 g, 1 for an age in years). 0 when none does (a value computed to full precision), and
    for scores and counts (at most twenty distinct whole numbers), which carry no rounding.
    """
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_datetime64_any_dtype(values):
        return 0.0
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if not len(x) or (np.all(x == np.round(x)) and len(np.unique(x)) <= _EXACT):
        return 0.0
    for unit in _UNITS:
        q = x / unit
        if np.mean(np.isclose(q, np.round(q), rtol=0.0, atol=1e-6)) >= 0.99:
            return unit
    return 0.0


def _levels(series: pd.Series) -> bool:
    """A variable checked level by level: text, yes/no, or codes (at most ten distinct whole numbers)."""
    if is_text_column(series) or pd.api.types.is_bool_dtype(series):
        return True
    x = pd.to_numeric(series, errors="coerce").dropna()
    return bool(len(x)) and bool((x == x.round()).all()) and x.nunique() <= _CODES


def _is_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    return is_text_column(series) and series.name in DataAnalyzer(series.to_frame()).date_cols


def _computed(formula: str, frame: pd.DataFrame) -> pd.Series | None:
    ok, result, _ = evaluate_formula_safely(formula, "__derived__", frame, return_series=True)
    return pd.Series(result, index=frame.index) if ok else None


def _floor(values: np.ndarray) -> np.ndarray:
    return 1e-9 * np.maximum(1.0, np.abs(values))


def _propagated(
    definition: DerivedVariable, rows: pd.DataFrame, inputs: list[str], values: np.ndarray
) -> np.ndarray:
    """Per row: half the variable's recording unit, plus what rounding each input to its own unit moves the formula, up
    or down, whichever moves it more (a BMI to 0.1 from a weight to 0.1 kg and a height to 1 cm: about 0.21; a curved
    formula moves more one way)."""
    tau = np.full(len(values), resolution(rows[definition.target]) / 2)
    for name in inputs:
        unit = resolution(rows[name])
        if not unit or is_text_column(rows[name]):
            continue
        shift = np.zeros(len(values))
        for half in (unit / 2, -unit / 2):
            moved = _computed(
                definition.formula, rows.assign(**{name: pd.to_numeric(rows[name]) + half})
            )
            if moved is not None:
                moved_by = np.abs(
                    pd.to_numeric(moved, errors="coerce").to_numpy(dtype=float) - values
                )
                shift = np.maximum(shift, np.nan_to_num(moved_by, nan=0.0))
        tau += shift
    return tau + _floor(values)


def _unit_hint(recorded: np.ndarray, values: np.ndarray) -> str | None:
    """A constant factor between the recorded values and the formula's in most disagreeing rows: another unit."""
    keep = np.isfinite(recorded) & np.isfinite(values) & (values != 0) & (recorded != 0)
    if keep.sum() < 3:
        return None
    ratios = recorded[keep] / values[keep]
    ratio = float(np.median(ratios))
    if ratio <= 0 or np.mean(np.abs(ratios / ratio - 1) <= _HINT_SPREAD) < _HINT_SHARE:
        return None
    power = round(np.log10(ratio))
    if power and abs(ratio / 10.0**power - 1) <= _HINT_SPREAD:
        return f"the recorded values are {10.0**power:,.6g} times the formula's: another unit?"
    named = next(
        (
            label
            for factor, label in _FACTORS.items()
            for f in (factor, 1 / factor)
            if abs(ratio / f - 1) <= _HINT_SPREAD
        ),
        None,
    )
    if named is None:
        return None
    return f"the recorded values are {ratio:.4g} times the formula's: another unit ({named})?"


def _agree(recorded: np.ndarray, values: np.ndarray, tau: np.ndarray | None) -> np.ndarray:
    if tau is None:  # text or codes: the same value
        return recorded.astype(str) == values.astype(str)
    return np.abs(recorded - values) <= tau


def _cannot(definition: DerivedVariable, reason: str) -> DerivedCheck:
    return DerivedCheck(definition, CANNOT, reason=reason)


def _multiple_of_one(
    definition: DerivedVariable, rows: pd.DataFrame, inputs: list[str], got: np.ndarray
) -> bool:
    """The formula is a constant times its one variable: a copy or a unit conversion, which agrees by construction."""
    if definition.kind == "conversion" or len(inputs) != 1 or _levels(rows[inputs[0]]):
        return False
    x = pd.to_numeric(rows[inputs[0]], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = got / x
    ratio = ratio[np.isfinite(ratio)]
    return bool(len(ratio)) and float(np.ptp(ratio)) <= 1e-9 * max(
        1.0, abs(float(np.median(ratio)))
    )


def _level_disagreeing(rows: pd.DataFrame, inputs: list[str], agree: np.ndarray) -> str | None:
    """The first level of a text or code variable whose rows give the recorded value less often than 90%: the formula
    is wrong for it (a factor for women only), however well the others agree."""
    for name in inputs:
        if not _levels(rows[name]):
            continue
        levels = rows[name].astype(str).to_numpy()
        for level in pd.unique(levels):
            in_level = levels == level
            if in_level.sum() >= LEVEL_ROWS and agree[in_level].mean() < LEVEL_SHARE:
                k, m = int(agree[in_level].sum()), int(in_level.sum())
                return f"it disagrees where {name} = {level}: {k:,} of {m:,} rows give the recorded value"
    return None


def _imputed_rows(
    index: pd.Index, imputed: Mapping[str, Any] | None, names: list[str]
) -> np.ndarray:
    """The rows where an imputation filled one of ``names`` (row ids per column, as the session keeps them)."""
    out = np.zeros(len(index), dtype=bool)
    for name in names:
        ids = (imputed or {}).get(name)
        if ids is not None and len(ids):
            out |= index.isin(np.asarray(ids))
    return out


def check_derived(
    definition: DerivedVariable, data: pd.DataFrame, *, imputed: Mapping[str, Any] | None = None
) -> DerivedCheck:
    """
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
    """
    target, formula = definition.target, definition.formula
    reading = read_formula(formula)
    if reading.problem:
        return _cannot(definition, reading.problem)
    if reading.power_caret:
        return _cannot(definition, CARET_IS_NOT_A_POWER)
    if target not in data.columns:
        return _cannot(definition, f"{target} is not in the data")
    if target in reading.names:
        return _cannot(definition, f"the formula uses {target}, the variable it computes")
    unknown = [name for name in reading.names if name not in data.columns]
    if unknown:
        return _cannot(definition, f"not in the data: {', '.join(unknown)}")
    inputs = list(reading.names)
    if not inputs:
        return _cannot(definition, "the formula uses no variable")
    other = other_rows(formula, data)
    if other:  # dates kept as text say so themselves
        return _cannot(
            definition,
            other if other.startswith("the dates of") else f"it uses other rows ({other})",
        )
    # the formula's columns only: each evaluation profiles the columns it is given
    data = data[list(dict.fromkeys([target, *inputs]))]
    trial = try_formula(formula, data, target=target)
    if not trial.ok or trial.values is None:
        return _cannot(definition, trial.problem or "it cannot be computed")

    recorded = data[target]
    present = data[inputs].notna().all(axis=1) & usable_values(trial.values)
    compared = recorded.notna() & present & ~_imputed_rows(data.index, imputed, [target, *inputs])
    n = int(compared.sum())
    rows = data.loc[compared]
    fills = not _is_date(recorded)
    warnings: list[str] = [] if fills else [f"{target} is a date: it is checked, never filled"]
    # a level of a text or code variable is checked when LEVEL_ROWS compared rows have it, as the level rule needs: the
    # rows with another one (rarer, or never compared) are not filled
    unchecked = pd.Series(False, index=data.index)
    levels: dict[str, frozenset[str]] = {}
    for name in inputs:
        # a code is a code in its recorded values: an imputed mean is no level
        if _levels(data.loc[~_imputed_rows(data.index, imputed, [name]), name]):
            counts = rows[name].astype(str).value_counts()
            levels[name] = frozenset(counts.index[counts >= LEVEL_ROWS])
            unchecked |= data[name].notna() & ~data[name].astype(str).isin(levels[name])
    base = {
        "inputs": tuple(inputs),
        "compared": n,
        "fills": fills,
        "fillable_rows": data.index[recorded.isna() & present & ~unchecked],
        "checked_levels": levels,
    }
    if n < MIN_ROWS:
        reason = f"{n:,} rows have {target} and every variable of its formula recorded: the check needs {MIN_ROWS}"
        return DerivedCheck(definition, TOO_FEW, reason=reason, warnings=tuple(warnings), **base)
    top = rows[target].astype(str).value_counts(normalize=True).iloc[0]
    if top > _CONSTANT_SHARE:
        reason = f"{target} takes one value in {top:.0%} of the rows compared: a fit by chance cannot be ruled out"
        return DerivedCheck(definition, TOO_FEW, reason=reason, warnings=tuple(warnings), **base)
    for name in inputs:
        share = rows[name].astype(str).value_counts(normalize=True).iloc[0]
        if share > _CONSTANT_SHARE:
            warnings.append(
                f"{name} takes one value in {share:.0%} of the rows: the formula is checked for it only"
            )

    textual = is_text_column(recorded) or not pd.api.types.is_numeric_dtype(trial.values)
    strict = True
    if textual:
        got, want, tau = trial.values[compared].to_numpy(), rows[target].to_numpy(), None
    else:
        got = pd.to_numeric(trial.values[compared], errors="coerce").to_numpy(dtype=float)
        want = pd.to_numeric(rows[target], errors="coerce").to_numpy(dtype=float)
        if definition.tolerance is not None:  # declared: not the variable's own rounding
            tau, strict = np.full(n, float(definition.tolerance)) + _floor(got), False
        else:
            # the variable's own rounding first: when nine rows in ten agree within it, the variable was
            # computed from the recorded values and the others disagree (an entry error of a few tenths)
            tau = resolution(rows[target]) / 2 + _floor(got)
            if _agree(want, got, tau).mean() < MOSTLY_SHARE:
                tau, strict = _propagated(definition, rows, inputs, got), False
    agree = _agree(want, got, tau)
    agreeing = int(agree.sum())
    share = agreeing / n
    rng = np.random.default_rng(_SEED)
    chance = float(
        np.mean([_agree(want[rng.permutation(n)], got, tau).mean() for _ in range(_PERMUTATIONS)])
    )
    kappa = (share - chance) / (1 - chance) if chance < 1 else 0.0

    if share >= AGREE_SHARE and kappa >= AGREE_KAPPA:
        verdict, reason = AGREES, None
    elif share >= MOSTLY_SHARE and kappa >= MOSTLY_KAPPA:
        other = "another definition (a cut-off)?" if textual else "another unit?"
        verdict, reason = MOSTLY, f"{n - agreeing:,} rows disagree: entry errors, or {other}"
    else:
        verdict, reason = DISAGREES, None
    # agreeing overall, but not for one level of a text or code variable: the formula is wrong for it
    level_reason = None if verdict == DISAGREES else _level_disagreeing(rows, inputs, agree)
    if level_reason:
        verdict, reason = DISAGREES, level_reason
    hint = None if verdict == AGREES or textual else _unit_hint(want[~agree], got[~agree])
    if not textual and _multiple_of_one(definition, rows, inputs, got):
        warnings.append(
            f"the formula is a multiple of {inputs[0]}: a copy or a unit conversion agrees by construction"
        )
    return DerivedCheck(
        definition,
        verdict,
        agreeing=agreeing,
        tolerance=None if tau is None else float(np.median(tau)),
        strict=strict,
        tolerance_high=None if tau is None else float(np.quantile(tau, 0.95)),
        reason=hint or reason,
        unit_hint=hint,
        warnings=tuple(warnings),
        kappa=float(kappa),
        disagreeing=rows.index[~agree],
        **base,
    )


def inverse_definition(definition: DerivedVariable, name: str) -> DerivedVariable:
    """The inverse of ``definition`` that recovers ``name``, as a definition of its own (checked like one)."""
    return DerivedVariable(name, definition.inverses[name], origin=definition.origin)


def check_inverses(
    definition: DerivedVariable, data: pd.DataFrame, *, imputed: Mapping[str, Any] | None = None
) -> dict[str, DerivedCheck]:
    """Each inverse of the definition checked as a definition of its own (its variable, its formula): the tolerance
    declared for the definition is in its variable's unit, not theirs."""
    return {
        name: check_derived(inverse_definition(definition, name), data, imputed=imputed)
        for name in definition.inverses
    }


# --- filling exactly -------------------------------------------------------------------------------------------------

# rounding the inputs may move an exact value by the variable's own precision, or 5% of its SD
EXACT_SHARE = 0.05
# an inverse can give a variable another formula needs: filled again until nothing more is
MAX_PASSES = 5
# a text or code variable with more checked values is a text the check cannot cover level by level (a free text, an
# identifier): a formula using it fills nothing, and its values never reach a trace
MAX_LEVELS = 20


@dataclass(frozen=True)
class ExactFill:
    """The missing values of one variable computed by a declared definition (its formula, or an inverse of it): one
    targeted imputation of the Enrich draft. ``target`` is the variable filled, the definition's own or a variable of
    its formula; ``decimals`` rounds the values as the variable is recorded (None: a text, or no precision found);
    ``levels`` limits the fill to the values a text or code variable took in the rows checked ({}: no limit)."""

    definition: DerivedVariable
    target: str
    formula: str
    filled: int
    exact: bool  # the formula's values count as observed
    decimals: int | None
    levels: Mapping[str, tuple[str, ...]]
    check: DerivedCheck  # the check that lets it fill
    single: int = 0  # of the values filled, those single-imputed: not exact, or computed from an imputed value

    @property
    def inverse(self) -> bool:
        return self.target != self.definition.target


@dataclass(frozen=True)
class FillPlan:
    """What filling exactly would do: the fills in order, why a declared definition fills nothing (by variable), the
    values each variable would still miss in the rows filled (by variable), and why nothing can be filled."""

    fills: tuple[ExactFill, ...] = ()
    unused: Mapping[str, str] = field(default_factory=dict)
    left: Mapping[str, int] = field(default_factory=dict)
    problem: str | None = None

    @property
    def filled(self) -> int:
        return sum(f.filled for f in self.fills)


def _decimals(recorded: pd.Series, unit: float) -> int | None:
    """The decimals a variable is written with: those of its recording unit (1 for 0.1, 0 for 1, -1 for 10); 0 for
    whole numbers (a score: its unit of 0 means no rounding slack, not full precision); None for a text or a variable
    recorded to full precision."""
    if unit:
        return int(-round(np.log10(unit)))
    if is_text_column(recorded) or pd.api.types.is_bool_dtype(recorded):
        return None
    values = pd.to_numeric(recorded, errors="coerce").dropna()
    return 0 if len(values) and bool((values == values.round()).all()) else None


def _exact(check: DerivedCheck, recorded: pd.Series, unit: float) -> bool:
    """
    The values count as observed when the data agree with the formula (an accepted definition only mostly agrees, or
    could not be checked: its values are estimates) and either the variable was computed from the recorded values (it
    agrees within its own rounding) or rounding its formula's variables moves it, in 95% of the rows, by at most the
    variable's own precision or 5% of its standard deviation: a height from a BMI to 0.1 and a weight to 0.1 kg moves
    by about 0.2 cm. A declared tolerance is never the variable's own rounding.
    """
    if check.verdict != AGREES:
        return False
    if check.tolerance is None or check.strict:  # a text, or within the variable's own rounding
        return True
    values = pd.to_numeric(recorded, errors="coerce").dropna()
    spread = float(values.std()) if len(values) > 1 else 0.0
    bound = check.tolerance_high if check.tolerance_high is not None else check.tolerance
    return bound - unit / 2 <= max(unit, EXACT_SHARE * spread)


def _unused(check: DerivedCheck) -> str | None:
    if not check.fills:
        return "a date is never filled"
    if check.usable:
        wide = [name for name, seen in check.checked_levels.items() if len(seen) > MAX_LEVELS]
        if wide:
            return f"{wide[0]} takes more than {MAX_LEVELS} values: a text the check cannot cover value by value"
        return None
    if check.verdict in (MOSTLY, TOO_FEW):
        return "the data do not fully agree with it, or cannot check it: accept it first (Use it all the same)"
    return "the data disagree with it" if check.verdict == DISAGREES else "it cannot be used"


def _fill(frame: pd.DataFrame, fill: ExactFill, rows: pd.Index | None) -> tuple[pd.DataFrame, int]:
    """One fill, as its step records it and the replay computes it (manage.reproduction_manager)."""
    from enrich.imputation_service import fill_with_formula

    return fill_with_formula(
        frame,
        fill.target,
        fill.formula,
        rows,
        complete_inputs=True,
        finite_only=True,
        decimals=fill.decimals,
        levels=fill.levels or None,
    )


def plan_exact_fills(
    definitions: Sequence[DerivedVariable],
    data: pd.DataFrame,
    rows: pd.Index | None = None,
    *,
    imputed: Mapping[str, Any] | None = None,
    single: Mapping[str, Any] | None = None,
    check: Callable[[DerivedVariable], DerivedCheck] | None = None,
    inverse_check: Callable[[DerivedVariable, str], DerivedCheck] | None = None,
) -> FillPlan:
    """
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
    """
    from enrich.imputation_service import formula_inputs

    check = check or (lambda d: check_derived(d, data, imputed=imputed))
    inverse_check = inverse_check or (
        lambda d, name: check_derived(inverse_definition(d, name), data, imputed=imputed)
    )
    definitions = list(definitions)
    order = fill_order(definitions, {d.target: read_formula(d.formula).names for d in definitions})
    if order is None:
        return FillPlan(
            problem="Two definitions compute each other: declare one as the other's inverse."
        )
    by_target = {d.target: d for d in definitions}
    sources: list[tuple[DerivedVariable, str, str, DerivedCheck]] = []
    unused: dict[str, str] = {}
    for target in order:
        definition = by_target[target]
        found = check(definition)
        reason = _unused(found)
        if reason:
            unused[target] = reason
            continue
        sources.append((definition, target, definition.formula, found))
        for name, formula in definition.inverses.items():
            inverse = inverse_check(definition, name)
            if inverse.verdict == AGREES and not _unused(inverse):
                sources.append((definition, name, formula, inverse))
    # how each variable is written, from its recorded values only (an imputed value is no record); whether a formula's
    # values count as observed, from its own check
    recorded: dict[str, tuple[pd.Series, float]] = {}
    for _, name, _, _ in sources:
        if name not in recorded and name in data.columns:
            values = data.loc[~_imputed_rows(data.index, imputed, [name]), name].dropna()
            recorded[name] = (values, 0.0 if is_text_column(data[name]) else resolution(values))
    exact = [name in recorded and _exact(found, *recorded[name]) for _, name, _, found in sources]
    first = {name: i for i, (_, name, _, _) in reversed(list(enumerate(sources)))}
    ranked = sorted(range(len(sources)), key=lambda i: (first[sources[i][1]], not exact[i], i))

    frame = data
    scope = data.index if rows is None else data.index.intersection(rows)
    estimated = {str(c): set(np.asarray(ids).tolist()) for c, ids in (single or {}).items()}
    fills: list[ExactFill] = []
    for _ in range(MAX_PASSES):
        before = len(fills)
        for i in ranked:
            definition, name, formula, found = sources[i]
            if name not in recorded or not frame.loc[scope, name].isna().any():
                continue
            # limited to the values a text or code variable took in the rows checked, when it takes another here
            missing = frame.loc[scope, name].isna()
            levels = {
                variable: tuple(sorted(seen))
                for variable, seen in found.checked_levels.items()
                if variable in frame.columns
                and (
                    missing
                    & frame.loc[scope, variable].notna()
                    & ~frame.loc[scope, variable].astype(str).isin(seen)
                ).any()
            }
            values, unit = recorded[name]
            fill = ExactFill(
                definition, name, formula, 0, exact[i], _decimals(values, unit), levels, found
            )
            filled_frame, n = _fill(frame, fill, rows)
            if not n:
                continue
            # single-imputed: every value of a formula that is not exact, and those computed from an estimate
            new = frame.index[frame[name].isna() & filled_frame[name].notna()]
            inputs = formula_inputs(formula, name, frame)
            of_estimates = {
                row for row in new.tolist() if any(row in estimated.get(c, ()) for c in inputs)
            }
            singles = set(new.tolist()) if not exact[i] else of_estimates
            estimated.setdefault(name, set()).update(singles)
            fills.append(replace(fill, filled=n, single=len(singles)))
            frame = filled_frame
        if len(fills) == before:
            break
    left = {name: int(frame.loc[scope, name].isna().sum()) for name in recorded}
    return FillPlan(tuple(fills), unused, {name: n for name, n in left.items() if n})


def fill_steps(
    plan: FillPlan, data: pd.DataFrame, rows: pd.Index | None, run: str
) -> list[tuple[pd.DataFrame, dict[str, Any], str]]:
    """
    The plan as steps of the Enrich draft, one targeted imputation per fill: the table once filled (computed again
    here by the call the replay makes), the step's parameters and its description. The parameters hold what the replay
    repeats rather than decides again (the formula computed, whether the values count as observed, their rounding,
    the levels a fill is limited to) and, as a record only, the definition used (core.derived.record): the replay never
    reads the declared list.
    """
    from core.derived import record
    from enrich.imputation_service import describe_formula_fill, formula_inputs, left_missing

    steps = []
    before = data
    for position, fill in enumerate(plan.fills, start=1):
        after, filled = _fill(before, fill, rows)
        params: dict[str, Any] = {
            "target_variable": fill.target,
            "formula": fill.formula,
            "filled_count": filled,
            "exact": fill.exact,
            "row_wise": True,
            "complete_inputs": True,
            "finite_only": True,
            "inputs": formula_inputs(fill.formula, fill.target, before),
        }
        if fill.decimals is not None:
            params["round_decimals"] = fill.decimals
        if fill.levels:
            params["only_levels"] = {name: list(values) for name, values in fill.levels.items()}
        run_info = {"id": run, "fill": position, "of": len(plan.fills)}
        params["derived"] = record(fill.definition, [fill.target], fill.check.summary(), run_info)
        description = describe_formula_fill(
            fill.target,
            filled,
            exact=fill.exact,
            n_rows=None if rows is None else len(before.index.intersection(rows)),
            left=left_missing(before, fill.target, fill.formula, rows),
            derived_of=fill.definition.target,
        )
        steps.append((after, params, description))
        before = after
    return steps


__all__ = [
    "AGREES",
    "CANNOT",
    "DISAGREES",
    "EXACT_SHARE",
    "MAX_LEVELS",
    "LABELS",
    "MIN_ROWS",
    "MOSTLY",
    "TOO_FEW",
    "DerivedCheck",
    "ExactFill",
    "FillPlan",
    "check_derived",
    "check_inverses",
    "fill_steps",
    "inverse_definition",
    "plan_exact_fills",
    "resolution",
]
