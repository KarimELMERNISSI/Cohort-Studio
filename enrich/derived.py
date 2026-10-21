"""
The check of a declared derived variable against the data (core.derived declares it). On the rows where the variable
and every variable of its formula have a value, the formula is computed here, by the evaluator of every formula, and
compared with the recorded value. Nothing is estimated and nothing leaves the machine; the disagreeing rows are kept
for the page to show, never written anywhere.

A definition the data disagree with, or one that cannot be checked, is never used to fill or impute unless the user
accepts it; agreeing with the data says nothing of the published reference, which a definition's note names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.derived import DerivedVariable
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
    strict: bool = True  # the tolerance of the variable's own rounding; else with its formula's variables' rounding
    reason: str | None = None  # why it disagrees, or cannot be used or checked
    unit_hint: str | None = None  # the disagreeing rows are another unit by a constant factor
    warnings: tuple[str, ...] = ()  # what the check could not test (a variable taking one value)
    fills: bool = True  # False for a date: dates are never imputed
    kappa: float | None = None
    disagreeing: pd.Index = field(default_factory=lambda: pd.Index([]), repr=False, compare=False)
    # missing values of the variable whose formula has every variable recorded, in a checked level
    fillable_rows: pd.Index = field(default_factory=lambda: pd.Index([]), repr=False, compare=False)

    @property
    def label(self) -> str:
        return LABELS[self.verdict]

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
    """Per row: half the variable's recording unit, plus what rounding each input to its own unit moves the formula
    (a BMI to 0.1 from a weight to 0.1 kg and a height to 1 cm: about 0.21)."""
    tau = np.full(len(values), resolution(rows[definition.target]) / 2)
    for name in inputs:
        unit = resolution(rows[name])
        if not unit or is_text_column(rows[name]):
            continue
        shifted = rows.assign(**{name: pd.to_numeric(rows[name]) + unit / 2})
        moved = _computed(definition.formula, shifted)
        if moved is not None:
            shift = np.abs(pd.to_numeric(moved, errors="coerce").to_numpy(dtype=float) - values)
            tau += np.nan_to_num(shift, nan=0.0)
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


def check_derived(definition: DerivedVariable, data: pd.DataFrame) -> DerivedCheck:
    """
    The definition checked against ``data`` (the Enrich data, every row: nothing is filled here).

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
    compared = recorded.notna() & present
    n = int(compared.sum())
    rows = data.loc[compared]
    fills = not _is_date(recorded)
    warnings: list[str] = [] if fills else [f"{target} is a date: it is checked, never filled"]
    # rows whose level of a text or code variable no compared row has: not checked, so not filled
    unchecked = pd.Series(False, index=data.index)
    for name in inputs:
        if _levels(data[name]):
            seen = set(rows[name].astype(str))
            unchecked |= data[name].notna() & ~data[name].astype(str).isin(seen)
    base = {
        "inputs": tuple(inputs),
        "compared": n,
        "fills": fills,
        "fillable_rows": data.index[recorded.isna() & present & ~unchecked],
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
        if definition.tolerance is not None:
            tau = np.full(n, float(definition.tolerance)) + _floor(got)
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
        verdict, reason = MOSTLY, f"{n - agreeing:,} rows disagree: entry errors, or another unit?"
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
        reason=hint or reason,
        unit_hint=hint,
        warnings=tuple(warnings),
        kappa=float(kappa),
        disagreeing=rows.index[~agree],
        **base,
    )


def check_inverses(definition: DerivedVariable, data: pd.DataFrame) -> dict[str, DerivedCheck]:
    """Each inverse of the definition checked as a definition of its own (its variable, its formula): the tolerance
    declared for the definition is in its variable's unit, not theirs."""
    return {
        name: check_derived(DerivedVariable(name, formula, origin=definition.origin), data)
        for name, formula in definition.inverses.items()
    }


__all__ = [
    "AGREES",
    "CANNOT",
    "DISAGREES",
    "LABELS",
    "MIN_ROWS",
    "MOSTLY",
    "TOO_FEW",
    "DerivedCheck",
    "check_derived",
    "check_inverses",
    "resolution",
]
