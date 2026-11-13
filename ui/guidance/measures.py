"""
Guidance for two tabs of Analyze > Hypothesis tests & epidemiology, "Diagnostic accuracy & agreement" and "Paired,
rates & stratified": the entries each tab offers (one analysis is open at a time), the words of their choices, which
columns each input offers, the value that reads as "present", and the plan shown before an analysis runs.

No Streamlit call: the pages render these, the tests check them. A plan says what Run will do and on how many rows,
then what was checked on the data (a direction the data contradict, two columns that do not hold the same values,
strata that add nothing, a reference that is the sample's own median). It advises, and never decides: what runs is
what the user chose. A plan without a step means that nothing can be computed yet, and says what is missing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from biostats.diagnostic import YOUDEN_MIN_CLASS, delong_auc
from biostats.reliability import by_value, category, category_order, number_of, rating
from biostats.survival import EVENT_WORDS
from core.registry import as_code
from ui.guidance.analysis import Advice
from ui.guidance.outcomes import ModelPlan, PlanStep, _spellings

# ----------------------------------------------------------------------------- the entries of the two tabs


@dataclass(frozen=True)
class Entry:
    id: str  # kept in the page's state and in the links that open it
    label: str
    gives: str  # what it answers, in one line: plain words first, the method second
    terms: tuple[str, ...]  # its words in the glossary (ui.guidance.glossary)


ACCURACY, MEASUREMENTS, RATERS = "accuracy", "measurements", "raters"
PAIRED, ONE_GROUP, STRATA, RATES = "paired", "one_group", "strata", "rates"

DIAGNOSTIC_ENTRIES: tuple[Entry, ...] = (
    Entry(
        ACCURACY,
        "Accuracy of a test (ROC)",
        "A score or marker against a reference standard: sensitivity, specificity, predictive values, the ROC curve "
        "and its AUC.",
        (
            "Sensitivity",
            "Specificity",
            "Positive / negative predictive value",
            "Likelihood ratio",
            "ROC curve and AUC",
            "Youden index",
        ),
    ),
    Entry(
        MEASUREMENTS,
        "Agreement of two measurements",
        "Two methods, raters or occasions measuring the same quantity: Bland-Altman limits of agreement and the "
        "intraclass correlation (ICC).",
        ("Bland-Altman plot", "Limits of agreement", "Intraclass correlation (ICC)"),
    ),
    Entry(
        RATERS,
        "Agreement of two raters (kappa)",
        "Two raters sorting the same participants into the same categories: Cohen's kappa, weighted when the "
        "categories are ordered grades.",
        ("Cohen's kappa", "Weighted kappa"),
    ),
)
PAIRED_ENTRIES: tuple[Entry, ...] = (
    Entry(
        PAIRED,
        "Paired measurements",
        "The same participants measured twice: the change from the first measurement "
        "to the second, for a number or a yes/no characteristic.",
        ("Paired t-test", "Wilcoxon signed-rank test", "McNemar's test", "Effect size"),
    ),
    Entry(
        ONE_GROUP,
        "One group against a reference",
        "A mean or a share compared with a value fixed beforehand: one-sample t-test, Wilcoxon signed-rank, exact "
        "binomial test.",
        (
            "One-sample t-test",
            "Wilcoxon signed-rank test",
            "Exact binomial test",
            "Confidence interval (CI)",
        ),
    ),
    Entry(
        STRATA,
        "Stratified (Mantel-Haenszel)",
        "A yes/no outcome compared between two groups within the levels of a third "
        "variable: odds and risk ratios adjusted for it (Mantel-Haenszel).",
        (
            "Mantel-Haenszel",
            "Breslow-Day test",
            "Odds ratio (OR)",
            "Relative risk (RR)",
            "Confounder",
        ),
    ),
    Entry(
        RATES,
        "Rates (from counts)",
        "A calculator for numbers you type, not the dataset: events per person-time, "
        "the ratio of two rates, observed against expected events.",
        ("Incidence rate", "Incidence rate ratio (IRR)", "Standardised mortality ratio (SMR)"),
    ),
)
ENTRY = {entry.id: entry for entry in (*DIAGNOSTIC_ENTRIES, *PAIRED_ENTRIES)}

# the second choice of an entry: ids kept in the page's state and in the links, with their labels
NUMBER, YES_NO = "number", "yes_no"
MEASURES = {NUMBER: "A number (continuous)", YES_NO: "Yes or no (binary)"}
MEAN, SHARE = "mean", "share"
COMPARED = {MEAN: "A mean (a measurement)", SHARE: "A share (a yes/no variable)"}
DATASET, TYPED = "dataset", "typed"
SOURCES = {DATASET: "The dataset", TYPED: "Numbers I type"}
ONE_RATE, TWO_RATES, EXPECTED = "one", "two", "expected"
CALCULATIONS = {
    ONE_RATE: "One rate",
    TWO_RATES: "Two rates and their ratio",
    EXPECTED: "Observed against expected (SMR, SIR)",
}
# values stored with results since before the entries: kept, their labels changed
NO_CUTOFF, FIXED, YOUDEN = "No cut-off", "Manual", "Youden-optimal (data-driven)"
CUTOFFS = {
    NO_CUTOFF: "None: curve and AUC only",
    FIXED: "A value fixed beforehand",
    YOUDEN: "Found on these data (Youden's J, exploratory)",
}
HIGHER, LOWER = "higher", "lower"
T_TEST, SIGNED_RANK = "Paired t-test", "Wilcoxon signed-rank"
CHANGES = {
    T_TEST: "Mean change (paired t-test)",
    SIGNED_RANK: "Typical change (Wilcoxon signed-rank)",
}
ONE_T = "One-sample t-test"
SUMMARIES = {
    ONE_T: "Mean (one-sample t-test)",
    SIGNED_RANK: "Typical value (Wilcoxon signed-rank)",
}
ABSOLUTE, CONSISTENCY = "ICC(2,1)", "ICC(3,1)"
ICC_FORMS = {
    ABSOLUTE: "The same value (absolute agreement, ICC(2,1))",
    CONSISTENCY: "The same ranking: a constant shift between raters does not count (consistency, ICC(3,1))",
}
ICC_WORDS = {
    ABSOLUTE: "the same value counting as agreement",
    CONSISTENCY: "the same ranking counting as agreement",
}
BLAND_ALTMAN, ICC = "bland_altman", "icc"
AGREEMENT = {BLAND_ALTMAN: "Bland-Altman limits", ICC: "ICC"}
NAMES, GRADES = "names", "grades"
SCALES = {NAMES: "Names without an order", GRADES: "Ordered grades"}
LINEAR, QUADRATIC = "linear", "quadratic"
NEAR_MISSES = {
    LINEAR: "by its distance (linear weights)",
    QUADRATIC: "by the square of its distance (quadratic weights)",
}
DEATHS, CASES = "deaths", "cases"
EVENT_KINDS = {DEATHS: "deaths (SMR)", CASES: "new cases (SIR)"}
RATIO_NAMES = {
    DEATHS: "Standardised mortality ratio (SMR)",
    CASES: "Standardised incidence ratio (SIR)",
}
PERSON_TIME = ("person-years", "person-months", "person-days")

HELP: dict[str, str] = {
    "question": "What you want to know. One analysis is shown at a time; each keeps its choices and its result while "
    "you open another.",
    "standard": "The column that says who has the condition and who does not, by the best method available. Two "
    "values.",
    "present": "The value of the reference standard that means the condition is present.",
    "index": "The measurement whose accuracy is assessed: a score or a marker, one number per participant.",
    "direction": "Which way the test points is known before the data: a marker that rises with the condition, or "
    "one that falls with it (an ejection fraction, a filtration rate). The medians beside it show what these data "
    "say.",
    "cutoff": "A cut-off found on the same data makes sensitivity and specificity look better than they are in new "
    "participants. Use a value fixed beforehand (a protocol, a guideline, the maker's threshold) when there is one.",
    "cutoff_value": "The threshold fixed beforehand, in the unit of the test.",
    "prevalence": "Predictive values depend on how common the condition is. Leave empty to read them at the share "
    "in these data; give another value for another setting (0.02 for 2%).",
    "measurements": "One column per method, rater or occasion, each measuring the same quantity on the same "
    "participants, in the same unit. Two columns give the Bland-Altman limits and the ICC; three or more, the ICC.",
    "analyses": "Bland-Altman says by how much two measurements differ, in their unit; the ICC, how well repeated "
    "measurements tell participants apart, from 0 to 1.",
    "icc_form": "Absolute agreement counts a constant shift between raters as disagreement; consistency does not. "
    "Choose by the use: if the measurements will be read as values, the same value.",
    "delta": "The largest difference between the two measurements that would not matter in use, in their unit, "
    "decided before looking. The result then says what share of the differences stays within it.",
    "rater_a": "The first rater's category for each participant.",
    "rater_b": "The second rater's category for the same participants, from the same categories.",
    "scale": "Names (blood groups, diagnoses) have no order: every disagreement counts the same. For ordered grades a "
    "difference of one grade counts less than a difference of two.",
    "order": "The grades from the lowest to the highest. The order belongs to the scale: put it right before reading "
    "the result. Remove a grade and add it again to move it to the end.",
    "near_miss": "Linear weights count a disagreement in proportion to its distance in grades; quadratic weights "
    "count distant disagreements more heavily.",
    "measure": "What was measured twice: a number (a blood pressure, a score) or a yes/no characteristic.",
    "first": "The value of each participant at the first visit or under the first condition. For matched pairs, the "
    "two members of each pair, one column each.",
    "second": "The value of the same participants at the second visit or under the second condition.",
    "change": "The mean change is the usual summary. The signed-rank summary is the typical change (pseudo-median): "
    "robust to outliers, and safer with few pairs or skewed changes.",
    "counted": "The value whose share is compared between the two measurements.",
    "compared": "What is compared with the reference: the mean of a measurement, or the share of participants with a "
    "characteristic.",
    "measurement": "The variable compared with the reference value.",
    "reference_value": "A value decided before looking at these data: a published mean, a target, a limit. Not this "
    "sample's own mean or median.",
    "summary": "The mean is the usual summary. The signed-rank summary is the typical value (pseudo-median): robust "
    "to outliers, and safer with few or skewed values.",
    "source": "Counts read from a yes/no column of the dataset, or counts you type (from a published table).",
    "share_variable": "A yes/no variable of the dataset.",
    "share_level": "The value whose share is compared with the reference.",
    "with_characteristic": "How many of the participants counted have the characteristic.",
    "participants": "The number of participants the share is computed on.",
    "reference_share": "The share expected from a published source or a target, between 0 and 1 (0.20 for 20%).",
    "outcome": "A yes/no outcome.",
    "event": "The value of the outcome counted as the event.",
    "compare": "The exposure or group compared, in two values.",
    "reference_group": "Ratios compare the other group with this one.",
    "adjust": "The analysis is done within each of its groups (strata), then combined. Choose a variable that comes "
    "before the exposure and may distort the comparison (site, sex, age group), not a consequence of the exposure. "
    "For several variables at once, use a logistic model (Survival & regression, Binary).",
    "calculate": "One rate, two rates and their ratio, or the events observed against those expected from reference "
    "rates.",
    "events": "The number of events in the group. Count each participant's first event.",
    "person_time": "The sum of every participant's time at risk in the group, in one unit.",
    "unit": "The unit the person-time is counted in. It is written beside every rate.",
    "per": "Rates are shown per this much person-time, for example per 1,000 person-years.",
    "observed": "The events observed in the cohort.",
    "expected": "The number expected if the reference rates applied: for each stratum (age band, sex, period), the "
    "reference rate times the cohort's person-time, summed.",
    "event_kind": "It names the ratio: an SMR when the events are deaths, an SIR for new cases.",
}

# ----------------------------------------------------------------------------- what the inputs offer

PRESENT_WORDS = frozenset(
    {*EVENT_WORDS, "positive", "present", "case", "abnormal", "detected", "pos", "+"}
)
MAX_CATEGORIES = 10  # a rater's column
MAX_STRATA = 8
LEVEL_MIN = 10  # participants below which a share rests on few
SMALL_STRATUM = 5
LEFT_OUT = 0.20  # the share of rows left out from which the plan cautions
FEW_PAIRS, FEW_VALUES = 30, 15
SKEWED = 1.0
UNRELATED = 0.3


def measurements(df: pd.DataFrame) -> list[str]:
    """
    The columns that can be a measurement, a score or an index test: numbers (not True/False) that hold three values
    or more. A whole-number score of a few values is one (a pain score from 1 to 5: the signed-rank test's textbook
    use), where the page's other lists read a number of ten values or fewer as categories.
    """
    found = []
    for column in df.columns:
        values = df[column]
        if pd.api.types.is_bool_dtype(values.dtype) or not pd.api.types.is_numeric_dtype(
            values.dtype
        ):
            continue
        if int(values.nunique(dropna=True)) >= 3:
            found.append(str(column))
    return found


def two_valued(df: pd.DataFrame) -> list[str]:
    """The columns that hold exactly two values: a reference standard, a yes/no characteristic, an exposure."""
    return [str(c) for c in df.columns if int(df[c].nunique(dropna=True)) == 2]


def rated(df: pd.DataFrame) -> list[str]:
    """The columns a rater's categories can be: 2 to 10 values."""
    return [str(c) for c in df.columns if 2 <= int(df[c].nunique(dropna=True)) <= MAX_CATEGORIES]


def strata(df: pd.DataFrame) -> list[str]:
    """
    The columns an association can be adjusted for: 2 to 8 groups. A text column's values are counted with spaces and
    capitals set aside, as Data > Clean merges its spellings: a site written seven ways for four sites is offered, and
    its plan then offers to clean it.
    """
    found = []
    for column in df.columns:
        values = df[column].dropna()
        n = int(values.nunique())
        if n > MAX_STRATA and not pd.api.types.is_numeric_dtype(values.dtype):
            n = len({" ".join(str(v).split()).casefold() for v in values.unique()})
        if 2 <= n <= MAX_STRATA:
            found.append(str(column))
    return found


def look_alikes(first: str | None, candidates: Sequence[str]) -> list[str]:
    """
    ``candidates`` without ``first``, those whose names resemble it at the top (the same first word, as
    ``sbp_baseline`` and ``sbp_6m``, then the longest common start): the second measurement is usually named after the
    first. A hint in the order of the list, never a selection.
    """
    rest = [c for c in candidates if c != first]
    if not first:
        return rest
    stem = str(first).replace("-", "_").split("_")[0].casefold()

    def close(name: str) -> tuple[int, int]:
        text, own = name.casefold(), str(first).casefold()
        common = next(
            (i for i, (a, b) in enumerate(zip(text, own, strict=False)) if a != b),
            min(len(text), len(own)),
        )
        same_word = bool(stem) and text.replace("-", "_").split("_")[0] == stem
        return (0 if same_word else 1, -common if common >= 3 else 0)

    return sorted(rest, key=close)  # stable: the file's order within each rank


def levels_of(values: pd.Series) -> list[Any]:
    """A column's values as the lists show them: numbers by value, anything else by name."""
    found = list(pd.unique(values.dropna()))
    try:
        return sorted(found)
    except TypeError:
        return sorted(found, key=str)


def present_value(levels: Sequence[Any]) -> Any | None:
    """
    The value that reads as "present", "event" or "yes" (Yes, 1, True, Positive, Present, Case, Abnormal, Detected,
    Died...), or None when none or several do: the last in sorted order took "Control" for Case / Control, "Normal"
    for Abnormal / Normal and "Survived" for Died / Survived.
    """
    reads = [level for level in levels if category(level).strip().casefold() in PRESENT_WORDS]
    if len(reads) != 1:
        return None
    # 1 reads as "present" beside 0 or False only: beside 2 the coding is unknown (1 = censored and 2 = event in
    # R's survival package), and the survival page refuses it rather than guess
    others = {category(level).strip().casefold() for level in levels if level is not reads[0]}
    if category(reads[0]).strip() == "1" and not others <= {"0", "false"}:
        return None
    return reads[0]


def group_medians(
    df: pd.DataFrame, outcome: str, present: Any, score: str
) -> tuple[float, float, int, int] | None:
    """The median of the test with the condition and without it, and the participants in each group."""
    frame = pd.DataFrame(
        {"y": df[outcome], "x": pd.to_numeric(df[score], errors="coerce")}
    ).dropna()
    has = frame["y"].map(category) == category(present)
    if not bool(has.any()) or bool(has.all()):
        return None
    return (
        float(frame.loc[has, "x"].median()),
        float(frame.loc[~has, "x"].median()),
        int(has.sum()),
        int((~has).sum()),
    )


def _value(value: Any) -> str:
    """A value of the data in a sentence, as code, a hidden space shown as the lists show it ("Yes␣"): written
    as code alone, "Yes" and "Yes " read the same in the sentence that refused them for being two values."""
    from ui.components.levels import show_level  # the lists' own reading of a level

    return as_code(show_level(value))


@dataclass(frozen=True)
class Pooled:
    """What the imputed datasets give a pooled test of some columns: their rows, the rows one dataset gives the
    test (those that hold every value, observed or imputed), and the values imputed per column."""

    rows: int
    used: int
    imputed: Mapping[str, int]
    datasets: int


def pooled_rows(mi: Any, columns: Sequence[str]) -> Pooled:
    """The rows a test pooled over ``mi`` (biostats.mi.MIData) reads on ``columns``."""
    first = mi.complete(1, list(columns)).apply(pd.to_numeric, errors="coerce")
    return Pooled(
        rows=len(first),
        used=int(first.notna().all(axis=1).sum()),
        # the cells given a value: a column left out of the imputation model can be listed with none
        imputed={
            c: int(mi.imputations[c][1].notna().sum()) if c in mi.imputations else 0
            for c in columns
        },
        datasets=int(mi.m),
    )


def _imputed(df: pd.DataFrame, columns: Sequence[str], pooled: Pooled) -> list[Advice]:
    """Under a pooled test, in place of "rows left out": the values imputed, and a column that was not imputed
    (the plan counted the complete rows and said the others were left out, over a test that reads them all)."""
    said: list[Advice] = []
    filled = [
        f"{n:,} value{'s' if n != 1 else ''} of {as_code(c)}"
        for c, n in pooled.imputed.items()
        if n
    ]
    if filled:
        one = len(filled) == 1 and filled[0].startswith("1 value ")
        said.append(
            Advice(
                "info",
                f"{' and '.join(filled)} {'is' if one else 'are'} imputed: each imputed dataset gives the test "
                f"{pooled.used:,} of the {pooled.rows:,} rows.",
            )
        )
    lacking = [
        c
        for c in dict.fromkeys(columns)
        if not pooled.imputed.get(c) and int(pd.to_numeric(df[c], errors="coerce").isna().sum())
    ]
    if lacking and pooled.used < pooled.rows:
        share = (pooled.rows - pooled.used) / pooled.rows
        names = " and ".join(as_code(c) for c in lacking)
        said.append(
            Advice(
                "caution" if share >= LEFT_OUT else "info",
                f"{names} {'is' if len(lacking) == 1 else 'are'} not imputed: each imputed dataset reads the "
                f"{pooled.used:,} rows that have {'it' if len(lacking) == 1 else 'them'} "
                f"({pooled.rows - pooled.used:,} of {pooled.rows:,} are left out).",
            )
        )
    return said


def decimals(value: float, digits: int) -> int:
    """The decimals that give ``value`` ``digits`` significant figures."""
    if value == 0 or not math.isfinite(value):
        return 0
    return max(0, digits - 1 - math.floor(math.log10(abs(value))))


def number(value: Any, digits: int = 3) -> str:
    """
    A number as a plan, a sentence or a result writes it: ``digits`` significant figures for its decimals, "n/a"
    when it has no value. Thousands are grouped at any size, the whole part is never rounded (60,962.5 read
    "61,000", and two means 1,200 apart read "243,700 to 244,900" beside "a change of 1,230"), and no exponent is
    written (0.00001 read "0", 93,656 "+9.37e+04", and a plan said that a score "runs from 174 to 2.802e+04").
    One rule for the plans and for the results: a number is not written two ways on one screen.
    """
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    amount = float(value)
    text = f"{amount:,.{min(12, decimals(amount, digits))}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _num(value: float) -> str:
    """A number of the data in a plan: four significant figures for its decimals."""
    return number(value, 4)


def _left_out(df: pd.DataFrame, columns: Sequence[str], used: int, more: str = "") -> list[Advice]:
    """The rows left out for a missing value, per column; a caution from a fifth of the rows."""
    rows = len(df)
    if used >= rows:
        return []
    lacking = [
        f"{int(df[c].isna().sum()):,} lack {as_code(c)}"
        for c in dict.fromkeys(columns)
        if int(df[c].isna().sum())
    ]
    share = (rows - used) / rows if rows else 0.0
    text = f"{rows - used:,} of {rows:,} rows are left out ({share:.0%})"
    text += f": {'; '.join(lacking)}." if lacking else "."
    return [Advice("caution" if share >= LEFT_OUT else "info", f"{text} {more}".strip())]


def _idle(rows: int, text: str) -> ModelPlan:
    """Nothing to run yet, and what is missing."""
    return ModelPlan(rows, (), (Advice("info", text),))


def _refused(rows: int, *advice: Advice) -> ModelPlan:
    """Nothing can be computed on these choices, and why."""
    return ModelPlan(rows, (), tuple(advice))


def _skew(values: np.ndarray) -> float:
    return float(stats.skew(values)) if len(values) > 2 and np.ptp(values) > 0 else 0.0


def _same_quantity(a: pd.Series, b: pd.Series) -> list[Advice]:
    """Two columns that are hardly related, or several-fold apart: two measurements of one quantity?"""
    found = []
    if len(a) > 2 and float(a.std()) > 0 and float(b.std()) > 0:
        r = float(np.corrcoef(a, b)[0, 1])
        if r < UNRELATED:
            found.append(
                Advice(
                    "caution",
                    f"{as_code(a.name)} and {as_code(b.name)} are hardly related (r = {r:.2f}): two "
                    "measurements of one quantity usually are. Check the columns.",
                )
            )
    ma, mb = float(a.mean()), float(b.mean())
    if ma > 0 and mb > 0 and not 0.5 <= ma / mb <= 2:
        found.append(
            Advice(
                "caution",
                f"Their means are several-fold apart ({_num(ma)} and {_num(mb)}): the same quantity, in the "
                "same unit?",
            )
        )
    return found


# ----------------------------------------------------------------------------- accuracy of a test


def plan_accuracy(
    df: pd.DataFrame,
    outcome: str | None,
    present: Any | None,
    score: str | None,
    direction: str | None,
    mode: str,
    cutoff: float | None,
    prevalence: float | None,
) -> ModelPlan:
    rows = len(df)
    if not outcome or not score:
        return _idle(rows, "Choose the reference standard and the test.")
    frame = pd.DataFrame(
        {"y": df[outcome], "x": pd.to_numeric(df[score], errors="coerce")}
    ).dropna()
    if frame.empty:
        return _refused(
            rows,
            Advice(
                "violated",
                f"No row holds both {as_code(outcome)} and a number in {as_code(score)}.",
            ),
        )
    held = levels_of(frame["y"])
    if len(held) != 2:
        said = Advice(
            "violated",
            f"{as_code(outcome)} holds {len(held)} value{'s' if len(held) != 1 else ''} in the rows where the test "
            "is recorded: a reference standard holds two (the condition present, or absent).",
        )
        spelt = _spellings(outcome, df[outcome])
        return _refused(rows, said, *([spelt] if spelt else []))
    if present is None or category(present) not in {category(v) for v in held}:
        return _idle(rows, "Choose the value that means the condition is present.")
    medians = group_medians(df, outcome, present, score)
    assert medians is not None  # two values, one of them chosen
    with_it, without, n_pos, n_neg = medians
    if direction not in (HIGHER, LOWER):
        return _idle(
            rows,
            "Say which way the test points: is the condition more likely when the test is higher, or lower? In "
            f"these data its median is {_num(with_it)} with the condition and {_num(without)} without.",
        )
    if mode == FIXED and cutoff is None:
        return _idle(rows, "Type the cut-off fixed beforehand, or choose another cut-off.")
    lower = direction == LOWER
    counts = f"{n_pos:,} with the condition, {n_neg:,} without"
    steps = [
        PlanStep(
            f"The accuracy of {as_code(score)} against {as_code(outcome)} = {as_code(present)}, {direction} values "
            f"pointing to the condition ({counts})",
            len(frame),
        )
    ]
    bound = "at most" if lower else "at least"
    if mode == NO_CUTOFF:
        steps.append(
            PlanStep(
                "The area under the ROC curve with its 95% interval, and the curve. No cut-off: choose one to read "
                "sensitivity and specificity",
                len(frame),
            )
        )
    elif mode == FIXED:
        steps.append(
            PlanStep(
                f"Sensitivity, specificity and predictive values with the test positive when it is {bound} "
                f"{_num(float(cutoff))}, a value fixed beforehand; the ROC curve and its AUC",  # type: ignore[arg-type]
                len(frame),
            )
        )
    else:
        steps.append(
            PlanStep(
                "Sensitivity, specificity and predictive values at the cut-off found on these data (the largest "
                "sensitivity + specificity - 1), with the range of the cut-offs found in bootstrap resamples; the "
                "ROC curve and its AUC",
                len(frame),
            )
        )
    advice: list[Advice] = []
    oriented = -frame["x"] if lower else frame["x"]
    labels = [category(v) == category(present) for v in frame["y"]]
    auc = delong_auc(labels, oriented.tolist(), pos_label=True)
    if auc["ci"].upper < 0.5:
        other = "lower" if not lower else "higher"
        advice.append(
            Advice(
                "caution",
                f"You said {direction} values point to the condition; here they are {other} in participants with it "
                f"(median {_num(with_it)} against {_num(without)}, AUC {auc['auc']:.2f}). Check the direction, or "
                "the value that means the condition is present.",
            )
        )
    elif auc["ci"].lower <= 0.5:
        # neither "fits" nor "contradicts": the same data read "the direction fits" for both directions
        advice.append(
            Advice(
                "info",
                f"These data do not show which way the test points (AUC {auc['auc']:.2f}, 95% CI "
                f"{auc['ci'].lower:.2f} to {auc['ci'].upper:.2f}; median {_num(with_it)} with the condition, "
                f"{_num(without)} without): it hardly separates the two groups.",
            )
        )
    if mode != NO_CUTOFF:
        for n, what in ((n_pos, "sensitivity"), (n_neg, "specificity")):
            if n < LEVEL_MIN:
                advice.append(
                    Advice(
                        "caution",
                        f"{what.capitalize()} rests on {n:,} participant{'s' if n != 1 else ''}: its interval is "
                        "wide.",
                    )
                )
    if mode == FIXED and cutoff is not None:
        low, high = float(frame["x"].min()), float(frame["x"].max())
        nobody = cutoff < low if lower else cutoff > high
        everybody = cutoff >= high if lower else cutoff <= low
        if nobody or everybody:
            side = ("below" if cutoff < low else "above") if nobody else ""
            advice.append(
                Advice(
                    "violated",
                    (
                        f"{_num(float(cutoff))} is {side} every {as_code(score)} ({_num(low)} to {_num(high)}): no "
                        "one would be called positive."
                        if nobody
                        else f"With the test positive when it is {bound} {_num(float(cutoff))}, everyone is called "
                        f"positive ({as_code(score)} runs from {_num(low)} to {_num(high)})."
                    )
                    + " The right variable, the right unit?",
                )
            )
    if mode == YOUDEN:
        text = (
            "A cut-off found on these data: the sensitivity and specificity read at it are optimistic, and their "
            f"intervals do not know that it was chosen (here {min(n_pos, n_neg):,} participants in the smaller "
            "group). Report it as exploratory."
        )
        if min(n_pos, n_neg) < 50:
            text += " With fewer than 50 participants in a group the optimism is no longer small."
        advice.append(Advice("caution", text))
        if min(n_pos, n_neg) < YOUDEN_MIN_CLASS:
            advice.append(
                Advice(
                    "info",
                    f"Fewer than {YOUDEN_MIN_CLASS} participants in a group: no range is drawn for the cut-off.",
                )
            )
    if mode != NO_CUTOFF:
        share = n_pos / (n_pos + n_neg)
        text = (
            f"Predictive values are read at the share with the condition in these data ({share:.1%}). That is the "
            "prevalence only if participants were taken as they came; where those with and without the condition "
            "were sampled apart, give the prevalence."
        )
        if prevalence is not None:
            text = (
                f"Predictive values are read at the share with the condition in these data ({share:.1%}) and at "
                f"the prevalence typed ({prevalence:.1%})."
            )
        advice.append(Advice("info", text))
    advice += _left_out(df, [outcome, score], len(frame))
    return ModelPlan(rows, tuple(steps), tuple(advice))


# ----------------------------------------------------------------------------- agreement of two measurements


def plan_measurements(
    df: pd.DataFrame, columns: Sequence[str], analyses: Sequence[str], model: str
) -> ModelPlan:
    rows = len(df)
    columns = [c for c in dict.fromkeys(columns) if c in df.columns]
    if len(columns) < 2:
        return _idle(rows, "Choose two or more columns that measure the same quantity.")
    frame = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    used = len(frame)
    if used < 3:
        return _refused(
            rows,
            Advice(
                "violated",
                f"{used:,} row{'s' if used != 1 else ''} hold every value: agreement needs three at least.",
            ),
            *_left_out(df, columns, used),
        )
    if all(float(frame[c].max()) == float(frame[c].min()) for c in columns):
        return _refused(
            rows,
            Advice(
                "violated",
                "Each column holds one value only in these rows: there is no difference between participants "
                "for an agreement to be read on.",
            ),
        )
    pair = len(columns) == 2
    steps: list[PlanStep] = []
    advice: list[Advice] = []
    if BLAND_ALTMAN in analyses and pair:
        steps.append(
            PlanStep(
                f"Bland-Altman: the mean of {as_code(columns[0])} - {as_code(columns[1])} and the limits of "
                "agreement",
                used,
            )
        )
    elif BLAND_ALTMAN in analyses:
        advice.append(
            Advice(
                "info",
                f"Bland-Altman limits compare two columns: with {len(columns)}, the ICC alone is computed.",
            )
        )
    if ICC in analyses:
        steps.append(
            PlanStep(
                f"The ICC of the {len(columns)} columns, {ICC_WORDS.get(model, model)}, one measurement each "
                f"({model})",
                used,
            )
        )
    if not steps:
        return ModelPlan(
            rows,
            (),
            (*advice, Advice("info", "Choose an analysis: Bland-Altman limits, the ICC, or both.")),
        )
    first = frame[columns[0]]
    for other in columns[1:]:
        advice += _same_quantity(first, frame[other])
    if pair and BLAND_ALTMAN in analyses:
        if used < 50:
            # the half-width of a limit's 95% interval (Bland & Altman 1986): t x sqrt(3 / n) SD. "About 0.65 SD"
            # was printed for any number of pairs below 30: it is 0.65 at 30, 1.24 at 10, 2.15 at 5
            width = float(stats.t.ppf(0.975, used - 1)) * (3 / used) ** 0.5
            advice.append(
                Advice(
                    "caution" if used < FEW_PAIRS else "info",
                    f"{used:,} pairs: each limit of agreement is known to about {width:.2f} SD of the "
                    "differences either way"
                    + (": the limits are rough." if used < FEW_PAIRS else "."),
                )
            )
        differences = (frame[columns[0]] - frame[columns[1]]).to_numpy(dtype=float)
        if abs(_skew(differences)) > SKEWED:
            advice.append(
                Advice(
                    "info",
                    "The differences are skewed: the limits (mean difference +/- 1.96 SD) may not hold 95% of them. "
                    "Read the plot.",
                )
            )
    if ICC in analyses:
        advice.append(
            Advice(
                "info",
                "The ICC depends on how much the participants differ: the same raters score lower in a group that "
                "is alike.",
            )
        )
    advice += _left_out(df, columns, used)
    return ModelPlan(rows, tuple(steps), tuple(advice))


# ----------------------------------------------------------------------------- agreement of two raters


def rater_categories(
    df: pd.DataFrame, first: str, second: str
) -> tuple[list[str], dict[str, int], dict[str, int], int]:
    """The categories either rater used on the participants rated by both (in the order proposed), how many
    participants each rater put in each, and the participants rated by both."""
    both = df[first].notna() & df[second].notna()
    valued = by_value(df[first], df[second])  # as the table of the result reads them
    a = pd.Series([rating(v, valued=valued) for v in df.loc[both, first]]).value_counts().to_dict()
    b = pd.Series([rating(v, valued=valued) for v in df.loc[both, second]]).value_counts().to_dict()
    return category_order(list({*a, *b})), a, b, int(both.sum())


_ROMAN = re.compile(r"^(x{0,3})(ix|iv|v?i{0,3})$")  # I to XXXIX: the grades and stages of a scale


def _roman(name: str) -> int | None:
    text = name.strip().casefold()
    found = _ROMAN.match(text) if text else None
    if found is None:
        return None
    tens, units = found.groups()
    ones = {"ix": 9, "iv": 4}.get(units, 5 * units.count("v") + units.count("i"))
    return 10 * len(tens) + ones


def grade_order(categories: Sequence[str]) -> list[str]:
    """
    The order of the grades when the grades themselves say it: numbers by value, Roman numerals by value (IX after
    V). Any other grades give no order: "high, low, medium, none" is the alphabet's, and a weighted kappa computed
    on it is another number (0.63 for 0.76 on the tutorial's ratings). The order of a scale written in words is
    the one input of this analysis the software cannot know: its user places each grade.
    """
    try:
        return sorted(
            categories, key=lambda name: (number_of(name), name)
        )  # a grade "nan" is no number
    except ValueError:
        values = [_roman(name) for name in categories]
        if any(value is None for value in values):
            return []
        return [name for _, name in sorted(zip(values, categories, strict=True))]  # type: ignore[type-var]


def _writings(
    df: pd.DataFrame, first: str, second: str, a: Mapping[str, int], b: Mapping[str, int]
) -> list[Advice]:
    """
    What a plan says of ratings written as numbers. A column of numbers beside a column of texts: the texts are
    read as numbers, and the plan names the writings that were read as one ("2.0" as 2). Two columns of texts:
    they are compared as written, and the plan names the writings that read as the same number and are kept as
    two categories ("250.0" and "250.00", "01" and "1"): codes, unless their user says otherwise by cleaning
    the column.
    """
    if by_value(df[first], df[second]):
        both = df[first].notna() & df[second].notna()
        read: dict[str, str] = {}
        where = None
        for column in (first, second):
            for value in df.loc[both, column].unique():
                if isinstance(value, str) and category(value) != value:
                    read.setdefault(value, category(value))
                    where = where or column
        if not read:
            return []
        shown = ", ".join(
            f"{_value(text)} as {_value(number)}" for text, number in list(read.items())[:3]
        )
        return [
            Advice(
                "info",
                f"{as_code(where)} holds texts beside a column of numbers: its values are read as numbers "
                f"({shown}{', ...' if len(read) > 3 else ''}).",
            )
        ]
    numbers: dict[float, list[str]] = {}
    for name in {*a, *b}:
        try:
            numbers.setdefault(number_of(name), []).append(name)
        except ValueError:
            continue
    apart = [sorted(names) for names in numbers.values() if len(names) > 1]
    if not apart:
        return []
    shown = "; ".join(" and ".join(_value(n) for n in names) for names in sorted(apart)[:3])
    odd = {name for names in apart for name in names if category(name) != name}
    where = next((c for c, own in ((first, a), (second, b)) if odd & set(own)), first)
    return [
        Advice(
            "caution",
            f"{shown} read as the same number and count as categories of their own: two columns of texts are "
            "compared as written, since codes are not numbers. If they are one grade, clean the column into a "
            "copy beside it, which the analysis then uses.",
            column=where,
        )
    ]


def _look_ordered(categories: Sequence[str]) -> bool:
    return len(grade_order(categories)) == len(categories) > 0


def plan_raters(
    df: pd.DataFrame,
    first: str | None,
    second: str | None,
    scale: str,
    order: Sequence[str] | None,
    weights: str,
) -> ModelPlan:
    rows = len(df)
    if not first or not second:
        return _idle(rows, "Choose the two raters' columns.")
    categories, a, b, used = rater_categories(df, first, second)
    if used < 2:
        return _refused(
            rows, Advice("violated", "Fewer than two participants are rated by both raters.")
        )
    shared = [name for name in categories if name in a and name in b]
    advice: list[Advice] = []
    folded: dict[str, list[str]] = {}
    for name in categories:
        folded.setdefault(" ".join(name.split()).casefold(), []).append(name)
    alike = [names for names in folded.values() if len(names) > 1]
    spelt: list[Advice] = []
    if alike:
        # a hidden space is shown as the lists show it: as code alone, `Yes` and `Yes ` read the same here
        shown = "; ".join(", ".join(_value(n) for n in names) for names in alike[:3])
        odd = {
            n for names in alike for n in names if n != " ".join(n.split())
        }  # written with hidden spaces
        where = next((c for c, own in ((first, a), (second, b)) if odd & set(own)), None)
        spelt.append(
            Advice(
                "caution",
                f"Categories written alike but not identical ({shown}): each counts as a category of its own until "
                "the column is cleaned into a copy beside it, which the analysis then uses.",
                column=where,
            )
        )
    spelt += _writings(df, first, second, a, b)
    if not shared:
        return _refused(
            rows,
            Advice(
                "violated",
                f"{as_code(first)} and {as_code(second)} share no category "
                f"({', '.join(_value(n) for n in list(a)[:3])} against "
                f"{', '.join(_value(n) for n in list(b)[:3])}): kappa would be 0 by construction. The two columns "
                "are coded differently (capitals, spaces, other words).",
            ),
            *spelt,
        )
    if len(categories) < 2:
        return _refused(
            rows,
            Advice(
                "violated",
                f"Both raters put every participant in {_value(categories[0])}: there is nothing kappa can "
                "compare.",
            ),
        )
    advice += spelt
    for column, own in ((first, a), (second, b)):
        if len(own) == 1:
            advice.append(
                Advice(
                    "caution",
                    f"{as_code(column)} puts every participant in {_value(next(iter(own)))}: kappa is 0 by "
                    "construction, whatever the other rater says. Read the share of same categories.",
                )
            )
    for name in categories:
        for column, own, other in ((first, a, b), (second, b, a)):
            if name in own and name not in other:
                advice.append(
                    Advice(
                        "info",
                        f"{_value(name)} is given by {as_code(column)} only ({own[name]:,} participant"
                        f"{'s' if own[name] != 1 else ''}).",
                    )
                )
    if used < FEW_PAIRS:
        advice.append(
            Advice("caution", f"{used:,} participants rated by both: kappa's interval is wide.")
        )
    # the same category for both: 95 of 100 in X for one and in Y for the other is no agreement at all
    most_a, most_b = max(a, key=lambda name: a[name]), max(b, key=lambda name: b[name])
    if most_a == most_b and a[most_a] / used >= 0.9 and b[most_b] / used >= 0.9:
        advice.append(
            Advice(
                "info",
                "Both raters put 90% or more of the participants in the same category: kappa can be low although the "
                "raters agree on most participants. Read the share of same categories beside it.",
            )
        )
    listed = ", ".join(_value(name) for name in categories)
    graded = scale == GRADES and len(categories) > 2
    if graded:
        stated = [str(name) for name in order or ()]
        missing = [name for name in categories if name not in stated]
        if missing or len(set(stated)) != len(stated):
            return _refused(
                rows,
                Advice(
                    "violated",
                    "Put every grade in the order of the scale, lowest first: the order lacks "
                    f"{', '.join(_value(n) for n in missing[:6]) or 'none, but repeats one'}. Grades written in "
                    "words carry no order of their own.",
                ),
                *advice,
            )
        advice.append(
            Advice(
                "info",
                f"Order used: {' < '.join(_value(n) for n in stated)}. Correct it if it is not the order of the "
                "scale.",
            )
        )
        step = PlanStep(
            f"Weighted kappa ({weights} weights) of {as_code(first)} and {as_code(second)}, ordered grades "
            f"({len(categories)} categories)",
            used,
        )
    else:
        if len(categories) > 2 and _look_ordered(categories):
            advice.append(
                Advice(
                    "info",
                    "These categories look ordered: with Ordered grades, a difference of one grade counts less "
                    "than a difference of two.",
                )
            )
        step = PlanStep(
            f"Cohen's kappa of {as_code(first)} and {as_code(second)}, categories without an order "
            f"({len(categories)} categories: {listed})",
            used,
        )
    advice += _left_out(df, [first, second], used)
    return ModelPlan(rows, (step,), tuple(advice))


# ----------------------------------------------------------------------------- paired measurements

WITHIN_ONE_GROUP = (
    "A change within one group is not the effect of a treatment: time, regression to the mean and who stayed also "
    "move it. To compare groups, adjust for the first measurement (Analysis tab, Multivariate)."
)


def plan_paired(
    df: pd.DataFrame,
    before: str | None,
    after: str | None,
    kind: str,
    pooled: Pooled | None = None,
) -> ModelPlan:
    """``pooled``: the test is pooled over imputed datasets, and reads their rows (the mean change only)."""
    rows = len(df) if pooled is None else pooled.rows
    if not before or not after:
        return _idle(rows, "Choose the first and the second measurement.")
    if before == after:
        return _refused(
            rows, Advice("violated", "The first and the second measurement are the same column.")
        )
    frame = df[[before, after]].apply(pd.to_numeric, errors="coerce").dropna()
    used = len(frame) if pooled is None else pooled.used
    if used < 2:
        return _refused(
            rows,
            Advice("violated", "Fewer than two participants have both measurements."),
            *(
                _left_out(df, [before, after], used)
                if pooled is None
                else _imputed(df, [before, after], pooled)
            ),
        )
    differences = (frame[after] - frame[before]).to_numpy(dtype=float)
    advice: list[Advice] = []
    if kind == SIGNED_RANK:
        same = int((differences == 0).sum())
        text = (
            f"The typical change {as_code(after)} - {as_code(before)} (pseudo-median) with its 95% interval, "
            "Wilcoxon signed-rank test"
        )
        if same:
            advice.append(
                Advice(
                    "caution" if same / used >= LEFT_OUT else "info",
                    f"{same:,} pair{'s' if same != 1 else ''} do{'es' if same == 1 else ''} not differ: the test "
                    f"leaves them out, by its rule, and reads {used - same:,}.",
                )
            )
        if used - same < 6:
            advice.append(
                Advice("caution", "Fewer than 6 pairs differ: no 95% interval can be given.")
            )
    else:
        text = f"The mean change {as_code(after)} - {as_code(before)} with its 95% interval, paired t-test"
        if pooled is not None:
            text += f", pooled over the {pooled.datasets} imputed datasets"
        if used < FEW_VALUES and abs(_skew(differences)) > SKEWED:
            advice.append(
                Advice(
                    "caution",
                    "Few pairs and skewed changes: the signed-rank summary is the safer one.",
                )
            )
    advice += _same_quantity(frame[before], frame[after])
    if pooled is None:
        advice += _left_out(
            df,
            [before, after],
            used,
            "The change is that of the participants measured twice; the others may differ.",
        )
    else:
        advice += _imputed(df, [before, after], pooled)
    return ModelPlan(rows, (PlanStep(text, used),), tuple(advice))


def paired_values(df: pd.DataFrame, before: str, after: str) -> tuple[set[str], set[str], int]:
    """The values each of two yes/no columns holds in the rows where both are recorded, and those rows' number."""
    both = df[before].notna() & df[after].notna()
    return (
        {category(v) for v in df.loc[both, before]},
        {category(v) for v in df.loc[both, after]},
        int(both.sum()),
    )


def plan_paired_yes_no(
    df: pd.DataFrame, before: str | None, after: str | None, counted: Any | None
) -> ModelPlan:
    from biostats.paired import paired_table

    rows = len(df)
    if not before or not after:
        return _idle(rows, "Choose the first and the second measurement.")
    if before == after:
        return _refused(
            rows, Advice("violated", "The first and the second measurement are the same column.")
        )
    first, second, used = paired_values(df, before, after)
    if first != second:
        return _refused(
            rows,
            Advice(
                "violated",
                f"{as_code(before)} holds {' and '.join(_value(v) for v in sorted(first))}, {as_code(after)} holds "
                f"{' and '.join(_value(v) for v in sorted(second))}: the test needs the same two values in both.",
            ),
            # the column written with hidden spaces is cleaned from the plan (a copy beside it, used in its place)
            *[
                Advice(
                    "caution",
                    f"{as_code(column)} holds values written with hidden spaces: clean it into a copy beside it, "
                    "which the analysis then uses.",
                    column=str(column),
                )
                for column, held in ((before, first), (after, second))
                if any(value != " ".join(value.split()) for value in held)
            ],
        )
    if len(first) != 2:
        return _refused(
            rows,
            Advice(
                "violated",
                f"The two columns hold {len(first)} value{'s' if len(first) != 1 else ''} in the rows where both are "
                "recorded: the test needs two.",
            ),
        )
    if counted is None or category(counted) not in first:
        return _idle(rows, "Choose the value whose share is compared.")
    table = paired_table(df[before], df[after], counted=counted).to_numpy()
    yes = as_code(category(counted))
    other = as_code(next(v for v in category_order(list(first)) if v != category(counted)))
    gained, lost = int(table[0, 1]), int(table[1, 0])
    changed = gained + lost
    advice = [
        Advice(
            "info",
            f"{changed:,} participant{'s' if changed != 1 else ''} changed ({lost:,} from {yes} to {other}, "
            f"{gained:,} from {other} to {yes}); {used - changed:,} did not.",
        )
    ]
    if not changed:
        advice.append(Advice("info", "No participant changed: there is nothing to test."))
    elif changed < LEVEL_MIN:
        advice.append(
            Advice(
                "caution",
                "The test rests on the participants who changed: with fewer than 10 it can show little.",
            )
        )
    advice += _left_out(df, [before, after], used)
    step = PlanStep(
        f"The change in the share with {yes}, {as_code(after)} against {as_code(before)}, McNemar's exact test",
        used,
    )
    return ModelPlan(rows, (step,), tuple(advice))


# ----------------------------------------------------------------------------- one group against a reference


def _decimals(value: float) -> int:
    text = f"{value:.10f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


def plan_one_mean(
    df: pd.DataFrame,
    column: str | None,
    reference: float | None,
    kind: str,
    pooled: Pooled | None = None,
) -> ModelPlan:
    """``pooled``: the test is pooled over imputed datasets, and reads their rows (the mean only)."""
    rows = len(df) if pooled is None else pooled.rows
    if not column:
        return _idle(rows, "Choose the measurement.")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    used = len(values)
    if used < 2:
        return _refused(rows, Advice("violated", f"{as_code(column)} holds fewer than two values."))
    low, high = float(values.min()), float(values.max())
    if reference is None:
        return _idle(
            rows,
            "Type the reference value: a value fixed before looking at these data. "
            f"{as_code(column)} here: mean {_num(float(values.mean()))}, from {_num(low)} to {_num(high)}.",
        )
    reference = float(reference)
    advice: list[Advice] = []
    if not low <= reference <= high:
        advice.append(
            Advice(
                "caution",
                f"{_num(reference)} is {'below' if reference < low else 'above'} every {as_code(column)} "
                f"({_num(low)} to {_num(high)}): the right variable, the right unit?",
            )
        )
    array = values.to_numpy(dtype=float)
    # the data's own summary typed as a reference: the mean for the mean, the median for the typical value. The mean
    # is compared one decimal finer than the values are recorded (7 "was" the mean of values averaging 6.552; 50
    # "was" the mean of ages averaging 50.43, which a t-test tells apart from 50). A median of whole numbers is a
    # round number, and so is many a reference fixed beforehand: a question, not a verdict
    recorded = max((_decimals(float(value)) for value in array[:2000]), default=0)
    name, own, finer = (
        ("median", float(values.median()), recorded)
        if kind == SIGNED_RANK
        else ("mean", float(values.mean()), recorded + 1)
    )
    digits = min(6, max(_decimals(reference), finer))
    if round(own, digits) == round(reference, digits):
        advice.append(
            Advice(
                "caution",
                f"{_num(reference)} is also the {name} of these values: was it read from them? A reference comes "
                "from outside the data.",
            )
        )
    if kind == SIGNED_RANK:
        same = int((array == reference).sum())
        text = (
            f"Is the typical value (pseudo-median) of {as_code(column)} different from {_num(reference)}? "
            "Wilcoxon signed-rank test"
        )
        if same:
            advice.append(
                Advice(
                    "info",
                    f"{same:,} value{'s' if same != 1 else ''} equal to the reference: the test leaves them out, by "
                    f"its rule, and reads {used - same:,}.",
                )
            )
    else:
        text = (
            f"Is the mean of {as_code(column)} different from {_num(reference)}? One-sample t-test"
        )
        if pooled is not None:
            used = pooled.used
            text += f", pooled over the {pooled.datasets} imputed datasets"
        if used < FEW_VALUES and abs(_skew(array)) > SKEWED:
            advice.append(
                Advice(
                    "caution",
                    "Few values and a skewed distribution: the signed-rank summary is the safer one.",
                )
            )
    advice += _left_out(df, [column], used) if pooled is None else _imputed(df, [column], pooled)
    return ModelPlan(rows, (PlanStep(text, used),), tuple(advice))


def share_counts(df: pd.DataFrame, column: str, level: Any) -> tuple[int, int]:
    """The participants with ``level`` and the participants with a value, in a yes/no column."""
    held = df[column].dropna().map(category)
    return int((held == category(level)).sum()), len(held)


def check_share(
    with_it: int | None, counted: int | None, reference: float | None
) -> tuple[bool, list[Advice]]:
    """Whether a share can be compared with a reference, and what to know of the numbers: read or typed."""
    if with_it is None or counted is None or reference is None:
        return False, []
    if counted < 1 or with_it < 0 or with_it > counted:
        return False, [
            Advice(
                "violated",
                f"{with_it:,} with the characteristic among {counted:,} participants: the first cannot exceed the "
                "second.",
            )
        ]
    if not 0 < reference < 1:
        return False, [
            Advice(
                "violated",
                "A reference share of 0 or 1 cannot be tested: one participant with the characteristic (or "
                "without it) contradicts it. Give a share between the two (0.20 for 20%).",
            )
        ]
    advice = []
    if counted < LEVEL_MIN:
        advice.append(
            Advice(
                "info",
                f"{counted:,} participant{'s' if counted != 1 else ''}: the interval of the share is wide.",
            )
        )
    return True, advice


def plan_share(
    df: pd.DataFrame, column: str | None, level: Any | None, reference: float | None
) -> ModelPlan:
    rows = len(df)
    if not column:
        return _idle(rows, "Choose the yes/no variable.")
    held = {category(v) for v in df[column].dropna()}
    if len(held) != 2:
        said = Advice(
            "violated",
            f"{as_code(column)} holds {len(held)} value{'s' if len(held) != 1 else ''}: a share is that of one "
            "value among two.",
        )
        spelt = _spellings(column, df[column])
        return _refused(rows, said, *([spelt] if spelt else []))
    if level is None or category(level) not in held:
        return _idle(rows, "Choose the value whose share is compared.")
    if reference is None:
        return _idle(rows, "Type the reference share: a share fixed before looking at these data.")
    with_it, counted = share_counts(df, column, level)
    ready, advice = check_share(with_it, counted, reference)
    if not ready:
        return _refused(rows, *advice)
    advice += _left_out(df, [column], counted)
    step = PlanStep(
        f"Is the share with {as_code(column)} = {as_code(category(level))} ({with_it:,} of {counted:,}) different "
        f"from {reference:.1%}? Exact binomial test",
        counted,
    )
    return ModelPlan(rows, (step,), tuple(advice))


# ----------------------------------------------------------------------------- an association within strata


def plan_strata(
    df: pd.DataFrame,
    outcome: str | None,
    event: Any | None,
    compare: str | None,
    reference: Any | None,
    stratum: str | None,
) -> ModelPlan:
    rows = len(df)
    if not outcome or not compare or not stratum:
        # what is still to choose, by name: opened from the 2x2 panel, only the variable to adjust for is
        missing = [
            what
            for chosen, what in (
                (outcome, "the outcome"),
                (compare, "what is compared"),
                (stratum, "the variable to adjust for"),
            )
            if not chosen
        ]
        listed = " and ".join(filter(None, (", ".join(missing[:-1]), missing[-1])))
        return _idle(rows, f"Choose {listed}.")
    if len({outcome, compare, stratum}) < 3:
        return _refused(
            rows,
            Advice(
                "violated",
                "The outcome, the comparison and the variable adjusted for are three different columns.",
            ),
        )
    frame = df[[compare, outcome, stratum]].dropna()
    used = len(frame)
    refused: list[Advice] = []
    for column, what in ((outcome, "outcome"), (compare, "comparison")):
        held = int(frame[column].astype(str).nunique())
        if held != 2:
            refused.append(
                Advice(
                    "violated",
                    f"The {what} {as_code(column)} holds {held} value{'s' if held != 1 else ''} in the rows used: it "
                    "needs two.",
                )
            )
            spelt = _spellings(column, df[column])
            if spelt:
                refused.append(spelt)
    groups = frame[stratum].astype(str)
    if int(groups.nunique()) < 2:
        refused.append(
            Advice(
                "violated",
                f"{as_code(stratum)} holds one group in the rows used: adjusting needs two at least.",
            )
        )
    if refused:
        return _refused(rows, *refused)
    out_levels, cmp_levels = (set(frame[c].astype(str)) for c in (outcome, compare))
    if event is None or str(event) not in out_levels:
        return _idle(rows, "Choose the value of the outcome counted as the event.")
    if reference is None or str(reference) not in cmp_levels:
        return _idle(rows, "Choose the reference group.")
    exposed_level = next(level for level in sorted(cmp_levels) if level != str(reference))
    exposed = frame[compare].astype(str) == exposed_level
    happened = frame[outcome].astype(str) == str(event)
    advice: list[Advice] = []
    spelt = _spellings(stratum, df[stratum])
    if spelt:
        advice.append(spelt)
    empty = small = 0
    sizes: list[int] = []
    expected = lowest = highest = 0.0
    for _, index in groups.groupby(groups).groups.items():
        e, y = exposed.loc[index], happened.loc[index]
        n1, n0, m1, n = int(e.sum()), int((~e).sum()), int(y.sum()), len(index)
        sizes.append(n)
        if not n1 or not n0 or not m1 or m1 == n:
            empty += 1  # no odds ratio of its own: the two tests do not read it (biostats.rates)
            continue
        if min(n1, n0) < SMALL_STRATUM:
            small += 1
        expected += n1 * m1 / n
        lowest += max(0, m1 - n0)
        highest += min(n1, m1)
    total = len(sizes)
    if empty:
        advice.append(
            Advice(
                "caution",
                f"{empty} of {total} strata hold{'s' if empty == 1 else ''} one group only or one outcome only: "
                f"{'it adds' if empty == 1 else 'they add'} nothing to the odds ratio, and the two tests read the "
                f"other {total - empty}."
                + (
                    " With fewer than two such strata, whether the strata differ cannot be tested."
                    if total - empty < 2
                    else ""
                ),
            )
        )
    if small:
        advice.append(
            Advice(
                "info",
                f"{small} of {total} strata {'has' if small == 1 else 'have'} fewer than {SMALL_STRATUM} "
                "participants in a group.",
            )
        )
    if total > empty and min(expected - lowest, highest - expected) < 5:
        advice.append(
            Advice(
                "info",
                "Few events for the test's chi-square approximation (Mantel-Fleiss criterion below 5): read its "
                "p-value with care.",
            )
        )
    if max(sizes) / used >= 0.9:
        advice.append(
            Advice("info", "One stratum holds 90% or more of the rows: little is adjusted.")
        )
    advice += _left_out(df, [compare, outcome, stratum], used)
    step = PlanStep(
        f"Odds and risk ratios of {as_code(outcome)} = {as_code(event)}, {as_code(exposed_level)} against "
        f"{as_code(reference)}, within the {total} groups of {as_code(stratum)}, then combined (strata of "
        f"{min(sizes):,} to {max(sizes):,} rows)",
        used,
    )
    return ModelPlan(rows, (step,), tuple(advice))


# ----------------------------------------------------------------------------- the calculators

FIRST_EVENTS = (
    "Count each participant's first event. With several events per participant the exact interval is too narrow: "
    "use a count model (Survival & regression, Count or rate)."
)
EXPECTED_AS_KNOWN = (
    "The expected number is taken as known. The ratio compares the cohort with the population that gave the "
    "reference rates, and with no other."
)
# "How to read this" of the calculators: what is to know whatever the numbers typed, said once under the result
# (above every result, it was read again at each number typed)
RATE_READING = (
    "A rate is events over the time at risk, not over a number of participants. " + FIRST_EVENTS
)
RATIO_READING = (
    RATE_READING
    + " The ratio compares two rates; its interval and its test are exact and agree: the interval "
    "leaves 1 out exactly when p is below 0.05."
)
EXPECTED_READING = (
    "An SMR or SIR compares the events observed in the cohort with those expected if the reference population's "
    "rates applied. " + EXPECTED_AS_KNOWN
)
SHARE_READING = (
    "The interval says how large the share may be; the test asks whether the reference is compatible with it. "
    "They are two methods (a Wilson interval, an exact binomial test): next to the reference they can part, "
    "and the result then says so. The reference is a value fixed beforehand, from outside these data."
)


def check_rates(
    calculation: str,
    events: int | None,
    person_time: float | None,
    reference_events: int | None,
    reference_time: float | None,
) -> tuple[bool, list[Advice]]:
    """Whether one rate, or two and their ratio, can be computed from the numbers typed, and what to know of them."""
    two = calculation == TWO_RATES
    typed = [events, person_time, *((reference_events, reference_time) if two else ())]
    if any(value is None for value in typed):
        return False, []
    times = [person_time, *((reference_time,) if two else ())]
    if any(float(t) <= 0 for t in times):  # type: ignore[arg-type]
        which = (
            ""
            if not two
            else (
                " for the group compared" if float(person_time) <= 0 else " for the reference group"  # type: ignore[arg-type]
            )
        )
        return False, [
            Advice(
                "violated",
                f"Type a person-time at risk above 0{which}: a rate is events over time.",
            )
        ]
    advice: list[Advice] = []
    if two and not reference_events and not events:
        advice.append(
            Advice(
                "info",
                "No event in either group: no ratio; each rate is given with its upper bound.",
            )
        )
    elif two and not reference_events:
        advice.append(
            Advice(
                "caution",
                "No event in the reference group: the ratio has no upper bound; only its lower bound says "
                "something.",
            )
        )
    counted = int(events or 0) + (int(reference_events or 0) if two else 0)
    if 0 < counted < LEVEL_MIN:
        advice.append(Advice("info", f"{counted} events in all: the intervals are wide."))
    if any(
        e is not None and t is not None and float(e) > float(t)
        for e, t in ((events, person_time), *(((reference_events, reference_time),) if two else ()))
    ):
        advice.append(
            Advice(
                "info",
                "More events than units of person-time: the right unit? Several events for one participant?",
            )
        )
    return True, advice


def check_expected(observed: int | None, expected: float | None) -> tuple[bool, list[Advice]]:
    """Whether observed events can be compared with an expected number, and what to know of the numbers typed."""
    if observed is None or expected is None:
        return False, []
    if float(expected) <= 0:
        return False, [
            Advice(
                "violated", "Type an expected number of events above 0: the ratio divides by it."
            )
        ]
    advice: list[Advice] = []
    if float(expected) < 1:
        advice.append(
            Advice("caution", "Fewer than 1 event expected: the ratio rests on almost nothing.")
        )
    elif float(expected) < 5:
        advice.append(Advice("info", "Fewer than 5 events expected: the interval is wide."))
    if not observed:
        advice.append(
            Advice("info", "No event observed: the ratio is 0, and its upper bound says something.")
        )
    return True, advice


__all__ = [
    "ABSOLUTE",
    "ACCURACY",
    "AGREEMENT",
    "BLAND_ALTMAN",
    "CALCULATIONS",
    "CASES",
    "CHANGES",
    "COMPARED",
    "CONSISTENCY",
    "CUTOFFS",
    "DATASET",
    "DEATHS",
    "DIAGNOSTIC_ENTRIES",
    "ENTRY",
    "EVENT_KINDS",
    "EXPECTED",
    "FIXED",
    "GRADES",
    "HELP",
    "HIGHER",
    "ICC",
    "ICC_FORMS",
    "LINEAR",
    "LOWER",
    "MEAN",
    "MEASUREMENTS",
    "MEASURES",
    "NAMES",
    "NEAR_MISSES",
    "NO_CUTOFF",
    "NUMBER",
    "ONE_GROUP",
    "ONE_RATE",
    "ONE_T",
    "PAIRED",
    "PAIRED_ENTRIES",
    "PERSON_TIME",
    "QUADRATIC",
    "RATERS",
    "RATES",
    "RATIO_NAMES",
    "SCALES",
    "SHARE",
    "SIGNED_RANK",
    "SOURCES",
    "STRATA",
    "SUMMARIES",
    "TWO_RATES",
    "TYPED",
    "T_TEST",
    "YES_NO",
    "YOUDEN",
    "Entry",
    "check_expected",
    "check_rates",
    "check_share",
    "group_medians",
    "levels_of",
    "look_alikes",
    "measurements",
    "paired_values",
    "plan_accuracy",
    "plan_measurements",
    "plan_one_mean",
    "plan_paired",
    "plan_paired_yes_no",
    "plan_raters",
    "plan_share",
    "plan_strata",
    "present_value",
    "rated",
    "rater_categories",
    "share_counts",
    "strata",
    "two_valued",
]
