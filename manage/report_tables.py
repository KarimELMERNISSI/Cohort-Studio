"""
The results of the study dossier as report blocks: tables, lines of text, references and, for survival
analyses, a Kaplan-Meier figure. One function, :func:`result_blocks`, reads every type of result the analysis
pages store (biostats results, pooled results over imputed datasets, the dictionaries of a few pages) and
returns plain data that the Word builder (:mod:`manage.dossier`) lays out.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pandas as pd

from utils.numbers import (
    area_parts,
    decimals,
    fixed,
    number,
    p_text,
    percent,
    prevalence_text,
    ratio_parts,
    share_decimals,
    share_parts,
    with_interval,
)

ESTIMATE_COLUMNS = {"hr": "hazard ratio", "odds_ratio": "odds ratio", "rate_ratio": "rate ratio"}


@dataclass
class ResultBlocks:
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    figure: bytes | None = None  # PNG
    warnings: list[str] = field(default_factory=list)
    # the words of the result's settings line that the block holds back (:func:`settings_written`)
    held_settings: tuple[str, ...] = ()
    # a calculation whose block holds its table back: the line of the numbers typed is then not written either
    # (the four counts of a 2x2 table are the table)
    held_numbers: bool = False



# how the participants entered the study, as the dossier writes it (the page's labels, in a sentence: held equal by
# tests/ui_layer/test_study_design.py); the ids are biostats.study_designs'
DESIGN_WORDS: dict[str, str] = {
    "unknown": "not stated: odds ratios only",
    "cohort": "followed over time (cohort)",
    "rct": "followed over time, groups formed by chance (randomised trial)",
    "cross-sectional": "at one point in time (cross-sectional)",
    "case-control": "by their outcome (case-control)",
}

def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except TypeError, ValueError:
        return str(value)
    if math.isnan(x):
        return "n/a"
    if math.isinf(x):
        return "inf"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"


def _points(share: float) -> str:
    """A difference of two shares in percentage points, one decimal, a half rounded away from zero, as the page
    writes it: "-10.6", never "-0.0"."""
    return fixed(100 * float(share), 1)


def _p(value: Any) -> str:
    try:
        x = float(value)
    except TypeError, ValueError:
        return "n/a"
    if math.isnan(x):
        return "n/a"
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def _peq(value: Any) -> str:
    """'p = 0.251' or 'p < 0.001', for sentences."""
    text = _p(value)
    return f"p {text}" if text.startswith("<") else f"p = {text}"


def _ci(ci: Any) -> str:
    """'low to high' from a CI object, a pair or a mapping."""
    if ci is None:
        return ""
    if hasattr(ci, "lower") and hasattr(ci, "upper"):
        low, high = ci.lower, ci.upper
    elif isinstance(ci, dict):
        low, high = ci.get("lower"), ci.get("upper")
    else:
        try:
            low, high = ci
        except TypeError, ValueError:
            return str(ci)
    return f"{_num(low)} to {_num(high)}"


def _test_line(test: Any) -> str:
    """One line for a TestResult: estimate with CI, statistic, df, p and n."""
    parts = []
    if test.estimate is not None:
        parts.append(f"{test.estimate_label or 'estimate'} = {_num(test.estimate)}")
        if test.estimate_ci is not None:
            parts[-1] += f" (95% CI {_ci(test.estimate_ci)})"
    if test.statistic is not None:
        stat = f"{test.statistic_label or 'statistic'} = {_num(test.statistic)}"
        if test.df:
            stat += ", df = " + ", ".join(_num(d, 4) for d in test.df)
        parts.append(stat)
    parts.append(f"{_peq(test.p_value)}")
    if test.n_per_group:
        parts.append(
            "n = "
            + ", ".join(f"{g}: {n}" for g, n in zip(test.groups, test.n_per_group, strict=False))
        )
    return "; ".join(parts)


def _test_blocks(test: Any) -> ResultBlocks:
    blocks = ResultBlocks(
        lines=[f"{test.name}: {_test_line(test)}"], references=list(test.references)
    )
    if test.mi is not None:
        blocks.lines.append(
            f"Pooled over {test.mi.m} imputed datasets with Rubin's rules: df = {_num(test.mi.df, 4)}, "
            f"FMI = {_num(test.mi.fmi)}, RIV = {_num(test.mi.riv)}."
        )
    for eff in test.effect_sizes:
        text = f"{eff.name} = {_num(eff.estimate)}"
        if eff.ci is not None:
            text += f" (95% CI {_ci(eff.ci)})"
        blocks.lines.append(text)
    blocks.warnings.extend(test.warnings)
    return blocks


def _model_table(table: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Term, ratio (or coefficient) with its 95% CI and p, from a model's coefficient table."""
    ratio = next((c for c in ESTIMATE_COLUMNS if c in table.columns), None)
    rows = []
    for _, r in table.iterrows():
        if str(r["term"]) == "const":
            continue
        estimate = r[ratio] if ratio else r.get("coef")
        rows.append(
            {
                "Term": str(r["term"]),
                (ESTIMATE_COLUMNS[ratio] if ratio else "Coefficient").capitalize(): _num(estimate),
                "95% CI": f"{_num(r['ci_low'])} to {_num(r['ci_high'])}",
                "p": _p(r["p"]),
            }
        )
    return ("Estimates", pd.DataFrame(rows))


def km_png(curves: list[tuple[str, Any, Any]], *, dpi: int = 160) -> bytes:
    """Kaplan-Meier curves (label, times, survival) as a step plot."""
    import numpy as np
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.0, 3.6))
    ax = fig.subplots()
    for label, times, survival in curves:
        t = np.concatenate([[0.0], np.asarray(times, dtype=float)])
        s = np.concatenate([[1.0], np.asarray(survival, dtype=float)])
        ax.step(t, s, where="post", label=str(label), linewidth=1.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time")
    ax.set_ylabel("survival probability")
    ax.grid(alpha=0.3)
    if len(curves) > 1:
        ax.legend(fontsize=8, frameon=False)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()


def _median(value: Any) -> str:
    """A median time: "not reached", or the time in its unit, without an exponent, to two decimals at most (a
    midpoint ends in .5: 1,119.5, which three figures rounded to 1,120)."""
    return "not reached" if value is None else f"{float(value):,.2f}".rstrip("0").rstrip(".")


def _median_ci(lower: Any, upper: Any) -> str:
    """The median's interval: "not reached" alone when neither end is reached (the lower one is reached first)."""
    return "not reached" if lower is None and upper is None else f"{_median(lower)} to {_median(upper)}"


# a group's median and its interval go into the dossier from this many participants: in smaller groups they are event
# times (the interval's lower limit the shortest), a participant's duration in a document that travels; the page keeps
# them
DOSSIER_MIN_N = 10
SMALL = f"not shown: fewer than {DOSSIER_MIN_N}"


def dossier_min_n() -> int:
    """
    The fewest participants whose values the dossier writes: a group's median, its interval and its curve; a cut-off
    found on the data, its range and the measures at it. One number for every rule of the dossier, so that it can be
    the study's own (the privacy rules set in one place) where it is ``DOSSIER_MIN_N`` today.
    """
    return DOSSIER_MIN_N


def _shown(n: Any) -> bool:
    """Whether a group's median goes into the dossier (a result pooled before the sizes were kept: as it was)."""
    return n is None or n >= dossier_min_n()


def _survival_blocks(res: Any) -> ResultBlocks:
    table = pd.DataFrame(
        [
            {
                "Group": c.label,
                "n": c.n,
                "Events": c.events,
                "Median survival": _median(c.median) if _shown(c.n) else SMALL,
                f"Median {c.level:.0%} CI": _median_ci(c.median_lower, c.median_upper)
                if _shown(c.n)
                else SMALL,
            }
            for c in res.curves
        ]
    )
    blocks = ResultBlocks(tables=[("Kaplan-Meier", table)], references=list(res.references))
    if res.logrank_statistic is not None:
        blocks.lines.append(
            f"Log-rank test: chi2 = {_num(res.logrank_statistic)}, df = {res.logrank_df}, {_peq(res.logrank_p)}"
        )
    # a small group's curve steps at its participants' event times: drawn only where its median is shown
    drawn = [c for c in res.curves if _shown(c.n)]
    if drawn:
        blocks.figure = km_png([(c.label, c.times, c.survival) for c in drawn])
    if len(drawn) < len(res.curves):
        blocks.lines.append(_not_drawn(len(res.curves) - len(drawn)))
    blocks.warnings.extend(res.warnings)
    return blocks


def _not_drawn(count: int) -> str:
    return (
        f"{count} group{'s' if count != 1 else ''} of fewer than {dossier_min_n()} "
        f"{'are' if count != 1 else 'is'} not drawn."
    )


def settings_written(settings: str, blocks: ResultBlocks) -> str:
    """A result's settings line as the dossier writes it: the words its block holds back replaced by "not written"
    (a cut-off whose value the block does not write stood in the settings line, as "3987" beside a title without
    "3,987")."""
    if not blocks.held_settings:
        return settings
    return ", ".join(
        "not written" if word in blocks.held_settings else word for word in settings.split(", ")
    )


def _few_called(acc: Any, least: int) -> str:
    """
    The side a cut-off calls fewer than ``least`` participants to ("positive", "negative"; "" when it calls enough to
    each). Found on the data, such a cut-off is the score of the participant who holds it, or of a few: at the largest
    score, the score of the one called positive, and every measure at it (sensitivity, the predictive values, the
    likelihood ratios) reads the small row of the 2x2 table, so that with the outcome's two totals it gives whether
    they had the condition.
    """
    positive, negative = acc.tp + acc.fp, acc.fn + acc.tn
    if positive < least:
        return "positive"
    return "negative" if negative < least else ""


def _thin_side(auc: dict, least: int) -> str:
    """
    The side of the reference standard that holds fewer than ``least`` participants ("have the condition", "are
    without the condition"; "" when both hold enough, or for a result kept before its sizes). The AUC ranks each of
    them among the others, and the measures at any cut-off read them one by one: with one participant with the
    condition among 400, "AUC 1 (1 to 1)" said that the largest score was theirs.
    """
    for name, side in (("n_positive", "have the condition"), ("n_negative", "are without the condition")):
        held = auc.get(name)
        if held is None or isinstance(held, bool):
            continue
        try:
            count = int(held)
        except (TypeError, ValueError):
            continue
        if count < least:
            return side
    return ""


def _youden_range(youden: dict, least: int) -> str:
    """
    The sentence of the range of a cut-off found on the data: the middle 95% of the cut-offs found in resamples. Each
    end is a score found in a resample, a participant's own when few lie beyond it: it is written only when ``least``
    participants or more lie on each side of it, and otherwise named by where it lies.
    """
    ci = youden["ci"]
    resamples = youden.get("resamples")
    among = f"In {resamples:,} bootstrap resamples" if resamples else "In the bootstrap resamples"
    sides = youden.get("ci_sides")
    if not sides:  # a result kept before its ends were counted: the page shows its range
        return (
            f"Cut-off chosen by Youden's J. {among}, the cut-offs found have a range the page shows; it is not "
            "written here."
        )
    ends, hidden = [], False
    for end, (below, above) in zip((ci.lower, ci.upper), sides, strict=True):
        if min(below, above) >= least:
            ends.append(_num(end))
            continue
        hidden = True
        ends.append("the smallest observed scores" if below < least else "the largest observed scores")
    said = f"Cut-off chosen by Youden's J. {among}, 95% of the cut-offs found lie between {ends[0]} and {ends[1]}"
    if hidden:
        said += f" (an end with fewer than {least} participants beyond it is not written)"
    return said + "."


def _pooled_frame(result: Any) -> pd.DataFrame:
    frame = result.to_frame()
    if frame.empty:
        return frame
    rows = []
    for _, r in frame.iterrows():
        if str(r["term"]) == "const":
            continue
        natural = r.get("scale") not in (None, "identity") and pd.notna(r.get("natural_estimate"))
        rows.append(
            {
                "Term": str(r["term"]),
                "Estimate": _num(r["natural_estimate"] if natural else r["estimate"]),
                "95% CI": f"{_num(r['natural_ci_low'] if natural else r['ci_low'])} to "
                f"{_num(r['natural_ci_high'] if natural else r['ci_high'])}",
                "p": _p(r["p_value"]),
                "df": _num(r["df"], 4),
                "FMI": _num(r["fmi"]),
            }
        )
    return pd.DataFrame(rows)


def _pooled_blocks(res: Any) -> ResultBlocks:
    blocks = ResultBlocks()
    blocks.lines.append(
        f"Pooled over {res.m} imputed datasets with Rubin's rules (Barnard-Rubin degrees of freedom)."
    )
    if res.kind == "test" and "result" in res.extra:
        inner = _test_blocks(res.extra["result"])
        blocks.lines.extend(inner.lines)
        blocks.references.extend(inner.references)
    elif res.kind == "survival":
        curves = res.extra.get("curves") or {}
        medians = res.extra.get("medians") or {}
        intervals = res.extra.get("median_ci") or {}  # a result pooled before: none
        sizes = res.extra.get("sizes") or {}  # likewise
        rows = []
        for label, curve in curves.items():
            shown = _shown(sizes.get(label))
            interval = _median_ci(*intervals[label]) if label in intervals else "n/a"
            rows.append(
                {
                    "Group": label,
                    "Median survival (pooled curve)": _median(medians.get(label)) if shown else SMALL,
                    "Median 95% CI (pooled band)": interval if shown else SMALL,
                    "Survival at last time": _num(curve["survival"].iloc[-1]) if shown else SMALL,
                }
            )
        blocks.tables.append(("Kaplan-Meier, pooled (log(-log) scale)", pd.DataFrame(rows)))
        drawn = {label: c for label, c in curves.items() if _shown(sizes.get(label))}
        if drawn:
            blocks.figure = km_png([(label, c["time"], c["survival"]) for label, c in drawn.items()])
        if len(drawn) < len(curves):
            blocks.lines.append(_not_drawn(len(curves) - len(drawn)))
    else:
        frame = _pooled_frame(res)
        if not frame.empty:
            blocks.tables.append(("Pooled estimates (ratios on their natural scale)", frame))
    if res.kind == "linear" and res.extra.get("adjusted_means") is not None:
        means = res.extra["adjusted_means"]
        blocks.tables.append(
            (
                "Adjusted means, pooled",
                pd.DataFrame(
                    {
                        "Group": means["group"],
                        "Mean": [_num(v) for v in means["mean"]],
                        "95% CI": [
                            f"{_num(a)} to {_num(b)}"
                            for a, b in zip(means["ci_low"], means["ci_high"], strict=True)
                        ],
                    }
                ),
            )
        )
    if res.tests:
        blocks.tables.append(
            (
                "Tests of several parameters",
                pd.DataFrame(
                    [
                        {
                            "Test": name,
                            "Method": t.method,
                            "F": _num(t.statistic),
                            "df": f"{_num(t.df1, 4)}, {_num(t.df2, 4)}",
                            "p": _p(t.p_value),
                        }
                        for name, t in res.tests.items()
                    ]
                ),
            )
        )
        for t in res.tests.values():
            blocks.references.extend(t.references)
    r2 = res.extra.get("r2")
    if r2 is not None:
        blocks.lines.append(f"R2 pooled with Harel's method: {_num(r2.estimate)}")
        blocks.references.extend(r2.references)
    blocks.warnings.extend(res.warnings)
    blocks.lines.extend(res.notes)
    cc = res.complete_case
    n = getattr(cc, "n", None) or getattr(cc, "n_total", None)
    if cc is not None and n:
        blocks.lines.append(f"Complete-case analysis for comparison: n = {n:,}.")
    return blocks


def _by_group(counts: Any) -> str:
    """The rows of each group, in the groups' order: "Control 390, Intervention 388"."""
    if not isinstance(counts, dict):
        return ""
    return ", ".join(f"{group} {int(n):,}" for group, n in counts.items())


def _univariate_blocks(value: dict) -> ResultBlocks:
    from biostats.multiplicity import adjust

    frame = pd.DataFrame(value.get("results") or [])
    blocks = ResultBlocks()
    if frame.empty:
        return blocks
    correction = value.get("correction") or "None"
    estimates = frame["Estimate"] if "Estimate" in frame.columns else pd.Series([None] * len(frame))
    rows = pd.DataFrame(
        {
            "Variable": frame["Variable"],
            # the rows each variable is analysed on, in all and in each group (STROBE item 12: a difference of two
            # means is read with its two n); results stored before they were kept say nothing
            "n": [f"{int(v):,}" for v in frame["N"]] if "N" in frame.columns else "",
            "n by group": [_by_group(v) for v in frame["N per group"]]
            if "N per group" in frame.columns
            else "",
            "Test": frame["Test Used"],
            "Estimate": [_num(v) for v in estimates],
            "Estimate CI": frame["Estimate CI"] if "Estimate CI" in frame.columns else "",
            # its direction (Intervention - Control); results stored before it was kept say nothing
            "Estimate of": frame["Estimate of"] if "Estimate of" in frame.columns else "",
            "p": [_p(v) for v in frame["P-Value"]],
        }
    )
    if correction != "None":
        rows[f"p ({correction})"] = [_p(v) for v in adjust(frame["P-Value"].tolist(), correction)]
    blocks.tables.append((f"Comparison by {value.get('group')}", rows))
    return blocks


def _pooled_comparison_blocks(value: dict) -> ResultBlocks:
    blocks = ResultBlocks()
    rows = []
    for target, item in (value.get("results") or {}).items():
        res = item.result
        if item.kind in ("welch", "student"):
            test = res.extra["result"]
            analysis = "Welch t-test" if item.kind == "welch" else "Student t-test (equal variances, chosen)"
            rows.append({"Variable": target, "Analysis": analysis, "Result": _test_line(test)})
            blocks.references.extend(test.references)
        elif item.kind in ("robust", "anova"):
            group = next(iter(res.tests))
            d1 = res.tests[group]
            rows.append(
                {
                    "Variable": target,
                    "Analysis": "Group means, unequal variances (Welch-type; D1)"
                    if item.kind == "robust"
                    else "Linear model on the groups, equal variances (chosen; D1)",
                    "Result": f"F = {_num(d1.statistic)}, df = {_num(d1.df1, 4)}, {_num(d1.df2, 4)}; {_peq(d1.p_value)}",
                }
            )
            # the test's own references (D1, Reiter's df), and for Welch-type means their variances and df
            blocks.references.extend(d1.references)
            if item.kind == "robust":
                blocks.references.extend(res.extra.get("references", []))
        elif item.kind == "twobytwo" and "odds_ratio" in res.table.names:
            t = res.term("odds_ratio")
            sensitivity = getattr(item, "haldane", "if_zero") == "always"
            rows.append(
                {
                    "Variable": target,
                    "Analysis": "Odds ratio, 0.5 added to every cell (sensitivity analysis)"
                    if sensitivity
                    else "Odds ratio",
                    "Result": f"OR = {_num(t.natural_estimate)} (95% CI {_ci(t.natural_ci)}); {_peq(t.p_value)}",
                }
            )
        elif item.kind == "twobytwo":
            # a zero cell in some imputed datasets: the ratios are not pooled. The pooled chi-square test stands in,
            # which every study design supports (a risk difference is no measure of a case-control sample)
            d2 = res.tests["chi_square"]
            rows.append(
                {
                    "Variable": target,
                    "Analysis": "Chi-square test (D2; odds ratio not pooled: a zero cell)",
                    "Result": f"F = {_num(d2.statistic)}, df = {_num(d2.df1, 4)}, {_num(d2.df2, 4)}; {_peq(d2.p_value)}",
                }
            )
            blocks.warnings.extend(
                w
                for w in res.warnings
                # the words of before named the risk ratio whatever the design
                if w.startswith(("The ratios are not pooled", "Odds ratio and risk ratio not pooled"))
            )
        else:
            d2 = res.tests["chi_square"]
            rows.append(
                {
                    "Variable": target,
                    "Analysis": "Chi-square test (D2)",
                    "Result": f"F = {_num(d2.statistic)}, df = {_num(d2.df1, 4)}, {_num(d2.df2, 4)}; {_peq(d2.p_value)}",
                }
            )
    for target, message in (value.get("errors") or {}).items():
        rows.append({"Variable": target, "Analysis": "not computed", "Result": message})
    if rows:
        blocks.tables.append(("Pooled comparisons", pd.DataFrame(rows)))
    return blocks


# how the participants of an accuracy study entered it, as the dossier writes it beside the predictive values (the
# ids are biostats.diagnostic's)
SAMPLING_WORDS: dict[str, str] = {
    "series": "The participants entered the study as a series, taken before the condition was known (consecutive or "
    "random)",
    "two_gate": "Participants with and without the condition were sampled apart (two-gate)",
    "not_stated": "How the participants entered the study is not stated",
}
# sampled apart, the spectrum of the participants is not the setting's (Lijmer et al. 1999)
TWO_GATE_SPECTRUM = (
    "Sensitivity and specificity may read higher than where the test will be used: sampled apart, those with the "
    "condition are often clearer cases, and those without healthier."
)


def sampling_line(
    sampling: str, prevalence: Any = None, *, read: bool = True, unequal: bool = False
) -> str:
    """How the participants entered the study, and where the predictive values of the block are read: said by what
    was written (a line that sent the reader to "a prevalence given" stood over a table without one). ``read``:
    whether the predictive values at the prevalence given have a value (a test that calls everyone, or no one,
    positive has none). ``unequal``: the reference standard was done unequally by test result; a series' share with
    the condition is then that of those who had it, and no prevalence."""
    from biostats.diagnostic import SERIES, TWO_GATE

    stem = SAMPLING_WORDS[sampling]
    given = None if prevalence is None or not read else prevalence_text(prevalence)
    caveat =f" {TWO_GATE_SPECTRUM}" if sampling == TWO_GATE else ""
    if sampling == SERIES:
        also = "" if given is None else f", and at the prevalence given ({given})"
        if unequal:
            return (
                f"{stem}: predictive values are read among those who had the reference standard, whose share with "
                f"the condition is not the prevalence{also}."
            )
        return f"{stem}: predictive values are read at their share with the condition{also}."
    if given is None:
        asked = "" if prevalence is not None else ", none was asked at a prevalence"
        return f"{stem}: no predictive value is written{asked}.{caveat}"
    return f"{stem}: predictive values are read at the prevalence given ({given}) only.{caveat}"


def _small_side(acc: Any, auc: Any, least: int) -> str:
    """
    The side of the reference standard that holds fewer than ``least`` participants ("have the condition", "are
    without the condition"; "" when both hold enough). Read on the 2x2 table whenever there is one, whatever the
    index test: a yes/no test has no AUC, and a result without the AUC's counts was read as "enough". Without a
    table, on the AUC's counts (:func:`_thin_side`).
    """
    if acc is not None:
        if acc.tp + acc.fn < least:
            return "have the condition"
        if acc.fp + acc.tn < least:
            return "are without the condition"
        return ""
    return _thin_side(auc, least) if isinstance(auc, dict) else ""


AUC_NAME = "Area under the ROC curve (AUC)"


def auc_row(auc: Mapping[str, Any]) -> tuple[str, str, str]:
    """
    The area under the ROC curve, as the page's headline and the dossier's block write it: its name, the area, and
    the bounds of its interval with the method ("0.897 to 0.946 (DeLong)"), the three with the same decimals
    (``utils.numbers.area_parts``). No interval where DeLong's is not one to write
    (``biostats.diagnostic.auc_interval_missing``, the rule the plan of the analysis reads too): "1.000 (95% CI 1.000
    to 1.000)" stood for two groups that do not overlap, as if the area were known exactly; the result's caution
    says why.
    """
    from biostats.diagnostic import auc_interval_missing

    area, bounds = area_parts(auc.get("auc"), None if auc_interval_missing(auc) else auc.get("ci"))
    return AUC_NAME, area, f"{bounds} (DeLong)" if bounds else ""


def measure_rows(
    acc: Any,
    prevalence: Any = None,
    *,
    unequal: bool = False,
    typed: bool = False,
    every: bool = False,
) -> list[tuple[str, str, str]]:
    """
    The measures of a 2x2 table, each with its estimate and the bounds of its interval: the rows of the page's
    table of every measure and of the dossier's block, from one list, so that a measure has one name and one number
    on the screen and in the document. The block read "Sensitivity | 0.861 | 0.781 to 0.916", "NPV, ..." and "LR- |
    0.164 | 0.1 to 0.266" under a page that read 86.1% (78.1% to 91.6%), "Negative predictive value, ..." and 0.100
    to 0.266. A share is a percentage (``utils.numbers.share_parts``), a ratio has the decimals of its bounds
    (``ratio_parts``).

    The rows are those the sampling supports: the predictive values at the share with the condition for a series
    only (sampled apart, that share is the sampling's), those at a prevalence given whatever the sampling.
    ``unequal``: the reference standard was done unequally by test result, and the rows name that share as the
    page's caption does. ``typed``: a table typed into the calculator, whose share is the table's. ``every``: the
    page's table, with Youden's J, the share classified correctly (a series), and the diagnostic odds ratio's row
    when it has no value; the dossier's block writes the others (a ratio without a value has its warning there).
    """
    from biostats.diagnostic import reads_share

    share = percent(acc.prevalence_observed, share_decimals(acc.prevalence_observed))
    at_share = reads_share(acc.sampling)
    where = (
        f"among those who had the reference standard ({share} with the condition)"
        if unequal
        else f"at the share in {'this table' if typed else 'these data'} ({share})"
    )
    # name, value, interval, whether it is a share
    rows: list[tuple[str, Any, Any, bool]] = [
        ("Sensitivity", acc.sensitivity, acc.sensitivity_ci, True),
        ("Specificity", acc.specificity, acc.specificity_ci, True),
    ]
    if at_share:
        rows += [
            (f"Positive predictive value, {where}", acc.ppv, acc.ppv_ci, True),
            (f"Negative predictive value, {where}", acc.npv, acc.npv_ci, True),
        ]
    if prevalence is not None and acc.ppv_at_prevalence is not None:
        given = prevalence_text(prevalence)
        rows += [
            (
                f"Positive predictive value, at a prevalence of {given}",
                acc.ppv_at_prevalence,
                getattr(acc, "ppv_at_prevalence_ci", None),
                True,
            ),
            (
                f"Negative predictive value, at a prevalence of {given}",
                acc.npv_at_prevalence,
                getattr(acc, "npv_at_prevalence_ci", None),
                True,
            ),
        ]
    rows += [
        ("Positive likelihood ratio (LR+)", acc.lr_positive, acc.lr_positive_ci, False),
        ("Negative likelihood ratio (LR-)", acc.lr_negative, acc.lr_negative_ci, False),
    ]
    if every or acc.dor is not None:
        rows.append(("Diagnostic odds ratio", acc.dor, acc.dor_ci, False))
    if every:
        rows.append(("Youden's J", acc.youden, None, False))
        if at_share:
            rows.append(("Share classified correctly", acc.accuracy, None, True))
    return [
        (name, *(share_parts(value, ci) if is_share else ratio_parts(value, ci)))
        for name, value, ci, is_share in rows
    ]


def _accuracy_blocks(value: dict) -> ResultBlocks:
    """
    The accuracy of a test against a reference standard: a score (its AUC; at a cut-off, the 2x2 table and its
    measures) or a yes/no test (the table and its measures; no AUC).

    One rule for a side of the reference standard, whatever the index test: with fewer than the dossier's threshold
    on a side, neither the table nor a measure is written (they read those participants one by one), and one line
    says so. A yes/no test is read as a cut-off typed beforehand: the side it calls does not count, its positive
    value being a code of the data, not a score found on them. The table goes with the measures: sensitivity,
    specificity and the predictive values give its cells back, so writing it adds nothing, and holding it back
    alone would protect nothing.
    """
    from biostats.diagnostic import FOUND_CUTOFF, LOWER, YES_NO_FORM
    from biostats.references import cite

    acc, auc = value["acc"], value.get("auc")
    yes_no = value.get("index_form") == YES_NO_FORM
    # a table typed into the calculator: no column, no value and no cut-off to name; one rule with the dataset's
    typed = bool(value.get("typed"))
    lower = value.get("direction") == LOWER
    least = dossier_min_n()
    found = value.get("mode") == FOUND_CUTOFF
    thin, few = _table_held(value, least)
    held_back = bool(thin or few)
    rows: list[tuple[str, str, str]] = []
    if auc is not None:
        # the page's headline, name for name and number for number
        rows.append((AUC_NAME, f"not shown: fewer than {least}", "") if thin else auc_row(auc))
    written = acc is not None and not held_back
    sampling = getattr(acc, "sampling", "") if acc is not None else ""
    # the reference standard done unequally by test result: the share with the condition is then that of those who
    # had it, and the rows and the sampling line name it so
    unequal = bool(getattr(value.get("verified"), "partial", False))
    if written:
        # the page's rows, name for name and number for number
        rows += measure_rows(acc, value.get("prevalence"), unequal=unequal, typed=typed)
    table = pd.DataFrame(rows, columns=["Measure", "Estimate", "95% CI"])
    # what the numbers are of: the value that means the condition, the way the test points, the cut-off; a table
    # typed names none of them (its heading and the line of its numbers have said what it is)
    if typed:
        title = ""
    elif yes_no:
        title = f"{value.get('score')} = {value.get('positive')} (positive) for {value.get('outcome')}"
    else:
        title = f"{value.get('score')} for {value.get('outcome')}"
    if not typed and value.get("pos_label") is not None:
        title += f" = {value.get('pos_label')}"
    called = "Test positive"
    if yes_no:
        called += f" ({value.get('positive')})"
    elif lower:
        title += ", lower values pointing to the condition"
    # a cut-off typed beforehand that calls fewer than ``least`` participants to one side stands among their scores
    # (typed at the largest score, it is the score of the one called positive): its value is not written, in the
    # title, in the table's row or in the settings line. The table and the measures are, as for any typed cut-off;
    # a yes/no test's positive value is a code of the data, and is out of this
    scored = not yes_no and not typed
    typed_few = _few_called(acc, least) if acc is not None and scored and not found else ""
    held_settings: tuple[str, ...] = ()
    if scored:
        if acc is None:
            title += " (no cut-off)"
        elif found and held_back:
            title += " (a cut-off found on these data, not written)"
        elif typed_few:
            title += (
                f" (a cut-off fixed beforehand, not written: it calls fewer than {least} participants "
                f"{typed_few})"
            )
            held_settings = (f"{float(value.get('cutoff')):g}",)  # as the settings line spells a number
        else:
            at = f"at {'most' if lower else 'least'} {_num(value.get('cutoff'))}"
            title += f" (positive {at})"
            called += f" ({at})"
    blocks = ResultBlocks(
        # a yes/no test beside a side too small has no row to write: its title stands as a line, no empty table
        tables=[(title, table)] if rows else [],
        lines=[] if rows or typed else [title],
        held_settings=held_settings,
        held_numbers=typed and held_back,
        references=([] if not written else list(acc.references))
        # the caution of partial verification is written with the measures only, and its reference with it
        + (cite("begg1983") if written and value.get("verification") else [])
        + ([] if thin or auc is None else list(auc.get("references") or [])),
    )
    if written:
        # the table the measures are read on (STARD item 23): the measures give its cells back, so it is written
        # wherever they are
        present = "Condition present" + (
            f" ({value.get('pos_label')})" if value.get("pos_label") is not None else ""
        )
        absent = "Condition absent" + (
            f" ({value.get('absent')})" if value.get("absent") is not None else ""
        )
        negative = "Test negative" + (f" ({value.get('negative')})" if yes_no else "")
        counts = pd.DataFrame(
            {
                "": [called, negative, "Total"],
                present: [acc.tp, acc.fn, acc.tp + acc.fn],
                absent: [acc.fp, acc.tn, acc.fp + acc.tn],
                "Total": [acc.tp + acc.fp, acc.fn + acc.tn, acc.tp + acc.fp + acc.fn + acc.tn],
            }
        )
        blocks.tables.append(
            ("The 2x2 table (rows: the test; columns: the reference standard)", counts)
        )
        if sampling in SAMPLING_WORDS:
            blocks.lines.append(
                sampling_line(
                    sampling,
                    value.get("prevalence"),
                    read=acc.ppv_at_prevalence is not None,
                    unequal=unequal,
                )
            )
    youden = value.get("youden") or {}
    if thin and typed:
        # true in the Word file and on the Study dossier page's card, which shows the counts typed
        blocks.lines.append(
            f"Fewer than {least} participants {thin} in this table: the table and its measures read them one by "
            "one, so the Word dossier writes neither, as for a table of the dataset; the calculator shows them."
        )
    elif thin and yes_no:
        blocks.lines.append(
            f"Fewer than {least} participants {thin}: the table and the measures of the test read them one by one, "
            "and are not written in the dossier; the page shows them."
        )
    elif thin:
        blocks.lines.append(
            f"Fewer than {least} participants {thin}: the AUC and the measures at a cut-off rank their "
            "scores among the others one by one, and are not written in the dossier; the page shows them."
        )
    elif found and few:
        # the engine's caution ("1 of 928 is called positive") gives the count: it is not copied either
        blocks.lines.append(
            f"The cut-off found on these data calls fewer than {least} participants {few}: its value, its "
            "range and the measures at it are not written in the dossier; the page shows them."
        )
    elif found:
        # the cut-off was chosen on these data: its range (drawn at random; the settings give the draws) and
        # the caution that goes with it
        if youden.get("ci") is not None:
            blocks.lines.append(_youden_range(youden, least))
        if youden.get("note"):
            blocks.lines.append(str(youden["note"]))
        if "cautions" not in value and youden.get("caution"):  # a result kept before its own words
            blocks.warnings.append(str(youden["caution"]))
    # in the words of the direction asked on the page (the engine's are of the marker it was given); they give
    # the AUC, or shares of the table, which are not written beside a side too small
    if not thin:
        blocks.warnings.extend(str(text) for text in value.get("cautions") or ())
    if written:
        # the reference standard done unequally by test result: its shares are read at the cut-off, and are written
        # only where the measures at it are
        if value.get("verification"):
            blocks.warnings.append(str(value["verification"]))
        blocks.warnings.extend(acc.warnings)
        if typed:
            blocks.warnings.append(TABLE_NOTE)
    return blocks


def _table_held(value: Mapping[str, Any], least: int) -> tuple[str, str]:
    """
    Why the dossier holds back the 2x2 table of an accuracy result and its measures, as two words ("" when it does
    not): the side of the reference standard that holds fewer than ``least`` participants (the AUC and the measures
    rank them one by one, and stay on the page), and, for a cut-off found on these data, the side it calls fewer
    than ``least`` participants to (its value, its range and the measures at it stay on the page; a cut-off typed
    beforehand is no participant's value). One test for the block and for what a table typed is compared with.
    """
    from biostats.diagnostic import FOUND_CUTOFF

    acc = value.get("acc")
    thin = _small_side(acc, value.get("auc"), least)
    found = value.get("mode") == FOUND_CUTOFF
    few = _few_called(acc, least) if found and acc is not None and not thin else ""
    return thin, few


def table_held_back(value: Any) -> tuple[int, ...] | None:
    """
    The four counts, sorted, of an accuracy result of the dataset whose 2x2 table the dossier does not write; None
    when it writes the table, or for a result without one. Counts only: a table typed that holds the same four, in
    any order (its rows or its columns swapped), is not written either (:func:`calculation_blocks`).
    """
    if not isinstance(value, Mapping) or value.get("acc") is None:
        return None
    if not any(_table_held(value, dossier_min_n())):
        return None
    acc = value["acc"]
    return tuple(sorted(int(cell) for cell in (acc.tp, acc.fp, acc.fn, acc.tn)))


# ----------------------------------------------------------------------------- calculations from numbers typed

# under a calculator's result, in place of "Computed on"
TYPED_SOURCE = "Not from the dataset: the analysis population and the imputation do not apply."
# what every calculation is called until its kind says more, and where its numbers came from when it was not said
CALCULATION = "Calculation from numbers typed"
SOURCE_NOT_GIVEN = "Source: not given."
# what each kind's numbers are taken to be: the software never saw where they come from, so each is an assumption
SHARE_NOTE = "The reference share is taken as fixed beforehand."
RATES_NOTE = (
    "The events are taken as first events, one per participant: if the numbers count repeated events of a "
    "participant, the intervals are too narrow."
)
STANDARDISED_NOTE = (
    "The expected number is taken as known; the comparison is with the population that gave the reference rates, "
    "and no other."
)
TABLE_NOTE = (
    "The table is taken as counting each participant once, with one test result and one reference standard: counts "
    "of lesions, segments or readings, several per participant, give intervals that are too narrow."
)
# a table typed that holds the four counts of a table the dossier does not write (the table of an accuracy result of
# the dataset, or another table typed that a side too small holds back): the plain copy of what is held back, in
# words that are true in the Word file and on the Study dossier page's card, which shows the counts
TABLE_OF_A_RESULT_HELD_BACK = (
    "These four counts are those of a table that the Word dossier does not write (a result of the dataset, or "
    "another table typed here): it writes neither this table nor its measures; the calculator shows them."
)


def table_cautions(acc: Any) -> list[str]:
    """What to know of a 2x2 table typed, beyond its measures' own warnings: said under the calculator's result and
    written with its block, from this one place. A test positive more often without the condition than with it
    points the other way: its two rows, or its two columns, may be exchanged. (A table typed with the condition in
    its rows is not caught here: it reads predictive values as sensitivity and specificity, and only its margins,
    written under the grid, show it.)"""
    said: list[str] = []
    with_it, without = acc.tp + acc.fn, acc.fp + acc.tn
    if with_it and without and acc.tp / with_it < acc.fp / without:
        said.append(
            f"In this table the test is positive for {percent(acc.tp / with_it)} of those with the condition and "
            f"{percent(acc.fp / without)} of those without it: it points the other way. Check that the two rows, or "
            "the two columns, are not exchanged."
        )
    return said


def source_line(note: str) -> str:
    """Where the numbers of a calculation come from, as a sentence: the user's words, closed with a full stop only
    when they have none (a reference ends with one more often than not: "1-10.." was written)."""
    if not note:
        return SOURCE_NOT_GIVEN
    return f"Source: {note}" + ("" if note.endswith((".", "?", "!")) else ".")


def _as_typed(value: Any) -> str:
    """A number as it was typed: its whole part grouped, every decimal kept, no exponent (18.4; 1,200;
    1,234,567.891234)."""
    if isinstance(value, int):
        return f"{value:,}"
    text = f"{Decimal(repr(float(value))):,f}"
    return text.removesuffix(".0")


def _counted(count: int, one: str, many: str) -> str:
    return f"{count:,} {one if count == 1 else many}"


def typed_title(spec: Any) -> str:
    """The title of a calculation from numbers typed, by what its numbers give: one rate or a ratio of two, a
    mortality or an incidence ratio."""
    from biostats.plan import kind_named
    from biostats.rates import has_ratio

    numbers, options = spec.typed or {}, spec.options
    if spec.kind == "rates":
        if "reference_events" not in numbers:
            return "Incidence rate (numbers typed)"
        if not has_ratio(numbers["events"], numbers["reference_events"]):
            return "Incidence rates (numbers typed)"  # no event in either group leaves no ratio
        return "Rate ratio (numbers typed)"
    if spec.kind == "standardised_ratio":
        which = "mortality" if options.get("events_are") == "deaths" else "incidence"
        return f"Standardised {which} ratio (numbers typed)"
    return kind_named(spec.kind).typed_title or CALCULATION


def typed_numbers(spec: Any) -> str:
    """
    The numbers of a calculation in words, each with its name, written afresh from its spec: what the page writes
    under the result after "From the numbers typed", and the dossier after "Numbers typed" (never bare values, and
    never a string the page made). Every kind computed from numbers typed has its words here (a test).
    """
    numbers, options = spec.typed or {}, spec.options
    if spec.kind == "proportion":
        said = (
            f"{numbers['count']:,} of {_counted(numbers['total'], 'participant', 'participants')} with the "
            "characteristic"
        )
        reference = options.get("reference")
        if reference is not None:
            said += f", against a reference share of {_as_typed(round(100 * float(reference), 10))}%"
        return said
    if spec.kind == "rates":
        unit, per = options.get("unit", "person-years"), options.get("per", 1000)
        first = f"{_counted(numbers['events'], 'event', 'events')} over {_as_typed(numbers['person_time'])} {unit}"
        if "reference_events" not in numbers:
            return f"{first}, rate per {per:,} {unit}"
        return (
            f"{first} (group compared) against {numbers['reference_events']:,} over "
            f"{_as_typed(numbers['reference_person_time'])} (reference group), rates per {per:,} {unit}"
        )
    if spec.kind == "standardised_ratio":
        deaths = options.get("events_are") == "deaths"
        observed = _counted(
            numbers["observed"],
            "observed death" if deaths else "observed new case",
            "observed deaths" if deaths else "observed new cases",
        )
        return f"{observed} against {_as_typed(numbers['expected'])} expected"
    if spec.kind == "diagnostic":
        said = (
            f"test positive, {numbers['tp']:,} with the condition and {numbers['fp']:,} without; test negative, "
            f"{numbers['fn']:,} with the condition and {numbers['tn']:,} without"
        )
        # what was asked with the table: two calculations of one table read alike without it
        entered = TABLE_SAMPLING.get(str(options.get("sampling")))
        if entered:
            said += f"; {entered}"
        prevalence = options.get("prevalence")
        if prevalence is not None:
            said += (
                f"; predictive values at a prevalence of {_as_typed(round(100 * float(prevalence), 10))}%"
            )
        return said
    raise ValueError(f"the {spec.kind} calculation has no words for its numbers")


# how the participants of a table typed entered their study, after its four counts (the ids are
# biostats.diagnostic's; nothing is added while it is not stated)
TABLE_SAMPLING: dict[str, str] = {
    "series": "participants taken as they came (a series)",
    "two_gate": "those with and without the condition sampled apart (two-gate)",
}


def _named(part: Any) -> dict[str, Any]:
    return {found.name: found.value for found in part.numbers}


def _level(numbers: dict[str, Any]) -> str:
    """ "95% CI", from the level the interval's record holds: no writer assumes it."""
    level = numbers.get("level")
    return "CI" if level is None else f"{_as_typed(round(100 * float(level), 6))}% CI"


def _share_block(record: Any) -> ResultBlocks:
    """A share against a reference, from its record: one sentence for a share counted in the dataset and for one of
    numbers typed, so that the two read alike. The share, its bounds and its reference take the decimals that tell
    a share from 0% and from 100% (``utils.numbers.share_decimals``)."""
    from biostats.references import cite

    share, test = _named(record.part("share")), _named(record.part("test:exact_binomial"))
    lower, upper, reference = share.get("lower"), share.get("upper"), test.get("reference")
    places = share_decimals(share["estimate"], lower, upper, reference)
    line = f"Share: {percent(share['estimate'], places)} ({share['count']:,} of {share['total']:,}"
    if lower is not None and upper is not None:
        line += (
            f"; {_level(share)} {fixed(100 * lower, places)}% to {fixed(100 * upper, places)}%, "
            f"{share.get('method')}"
        )
    line += ")"
    if reference is not None:
        gap = 100 * (float(share["estimate"]) - float(reference))
        apart = (
            "equal to the reference"
            if gap == 0
            else f"{fixed(abs(gap), max(places, decimals(abs(gap), 2)))} points "
            f"{'below' if gap < 0 else 'above'} the reference"
        )
        # the reference with the decimals that show it: beside a share far from it, 0.04% read "0.0%", a value
        # the page refuses to take
        shown = percent(reference, max(places, share_decimals(reference)))
        line += (
            f"; against a reference of {shown}: p {p_text(test['p'])} (exact binomial test, "
            f"two-sided); {apart}"
        )
    return ResultBlocks(
        lines=[line + "."],
        warnings=[*record.warnings, SHARE_NOTE],
        references=cite("wilson1927"),
    )


def _rates_block(record: Any) -> ResultBlocks:
    """One rate, or two and their ratio, from their record: the rates per the multiplier and the unit the record
    holds, the ratio with the one bound it has when the reference group has no event, and the intervals' methods."""
    from biostats.references import cite

    parts = {part.id: part for part in record.parts}
    first = _named(parts["rate#0"])
    per, unit, two = first["per"], first.get("unit"), "rate#1" in parts
    if unit is None:  # never guessed: 12.50 per 1,000 person-days is 4,566 per 1,000 person-years
        raise ValueError("a rate is not written without the unit of its person-time")
    rows: list[tuple[str, str, str]] = []
    for part_id, label in (
        ("rate#0", "Rate, group compared" if two else "Rate"),
        ("rate#1", "Rate, reference group"),
    ):
        if part_id not in parts:
            continue
        rate = _named(parts[part_id])
        value, bounds = with_interval(
            rate["estimate"] * per,
            SimpleNamespace(lower=rate["lower"] * per, upper=rate["upper"] * per),
        )
        rows.append((f"{label}, per {per:,} {unit}", value, bounds.removeprefix("95% CI ")))
    lines: list[str] = []
    methods = [f"{first['method']} for the rate{'s' if two else ''}"]
    cited = ["garwood1936"]
    if "ratio" in parts:
        ratio = _named(parts["ratio"])
        if ratio["bounded"]:
            value, bounds = with_interval(
                ratio["estimate"], SimpleNamespace(lower=ratio["lower"], upper=ratio["upper"])
            )
            rows.append(("Rate ratio", value, bounds.removeprefix("95% CI ")))
        else:  # no event in the reference group: the lower end is what the data say
            rows.append(("Rate ratio", "no upper bound", f"at least {number(ratio['lower'])}"))
        lines += [
            f"p {p_text(ratio['p'])} ({ratio['test']} test, twice the smaller tail).",
            f"Rate difference: {number(ratio['difference'] * per)} per {per:,} {unit} (no interval is given for "
            "it).",
        ]
        methods.append(f"{ratio['method']} for the ratio")
        cited.append("clopper1934")
    lines.append("Intervals: " + "; ".join(methods) + ".")
    return ResultBlocks(
        tables=[("", pd.DataFrame(rows, columns=["Measure", "Estimate", _level(first)]))],
        lines=lines,
        warnings=[*record.warnings, RATES_NOTE],
        references=cite(*cited),
    )


def _standardised_block(record: Any) -> ResultBlocks:
    """Observed against expected, from its record: a mortality or an incidence ratio by what the record says the
    events are, the test named by its rule (R's ``poisson.test`` computes its p-value another way)."""
    from biostats.references import cite

    ratio = _named(record.part("ratio"))
    if ratio.get("events_are") not in ("deaths", "cases"):  # never guessed: it names the ratio
        raise ValueError("a standardised ratio is not written without what its events are")
    name = f"Standardised {'mortality' if ratio['events_are'] == 'deaths' else 'incidence'} ratio"
    value, bounds = with_interval(
        ratio["estimate"], SimpleNamespace(lower=ratio["lower"], upper=ratio["upper"])
    )
    return ResultBlocks(
        lines=[
            f"{name} = {value} ({bounds.replace('95% CI', _level(ratio), 1)}), p {p_text(ratio['p'])} (exact "
            f"Poisson test, twice the smaller tail); {ratio['observed']:,} observed, "
            f"{_as_typed(ratio['expected'])} expected."
        ],
        warnings=[*record.warnings, STANDARDISED_NOTE],
        references=cite("ulm1990", "garwood1936"),
    )


# The blocks written from a result's record, by kind (the ids are biostats.plan.KINDS'): one block per kind whatever
# its source, written from the record alone (it holds every number and every word its block needs), and the only
# blocks a document may write from disclosed numbers. The results of the dataset whose kind is not here keep
# :func:`result_blocks`.
RECORD_BLOCKS: dict[str, Callable[[Any], ResultBlocks]] = {
    "proportion": _share_block,
    "rates": _rates_block,
    "standardised_ratio": _standardised_block,
}


def record_blocks(record: Any) -> ResultBlocks:
    """The report blocks of a result from its record (``biostats.plan.ResultRecord``), for the kinds of
    ``RECORD_BLOCKS``."""
    build = RECORD_BLOCKS.get(record.kind)
    if build is None:
        raise ValueError(f"no block is written from the record of the {record.kind} kind")
    return build(record)


# The kinds computed from numbers typed that the dossier writes: those of RECORD_BLOCKS, and the 2x2 table typed,
# which goes through the accuracy block of the dataset's results (one rule for a side too small, never a copy).
TABLE_KIND = "diagnostic"
CALCULATION_KINDS: frozenset[str] = frozenset({*RECORD_BLOCKS, TABLE_KIND})


def result_title(key: str, value: Any, title: str) -> str:
    """
    The title of a result of the dataset in the dossier's list and in the Word file: its kind's ``title``, and for
    the accuracy of a test the form of its index test with it ("Diagnostic accuracy of a score", "... of a yes/no
    test"). A score and a yes/no test of the same columns are two items of the dossier, and read as one under one
    title; a table typed is named as the yes/no test it is ("... of a yes/no test (numbers typed)").
    """
    if key != TABLE_KIND or not isinstance(value, Mapping):
        return title
    from biostats.diagnostic import YES_NO_FORM

    form = "a yes/no test" if value.get("index_form") == YES_NO_FORM else "a score"
    return f"{title} of {form}"


def calculation_blocks(
    spec: Any, result: Any, record: Any, *, held: Collection[tuple[int, ...]] = ()
) -> ResultBlocks:
    """
    The report blocks of a calculation from numbers typed, computed again from its spec: from its record for a
    share, rates and a standardised ratio, which the dossier writes as typed.

    A 2x2 table typed is counted as people's, under the rule of a table of the dataset: with fewer than the
    dossier's threshold on a side of the reference standard, neither the table nor a measure is written. Nor are
    they when the table holds the four counts, in any order, of a table the dossier does not write (``held``: the
    sorted counts of each, an accuracy result of the dataset or another table typed): the plain copy of what is
    held back. A block that holds its table back says so (``held_numbers``), and the line of the numbers typed is
    not written either.

    For the full dossier only: the table is written from its result, through the accuracy block of the dataset's
    results. A document written from disclosed records (the package's sharing mode) must not call this: the kind
    stays out of ``RECORD_BLOCKS``, so that ``record_blocks`` refuses it, until the accuracy block reads records.
    """
    if spec.kind != TABLE_KIND:
        return record_blocks(record)
    blocks = _accuracy_blocks(
        {
            "acc": result,
            "typed": True,
            "prevalence": result.prevalence_assumed,
            "cautions": table_cautions(result),
        }
    )
    cells = tuple(sorted(int(cell) for cell in (result.tp, result.fp, result.fn, result.tn)))
    if not blocks.held_numbers and cells in set(held):
        return ResultBlocks(lines=[TABLE_OF_A_RESULT_HELD_BACK], held_numbers=True)
    return blocks


def result_blocks(value: Any) -> ResultBlocks:
    """The report blocks of one stored result (unknown types give a short text)."""
    from biostats._types import TestResult
    from biostats.agreement import SLOPE_READING, BlandAltman
    from biostats.linear import AncovaResult, LinearModelResult
    from biostats.rates import StratifiedResult, stratum_differences
    from biostats.regression import APPARENT_AUC, APPARENT_AUC_NOTE, RegressionResult
    from biostats.reliability import NEAR_MISSES, ICCResult, KappaResult
    from biostats.study_designs import DIFFERENCE, RATIO, design_named
    from biostats.survival import CoxResult, SurvivalComparison

    if hasattr(value, "m_total") and hasattr(value, "tests"):
        return _pooled_blocks(value)
    if isinstance(value, TestResult):
        return _test_blocks(value)
    if isinstance(value, CoxResult):
        blocks = ResultBlocks(tables=[_model_table(value.table)], references=list(value.references))
        line = f"n = {value.n:,}, events = {value.events:,}, ties: {value.ties}"
        if value.concordance is not None:
            line += f", concordance = {_num(value.concordance)}"
        blocks.lines.append(line)
        if value.ph_test is not None:
            blocks.tables.append(
                (
                    "Proportional-hazards test (Grambsch-Therneau)",
                    pd.DataFrame(
                        {
                            "Term": value.ph_test["term"],
                            "Statistic": [_num(v) for v in value.ph_test["statistic"]],
                            "p": [_p(v) for v in value.ph_test["p"]],
                        }
                    ),
                )
            )
        blocks.warnings.extend(value.warnings)
        return blocks
    if isinstance(value, RegressionResult):
        blocks = ResultBlocks(tables=[_model_table(value.table)], references=list(value.references))
        line = f"{value.kind.replace('_', ' ')} regression: n = {value.n:,}"
        if value.events is not None:
            line += f", events = {value.events:,}"
        if value.auc is not None:
            line += f", {APPARENT_AUC} = {_num(value.auc)}" + (
                f" (95% CI {_ci(value.auc_ci)})" if value.auc_ci else ""
            )
        blocks.lines.append(line)
        if value.auc is not None:
            blocks.lines.append(APPARENT_AUC_NOTE)
        blocks.warnings.extend(value.warnings)
        return blocks
    if isinstance(value, AncovaResult | LinearModelResult):
        type2 = value.type2
        blocks = ResultBlocks(
            tables=[
                (
                    "Type II tests",
                    pd.DataFrame(
                        {
                            "Source": type2["source"],
                            "df": [_num(v, 4) for v in type2["df"]],
                            "F": [_num(v) for v in type2["F"]],
                            "p": [_p(v) for v in type2["p"]],
                            "Partial eta squared": [_num(v) for v in type2["partial_eta2"]],
                        }
                    ),
                )
            ],
            references=list(value.references),
        )
        means = getattr(value, "adjusted_means", None)
        if means is not None:
            blocks.tables.append(
                (
                    "Adjusted means",
                    pd.DataFrame(
                        {
                            "Group": means["group"],
                            "Mean": [_num(v) for v in means["mean"]],
                            "95% CI": [
                                f"{_num(a)} to {_num(b)}"
                                for a, b in zip(means["ci_low"], means["ci_high"], strict=True)
                            ],
                        }
                    ),
                )
            )
        blocks.lines.append(
            f"n = {value.n:,}, R2 = {_num(value.r2)}, adjusted R2 = {_num(value.r2_adj)}"
        )
        blocks.warnings.extend(value.warnings)
        return blocks
    if isinstance(value, SurvivalComparison):
        return _survival_blocks(value)
    if isinstance(value, StratifiedResult):
        blocks = ResultBlocks(references=list(value.references))
        # which group stands above the line: an odds ratio of 0.63 is 1.59 read the other way round
        compared, reference = getattr(value, "compared", ""), getattr(value, "reference", "")
        if compared and reference:
            event = getattr(value, "event", "")
            blocks.lines.append(
                f"Ratios of {compared} against {reference} (the reference group)"
                + (f"; event: {event}" if event else "")
            )
        strata = f"{value.n_strata} strata"
        tested = getattr(value, "n_tested", None)
        if tested is not None and tested < value.n_strata:  # by their number: one is no two to compare
            strata += (
                f", of which the two tests read the {tested} that hold both groups and both outcomes"
                if tested >= 2
                else ", of which one holds both groups and both outcomes: the test of the association reads "
                "it, and whether the strata differ needs two"
                if tested == 1
                else ", none of which holds both groups and both outcomes"
            )
        # how the participants entered the study, and only the measures it supports: never the "risk ratio" of a
        # case-control sample; a result kept before the design was asked is written as it was then
        design = getattr(value, "design", "")
        sampling = design_named(design)
        blocks.lines.append(
            "Study design not stated: odds ratios only"  # one colon, as the page's sentence
            if design == "unknown"
            else f"Study design: {DESIGN_WORDS[design]}"
            if design
            else "Study design: not recorded (computed before the design was asked)"
        )
        lines = [
            f"Mantel-Haenszel odds ratio = {_num(value.or_mh)} (95% CI {_ci(value.or_mh_ci)}); crude OR = {_num(value.crude_or)}"
        ]
        if sampling.supports(RATIO):
            prevalence = sampling.shares == "prevalence"
            ratio, short = ("prevalence ratio", "PR") if prevalence else ("risk ratio", "RR")
            lines.append(
                f"Mantel-Haenszel {ratio} = {_num(value.rr_mh)} (95% CI {_ci(value.rr_mh_ci)}); crude {short} = "
                f"{_num(value.crude_rr)}"
            )
            difference = getattr(value, "rd_mh_ci", None)
            rd = float(getattr(value, "rd_mh", float("nan")))
            if design and sampling.supports(DIFFERENCE) and math.isfinite(rd):
                # in percentage points, as the page: a proportion here (-0.106) beside "-9.9 points" there
                words = "prevalence difference" if prevalence else "risk difference"
                interval = (
                    "no interval: every stratum lies at the edge, a difference of -100 or +100 points, where "
                    "Sato's variance is 0"
                    if difference is None
                    else f"95% CI {_points(difference.lower)} to {_points(difference.upper)}; Sato's variance"
                )
                line = f"Mantel-Haenszel {words} = {_points(rd)} percentage points ({interval})"
                own = stratum_differences(value.per_stratum)
                if len(own) >= 2:
                    low, high = _points(own.min()), _points(own.max())
                    line += (
                        f". The strata's own are {'both' if len(own) == 2 else 'all'} {low} points"
                        if low == high
                        else f". The strata's own run from {low} to {high} points: one combined difference sums "
                        f"them up only when they are alike; with a common {ratio}, differences change with each "
                        f"stratum's {'prevalence' if prevalence else 'risk'}"
                    )
                lines.append(line)
        tests = (
            f"Cochran-Mantel-Haenszel test of the association: chi2 = {_num(value.mh_test_statistic)}, "
            f"{_peq(value.mh_test_p)}; Breslow-Day test that the odds ratios of the strata are alike: chi2 = "
            f"{_num(value.breslow_day_statistic)}, {_peq(value.breslow_day_p)}"
        )
        woolf = getattr(value, "woolf_p", float("nan"))
        if design and sampling.supports(RATIO) and math.isfinite(woolf):
            noun = "prevalence ratios" if sampling.shares == "prevalence" else "risk ratios"
            read = getattr(value, "n_woolf", None)
            tests += (
                f"; Woolf test that the {noun} of the strata are alike: chi2 = {_num(value.woolf_statistic)}, "
                f"{_peq(woolf)}"
                + (f", on {read} strata (those with an event in both groups)" if read is not None else "")
            )
            # the analysis's strata named as such: after Woolf's, "3 strata" read as the test's
            strata = f"the analysis: {strata}"
        lines.append(f"{tests}; {strata}")
        blocks.lines.extend(lines)
        blocks.warnings.extend(value.warnings)
        return blocks
    if isinstance(value, BlandAltman):
        blocks = ResultBlocks(references=list(value.references))
        # the slope with its two readings, as the page: "Proportional bias" named one of them
        slope = (
            f"Slope of the difference on the mean: {_num(value.proportional_bias_slope)} per unit, "
            f"{_peq(value.proportional_bias_p)}"
        )
        if value.proportional_bias_p < 0.05:
            slope += f". The difference changes with the size of the measurement: {SLOPE_READING}"
        blocks.lines.extend(
            [
                f"n = {value.n:,}; mean difference = {_num(value.bias)} (95% CI {_ci(value.bias_ci)}), SD of the "
                f"differences = {_num(value.sd)}",
                f"Limits of agreement: {_num(value.loa_lower)} (95% CI {_ci(value.loa_lower_ci)}) to "
                f"{_num(value.loa_upper)} (95% CI {_ci(value.loa_upper_ci)}); {_num(value.within_loa_pct)}% of the "
                "differences within",
                slope,
            ]
        )
        # the line above says it with its numbers; a result kept before holds the warning's first words
        blocks.warnings.extend(
            text
            for text in value.warnings
            if not text.startswith(("the difference changes with the size", "proportional bias"))
        )
        return blocks
    if isinstance(value, ICCResult):
        return ResultBlocks(
            lines=[
                f"ICC = {_num(value.icc)} (95% CI {_ci(value.ci)}); {value.model}; "
                f"{value.n_subjects:,} participants, {value.n_raters} columns",
                f"F = {_num(value.f_statistic)}, df = {value.df1:.0f}, {value.df2:.0f}, {_peq(value.p_value)}",
            ],
            references=list(value.references),
        )
    if isinstance(value, dict):
        if isinstance(value.get("kappa"), KappaResult) and "table" in value:
            # the order of the grades is part of a weighted kappa: written with it
            kappa, order = value["kappa"], [str(name) for name in value.get("order") or ()]
            weights = value.get("weights")
            name = f"Weighted kappa ({weights} weights)" if weights else "Cohen's kappa"
            counts = value["table"].to_numpy()
            one_category = 1 in (int((counts.sum(axis=1) > 0).sum()), int((counts.sum(axis=0) > 0).sum()))
            lines = [
                # without a standard error there is no interval ("95% CI -1 to 1" was written for a kappa of 0);
                # read on the table, so that a result kept before says it too
                f"{name} = {_num(kappa.kappa)} "
                + (
                    "(no interval: one rater uses one category only)"
                    if one_category
                    else f"(95% CI {_ci(kappa.ci)})"
                    if math.isfinite(kappa.ci.lower) and math.isfinite(kappa.ci.upper)
                    else "(no interval on this table)"
                )
                + f"; {kappa.n:,} participants rated by both",
                f"Same category for {_num(100 * kappa.observed_agreement)}% of them "
                f"({_num(100 * kappa.expected_agreement)}% expected by chance)",
            ]
            if weights:
                # the two shares give the unweighted kappa (0.706 beside a weighted 0.824): said with them
                lines.append(NEAR_MISSES)
            if weights and order:
                lines.append("Order of the grades, lowest first: " + ", ".join(order))
            # how the ratings were read as categories (texts as numbers, or codes kept apart), as the page says
            lines.extend(str(note).replace("`", "") for note in value.get("notes") or ())
            return ResultBlocks(
                lines=lines,
                tables=[("Ratings (rows: first rater; columns: second rater)", value["table"].reset_index())],
                references=list(kappa.references),
            )
        if "acc" in value and "auc" in value:
            return _accuracy_blocks(value)
        if "res" in value and "table" in value:
            blocks = _test_blocks(value["res"])
            blocks.tables.insert(0, ("Paired table", value["table"].reset_index()))
            return blocks
        if "results" in value and "errors" in value:
            return _pooled_comparison_blocks(value)
        if "results" in value:
            return _univariate_blocks(value)
    return ResultBlocks(lines=[f"{type(value).__name__} (no report layout)"])


__all__ = [
    "CALCULATION",
    "DOSSIER_MIN_N",
    "RATES_NOTE",
    "RECORD_BLOCKS",
    "SAMPLING_WORDS",
    "SOURCE_NOT_GIVEN",
    "TWO_GATE_SPECTRUM",
    "TYPED_SOURCE",
    "ResultBlocks",
    "dossier_min_n",
    "km_png",
    "record_blocks",
    "result_blocks",
    "sampling_line",
    "settings_written",
    "source_line",
    "typed_numbers",
    "typed_title",
]
