"""
The results of the study dossier as report blocks: tables, lines of text, references and, for survival
analyses, a Kaplan-Meier figure. One function, :func:`result_blocks`, reads every type of result the analysis
pages store (biostats results, pooled results over imputed datasets, the dictionaries of a few pages) and
returns plain data that the Word builder (:mod:`manage.dossier`) lays out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import pandas as pd

ESTIMATE_COLUMNS = {"hr": "hazard ratio", "odds_ratio": "odds ratio", "rate_ratio": "rate ratio"}


@dataclass
class ResultBlocks:
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    figure: bytes | None = None  # PNG
    warnings: list[str] = field(default_factory=list)


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
    return "not reached" if value is None else _num(value)


def _survival_blocks(res: Any) -> ResultBlocks:
    table = pd.DataFrame(
        [
            {"Group": c.label, "n": c.n, "Events": c.events, "Median survival": _median(c.median)}
            for c in res.curves
        ]
    )
    blocks = ResultBlocks(tables=[("Kaplan-Meier", table)], references=list(res.references))
    if res.logrank_statistic is not None:
        blocks.lines.append(
            f"Log-rank test: chi2 = {_num(res.logrank_statistic)}, df = {res.logrank_df}, {_peq(res.logrank_p)}"
        )
    blocks.figure = km_png([(c.label, c.times, c.survival) for c in res.curves])
    blocks.warnings.extend(res.warnings)
    return blocks


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
        blocks.tables.append(
            (
                "Kaplan-Meier, pooled (log(-log) scale)",
                pd.DataFrame(
                    [
                        {
                            "Group": label,
                            "Median survival (pooled curve)": _median(medians.get(label)),
                            "Survival at last time": _num(curve["survival"].iloc[-1]),
                        }
                        for label, curve in curves.items()
                    ]
                ),
            )
        )
        blocks.figure = km_png([(label, c["time"], c["survival"]) for label, c in curves.items()])
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
            "Test": frame["Test Used"],
            "Estimate": [_num(v) for v in estimates],
            "Estimate CI": frame["Estimate CI"] if "Estimate CI" in frame.columns else "",
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
        if item.kind == "welch":
            test = res.extra["result"]
            rows.append(
                {"Variable": target, "Analysis": "Welch t-test", "Result": _test_line(test)}
            )
            blocks.references.extend(test.references)
        elif item.kind == "anova":
            group = next(iter(res.tests))
            d1 = res.tests[group]
            rows.append(
                {
                    "Variable": target,
                    "Analysis": "Linear model on the groups (D1)",
                    "Result": f"F = {_num(d1.statistic)}, df = {_num(d1.df1, 4)}, {_num(d1.df2, 4)}; {_peq(d1.p_value)}",
                }
            )
        elif item.kind == "twobytwo":
            t = res.term("odds_ratio")
            rows.append(
                {
                    "Variable": target,
                    "Analysis": "Odds ratio",
                    "Result": f"OR = {_num(t.natural_estimate)} (95% CI {_ci(t.natural_ci)}); {_peq(t.p_value)}",
                }
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


def result_blocks(value: Any) -> ResultBlocks:
    """The report blocks of one stored result (unknown types give a short text)."""
    from biostats._types import TestResult
    from biostats.agreement import BlandAltman
    from biostats.linear import AncovaResult, LinearModelResult
    from biostats.rates import StratifiedResult
    from biostats.regression import RegressionResult
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
            line += f", AUC = {_num(value.auc)}" + (
                f" (95% CI {_ci(value.auc_ci)})" if value.auc_ci else ""
            )
        blocks.lines.append(line)
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
        blocks.lines.extend(
            [
                f"Mantel-Haenszel odds ratio = {_num(value.or_mh)} (95% CI {_ci(value.or_mh_ci)}); crude OR = {_num(value.crude_or)}",
                f"Mantel-Haenszel risk ratio = {_num(value.rr_mh)} (95% CI {_ci(value.rr_mh_ci)})",
                f"Mantel-Haenszel test: chi2 = {_num(value.mh_test_statistic)}, {_peq(value.mh_test_p)}; "
                f"Breslow-Day homogeneity: chi2 = {_num(value.breslow_day_statistic)}, {_peq(value.breslow_day_p)}; "
                f"{value.n_strata} strata",
            ]
        )
        blocks.warnings.extend(value.warnings)
        return blocks
    if isinstance(value, BlandAltman):
        blocks = ResultBlocks(references=list(value.references))
        blocks.lines.extend(
            [
                f"n = {value.n:,}; bias = {_num(value.bias)} (95% CI {_ci(value.bias_ci)}), SD = {_num(value.sd)}",
                f"Limits of agreement: {_num(value.loa_lower)} (95% CI {_ci(value.loa_lower_ci)}) to "
                f"{_num(value.loa_upper)} (95% CI {_ci(value.loa_upper_ci)}); {_num(value.within_loa_pct)}% within",
                f"Proportional bias: slope = {_num(value.proportional_bias_slope)}, {_peq(value.proportional_bias_p)}",
            ]
        )
        blocks.warnings.extend(value.warnings)
        return blocks
    if isinstance(value, dict):
        if "acc" in value and "auc" in value:
            acc, auc = value["acc"], value["auc"]
            table = pd.DataFrame(
                [
                    ("AUC (DeLong)", _num(auc.get("auc")), _ci(auc.get("ci"))),
                    ("Sensitivity", _num(acc.sensitivity), _ci(acc.sensitivity_ci)),
                    ("Specificity", _num(acc.specificity), _ci(acc.specificity_ci)),
                    ("PPV", _num(acc.ppv), _ci(acc.ppv_ci)),
                    ("NPV", _num(acc.npv), _ci(acc.npv_ci)),
                    ("LR+", _num(acc.lr_positive), _ci(acc.lr_positive_ci)),
                    ("LR-", _num(acc.lr_negative), _ci(acc.lr_negative_ci)),
                ],
                columns=["Measure", "Estimate", "95% CI"],
            )
            blocks = ResultBlocks(
                tables=[
                    (
                        f"{value.get('score')} for {value.get('outcome')} (cut-off {_num(value.get('cutoff'))})",
                        table,
                    )
                ],
                references=list(acc.references) + list(auc.get("references") or []),
            )
            blocks.warnings.extend(acc.warnings)
            return blocks
        if "res" in value and "table" in value:
            blocks = _test_blocks(value["res"])
            blocks.tables.insert(0, ("Paired table", value["table"].reset_index()))
            return blocks
        if "results" in value and "errors" in value:
            return _pooled_comparison_blocks(value)
        if "results" in value:
            return _univariate_blocks(value)
    return ResultBlocks(lines=[f"{type(value).__name__} (no report layout)"])


__all__ = ["ResultBlocks", "km_png", "result_blocks"]
