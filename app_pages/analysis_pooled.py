"""
Analyze page with multiply imputed data (follow-up programme M2): group comparisons and ANCOVA run on the m
imputed datasets of the analysis population and pooled with Rubin's rules (biostats.mi.analyze_mi).

Group comparison, per variable:
- numeric, two groups: Welch t-test, the mean difference pooled (Barnard-Rubin df);
- numeric, three or more groups: a linear model on the groups, the group means pooled and the group effect
  tested by D1 (pooled Wald F; equal variances, as in a one-way ANOVA);
- two categories, two groups: odds ratio and risk ratio (log scale) and risk difference pooled, the
  chi-square by D2;
- other categorical variables: the chi-square test of independence by D2.
Rank tests and Fisher's exact test have no pooling rule: the page gives the reason and the alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from biostats.contingency import (
    TwoByTwo,
    analyze_twobytwo,
    chi_square_independence,
    detect_event_level,
    detect_reference_level,
)
from biostats.inference import compare_groups
from biostats.linear import ancova
from biostats.mi import analyze_mi, not_poolable
from biostats.multiplicity import UI_OPTIONS as CORRECTION_OPTIONS
from biostats.multiplicity import adjust as adjust_pvalues
from ui.components import error_card, result_card
from ui.components.mi_mode import mi_token, show_mi_notes, show_pooled_terms, show_tests
from ui.progress import mark_step
from ui.results import provenance, recall, remember
from ui.state import get_state

MAX_GROUP_LEVELS = 10
NOT_POOLED = (
    ("mann_whitney", "Mann-Whitney U"),
    ("kruskal_wallis", "Kruskal-Wallis"),
    ("fisher_exact", "Fisher's exact test"),
)
ANALYSIS_NAMES = {
    "welch": "Welch t-test",
    "anova": "Linear model on the groups (D1)",
    "twobytwo": "2x2: odds ratio (Wald on the pooled log OR)",
    "chisq": "Chi-square test of independence (D2)",
}


@dataclass
class Pooled:
    """One variable compared across the groups: the kind of analysis, the pooled result and its orientation."""

    kind: str
    result: Any  # biostats.mi.analyze.MIResult
    reference: str | None = None
    comparison: str | None = None
    event: str | None = None


def _levels(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique())


def _in_groups(data: pd.DataFrame, group: str, levels: list[str]) -> pd.DataFrame:
    return data[data[group].astype(str).isin(levels)]


def _columns(mi, analyzer) -> tuple[list[str], list[str], list[str]]:
    """Numeric and categorical columns of the imputed data, and the grouping candidates."""
    in_base = set(mi.base.columns)
    numeric = [c for c in analyzer.numeric_cols if c in in_base]
    categorical = [c for c in dict.fromkeys([*analyzer.categorical_cols, *analyzer.binary_cols]) if c in in_base]
    groupers = [
        c for c in dict.fromkeys([*categorical, *numeric]) if 2 <= mi.base[c].nunique(dropna=True) <= MAX_GROUP_LEVELS
    ]
    return numeric, categorical, groupers


def compare_pooled(mi, group: str, levels: list[str], target: str, *, numeric: bool) -> Pooled:
    """Compare ``target`` across the ``levels`` of ``group`` on every imputed dataset and pool the result."""
    columns = [group, target]
    if numeric and len(levels) == 2:
        reference = detect_reference_level(levels) or levels[0]
        comparison = next(lvl for lvl in levels if lvl != reference)

        def welch(data: pd.DataFrame):
            data = data.dropna(subset=columns)
            key = data[group].astype(str)
            groups = {lvl: data.loc[key == lvl, target] for lvl in (reference, comparison)}
            return compare_groups(groups, kind="welch", reference=reference, effect_sizes=False)

        return Pooled("welch", analyze_mi(welch, mi, columns=columns), reference, comparison)
    if numeric:
        return Pooled(
            "anova",
            analyze_mi(
                lambda data: ancova(_in_groups(data, group, levels), target, group, levels={group: levels}),
                mi,
                columns=columns,
            ),
            reference=levels[0],
        )
    outcome_levels = _levels(mi.base[target])
    if len(levels) == 2 and len(outcome_levels) == 2:
        reference = detect_reference_level(levels) or levels[0]
        event = detect_event_level(outcome_levels) or outcome_levels[-1]
        return Pooled(
            "twobytwo",
            analyze_mi(
                lambda data: analyze_twobytwo(
                    TwoByTwo.from_crosstab(
                        _in_groups(data, group, levels), group, target, exposure_ref=reference, event_level=event
                    )
                ),
                mi,
                columns=columns,
            ),
            reference,
            next(lvl for lvl in levels if lvl != reference),
            event,
        )
    return Pooled(
        "chisq",
        analyze_mi(
            lambda data: chi_square_independence(
                _in_groups(data, group, levels), group, target, levels={group: levels, target: outcome_levels}
            ),
            mi,
            columns=columns,
        ),
    )


def _summary_row(target: str, item: Pooled, group: str) -> dict[str, Any]:
    res = item.result
    row: dict[str, Any] = {"variable": target, "analysis": ANALYSIS_NAMES[item.kind], "estimate_label": ""}
    if item.kind == "welch":
        test = res.extra["result"]
        row |= {
            "estimate_label": f"mean difference ({item.comparison} - {item.reference})",
            "estimate": test.estimate,
            "ci_low": test.estimate_ci.lower,
            "ci_high": test.estimate_ci.upper,
            "test": f"t = {test.statistic:.2f}, df = {test.df[0]:.1f}",
            "p": test.p_value,
            "fmi": test.mi.fmi,
        }
    elif item.kind == "anova":
        d1 = res.tests[group]
        row |= {
            "test": f"F({d1.df1:.0f}, {d1.df2:.1f}) = {d1.statistic:.2f}",
            "p": d1.p_value,
            "fmi": max(t.fmi for t in res.table.terms if t.term != "const"),
        }
    elif item.kind == "twobytwo":
        term = res.term("odds_ratio")
        row |= {
            "estimate_label": f"odds ratio of {item.event} ({item.comparison} vs {item.reference})",
            "estimate": term.natural_estimate,
            "ci_low": term.natural_ci.lower,
            "ci_high": term.natural_ci.upper,
            "test": f"z = {term.statistic:.2f}, df = {term.df:.1f}",
            "p": term.p_value,
            "fmi": term.fmi,
        }
    else:
        d2 = res.tests["chi_square"]
        row |= {"test": f"F({d2.df1:.0f}, {d2.df2:.1f}) = {d2.statistic:.2f}", "p": d2.p_value}
    return row


def _show_summary(results: dict[str, Pooled], errors: dict[str, str], group: str, correction: str) -> None:
    rows = [_summary_row(target, item, group) for target, item in results.items()]
    rows += [{"variable": target, "analysis": "not computed", "note": message} for target, message in errors.items()]
    frame = pd.DataFrame(rows)
    if "p" in frame and correction != "None":
        frame.insert(frame.columns.get_loc("p") + 1, "p_adj", adjust_pvalues(frame["p"].tolist(), correction))
    number = st.column_config.NumberColumn
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "variable": "Variable",
            "analysis": "Pooled analysis",
            "estimate_label": st.column_config.TextColumn("Estimate", width="medium"),
            "estimate": number("Value", format="%.3f"),
            "ci_low": number("95% CI low", format="%.3f"),
            "ci_high": number("95% CI high", format="%.3f"),
            "test": "Test",
            "p": number("p", format="%.4f"),
            "p_adj": number(f"p ({correction})", format="%.4f"),
            "fmi": number("FMI", format="%.3f", help="Fraction of missing information (largest term for a factor)"),
            "note": st.column_config.TextColumn("Why not computed", width="large"),
        },
        alt="Pooled group comparisons",
    )
    if correction != "None":
        st.caption(f"Multiple testing: {correction}, over the variables tested in this run.")


def _twobytwo_frame(res) -> pd.DataFrame:
    frame = res.to_frame().set_index("term")
    rows = []
    for name, label in (
        ("odds_ratio", "Odds ratio"),
        ("relative_risk", "Risk ratio"),
        ("risk_difference", "Risk difference"),
    ):
        t = frame.loc[name]
        log = t["scale"] == "log"
        rows.append(
            {
                "measure": label,
                "estimate": t["natural_estimate"] if log else t["estimate"],
                "ci_low": t["natural_ci_low"] if log else t["ci_low"],
                "ci_high": t["natural_ci_high"] if log else t["ci_high"],
                "p": t["p_value"],
                "df": t["df"],
                "fmi": t["fmi"],
            }
        )
    return pd.DataFrame(rows)


def _show_complete_twobytwo(cc) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {"measure": "Odds ratio", "estimate": cc.odds_ratio, "ci_low": cc.odds_ratio_ci.lower,
                 "ci_high": cc.odds_ratio_ci.upper},
                {"measure": "Risk ratio", "estimate": cc.relative_risk, "ci_low": cc.relative_risk_ci.lower,
                 "ci_high": cc.relative_risk_ci.upper},
                {"measure": "Risk difference", "estimate": cc.risk_difference,
                 "ci_low": cc.risk_difference_ci.lower, "ci_high": cc.risk_difference_ci.upper},
            ]
        ).style.format({"estimate": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}"}),
        hide_index=True,
    )
    st.caption(f"n = {cc.n_total}; {cc.primary_test}, p = {cc.primary_pvalue:.4g}")


def _show_means(means: pd.DataFrame, *, label: str) -> None:
    number = st.column_config.NumberColumn
    st.dataframe(
        means,
        hide_index=True,
        column_config={
            "group": "Group",
            "mean": number(label, format="%.3f"),
            "se": number("SE", format="%.3f"),
            "ci_low": number("95% CI low", format="%.3f"),
            "ci_high": number("95% CI high", format="%.3f"),
            "df": number("df", format="%.1f"),
            "fmi": number("FMI", format="%.3f"),
        },
        alt=label,
    )


def _show_detail(target: str, item: Pooled, group: str) -> None:
    res = item.result
    if item.kind == "welch":
        result_card(res.extra["result"])
        show_mi_notes(res, complete_case=result_card)
    elif item.kind == "anova":
        _show_means(res.extra["adjusted_means"], label=f"Mean of {target} (pooled)")
        show_tests(res, title=f"Group effect of {group} (D1)")
        st.markdown(f"**Differences from {item.reference}**")
        show_pooled_terms(res, alt="Pooled differences between groups")
        st.caption(
            "With imputed data the group means are compared with a linear model (equal variances, as a one-way "
            "ANOVA): Welch's ANOVA and pairwise post-hoc tests have no pooling rule."
        )
        show_mi_notes(
            res,
            complete_case=lambda cc: _show_means(cc.adjusted_means, label=f"Mean of {target} (complete cases)"),
        )
    elif item.kind == "twobytwo":
        st.markdown(f"**{item.comparison} compared with {item.reference}; event: {target} = {item.event}**")
        number = st.column_config.NumberColumn
        st.dataframe(
            _twobytwo_frame(res),
            hide_index=True,
            column_config={
                "measure": "Measure",
                "estimate": number("Estimate", format="%.3f"),
                "ci_low": number("95% CI low", format="%.3f"),
                "ci_high": number("95% CI high", format="%.3f"),
                "p": number("p", format="%.4f"),
                "df": number("df", format="%.1f"),
                "fmi": number("FMI", format="%.3f"),
            },
            alt="Pooled measures of association",
        )
        show_tests(res, title="Chi-square test (D2)")
        show_mi_notes(res, complete_case=_show_complete_twobytwo)
    else:
        show_tests(res, title=f"Chi-square test of independence of {group} and {target} (D2)")
        show_mi_notes(
            res,
            complete_case=lambda cc: st.caption(
                f"chi2 = {cc.statistic:.2f}, df = {cc.df[0]:.0f}, p = {cc.p_value:.4g}, n = {cc.n_total}"
            ),
        )


def run_pooled_comparison(mi, analyzer) -> None:
    st.header("Group comparison, pooled over the imputed datasets")
    numeric, categorical, groupers = _columns(mi, analyzer)
    with st.expander("Analysis configuration", expanded=True):
        c1, c2 = st.columns(2)
        group = c1.selectbox(
            "Grouping variable",
            ["None", *groupers],
            key="mi_cmp_group",
            help=f"A categorical variable with 2 to {MAX_GROUP_LEVELS} levels in the analysis population.",
        )
        levels: list[str] = []
        if group != "None":
            all_levels = _levels(mi.base[group])
            levels = c1.multiselect("Groups to include", all_levels, default=all_levels, key=f"mi_cmp_levels::{group}")
        candidates = [c for c in [*numeric, *categorical] if c != group]
        targets = c2.multiselect(
            "Variables to compare",
            candidates,
            default=[c for c in numeric if c != group][:5],
            key="mi_cmp_targets",
            help="Variables created after the imputation are not in the imputed datasets.",
        )
        correction = c2.selectbox("Multiple testing correction", list(CORRECTION_OPTIONS), key="mi_cmp_correction")
        st.caption(
            "Numeric variables: Welch t-test (two groups) or a linear model with a D1 test (three or more groups). "
            "Categorical variables: odds ratio, risk ratio and risk difference (two groups, two categories) or the "
            "chi-square test by D2."
        )
    with st.expander("Rank tests and Fisher's exact test are not pooled"):
        for key, label in NOT_POOLED:
            entry = not_poolable(key)
            if entry is not None:
                st.markdown(f"**{label}.** {entry.reason} {entry.alternative}")
    if group == "None" or not targets:
        st.info("Choose the grouping variable and the variables to compare, then run the pooled analysis.")
        return
    if len(levels) < 2:
        st.warning("Select at least two groups.")
        return

    config = (group, tuple(levels), tuple(targets), correction, mi_token(get_state()))
    if st.button("Run pooled analysis", type="primary", key="mi_cmp_run"):
        results: dict[str, Pooled] = {}
        errors: dict[str, str] = {}
        bar = st.progress(0.0, text="Pooling")
        for i, target in enumerate(targets):
            try:
                results[target] = compare_pooled(mi, group, levels, target, numeric=target in numeric)
            except Exception as exc:
                errors[target] = f"{type(exc).__name__}: {exc}"
            bar.progress((i + 1) / len(targets), text=f"Pooling: {target}")
        bar.empty()
        remember("univariate_mi", {"results": results, "errors": errors}, config=config, n_analysed=len(mi.base))
        mark_step("analyse", f"pooled group comparison of {len(targets)} variable(s) by {group}")
    stored = recall("univariate_mi", config=config)
    if stored is None:
        st.info("Click Run pooled analysis.")
        return
    st.subheader("Summary")
    _show_summary(stored["results"], stored["errors"], group, correction)
    provenance("univariate_mi")
    if stored["results"]:
        st.subheader("Detail")
        target = st.selectbox("Variable", list(stored["results"]), key="mi_cmp_detail")
        _show_detail(target, stored["results"][target], group)


def run_pooled_ancova(mi, analyzer) -> None:
    st.header("ANCOVA, pooled over the imputed datasets")
    numeric, _, groupers = _columns(mi, analyzer)
    if not numeric or not groupers:
        st.info("ANCOVA needs a numeric outcome and a grouping variable.")
        return
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target variable (Y)", numeric, key="mi_anc_target")
    group = c2.selectbox("Grouping variable (factor)", [g for g in groupers if g != target], key="mi_anc_group")
    covariates = c3.multiselect(
        "Covariates (confounders)", [c for c in numeric if c not in (target, group)], key="mi_anc_covs"
    )
    config = (target, group, tuple(covariates), mi_token(get_state()))
    if st.button("Run pooled ANCOVA", type="primary", key="mi_anc_run"):
        try:
            levels = mi.levels([group, *covariates]) | {group: _levels(mi.base[group])}
            res = analyze_mi(
                lambda data: ancova(data, target, group, covariates, levels=levels),
                mi,
                columns=[target, group, *covariates],
            )
            remember("ancova_mi", res, config=config, n_analysed=len(mi.base))
            mark_step("analyse", f"pooled ANCOVA of {target}")
        except Exception as exc:
            error_card("The pooled ANCOVA could not be computed.", exc)
    res = recall("ancova_mi", config=config)
    if res is None:
        return
    d1 = res.tests.get(group)
    if d1 is not None:
        st.metric(
            f"Group effect of {group}, adjusted (D1)",
            f"p = {d1.p_value:.4g}",
            help=f"F({d1.df1:.0f}, {d1.df2:.1f}) = {d1.statistic:.3f}; pooled Wald test of the group terms.",
        )
    st.markdown("**Adjusted means (pooled least-squares means)**")
    _show_means(res.extra["adjusted_means"], label=f"Adjusted mean of {target}")
    show_tests(res, title="Type II tests: one per covariate (D1)")
    st.markdown("**Coefficients**")
    show_pooled_terms(res, alt="Pooled ANCOVA coefficients")
    r2 = res.extra.get("r2")
    if r2 is not None:
        ci = f" (95% CI {r2.ci.lower:.3f} to {r2.ci.upper:.3f})" if r2.ci is not None else ""
        st.caption(f"R2 pooled with Harel's method: {r2.estimate:.3f}{ci}. Partial eta squared is not pooled.")

    def complete_case(cc) -> None:
        _show_means(cc.adjusted_means, label=f"Adjusted mean of {target} (complete cases)")
        st.dataframe(
            cc.type2.style.format({"df": "{:.0f}", "sum_sq": "{:.2f}", "F": "{:.3f}", "p": "{:.4f}",
                                   "partial_eta2": "{:.3f}"}, na_rep=""),
            hide_index=True,
        )

    show_mi_notes(res, complete_case=complete_case)
    provenance("ancova_mi")


__all__ = ["Pooled", "compare_pooled", "run_pooled_ancova", "run_pooled_comparison"]
