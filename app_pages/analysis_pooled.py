"""
Analyze page with multiply imputed data (follow-up programme M2): group comparisons and ANCOVA run on the m
imputed datasets of the analysis population and pooled with Rubin's rules (biostats.mi.analyze_mi).

Group comparison, per variable:
- numeric, two groups: Welch t-test, the mean difference pooled (Barnard-Rubin df);
- numeric, three or more groups: a linear model on the groups with robust (HC3) variances, the group means
  pooled and the group effect tested by D1 (pooled Wald F). Like Welch's ANOVA on the observed data, it does not
  assume equal variances; Welch's ANOVA itself has no pooling rule;
- the analysis can be chosen instead (*Choose the method*): Student's t-test for two groups, the equal-variance
  linear model (one-way ANOVA) for three or more, each with a warning;
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
from ui.components.levels import level_warning, show_level
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
    "student": "Student t-test (equal variances, chosen)",
    "hc3": "Linear model on the groups, robust variances (HC3; D1)",
    "anova": "Linear model on the groups, equal variances (chosen; D1)",
    "twobytwo": "2x2: odds ratio (Wald on the pooled log OR)",
    "chisq": "Chi-square test of independence (D2)",
}
# The pooled analyses a user can choose instead of the recommended one (numeric variable, two groups or three or
# more); the others have a single pooled analysis, and rank tests and Fisher's exact test none (NOT_POOLED)
TWO_GROUP_TESTS = {"welch": "Welch t-test (recommended)", "student": "Student t-test (equal variances)"}
STUDENT_WARNING = (
    "Student's t-test assumes equal variances in the two groups. Welch's t-test does not, and loses little power "
    "when they are equal (Delacre 2017): it is the recommended test. Report why you chose Student's."
)
MORE_GROUP_TESTS = {
    "hc3": "Linear model, robust variances (HC3, recommended)",
    "anova": "Linear model, equal variances (one-way ANOVA)",
}
ANOVA_WARNING = (
    "The one-way ANOVA assumes equal variances in the groups. The linear model with robust (HC3) variances does "
    "not, as Welch's ANOVA does not on the observed data, and stays valid when they are equal: it is the "
    "recommended analysis. Report why you chose the equal-variance model."
)


@dataclass
class Pooled:
    """One variable compared across the groups: the kind of analysis, the pooled result and its orientation."""

    kind: str
    result: Any  # biostats.mi.analyze.MIResult
    reference: str | None = None
    comparison: str | None = None
    event: str | None = None
    haldane: str = "if_zero"  # 2x2 tables: "always" when 0.5 was added to every cell (a sensitivity analysis)


def _levels(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique())


def _in_groups(data: pd.DataFrame, group: str, levels: list[str]) -> pd.DataFrame:
    return data[data[group].astype(str).isin(levels)]


def _two_per_group(data: pd.DataFrame, group: str, levels: list[str]) -> None:
    """The robust (HC3) model gives each group its own variance, which a group of one value does not have."""
    sizes = data[group].astype(str).value_counts()
    single = [show_level(level) for level in levels if sizes.get(level, 0) == 1]
    if single:
        has = "has a single value" if len(single) == 1 else "have a single value each"
        raise ValueError(
            f"{', '.join(single)} {has}, while the robust (HC3) model needs two or more per group: deselect it in "
            "Groups to include or merge it with another group (it may be a spelling variant), or choose the "
            "equal-variance model."
        )


def _columns(mi, analyzer) -> tuple[list[str], list[str], list[str]]:
    """Numeric and categorical columns of the imputed data, and the grouping candidates."""
    in_base = set(mi.base.columns)
    numeric = [c for c in analyzer.numeric_cols if c in in_base]
    categorical = [c for c in dict.fromkeys([*analyzer.categorical_cols, *analyzer.binary_cols]) if c in in_base]
    groupers = [
        c for c in dict.fromkeys([*categorical, *numeric]) if 2 <= mi.base[c].nunique(dropna=True) <= MAX_GROUP_LEVELS
    ]
    return numeric, categorical, groupers


def compare_pooled(
    mi,
    group: str,
    levels: list[str],
    target: str,
    *,
    numeric: bool,
    haldane: str = "if_zero",
    two_groups: str = "welch",
    more_groups: str = "hc3",
) -> Pooled:
    """
    Compare ``target`` across the ``levels`` of ``group`` on every imputed dataset and pool the result.

    ``haldane="always"`` adds 0.5 to every cell of every 2x2 table (a sensitivity analysis): without it, a zero
    cell in some imputed datasets leaves the odds and risk ratios unpooled. ``two_groups`` is the t-test of a
    numeric variable in two groups: Welch's (recommended) or Student's (equal variances), both pooled the same way.
    ``more_groups`` is the linear model of a numeric variable in three groups or more: with robust HC3 variances
    (recommended) or equal variances (one-way ANOVA), both tested by D1.
    """
    if two_groups not in TWO_GROUP_TESTS:
        raise ValueError(f"two_groups is one of {list(TWO_GROUP_TESTS)}, not {two_groups!r}")
    if more_groups not in MORE_GROUP_TESTS:
        raise ValueError(f"more_groups is one of {list(MORE_GROUP_TESTS)}, not {more_groups!r}")
    columns = [group, target]
    if numeric and len(levels) == 2:
        reference = detect_reference_level(levels) or levels[0]
        comparison = next(lvl for lvl in levels if lvl != reference)

        def t_test(data: pd.DataFrame):
            data = data.dropna(subset=columns)
            key = data[group].astype(str)
            groups = {lvl: data.loc[key == lvl, target] for lvl in (reference, comparison)}
            res = compare_groups(groups, kind=two_groups, reference=reference, effect_sizes=False)
            # the page says which test was chosen, and why Welch is recommended: no note on every dataset
            return res.model_copy(update={"warnings": [w for w in res.warnings if "selected manually" not in w]})

        return Pooled(two_groups, analyze_mi(t_test, mi, columns=columns), reference, comparison)
    if numeric:
        cov_type = "HC3" if more_groups == "hc3" else "nonrobust"

        def linear_model(data: pd.DataFrame):
            data = _in_groups(data, group, levels)
            if cov_type == "HC3":
                _two_per_group(data.dropna(subset=columns), group, levels)
            return ancova(data, target, group, levels={group: levels}, cov_type=cov_type)

        return Pooled(more_groups, analyze_mi(linear_model, mi, columns=columns), reference=levels[0])
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
                    ),
                    haldane="always" if haldane == "always" else "if_zero",
                ),
                mi,
                columns=columns,
            ),
            reference,
            next(lvl for lvl in levels if lvl != reference),
            event,
            haldane="always" if haldane == "always" else "if_zero",
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
    if item.kind in ("welch", "student"):
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
    elif item.kind in ("hc3", "anova"):
        d1 = res.tests[group]
        row |= {
            "test": f"F({d1.df1:.0f}, {d1.df2:.1f}) = {d1.statistic:.2f}",
            "p": d1.p_value,
            "fmi": max(t.fmi for t in res.table.terms if t.term != "const"),
        }
    elif item.kind == "twobytwo" and "odds_ratio" in res.table.names:
        term = res.term("odds_ratio")
        if item.haldane == "always":
            row["analysis"] = "2x2: odds ratio, 0.5 added to every cell (sensitivity analysis)"
        row |= {
            "estimate_label": f"odds ratio of {item.event} ({item.comparison} vs {item.reference})",
            "estimate": term.natural_estimate,
            "ci_low": term.natural_ci.lower,
            "ci_high": term.natural_ci.upper,
            "test": f"z = {term.statistic:.2f}, df = {term.df:.1f}",
            "p": term.p_value,
            "fmi": term.fmi,
        }
    elif item.kind == "twobytwo":  # a zero cell in some imputed datasets: only the risk difference is pooled
        term = res.term("risk_difference")
        row |= {
            "analysis": "2x2: risk difference (odds ratio not pooled: a zero cell)",
            "estimate_label": f"risk difference of {item.event} ({item.comparison} - {item.reference})",
            "estimate": term.estimate,
            "ci_low": term.ci.lower,
            "ci_high": term.ci.upper,
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
        if name not in frame.index:  # not pooled: the warning below the table says why
            rows.append({"measure": f"{label}: not pooled (a zero cell)"})
            continue
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
    if item.kind in ("welch", "student"):
        if item.kind == "student":
            st.warning(STUDENT_WARNING, icon=":material/rule:")
        result_card(res.extra["result"])
        show_mi_notes(res, complete_case=result_card)
    elif item.kind in ("hc3", "anova"):
        if item.kind == "anova":
            st.warning(ANOVA_WARNING, icon=":material/rule:")
        _show_means(res.extra["adjusted_means"], label=f"Mean of {target} (pooled)")
        show_tests(res, title=f"Group effect of {group} (D1)")
        st.markdown(f"**Differences from {item.reference}**")
        show_pooled_terms(res, alt="Pooled differences between groups")
        st.caption(
            (
                "With imputed data the group means are compared with a linear model whose variances are robust "
                "(HC3): each group keeps its own variance, as in Welch's ANOVA on the observed data, which has no "
                "pooling rule."
                if item.kind == "hc3"
                else "With imputed data the group means are compared with a linear model with equal variances "
                "(chosen), as a one-way ANOVA."
            )
            + " Pairwise post-hoc tests (Tukey, Games-Howell) have no pooling rule: the differences from "
            f"{item.reference} are shown instead."
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
            levels = c1.multiselect(
                "Groups to include",
                all_levels,
                default=all_levels,
                format_func=show_level,  # "Luxembourg " and "Luxembourg" are two groups: show the space
                key=f"mi_cmp_levels::{group}",
            )
            level_warning(group, all_levels)
        candidates = [c for c in [*numeric, *categorical] if c != group]
        targets = c2.multiselect(
            "Variables to compare",
            candidates,
            default=[c for c in numeric if c != group][:5],
            key="mi_cmp_targets",
            help="Variables created after the imputation are not in the imputed datasets.",
        )
        correction = c2.selectbox("Multiple testing correction", list(CORRECTION_OPTIONS), key="mi_cmp_correction")
        haldane = (
            "always"
            if c2.checkbox(
                "Zero cells: add 0.5 to every cell of every 2x2 table (sensitivity analysis)",
                key="mi_cmp_haldane",
                help="A zero cell gets the +0.5 correction in its own imputed dataset only, so by default the odds "
                "and risk ratios are then not pooled (the risk difference is). Adding 0.5 everywhere keeps one "
                "estimator across the imputed datasets; report it as a sensitivity analysis.",
            )
            else "if_zero"
        )
        selection = c1.radio(
            "Test selection",
            ["Recommended", "Choose the method"],
            horizontal=True,
            key="mi_cmp_selection",
            help="The analyses offered are those with a pooling rule over the imputed datasets.",
        )
        two_groups, more_groups = "welch", "hc3"
        if selection == "Choose the method":
            two_groups = c1.selectbox(
                "Numeric variable, two groups",
                list(TWO_GROUP_TESTS),
                format_func=TWO_GROUP_TESTS.get,
                key="mi_cmp_two_groups",
            )
            more_groups = c1.selectbox(
                "Numeric variable, three groups or more",
                list(MORE_GROUP_TESTS),
                format_func=MORE_GROUP_TESTS.get,
                key="mi_cmp_more_groups",
            )
            if two_groups != "welch":
                st.warning(STUDENT_WARNING, icon=":material/rule:")
            if more_groups != "hc3":
                st.warning(ANOVA_WARNING, icon=":material/rule:")
            st.caption(
                "Categorical variables have one pooled analysis: odds ratio, risk ratio and risk difference (two "
                "groups, two categories), otherwise the chi-square test by D2. Rank tests and Fisher's exact test "
                "have no pooling rule (below): run them on the observed data."
            )
        else:
            st.caption(
                "Numeric variables: Welch t-test (two groups) or a linear model with robust (HC3) variances and a D1 "
                "test (three or more groups); neither assumes equal variances. Categorical variables: odds ratio, "
                "risk ratio and risk difference (two groups, two categories) or the chi-square test by D2."
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

    config = (
        group, tuple(levels), tuple(targets), correction, haldane, two_groups, more_groups, mi_token(get_state())
    )
    if st.button("Run pooled analysis", type="primary", key="mi_cmp_run"):
        results: dict[str, Pooled] = {}
        errors: dict[str, str] = {}
        bar = st.progress(0.0, text="Pooling")
        for i, target in enumerate(targets):
            try:
                results[target] = compare_pooled(
                    mi,
                    group,
                    levels,
                    target,
                    numeric=target in numeric,
                    haldane=haldane,
                    two_groups=two_groups,
                    more_groups=more_groups,
                )
            except Exception as exc:
                errors[target] = f"{type(exc).__name__}: {exc}"
            bar.progress((i + 1) / len(targets), text=f"Pooling: {target}")
        bar.empty()
        remember(
            "univariate_mi",
            {"results": results, "errors": errors},
            config=config,
            n_analysed=len(mi.base),
            columns=[group, *targets],
        )
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
            remember("ancova_mi", res, config=config, n_analysed=len(mi.base), columns=[target, group, *covariates])
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
