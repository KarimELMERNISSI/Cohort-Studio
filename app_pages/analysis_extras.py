"""Paired tests, incidence rates, stratified tables, standardised ratios and reliability (audit S11)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from biostats.paired import (
    mcnemar_test,
    one_sample_proportion,
    one_sample_t,
    one_sample_wilcoxon,
    paired_t,
    wilcoxon_signed_rank,
)
from biostats.rates import incidence_rate, mantel_haenszel, rate_ratio, standardised_ratio
from biostats.reliability import cohens_kappa, icc
from ui.components import error_card, help_popover, result_card, section, status_chip
from ui.components.mi_mode import mi_token, show_mi_notes
from ui.progress import mark_step
from ui.results import provenance, recall, remember
from ui.state import get_state


def _ci(ci) -> str:
    return f"{ci.lower:.3g} to {ci.upper:.3g}"


OBSERVED_NOTE = "Not pooled over the imputed datasets: this analysis uses the observed data (complete cases)."


def _not_pooled(analysis: str) -> None:
    """The reason a test has no pooling rule, and what to run instead (biostats.mi.not_poolable)."""
    from biostats.mi import not_poolable

    entry = not_poolable(analysis)
    if entry is not None:
        st.info(f"Not pooled with imputed data. {entry.reason} {entry.alternative}", icon=":material/block:")


def _show_test(res, key: str, *, pooled: bool) -> None:
    """A stored test result: the pooled test with its notes and complete-case comparison, or the observed one."""
    if res is None:
        return
    if pooled:
        result_card(res.extra["result"])
        show_mi_notes(res, complete_case=result_card)
    else:
        result_card(res)
    provenance(key)


def run_paired_and_rates(df: pd.DataFrame, analyzer, *, mi=None) -> None:
    """Paired and one-sample tests (pooled over the imputed datasets ``mi`` when given), McNemar, rates,
    Mantel-Haenszel and standardised ratios."""
    numeric = list(analyzer.numeric_cols)
    binary = list(analyzer.binary_cols)
    small_cats = [
        c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 8 and c not in numeric
    ]

    section(
        "Paired measurements",
        "Two columns measured on the same subjects (before/after, two methods).",
    )
    if len(numeric) >= 2:
        c1, c2, c3 = st.columns(3)
        before = c1.selectbox(
            "Before / method A",
            numeric,
            key="pair_before",
            help="First measurement of each subject.",
        )
        after = c2.selectbox(
            "After / method B",
            [c for c in numeric if c != before],
            key="pair_after",
            help="Second measurement of the same subject (differences are after - before).",
        )
        kind = c3.radio(
            "Test",
            ["Paired t-test", "Wilcoxon signed-rank"],
            key="pair_kind",
            horizontal=True,
            help="Paired t for roughly symmetric differences; Wilcoxon for skewed differences or outliers.",
        )
        if mi is not None and not kind.startswith("Paired"):
            _not_pooled("wilcoxon_signed_rank")
        else:
            key, config = ("paired_mi", (before, after, kind, mi_token(get_state()))) if mi is not None else (
                "paired",
                (before, after, kind),
            )
            if st.button("Run paired test", key="pair_run", type="primary"):
                try:
                    if mi is not None:
                        from biostats.mi import analyze_mi

                        res = analyze_mi(lambda d: paired_t(d[after], d[before]), mi, columns=[before, after])
                        remember(key, res, config=config, n_analysed=len(mi.base), columns=[before, after])
                    else:
                        res = (
                            paired_t(df[after], df[before])
                            if kind.startswith("Paired")
                            else wilcoxon_signed_rank(df[after], df[before])
                        )
                        remember(key, res, config=config, columns=[before, after])
                    mark_step("analyse", "paired test")
                except Exception as exc:
                    error_card("The paired test could not be computed.", exc)
            _show_test(recall(key, config=config), key, pooled=mi is not None)
    else:
        st.info("Two numeric columns are needed.")

    section(
        "One sample vs a reference value",
        "Compare one measurement with a known value (e.g. a population mean).",
    )
    if numeric:
        c1, c2, c3 = st.columns(3)
        col = c1.selectbox("Variable", numeric, key="one_col", help="The measurement to compare.")
        ref = c2.number_input(
            "Reference value",
            value=float(pd.to_numeric(df[col], errors="coerce").median() or 0.0),
            key="one_ref",
            help="The fixed value the mean (or median) is compared with.",
        )
        kind = c3.radio(
            "Test ",
            ["One-sample t-test", "Wilcoxon signed-rank"],
            key="one_kind",
            horizontal=True,
            help="t-test for roughly symmetric data; Wilcoxon otherwise.",
        )
        if mi is not None and not kind.startswith("One"):
            _not_pooled("wilcoxon_signed_rank")
        else:
            key, config = ("one_sample_mi", (col, ref, kind, mi_token(get_state()))) if mi is not None else (
                "one_sample",
                (col, ref, kind),
            )
            if st.button("Run one-sample test", key="one_run", type="primary"):
                try:
                    if mi is not None:
                        from biostats.mi import analyze_mi

                        res = analyze_mi(lambda d: one_sample_t(d[col], ref), mi, columns=[col])
                        remember(key, res, config=config, n_analysed=len(mi.base), columns=[col])
                    else:
                        res = (
                            one_sample_t(df[col], ref)
                            if kind.startswith("One")
                            else one_sample_wilcoxon(df[col], ref)
                        )
                        remember(key, res, config=config, columns=[col])
                    mark_step("analyse", "one-sample test")
                except Exception as exc:
                    error_card("The one-sample test could not be computed.", exc)
            _show_test(recall(key, config=config), key, pooled=mi is not None)

    section("Paired binary outcomes (McNemar) and one-sample proportion")
    if mi is not None:
        st.caption(OBSERVED_NOTE)
    if len(binary) >= 2:
        c1, c2 = st.columns(2)
        b1 = c1.selectbox(
            "Before (binary)",
            binary,
            key="mc_before",
            help="First binary measurement of each subject.",
        )
        b2 = c2.selectbox(
            "After (binary)",
            [c for c in binary if c != b1],
            key="mc_after",
            help="Second binary measurement of the same subject.",
        )
        if st.button("Run McNemar's test", key="mc_run", type="primary"):
            try:
                table = pd.crosstab(df[b1].astype(str), df[b2].astype(str))
                if table.shape != (2, 2):
                    raise ValueError("both columns must have exactly two levels on complete pairs")
                remember(
                    "mcnemar",
                    {"res": mcnemar_test(table.to_numpy()), "table": table},
                    config=(b1, b2),
                    n_analysed=int(table.to_numpy().sum()),
                    columns=[b1, b2],
                )
                mark_step("analyse", "McNemar test")
            except Exception as exc:
                error_card("McNemar's test could not be computed.", exc)
        cached = recall("mcnemar", config=(b1, b2))
        if cached is not None:
            st.dataframe(cached["table"], width="stretch")
            result_card(cached["res"])
            provenance("mcnemar")
    p1, p2, p3 = st.columns(3)
    successes = p1.number_input(
        "Successes",
        min_value=0,
        value=30,
        step=1,
        key="prop_k",
        help="Observed number of successes.",
    )
    n_total = p2.number_input(
        "Trials", min_value=1, value=100, step=1, key="prop_n", help="Number of trials or subjects."
    )
    p_ref = p3.number_input(
        "Reference proportion",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        key="prop_ref",
        help="Hypothesised proportion.",
    )
    if st.button("Run exact binomial test", key="prop_run"):
        result_card(one_sample_proportion(int(successes), int(n_total), float(p_ref)))

    section(
        "Incidence rates and rate ratio",
        "Events over person-time with exact Poisson intervals; two groups give the rate ratio.",
    )
    r1, r2, r3, r4, r5 = st.columns(5)
    e1 = r1.number_input(
        "Events (exposed)",
        min_value=0,
        value=15,
        step=1,
        key="rate_e1",
        help="Number of events in the exposed group.",
    )
    t1 = r2.number_input(
        "Person-time (exposed)",
        min_value=0.0,
        value=1200.0,
        key="rate_t1",
        help="Total follow-up of the exposed group.",
    )
    e0 = r3.number_input(
        "Events (reference)",
        min_value=0,
        value=4,
        step=1,
        key="rate_e0",
        help="Number of events in the reference group (0 for a single rate).",
    )
    t0 = r4.number_input(
        "Person-time (reference)",
        min_value=0.0,
        value=1000.0,
        key="rate_t0",
        help="Total follow-up of the reference group.",
    )
    per = r5.selectbox(
        "Per", [100, 1000, 10000, 100000], index=1, key="rate_per", help="Multiplier for display."
    )
    if st.button("Compute rates", key="rate_run"):
        try:
            if t0 > 0 and (e0 > 0 or e1 > 0):
                rr = rate_ratio(int(e1), float(t1), int(e0), float(t0), per=per)
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Rate (exposed) per {per}", f"{rr.exposed.rate * per:.2f}")
                m1.caption(
                    f"95% CI {rr.exposed.ci.lower * per:.2f} to {rr.exposed.ci.upper * per:.2f}"
                )
                m2.metric(f"Rate (reference) per {per}", f"{rr.reference.rate * per:.2f}")
                m2.caption(
                    f"95% CI {rr.reference.ci.lower * per:.2f} to {rr.reference.ci.upper * per:.2f}"
                )
                m3.metric(
                    "Rate ratio",
                    f"{rr.ratio:.3f}",
                    help=f"exact conditional test p = {rr.p_value:.4g}",
                )
                m3.caption(f"95% CI {_ci(rr.ci)} ({rr.ci.method})")
                st.caption("Method: " + " | ".join(rr.references))
            else:
                rate = incidence_rate(int(e1), float(t1), per=per)
                st.metric(
                    f"Rate per {per}",
                    f"{rate.rate * per:.2f}",
                    help=f"95% CI {rate.ci.lower * per:.2f} to {rate.ci.upper * per:.2f} ({rate.ci.method})",
                )
        except Exception as exc:
            error_card("The rates could not be computed.", exc)

    section(
        "Stratified 2x2 tables (Mantel-Haenszel)",
        "Pooled odds and risk ratios adjusted for a stratifying variable; Breslow-Day tests homogeneity.",
    )
    if mi is not None:
        st.caption(OBSERVED_NOTE)
    if len(binary) >= 2 and small_cats:
        c1, c2, c3 = st.columns(3)
        exposure = c1.selectbox(
            "Exposure (binary)",
            binary,
            key="mh_exposure",
            help="The factor whose effect is estimated.",
        )
        outcome = c2.selectbox(
            "Outcome (binary)",
            [c for c in binary if c != exposure],
            key="mh_outcome",
            help="The event.",
        )
        stratum = c3.selectbox(
            "Stratum",
            [c for c in small_cats if c not in (exposure, outcome)],
            key="mh_stratum",
            help="The variable to adjust for (2 to 8 levels).",
        )
        l1, l2 = st.columns(2)
        exp_levels = sorted(df[exposure].dropna().astype(str).unique())
        out_levels = sorted(df[outcome].dropna().astype(str).unique())
        exp_level = l1.selectbox(
            "Exposed level",
            exp_levels,
            index=len(exp_levels) - 1,
            key="mh_exp_level",
            help="Level counted as exposed.",
        )
        ev_level = l2.selectbox(
            "Event level",
            out_levels,
            index=len(out_levels) - 1,
            key="mh_ev_level",
            help="Level counted as the event.",
        )
        if st.button("Run Mantel-Haenszel", key="mh_run", type="primary"):
            try:
                remember(
                    "mantel_haenszel",
                    mantel_haenszel(
                        df,
                        exposure,
                        outcome,
                        stratum,
                        exposure_level=exp_level,
                        event_level=ev_level,
                    ),
                    config=(exposure, outcome, stratum, exp_level, ev_level),
                    columns=[exposure, outcome, stratum],
                )
                mark_step("analyse", "Mantel-Haenszel analysis")
            except Exception as exc:
                error_card("The stratified analysis could not be computed.", exc)
        res = recall("mantel_haenszel", config=(exposure, outcome, stratum, exp_level, ev_level))
        if res is not None:
            provenance("mantel_haenszel")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("OR (Mantel-Haenszel)", f"{res.or_mh:.3f}")
            m1.caption(f"95% CI {_ci(res.or_mh_ci)}")
            m2.metric("RR (Mantel-Haenszel)", f"{res.rr_mh:.3f}")
            m2.caption(f"95% CI {_ci(res.rr_mh_ci)}")
            m3.metric(
                "Crude OR",
                f"{res.crude_or:.3f}",
                help="Unadjusted odds ratio from the pooled table.",
            )
            m4.metric(
                "MH test p", f"{res.mh_test_p:.4g}", help=f"chi2 = {res.mh_test_statistic:.3f}"
            )
            with m4:
                status_chip(
                    "homogeneous across strata"
                    if res.breslow_day_p >= 0.05
                    else "heterogeneous across strata",
                    "ok" if res.breslow_day_p >= 0.05 else "caution",
                )
            st.caption(
                f"Breslow-Day test of homogeneity: chi2 = {res.breslow_day_statistic:.3f}, p = {res.breslow_day_p:.4g} "
                "(a pooled ratio is only meaningful when the stratum-specific ratios are similar)."
            )
            st.dataframe(
                res.per_stratum.style.format(
                    {"odds_ratio": "{:.3f}", "risk_ratio": "{:.3f}"}, na_rep="n/a"
                ),
                hide_index=True,
                width="stretch",
            )
            for w in res.warnings:
                st.warning(w)
            st.caption("Method: " + " | ".join(res.references))
    else:
        st.info("Two binary columns and a stratum variable with 2 to 8 levels are needed.")

    section(
        "Standardised mortality / incidence ratio",
        "Observed events over the expected count from reference rates.",
    )
    s1, s2 = st.columns(2)
    observed = s1.number_input(
        "Observed events",
        min_value=0,
        value=25,
        step=1,
        key="smr_obs",
        help="Events observed in the cohort.",
    )
    expected = s2.number_input(
        "Expected events",
        min_value=0.0,
        value=18.4,
        key="smr_exp",
        help="Sum of reference rates times the cohort's person-time by stratum.",
    )
    if st.button("Compute SMR", key="smr_run"):
        try:
            smr = standardised_ratio(int(observed), float(expected))
            st.metric("SMR", f"{smr.ratio:.3f}", help=f"exact Poisson p = {smr.p_value:.4g}")
            st.caption(
                f"95% CI {_ci(smr.ci)} ({smr.ci.method}). Method: " + " | ".join(smr.references)
            )
        except Exception as exc:
            error_card("The ratio could not be computed.", exc)
    help_popover(
        "Rates need person-time, not head counts. The Mantel-Haenszel ratio adjusts for one stratifying variable; use "
        "regression for several. An SMR compares the cohort with the population that supplied the expected count only."
    )


def run_reliability(df: pd.DataFrame, analyzer) -> None:
    numeric = list(analyzer.numeric_cols)
    cats = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 10]

    section(
        "Inter-rater agreement (Cohen's kappa)",
        "Two raters' categorical ratings of the same subjects (same categories).",
    )
    if len(cats) >= 2:
        c1, c2, c3 = st.columns(3)
        r1 = c1.selectbox("Rater A", cats, key="kappa_a", help="Ratings of the first rater.")
        r2 = c2.selectbox(
            "Rater B",
            [c for c in cats if c != r1],
            key="kappa_b",
            help="Ratings of the second rater.",
        )
        weights = c3.selectbox(
            "Weights",
            ["none", "linear", "quadratic"],
            key="kappa_w",
            help="Weighted kappa for ordered categories.",
        )
        if st.button("Compute kappa", key="kappa_run", type="primary"):
            try:
                a, b = df[r1].astype(str), df[r2].astype(str)
                levels = sorted(set(a.dropna()) | set(b.dropna()))
                table = pd.crosstab(
                    pd.Categorical(a, categories=levels),
                    pd.Categorical(b, categories=levels),
                    dropna=False,
                )
                res = cohens_kappa(table.to_numpy(), weights=None if weights == "none" else weights)
                m1, m2 = st.columns(2)
                m1.metric("Kappa", f"{res.kappa:.3f}", help=f"SE {res.se:.3f}; n = {res.n}")
                m1.caption(f"95% CI {_ci(res.ci)}; {res.interpretation}")
                m2.metric(
                    "Observed agreement",
                    f"{res.observed_agreement:.3f}",
                    help=f"expected by chance {res.expected_agreement:.3f}",
                )
                st.dataframe(table, width="stretch")
                st.caption("Method: " + " | ".join(res.references))
                mark_step("analyse", "agreement (kappa)")
            except Exception as exc:
                error_card("Kappa could not be computed.", exc)

    section(
        "Intraclass correlation (ICC)",
        "Reliability of continuous measurements across raters or occasions (columns = raters).",
    )
    if len(numeric) >= 2:
        raters = st.multiselect(
            "Rater / occasion columns",
            numeric,
            default=numeric[:2],
            key="icc_cols",
            help="Each column is one rater or measurement occasion of the same subjects.",
        )
        model = st.radio(
            "Model",
            ["ICC(2,1)", "ICC(3,1)"],
            key="icc_model",
            horizontal=True,
            help="ICC(2,1): absolute agreement, raters are a random sample. ICC(3,1): consistency, these specific raters.",
        )
        if st.button("Compute ICC", key="icc_run", type="primary", disabled=len(raters) < 2):
            try:
                res = icc(df[raters], model=model)
                st.metric(
                    "ICC",
                    f"{res.icc:.3f}",
                    help=f"F = {res.f_statistic:.3f}, df = {res.df1:.0f}, {res.df2:.0f}, p = {res.p_value:.4g}",
                )
                st.caption(
                    f"95% CI {_ci(res.ci)}; {res.model}; {res.interpretation}; {res.n_subjects} subjects, {res.n_raters} raters"
                )
                st.caption("Method: " + " | ".join(res.references))
                mark_step("analyse", "reliability (ICC)")
            except Exception as exc:
                error_card("The ICC could not be computed.", exc)


__all__ = ["run_paired_and_rates", "run_reliability"]
