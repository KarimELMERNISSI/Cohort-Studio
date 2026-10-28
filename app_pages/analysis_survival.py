"""Survival & regression page body: Kaplan-Meier, log-rank, Cox, logistic and count regression (audit S11).

With multiply imputed data (``mi``), every analysis is run on the m imputed datasets and pooled
(biostats.mi.analyze_mi): Kaplan-Meier curves on the cloglog scale with the median read from the pooled
curve, the log-rank test by D2, hazard, odds and rate ratios with Rubin's rules and a D1 test per factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from biostats._types import TestResult
from biostats.regression import (
    APPARENT_AUC,
    APPARENT_AUC_NOTE,
    count_regression,
    logistic_regression,
)
from biostats.survival import compare_survival, cox_regression
from ui.components import error_card, help_popover, result_card, section
from ui.components.mi_mode import mi_token, show_mi_notes, show_pooled_terms, show_tests
from ui.progress import mark_step
from ui.results import provenance, recall, remember
from ui.state import get_state

PH_TEST_IMPUTATIONS = 5  # the proportional-hazards test (lifelines) runs on the first imputed datasets only


def _group_candidates(df: pd.DataFrame, analyzer, max_levels: int = 8) -> list[str]:
    numeric = set(analyzer.numeric_cols)
    out = []
    for col in df.columns:
        n = int(df[col].nunique(dropna=True))
        if 2 <= n <= max_levels and (col not in numeric or n <= 3):
            out.append(col)
    return out


def _km_figure(curves) -> go.Figure:
    fig = go.Figure()
    for curve in curves:
        times = np.concatenate([[0.0], curve.times])
        surv = np.concatenate([[1.0], curve.survival])
        lower = np.concatenate([[1.0], curve.lower])
        upper = np.concatenate([[1.0], curve.upper])
        fig.add_trace(
            go.Scatter(
                x=times,
                y=surv,
                mode="lines",
                line={"shape": "hv"},
                name=f"{curve.label} (n={curve.n}, events={curve.events})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                line={"width": 0, "shape": "hv"},
                fillcolor="rgba(31,95,139,0.10)",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        xaxis_title="time",
        yaxis_title="survival probability",
        yaxis={"range": [0, 1.02]},
        height=440,
    )
    return fig


def _pooled_km_figure(curves: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for label, curve in curves.items():
        times = np.concatenate([[0.0], curve["time"].to_numpy()])
        surv = np.concatenate([[1.0], curve["survival"].to_numpy()])
        band = curve.dropna(subset=["lower", "upper"])
        fig.add_trace(go.Scatter(x=times, y=surv, mode="lines", line={"shape": "hv"}, name=f"{label} (pooled)"))
        if not band.empty:
            t = band["time"].to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([t, t[::-1]]),
                    y=np.concatenate([band["upper"].to_numpy(), band["lower"].to_numpy()[::-1]]),
                    fill="toself",
                    line={"width": 0, "shape": "hv"},
                    fillcolor="rgba(31,95,139,0.10)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    fig.update_layout(xaxis_title="time", yaxis_title="survival probability", yaxis={"range": [0, 1.02]}, height=440)
    return fig


def _median_text(median: float | None) -> str:
    """The median survival time, or 'not reached' when the curve stays above 0.5."""
    return "not reached" if median is None else f"{median:.3g}"


def _table(df: pd.DataFrame, formats: dict) -> None:
    st.dataframe(df.style.format(formats, na_rep=""), hide_index=True, width="stretch")


def _default_index(options: list[str], hints: tuple[str, ...]) -> int:
    """Index of the first option whose name contains a hint (e.g. 'time', 'died'), else 0."""
    for i, name in enumerate(options):
        if any(h in str(name).lower() for h in hints):
            return i
    return 0


TIME_HINTS = ("time", "days", "months", "years", "duration", "follow", "fu_")
EVENT_HINTS = ("died", "death", "dead", "event", "status", "relapse", "fail")

COX_FORMATS = {"log_hr": "{:.3f}", "se": "{:.3f}", "hr": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}", "z": "{:.2f}", "p": "{:.4f}"}
LOGIT_FORMATS = {"coef": "{:.3f}", "se": "{:.3f}", "odds_ratio": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}", "z": "{:.2f}", "p": "{:.4f}"}
COUNT_FORMATS = {"coef": "{:.3f}", "se": "{:.3f}", "rate_ratio": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}", "z": "{:.2f}", "p": "{:.4f}"}


def _show_survival(res) -> None:
    st.plotly_chart(_km_figure(res.curves), width="stretch")
    _table(
        pd.DataFrame(
            [
                {
                    "group": c.label,
                    "n": c.n,
                    "events": c.events,
                    "median survival": _median_text(c.median),
                    "survival at last time": float(c.survival[-1]),
                }
                for c in res.curves
            ]
        ),
        {"survival at last time": "{:.3f}"},
    )
    if res.logrank_statistic is not None:
        result_card(
            TestResult(
                name="Log-rank test",
                family="nonparametric",
                statistic=res.logrank_statistic,
                statistic_label="chi2",
                df=[float(res.logrank_df)],
                p_value=res.logrank_p,
                groups=[c.label for c in res.curves],
                n_per_group=[c.n for c in res.curves],
                references=res.references,
            )
        )
    for w in res.warnings:
        st.warning(w)


def _show_pooled_survival(res) -> None:
    curves = res.extra["curves"]
    st.plotly_chart(_pooled_km_figure(curves), width="stretch")
    _table(
        pd.DataFrame(
            [
                {
                    "group": label,
                    "median survival (pooled curve)": _median_text(res.extra["medians"].get(label)),
                    "survival at last time": float(curve["survival"].iloc[-1]),
                }
                for label, curve in curves.items()
            ]
        ),
        {"survival at last time": "{:.3f}"},
    )
    test = res.tests.get("log_rank")
    if test is not None:
        result_card(
            TestResult(
                name="Log-rank test, pooled (D2)",
                family="nonparametric",
                statistic=test.statistic,
                statistic_label="F",
                df=[test.df1, test.df2],
                p_value=test.p_value,
                groups=list(curves),
                warnings=[test.note],
                references=test.references,
            )
        )
    show_mi_notes(res, complete_case=_show_survival)


def run_survival(df: pd.DataFrame, analyzer, *, mi=None) -> None:
    numeric = list(analyzer.numeric_cols)
    binary = list(analyzer.binary_cols)
    groups = _group_candidates(df, analyzer)
    token = mi_token(get_state()) if mi is not None else ""

    section(
        "Kaplan-Meier survival and the log-rank test",
        "Follow-up time, an event indicator (1/Yes = event, 0/No = censored) and an optional grouping variable.",
    )
    if not numeric or not binary:
        st.info("A numeric follow-up time and a binary event indicator are needed.")
    else:
        c1, c2, c3 = st.columns(3)
        time_col = c1.selectbox(
            "Follow-up time",
            numeric,
            index=_default_index(numeric, TIME_HINTS),
            key="surv_time",
            help="Time from entry to the event or to censoring (any unit, non-negative).",
        )
        event_col = c2.selectbox(
            "Event indicator",
            binary,
            index=_default_index(binary, EVENT_HINTS),
            key="surv_event",
            help="1 / Yes / True = the event occurred; 0 / No / False = censored.",
        )
        group_col = c3.selectbox(
            "Group (optional)",
            ["(none)", *[g for g in groups if g not in (time_col, event_col)]],
            key="surv_group",
            help="Curves per level and the log-rank test between them.",
        )
        group = None if group_col == "(none)" else group_col
        key, config = ("survival_mi", (time_col, event_col, group_col, token)) if mi is not None else (
            "survival",
            (time_col, event_col, group_col),
        )
        if st.button("Run survival analysis", type="primary", key="surv_run"):
            try:
                if mi is not None:
                    from biostats.mi import analyze_mi

                    columns = [time_col, event_col, *([group] if group else [])]
                    res = analyze_mi(lambda d: compare_survival(d, time_col, event_col, group), mi, columns=columns)
                    remember(key, res, config=config, n_analysed=len(mi.base), columns=columns)
                else:
                    res = compare_survival(df, time_col, event_col, group)
                    remember(
                        key,
                        res,
                        config=config,
                        n_analysed=sum(c.n for c in res.curves),
                        columns=[time_col, event_col, group or ""],
                    )
                mark_step("analyse", "survival analysis")
            except Exception as exc:
                error_card("The survival analysis could not be computed.", exc)
        res = recall(key, config=config)
        if res is not None:
            (_show_pooled_survival if mi is not None else _show_survival)(res)
            provenance(key)
            help_popover(
                "Each step down is an event; vertical ticks would mark censoring. The band is the pointwise 95% interval. "
                "The log-rank test weights every time equally and loses power when curves cross; report medians and the "
                "hazard ratio from the Cox model with it. With imputed data the curves are pooled on the "
                "log(-log S) scale at every time any curve steps, and the median is read from the pooled curve."
            )

    section(
        "Cox proportional-hazards regression",
        "Hazard ratios adjusted for several covariates; categorical covariates use their first level as reference.",
    )
    if numeric and binary:
        c1, c2 = st.columns(2)
        cox_time = c1.selectbox(
            "Follow-up time ",
            numeric,
            index=_default_index(numeric, TIME_HINTS),
            key="cox_time",
            help="Same meaning as above.",
        )
        cox_event = c2.selectbox(
            "Event indicator ",
            binary,
            index=_default_index(binary, EVENT_HINTS),
            key="cox_event",
            help="Same meaning as above.",
        )
        covariates = st.multiselect(
            "Covariates",
            [c for c in df.columns if c not in (cox_time, cox_event) and (c in numeric or c in groups)],
            key="cox_covs",
            help="Numeric covariates enter linearly; categorical ones are one-hot coded.",
        )
        key, config = ("cox_mi", (cox_time, cox_event, tuple(covariates), token)) if mi is not None else (
            "cox",
            (cox_time, cox_event, tuple(covariates)),
        )
        if st.button("Fit Cox model", key="cox_run", type="primary", disabled=not covariates):
            try:
                if mi is not None:
                    from biostats.mi import analyze_mi

                    levels = mi.levels(covariates)
                    fitted = analyze_mi(
                        lambda d, i: cox_regression(
                            d, cox_time, cox_event, covariates, levels=levels, diagnostics=i <= PH_TEST_IMPUTATIONS
                        ),
                        mi,
                        columns=[cox_time, cox_event, *covariates],
                    )
                    remember(
                        key, fitted, config=config, n_analysed=len(mi.base), columns=[cox_time, cox_event, *covariates]
                    )
                else:
                    fitted = cox_regression(df, cox_time, cox_event, covariates)
                    remember(
                        key, fitted, config=config, n_analysed=fitted.n, columns=[cox_time, cox_event, *covariates]
                    )
                mark_step("analyse", "Cox model")
            except Exception as exc:
                error_card("The Cox model could not be fitted.", exc)
        res = recall(key, config=config)
        if res is not None and mi is not None:
            show_pooled_terms(res, ratio_label="hazard_ratio", alt="Pooled hazard ratios")
            show_tests(res, title="Tests of each factor (all its levels at once)")
            st.caption(
                f"The proportional-hazards test ran on the first {PH_TEST_IMPUTATIONS} imputed datasets; its warnings "
                "say on how many it flagged a term."
            )
            show_mi_notes(res, complete_case=lambda cc: _table(cc.table, COX_FORMATS))
            provenance(key)
        elif res is not None:
            st.caption(
                f"n = {res.n}, events = {res.events}, ties: {res.ties}"
                + (f", concordance = {res.concordance:.3f}" if res.concordance is not None else "")
            )
            _table(res.table, COX_FORMATS)
            if res.ph_test is not None:
                with st.expander("Proportional-hazards test (Grambsch-Therneau)"):
                    _table(res.ph_test, {"statistic": "{:.3f}", "p": "{:.4f}"})
            for w in res.warnings:
                st.warning(w)
            st.caption("Method: " + " | ".join(res.references))
            provenance(key)

    section(
        "Logistic regression (binary outcome)",
        "Adjusted odds ratios with Wald intervals and the discrimination (AUC) of the fitted model.",
    )
    if binary:
        c1, c2 = st.columns(2)
        outcome = c1.selectbox(
            "Outcome",
            binary,
            key="logit_outcome",
            help="Binary outcome; the event level is the second sorted level unless chosen.",
        )
        levels_of_outcome = sorted(df[outcome].dropna().astype(str).unique())
        event_level = c2.selectbox(
            "Event level",
            levels_of_outcome,
            index=len(levels_of_outcome) - 1,
            key="logit_event",
            help="Level modelled as the event.",
        )
        covs = st.multiselect(
            "Covariates ",
            [c for c in df.columns if c != outcome and (c in numeric or c in groups)],
            key="logit_covs",
            help="Numeric covariates enter linearly; categorical ones are one-hot coded.",
        )
        key, config = ("logistic_mi", (outcome, event_level, tuple(covs), token)) if mi is not None else (
            "logistic",
            (outcome, event_level, tuple(covs)),
        )
        if st.button("Fit logistic model", key="logit_run", type="primary", disabled=not covs):
            try:
                if mi is not None:
                    from biostats.mi import analyze_mi

                    levels = mi.levels(covs)
                    fitted = analyze_mi(
                        lambda d: logistic_regression(d, outcome, covs, event_level=event_level, levels=levels),
                        mi,
                        columns=[outcome, *covs],
                    )
                    remember(key, fitted, config=config, n_analysed=len(mi.base), columns=[outcome, *covs])
                else:
                    fitted = logistic_regression(df, outcome, covs, event_level=event_level)
                    remember(key, fitted, config=config, n_analysed=fitted.n, columns=[outcome, *covs])
                mark_step("analyse", "logistic regression")
            except Exception as exc:
                error_card("The logistic model could not be fitted.", exc)
        res = recall(key, config=config)
        if res is not None and mi is not None:
            show_pooled_terms(res, ratio_label="odds_ratio", alt="Pooled odds ratios")
            show_tests(res, title="Tests of each factor (all its levels at once)")
            show_mi_notes(res, complete_case=lambda cc: _table(cc.table, LOGIT_FORMATS))
            provenance(key)
        elif res is not None:
            auc_text = (
                f", {APPARENT_AUC} = {res.auc:.3f} (95% CI {res.auc_ci[0]:.3f} to {res.auc_ci[1]:.3f})"
                if res.auc is not None and res.auc_ci
                else ""
            )
            st.caption(
                f"n = {res.n}, events = {res.events}, pseudo-R2 = {res.pseudo_r2:.3f}{auc_text}; reference levels: {res.reference_levels}"
            )
            if res.auc is not None:
                st.caption(APPARENT_AUC_NOTE)
            _table(res.table, LOGIT_FORMATS)
            for w in res.warnings:
                st.warning(w)
            st.caption("Method: " + " | ".join(res.references))
            provenance(key)

    section(
        "Count regression (Poisson / negative binomial)",
        "Incidence-rate ratios for event counts, with an optional person-time offset.",
    )
    if numeric:
        c1, c2, c3 = st.columns(3)
        count_col = c1.selectbox(
            "Count outcome",
            numeric,
            key="pois_count",
            help="Non-negative integer counts per subject or unit.",
        )
        exposure = c2.selectbox(
            "Exposure / person-time (optional)",
            ["(none)", *[c for c in numeric if c != count_col]],
            key="pois_exposure",
            help="Enters as log(exposure) offset so coefficients are rate ratios.",
        )
        family = c3.radio(
            "Family",
            ["poisson", "negative_binomial"],
            key="pois_family",
            horizontal=True,
            help="Use the negative binomial when the Poisson model is over-dispersed (Pearson chi2/df well above 1).",
        )
        covs = st.multiselect(
            "Covariates  ",
            [c for c in df.columns if c not in (count_col, exposure) and (c in numeric or c in groups)],
            key="pois_covs",
            help="Numeric covariates enter linearly; categorical ones are one-hot coded.",
        )
        exposure_col = None if exposure == "(none)" else exposure
        key, config = ("count_mi", (count_col, exposure, family, tuple(covs), token)) if mi is not None else (
            "count",
            (count_col, exposure, family, tuple(covs)),
        )
        if st.button("Fit count model", key="pois_run", type="primary", disabled=not covs):
            try:
                if mi is not None:
                    from biostats.mi import analyze_mi

                    levels = mi.levels(covs)
                    fitted = analyze_mi(
                        lambda d: count_regression(
                            d, count_col, covs, exposure_col=exposure_col, family=family, levels=levels
                        ),
                        mi,
                        columns=[count_col, *covs, *([exposure_col] if exposure_col else [])],
                    )
                    remember(
                        key,
                        fitted,
                        config=config,
                        n_analysed=len(mi.base),
                        columns=[count_col, *covs, exposure_col or ""],
                    )
                else:
                    fitted = count_regression(df, count_col, covs, exposure_col=exposure_col, family=family)
                    remember(
                        key, fitted, config=config, n_analysed=fitted.n, columns=[count_col, *covs, exposure_col or ""]
                    )
                mark_step("analyse", "count regression")
            except Exception as exc:
                error_card("The count model could not be fitted.", exc)
        res = recall(key, config=config)
        if res is not None and mi is not None:
            show_pooled_terms(res, ratio_label="rate_ratio", alt="Pooled rate ratios")
            show_tests(res, title="Tests of each factor (all its levels at once)")
            show_mi_notes(res, complete_case=lambda cc: _table(cc.table, COUNT_FORMATS))
            provenance(key)
        elif res is not None:
            disp = f", dispersion = {res.dispersion:.2f}" if res.dispersion is not None else ""
            st.caption(f"n = {res.n}, events = {res.events}, AIC = {res.aic:.1f}{disp}")
            _table(res.table, COUNT_FORMATS)
            for w in res.warnings:
                st.warning(w)
            st.caption("Method: " + " | ".join(res.references))
            provenance(key)


__all__ = ["run_survival"]
