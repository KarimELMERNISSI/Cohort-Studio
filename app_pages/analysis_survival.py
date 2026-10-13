"""Survival & regression page body: Kaplan-Meier, log-rank, Cox, logistic and count regression (audit S11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from biostats._types import TestResult
from biostats.regression import count_regression, logistic_regression
from biostats.survival import compare_survival, cox_regression
from ui.components import error_card, help_popover, result_card, section


def _group_candidates(df: pd.DataFrame, analyzer, max_levels: int = 8) -> list[str]:
    numeric = set(analyzer.numeric_cols)
    out = []
    for col in df.columns:
        n = df[col].nunique(dropna=True)
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


def run_survival(df: pd.DataFrame, analyzer) -> None:
    numeric = list(analyzer.numeric_cols)
    binary = list(analyzer.binary_cols)
    groups = _group_candidates(df, analyzer)

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
        if st.button("Run survival analysis", type="primary", key="surv_run"):
            try:
                res = compare_survival(
                    df, time_col, event_col, None if group_col == "(none)" else group_col
                )
                st.session_state["surv_result"] = {
                    "res": res,
                    "key": (time_col, event_col, group_col),
                }
            except Exception as exc:
                error_card("The survival analysis could not be computed.", exc)
        cached = st.session_state.get("surv_result")
        if cached and cached["key"] == (time_col, event_col, group_col):
            res = cached["res"]
            st.plotly_chart(_km_figure(res.curves), width="stretch")
            _table(
                pd.DataFrame(
                    [
                        {
                            "group": c.label,
                            "n": c.n,
                            "events": c.events,
                            "median survival": c.median,
                            "survival at last time": float(c.survival[-1]),
                        }
                        for c in res.curves
                    ]
                ),
                {"median survival": "{:.3g}", "survival at last time": "{:.3f}"},
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
            help_popover(
                "Each step down is an event; vertical ticks would mark censoring. The band is the pointwise 95% interval. "
                "The log-rank test weights every time equally and loses power when curves cross; report medians and the "
                "hazard ratio from the Cox model with it."
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
            [
                c
                for c in df.columns
                if c not in (cox_time, cox_event) and (c in numeric or c in groups)
            ],
            key="cox_covs",
            help="Numeric covariates enter linearly; categorical ones are one-hot coded.",
        )
        if st.button("Fit Cox model", key="cox_run", type="primary", disabled=not covariates):
            try:
                st.session_state["cox_result"] = {
                    "res": cox_regression(df, cox_time, cox_event, covariates),
                    "key": (cox_time, cox_event, tuple(covariates)),
                }
            except Exception as exc:
                error_card("The Cox model could not be fitted.", exc)
        cached = st.session_state.get("cox_result")
        if cached and cached["key"] == (cox_time, cox_event, tuple(covariates)):
            res = cached["res"]
            st.caption(
                f"n = {res.n}, events = {res.events}, ties: {res.ties}"
                + (f", concordance = {res.concordance:.3f}" if res.concordance is not None else "")
            )
            _table(
                res.table,
                {
                    "log_hr": "{:.3f}",
                    "se": "{:.3f}",
                    "hr": "{:.3f}",
                    "ci_low": "{:.3f}",
                    "ci_high": "{:.3f}",
                    "z": "{:.2f}",
                    "p": "{:.4f}",
                },
            )
            if res.ph_test is not None:
                with st.expander("Proportional-hazards test (Grambsch-Therneau)"):
                    _table(res.ph_test, {"statistic": "{:.3f}", "p": "{:.4f}"})
            for w in res.warnings:
                st.warning(w)
            st.caption("Method: " + " | ".join(res.references))

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
        levels = sorted(df[outcome].dropna().astype(str).unique())
        event_level = c2.selectbox(
            "Event level",
            levels,
            index=len(levels) - 1,
            key="logit_event",
            help="Level modelled as the event.",
        )
        covs = st.multiselect(
            "Covariates ",
            [c for c in df.columns if c != outcome and (c in numeric or c in groups)],
            key="logit_covs",
            help="Numeric covariates enter linearly; categorical ones are one-hot coded.",
        )
        if st.button("Fit logistic model", key="logit_run", type="primary", disabled=not covs):
            try:
                st.session_state["logit_result"] = {
                    "res": logistic_regression(df, outcome, covs, event_level=event_level),
                    "key": (outcome, event_level, tuple(covs)),
                }
            except Exception as exc:
                error_card("The logistic model could not be fitted.", exc)
        cached = st.session_state.get("logit_result")
        if cached and cached["key"] == (outcome, event_level, tuple(covs)):
            res = cached["res"]
            auc_text = (
                f", AUC = {res.auc:.3f} (95% CI {res.auc_ci[0]:.3f} to {res.auc_ci[1]:.3f})"
                if res.auc is not None and res.auc_ci
                else ""
            )
            st.caption(
                f"n = {res.n}, events = {res.events}, pseudo-R2 = {res.pseudo_r2:.3f}{auc_text}; reference levels: {res.reference_levels}"
            )
            _table(
                res.table,
                {
                    "coef": "{:.3f}",
                    "se": "{:.3f}",
                    "odds_ratio": "{:.3f}",
                    "ci_low": "{:.3f}",
                    "ci_high": "{:.3f}",
                    "z": "{:.2f}",
                    "p": "{:.4f}",
                },
            )
            for w in res.warnings:
                st.warning(w)
            st.caption("Method: " + " | ".join(res.references))

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
            [
                c
                for c in df.columns
                if c not in (count_col, exposure) and (c in numeric or c in groups)
            ],
            key="pois_covs",
            help="Numeric covariates enter linearly; categorical ones are one-hot coded.",
        )
        if st.button("Fit count model", key="pois_run", type="primary", disabled=not covs):
            try:
                st.session_state["pois_result"] = {
                    "res": count_regression(
                        df,
                        count_col,
                        covs,
                        exposure_col=None if exposure == "(none)" else exposure,
                        family=family,
                    ),
                    "key": (count_col, exposure, family, tuple(covs)),
                }
            except Exception as exc:
                error_card("The count model could not be fitted.", exc)
        cached = st.session_state.get("pois_result")
        if cached and cached["key"] == (count_col, exposure, family, tuple(covs)):
            res = cached["res"]
            disp = f", dispersion = {res.dispersion:.2f}" if res.dispersion is not None else ""
            st.caption(f"n = {res.n}, events = {res.events}, AIC = {res.aic:.1f}{disp}")
            _table(
                res.table,
                {
                    "coef": "{:.3f}",
                    "se": "{:.3f}",
                    "rate_ratio": "{:.3f}",
                    "ci_low": "{:.3f}",
                    "ci_high": "{:.3f}",
                    "z": "{:.2f}",
                    "p": "{:.4f}",
                },
            )
            for w in res.warnings:
                st.warning(w)
            st.caption("Method: " + " | ".join(res.references))


__all__ = ["run_survival"]
