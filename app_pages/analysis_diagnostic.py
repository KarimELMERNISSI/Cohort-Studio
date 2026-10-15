"""Diagnostic accuracy and method agreement tab of the Analyze page (audit S10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from biostats.agreement import bland_altman
from biostats.diagnostic import confusion_at, delong_auc, diagnostic_accuracy, youden_cutoff
from ui.components import error_card, help_popover, section, status_chip
from ui.results import provenance, recall, remember


def _ci_text(ci) -> str:
    return "" if ci is None else f"{ci.lower:.3f} to {ci.upper:.3f}"


def _roc_figure(y_true: pd.Series, y_score: pd.Series, pos_label, auc: dict, cutoff: float | None) -> go.Figure:
    from sklearn.metrics import roc_curve

    y = (y_true == pos_label).astype(int).to_numpy()
    fpr, tpr, thr = roc_curve(y, y_score.to_numpy())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {auc['auc']:.3f}, 95% CI {_ci_text(auc['ci'])})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line={"dash": "dash"}))
    if cutoff is not None and len(thr):
        i = int(np.argmin(np.abs(thr - cutoff)))
        fig.add_trace(go.Scatter(x=[fpr[i]], y=[tpr[i]], mode="markers", marker={"size": 12}, name=f"cut-off {cutoff:.3g}"))
    fig.update_layout(xaxis_title="1 - specificity", yaxis_title="sensitivity", xaxis={"range": [0, 1]}, yaxis={"range": [0, 1.02]},
                      height=420)
    return fig


def _bland_altman_figure(ba, label_a: str, label_b: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ba.means, y=ba.differences, mode="markers", name="differences"))
    x0, x1 = float(np.min(ba.means)), float(np.max(ba.means))
    for y, name, dash in ((ba.bias, f"bias {ba.bias:.3g}", "dash"), (ba.loa_upper, f"upper LoA {ba.loa_upper:.3g}", "dot"),
                          (ba.loa_lower, f"lower LoA {ba.loa_lower:.3g}", "dot")):
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode="lines", name=name, line={"dash": dash}))
    for ci in (ba.bias_ci, ba.loa_upper_ci, ba.loa_lower_ci):
        fig.add_trace(go.Scatter(x=[x0, x1, x1, x0], y=[ci.lower, ci.lower, ci.upper, ci.upper], fill="toself", mode="none",
                                 fillcolor="rgba(31,95,139,0.10)", showlegend=False, hoverinfo="skip"))
    fig.update_layout(xaxis_title=f"mean of {label_a} and {label_b}", yaxis_title=f"{label_a} - {label_b}", height=420)
    return fig


def run_diagnostic_accuracy(df: pd.DataFrame, analyzer) -> None:
    section("Diagnostic accuracy of a score or marker",
            "Sensitivity, specificity and predictive values with Wilson intervals; likelihood ratios with Simel intervals; AUC with the DeLong interval.")
    binary = list(analyzer.binary_cols)
    numeric = [c for c in analyzer.numeric_cols]
    if not binary or not numeric:
        st.info("This analysis needs one binary reference standard and one numeric score.")
    else:
        c1, c2, c3 = st.columns(3)
        outcome = c1.selectbox("Reference standard (binary outcome)", binary, key="diag_outcome",
                               help="The true disease status or event, coded with two levels.")
        score = c2.selectbox("Index test (score or marker)", [c for c in numeric if c != outcome], key="diag_score",
                             help="Higher values are assumed to indicate the positive level.")
        levels = sorted(df[outcome].dropna().unique(), key=str)
        pos_label = c3.selectbox("Positive (disease) level", levels, index=len(levels) - 1, key="diag_pos",
                                 help="The level of the reference standard that counts as the condition present.")
        c4, c5, c6 = st.columns(3)
        mode = c4.radio("Cut-off", ["Youden-optimal (data-driven)", "Manual"], key="diag_mode", horizontal=True,
                        help="The Youden cut-off is chosen on this data and is therefore optimistic; a pre-specified manual cut-off is preferable.")
        clean = df[[outcome, score]].dropna()
        clean = clean[clean[score].apply(lambda v: isinstance(v, (int, float, np.number)))]
        default_cut = float(clean[score].median()) if len(clean) else 0.0
        manual = c5.number_input("Manual cut-off (score >= cut-off is positive)", value=default_cut, key="diag_cut",
                                 help="Used when the cut-off mode is Manual.")
        observed_prev = float((clean[outcome] == pos_label).mean()) if len(clean) else 0.5
        prevalence = c6.number_input("Prevalence for predictive values", min_value=0.0, max_value=1.0, value=round(observed_prev, 3),
                                     step=0.01, key="diag_prev",
                                     help="PPV and NPV depend on the prevalence; enter the prevalence of the target population "
                                          "(default: the proportion observed in this dataset).")
        diag_config = (outcome, score, str(pos_label), mode, float(manual), float(prevalence))
        if st.button("Run diagnostic accuracy", type="primary", key="diag_run"):
            try:
                if clean[outcome].nunique() != 2:
                    raise ValueError("the reference standard must have exactly two levels after dropping missing values")
                auc = delong_auc(clean[outcome].tolist(), clean[score].tolist(), pos_label=pos_label)
                youden = youden_cutoff(clean[outcome].tolist(), clean[score].tolist(), pos_label=pos_label, n_boot=300)
                cutoff = youden["cutoff"] if mode.startswith("Youden") else float(manual)
                tp, fp, fn, tn = confusion_at(clean[outcome].tolist(), clean[score].tolist(), cutoff, pos_label=pos_label)
                acc = diagnostic_accuracy(tp, fp, fn, tn, prevalence=float(prevalence))
                remember("diagnostic",
                         {"auc": auc, "youden": youden, "cutoff": cutoff, "acc": acc, "outcome": outcome,
                          "score": score, "pos_label": pos_label, "mode": mode, "n": len(clean),
                          "fig": _roc_figure(clean[outcome], clean[score], pos_label, auc, cutoff)},
                         config=diag_config, n_analysed=len(clean), columns=[outcome, score])
            except Exception as exc:
                error_card("Diagnostic accuracy could not be computed.", exc)
        res = recall("diagnostic", config=diag_config)
        if res is not None:
            acc, auc = res["acc"], res["auc"]
            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AUC (DeLong 95% CI)", f"{auc['auc']:.3f}", help=f"95% CI {_ci_text(auc['ci'])}; {auc['n_positive']} positives, {auc['n_negative']} negatives")
                m1.caption(f"95% CI {_ci_text(auc['ci'])}")
                m2.metric("Cut-off", f"{res['cutoff']:.3g}", help=res["mode"])
                if res["mode"].startswith("Youden") and res["youden"]["ci"] is not None:
                    m2.caption(f"bootstrap 95% CI {_ci_text(res['youden']['ci'])}")
                m3.metric("Sensitivity", f"{acc.sensitivity:.3f}")
                m3.caption(f"95% CI {_ci_text(acc.sensitivity_ci)}")
                m4.metric("Specificity", f"{acc.specificity:.3f}")
                m4.caption(f"95% CI {_ci_text(acc.specificity_ci)}")
                table = pd.DataFrame([
                    ("Sensitivity", acc.sensitivity, _ci_text(acc.sensitivity_ci)),
                    ("Specificity", acc.specificity, _ci_text(acc.specificity_ci)),
                    ("PPV (observed prevalence)", acc.ppv, _ci_text(acc.ppv_ci)),
                    ("NPV (observed prevalence)", acc.npv, _ci_text(acc.npv_ci)),
                    (f"PPV at prevalence {acc.prevalence_assumed:.3g}", acc.ppv_at_prevalence, ""),
                    (f"NPV at prevalence {acc.prevalence_assumed:.3g}", acc.npv_at_prevalence, ""),
                    ("LR+", acc.lr_positive, _ci_text(acc.lr_positive_ci)),
                    ("LR-", acc.lr_negative, _ci_text(acc.lr_negative_ci)),
                    ("Diagnostic odds ratio", acc.dor, _ci_text(acc.dor_ci)),
                    ("Youden's J", acc.youden, ""),
                    ("Accuracy", acc.accuracy, ""),
                ], columns=["measure", "estimate", "95% CI"])
                st.dataframe(table.style.format({"estimate": "{:.3f}"}, na_rep="n/a"), hide_index=True, width="stretch")
                conf = pd.DataFrame({"test positive": [acc.tp, acc.fp], "test negative": [acc.fn, acc.tn]},
                                    index=[f"{res['pos_label']} (positive)", "negative"])
                st.dataframe(conf, width="stretch")
                for w in acc.warnings:
                    st.warning(w)
                if res["mode"].startswith("Youden"):
                    st.caption(res["youden"]["note"])
                st.caption("Method: " + " | ".join(acc.references + auc["references"]))
                help_popover("Sensitivity and specificity describe the test; predictive values describe the patient and change with the "
                             "prevalence. Likelihood ratios convert pre-test odds into post-test odds and do not depend on prevalence. "
                             "The AUC summarises discrimination across all cut-offs; its interval shows the precision.")
            st.plotly_chart(res["fig"], width="stretch")
            provenance("diagnostic")

    section("Agreement between two measurement methods (Bland-Altman)",
            "Bias and limits of agreement with confidence intervals; a proportional-bias check (Bland & Altman 1986).")
    if len(numeric) < 2:
        st.info("Two numeric columns measuring the same quantity are needed.")
        return
    a1, a2 = st.columns(2)
    method_a = a1.selectbox("Method A", numeric, key="ba_a", help="First measurement method (differences are A - B).")
    method_b = a2.selectbox("Method B", [c for c in numeric if c != method_a], key="ba_b", help="Second measurement method.")
    if st.button("Run Bland-Altman", key="ba_run"):
        try:
            ba = bland_altman(df[method_a], df[method_b])
            remember("bland_altman", ba, config=(method_a, method_b), n_analysed=ba.n, columns=[method_a, method_b])
        except Exception as exc:
            error_card("Bland-Altman could not be computed.", exc)
    ba = recall("bland_altman", config=(method_a, method_b))
    if ba is not None:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Bias (A - B)", f"{ba.bias:.3g}")
            m1.caption(f"95% CI {_ci_text(ba.bias_ci)}")
            m2.metric("Lower limit of agreement", f"{ba.loa_lower:.3g}")
            m2.caption(f"95% CI {_ci_text(ba.loa_lower_ci)}")
            m3.metric("Upper limit of agreement", f"{ba.loa_upper:.3g}")
            m3.caption(f"95% CI {_ci_text(ba.loa_upper_ci)}")
            m4.metric("Within limits", f"{ba.within_loa_pct:.1f}%", help=f"n = {ba.n}; SD of differences {ba.sd:.3g} (ddof = 1)")
            with m4:
                status_chip("proportional bias" if ba.proportional_bias_p < 0.05 else "no proportional bias",
                            "caution" if ba.proportional_bias_p < 0.05 else "ok")
            for w in ba.warnings:
                st.warning(w)
            st.caption("Method: " + " | ".join(ba.references))
            help_popover("The limits of agreement are the range within which 95% of differences between the two methods are expected. "
                         "Whether that range is acceptable is a clinical judgement, not a statistical one; the intervals around the "
                         "limits show how precisely they are estimated.")
        st.plotly_chart(_bland_altman_figure(ba, method_a, method_b), width="stretch")
        provenance("bland_altman")


__all__ = ["run_diagnostic_accuracy"]
