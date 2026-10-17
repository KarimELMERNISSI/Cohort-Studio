"""
Multiple imputation on the Enrich page (follow-up programme M2): a guided panel in five steps.

1. Population and missingness: what will be imputed, on which rows, how many imputations.
2. Imputation model: the variables (the outcome of the planned analyses always in) and a method each.
3. Settings: m, iterations, seed, strata, the auxiliary of a Cox analysis, parallel processes.
4. Run and diagnose: chain traces, R-hat, observed against imputed values.
5. Use, save, export: status against the analysis population, versions, mice and Stata files, methods text.

The imputation never changes the dataset: it is kept beside it (``AppState.set_mi``), recorded in the trace
and re-computed by the replay. The panel runs as a fragment so its widgets do not rerun the whole page.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from biostats._types import RANDOM_STATE
from biostats.mi.report import (
    default_model_columns,
    incomplete_share,
    methods_paragraph,
    missingness_summary,
)
from biostats.mi.spec import ImputationSpec, default_method, is_numeric, suggest_m
from core.session import PopulationError
from ui.components import error_card, help_popover
from ui.state import get_state

DO_NOT_IMPUTE = "do not impute"
METHOD_CHOICES = ["pmm", "norm", "rf", "logreg", "polyreg", DO_NOT_IMPUTE]
PARALLEL_WORK = (
    5e7  # rows x variables x m x iterations above which several processes pay for their start-up
)

STATUS_BADGE = {
    "fresh": (":material/check_circle:", "green", "Fits the analysis population"),
    "by_subset": (
        ":material/check_circle:",
        "green",
        "Fits the analysis population (whole strata)",
    ),
    "subset": (":material/warning:", "orange", "Imputed on more rows than the analysis population"),
    "stale": (":material/error:", "red", "Out of date"),
}


def _status_badge(state) -> None:
    current = state.mi_status()
    if current is None:
        return
    icon, color, label = STATUS_BADGE[current.kind]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(f"Multiple imputation: {label}", icon=icon, color=color)
        st.caption(f"m = {state.mi.mi.m} · {current.reason}")


@st.fragment
def render_multiple_imputation() -> None:
    state = get_state()
    if not state.has_data:
        st.info("Load a dataset first.")
        return
    try:
        frame = state.analysis_frame()
    except PopulationError as exc:
        st.error(f"The analysis population cannot be evaluated: {exc}")
        return

    st.markdown("### Multiple imputation (for inference)")
    st.caption(
        "Creates several completed versions of the analysis population; each analysis is run on every version and "
        "the results are combined with Rubin's rules, so that confidence intervals and p-values include the "
        "uncertainty due to the missing values. The dataset itself is not changed."
    )
    _status_badge(state)

    candidates = default_model_columns(frame)
    summary = missingness_summary(frame, candidates)

    # --- 1. population and missingness ---------------------------------------------------------------
    with st.container(border=True):
        st.markdown("**1. Population and missingness**")
        filled = state.single_imputed_counts(candidates)
        if filled:
            st.warning(
                f"{sum(filled.values()):,} values of these variables were filled by single imputation earlier "
                f"({', '.join(f'{c}: {n:,}' for c, n in filled.items())}). The multiple imputation treats them "
                "as observed, which understates the uncertainty. Impute the data as they were before the single "
                "imputation: reload a version saved before it, or replay the trace up to the step before it.",
                icon=":material/warning:",
            )
        exact = state.exact_imputed_counts(candidates)
        if exact:
            st.info(
                f"{sum(exact.values()):,} values of these variables were imputed earlier by an exact formula "
                f"({', '.join(f'{c}: {n:,}' for c, n in exact.items())}). The other variables determine them, so "
                "the multiple imputation keeps them as observed values and has fewer cells to impute.",
                icon=":material/calculate:",
            )
        if summary.empty:
            st.success(
                "No variable of the analysis population has missing values: there is nothing to impute."
            )
            return
        population = state.population
        share = incomplete_share(frame, candidates)
        hint = suggest_m(frame, candidates)
        with st.container(horizontal=True):
            st.metric(
                "Analysis population",
                f"{len(frame):,} rows",
                help=population.label if population else None,
            )
            st.metric(
                "Incomplete rows",
                f"{share:.0%}",
                help="Rows with at least one missing value in the model.",
            )
            st.metric(
                "Suggested imputations",
                hint,
                help="White, Royston & Wood (2011): at least the percentage of incomplete rows.",
            )
        st.dataframe(
            summary.drop(columns="method"),
            hide_index=True,
            column_config={
                "column": "Variable",
                "missing": st.column_config.NumberColumn("Missing", format="localized"),
                "share": st.column_config.ProgressColumn(
                    "Share", format="percent", min_value=0.0, max_value=1.0
                ),
                "kind": "Type",
            },
            alt="Missing values per variable of the analysis population",
        )
        st.caption(
            "The imputation assumes the values are missing at random given the variables of the model: include the "
            "variables that predict whether a value is missing and every variable of the planned analyses."
        )

    # --- 2. model -------------------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("**2. Imputation model**")
        columns = st.multiselect(
            "Variables in the imputation model",
            options=candidates,
            default=candidates,
            key="mi_columns",
            help="Every variable of the model predicts the others. Identifiers, dates and free text are left out.",
        )
        outcomes = st.multiselect(
            "Outcome of the planned analyses",
            options=candidates,
            key="mi_outcomes",
            help="Always kept in the imputation model: leaving the outcome out biases its associations toward the "
            "null (Moons et al. 2006; White, Royston & Wood 2011).",
        )
        model = list(dict.fromkeys([*columns, *outcomes]))
        table = summary[summary["column"].isin(model)].copy()
        if table.empty:
            st.info("None of the chosen variables has missing values.")
            return
        edited = st.data_editor(
            table[["column", "missing", "kind", "method"]],
            hide_index=True,
            disabled=["column", "missing", "kind"],
            column_config={
                "column": "Variable",
                "missing": st.column_config.NumberColumn("Missing", format="localized"),
                "kind": "Type",
                "method": st.column_config.SelectboxColumn(
                    "Method",
                    options=METHOD_CHOICES,
                    required=True,
                    help="pmm, norm or rf for numbers; logreg (two categories), polyreg or rf for categories.",
                ),
            },
            key=f"mi_methods::{'|'.join(table['column'])}",
            alt="Imputation method per incomplete variable",
        )
        help_popover(
            "**pmm** (predictive mean matching, the default for numbers): imputes values observed in similar rows, "
            "so imputations stay plausible (bounds, integers, skewness).\n\n**norm**: Bayesian linear regression, "
            "for unbounded, roughly normal numbers.\n\n**logreg / polyreg**: logistic and multinomial regression "
            "for categories.\n\n**rf**: random-forest donors (Doove et al. 2014); captures interactions and "
            "non-linear relations the model does not spell out, at the cost of intervals that can be slightly too "
            "narrow in small samples. Useful as a sensitivity analysis.",
            label="Which method",
        )

    # --- 3. settings ----------------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("**3. Settings**")
        with st.container(horizontal=True):
            m = int(st.number_input("Imputations (m)", 2, 500, min(max(hint, 5), 100), key="mi_m"))
            maxit = int(
                st.number_input(
                    "Iterations",
                    1,
                    200,
                    10,
                    key="mi_maxit",
                    help="5 to 20 is usual; check the traces.",
                )
            )
            seed = int(
                st.number_input(
                    "Seed",
                    0,
                    2**31 - 1,
                    RANDOM_STATE,
                    key="mi_seed",
                    help="Same seed, same imputations.",
                )
            )
        strata = [
            c
            for c in frame.columns
            if not is_numeric(frame[c])
            and frame[c].notna().all()
            and 2 <= frame[c].astype(str).nunique() <= 10
        ]
        by = st.selectbox(
            "Impute separately within",
            ["(no strata)", *strata],
            key="mi_by",
            help="For example the trial arm: each level is imputed with its own model.",
        )
        complete_numeric = [
            c for c in frame.columns if is_numeric(frame[c]) and frame[c].notna().all()
        ]
        binary = [
            c
            for c in frame.columns
            if frame[c].notna().all() and frame[c].astype(str).nunique() == 2
        ]
        survival = None
        if st.toggle(
            "A Cox model is planned",
            key="mi_cox",
            help="Adds the event indicator and the Nelson-Aalen "
            "cumulative hazard as predictors instead of the follow-up time (White & Royston 2009).",
        ):
            with st.container(horizontal=True):
                time_col = st.selectbox("Follow-up time", complete_numeric, key="mi_time")
                event_col = st.selectbox(
                    "Event indicator",
                    binary,
                    key="mi_event",
                    help="1, true, yes, event or died count as an event.",
                )
            if time_col and event_col and time_col != event_col:
                survival = (time_col, event_col)
        work = len(frame) * len(model) * m * maxit
        parallel = st.toggle(
            "Use several processor cores",
            value=work > PARALLEL_WORK,
            key="mi_parallel",
            help="Same result; worth it for large imputations (starting the processes takes several seconds).",
        )

    methods = {
        row["column"]: ("" if row["method"] == DO_NOT_IMPUTE else row["method"])
        for _, row in edited.iterrows()
        if row["method"] != default_method(frame[row["column"]])
    }
    if st.button(
        "Run the multiple imputation", type="primary", icon=":material/play_arrow:", key="mi_run"
    ):
        try:
            spec = ImputationSpec(
                columns=model,
                methods=methods,
                m=m,
                maxit=maxit,
                seed=seed,
                by=None if by == "(no strata)" else by,
                survival=survival,
            )
            bar = st.progress(0.0, text="Imputing")
            from biostats.mi import impute

            run = impute(
                frame,
                spec,
                progress=lambda f: bar.progress(f, text=f"Imputing: {f:.0%}"),
                workers=max(1, (os.cpu_count() or 2) - 1) if parallel else 1,
            )
            state.set_mi(run)
        except Exception as exc:
            error_card("The multiple imputation failed.", exc)
        else:
            st.rerun(scope="app")

    stored = state.mi
    if stored is None:
        return
    _diagnostics(stored)
    _use(state, stored)


def _trace_figure(trace: dict, name: str) -> go.Figure:
    means = np.asarray(trace["mean"], dtype=float)
    fig = go.Figure()
    for chain in range(means.shape[1]):
        fig.add_scatter(
            x=np.arange(1, len(means) + 1), y=means[:, chain], mode="lines", showlegend=False
        )
    fig.update_layout(
        height=260,
        margin=dict(t=10, b=10),
        xaxis_title="Iteration",
        yaxis_title=f"Mean of the imputed {name}",
    )
    return fig


def _distribution_figure(observed: pd.Series, imputed: pd.Series, numeric: bool) -> go.Figure:
    fig = go.Figure()
    if numeric:
        fig.add_histogram(
            x=observed.astype(float), name="Observed", histnorm="probability density", opacity=0.6
        )
        fig.add_histogram(
            x=imputed.astype(float), name="Imputed", histnorm="probability density", opacity=0.6
        )
        fig.update_layout(barmode="overlay")
    else:
        for label, values in (("Observed", observed), ("Imputed", imputed)):
            share = values.astype(str).value_counts(normalize=True).sort_index()
            fig.add_bar(x=share.index, y=share.to_numpy(), name=label)
    fig.update_layout(
        height=280, margin=dict(t=10, b=10), yaxis_title="Density" if numeric else "Share"
    )
    return fig


def _diagnostics(stored) -> None:
    with st.container(border=True):
        st.markdown("**4. Diagnostics**")
        diag = stored.diagnostics
        rhat = pd.DataFrame(
            {"variable": list(diag.get("rhat", {})), "rhat": list(diag.get("rhat", {}).values())}
        )
        if not rhat.empty:
            high = rhat[rhat["rhat"] > 1.1]
            if not high.empty:
                st.warning(
                    "The chains of "
                    + ", ".join(high["variable"])
                    + " have not mixed well (R-hat above 1.1): "
                    "increase the iterations."
                )
        imputed = list(stored.mi.imputations)
        if not imputed:
            return
        variable = st.selectbox("Variable", imputed, key="mi_diag_variable")
        traces = diag.get("trace", {})
        names = [k for k in traces if k == variable or k.startswith(f"{variable} [")]
        left, right = st.columns(2)
        with left:
            st.caption(
                "Chain means per iteration (each line one imputation): they should mix without a trend."
            )
            for name in names:
                st.plotly_chart(
                    _trace_figure(traces[name], variable), width="stretch", key=f"mi_trace::{name}"
                )
        with right:
            st.caption(
                "Observed against imputed values: differences are expected when the data are not MCAR."
            )
            observed = stored.mi.base[variable].dropna()
            values = stored.mi.imputations[variable].stack()
            numeric = is_numeric(stored.mi.base[variable])
            st.plotly_chart(
                _distribution_figure(observed, values, numeric),
                width="stretch",
                key="mi_distribution",
            )
        if not rhat.empty:
            st.dataframe(
                rhat,
                hide_index=True,
                column_config={
                    "variable": "Variable",
                    "rhat": st.column_config.NumberColumn("R-hat", format="%.3f"),
                },
                alt="Potential scale reduction of the chain means per variable",
            )
        dropped = diag.get("dropped") or {}
        if dropped:
            with st.expander("Predictors left out (constant, collinear or dependent)"):
                for target, cols in dropped.items():
                    st.markdown(f"**{target}**: {', '.join(cols)}")
        for warning in diag.get("warnings") or []:
            st.warning(warning)


def _use(state, stored) -> None:
    with st.container(border=True):
        st.markdown("**5. Use, save, export**")
        current = state.mi_status()
        if current is not None and not current.usable:
            st.warning(
                f"{current.reason}. Run the imputation again on the current analysis population."
            )
        with st.container(horizontal=True):
            if st.button(
                "Save a version with this imputation",
                icon=":material/save:",
                key="mi_save",
                disabled=current is None or not current.usable,
            ):
                ok, msg, _ = state.save_version()
                (st.success if ok else st.error)(msg)
            st.download_button(
                "mice long (.csv)",
                data=lambda: stored.mi.long().to_csv(index=False).encode("utf-8"),
                file_name="imputations_mice_long.csv",
                mime="text/csv",
                icon=":material/download:",
                on_click="ignore",
                key="mi_export_mice",
                help="Columns .imp (0 = observed) and .id: in R, mice::as.mids(read.csv(...)).",
            )
            st.download_button(
                "Stata flong (.csv)",
                data=lambda: stored.mi.flong().to_csv(index=False).encode("utf-8"),
                file_name="imputations_stata_flong.csv",
                mime="text/csv",
                icon=":material/download:",
                on_click="ignore",
                key="mi_export_stata",
                help="In Stata: import delimited, then mi import flong, m(_mi_m) id(_mi_id) imputed(...).",
            )
            if st.button("Discard the imputation", icon=":material/delete:", key="mi_discard"):
                state.clear_mi()
                st.rerun(scope="app")
        st.markdown("Methods paragraph")
        st.code(
            methods_paragraph(
                stored.binding,
                n_rows=stored.binding.n_rows,
                incomplete=incomplete_share(stored.mi.base, stored.binding.columns),
                info=dict(stored.mi.info),
            ),
            language=None,
            wrap_lines=True,
        )


__all__ = ["render_multiple_imputation"]
