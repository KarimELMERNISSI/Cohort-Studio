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
from core.derived import decided_on_data, record
from core.registry import as_code
from core.session import PopulationError
from enrich.derived import DERIVED_METHOD, derivation_steps, impute_and_derive, recomputed
from ui.components import error_card, help_popover
from ui.state import get_state

DO_NOT_IMPUTE = "do not impute"
DERIVE_KEY = "mi_derive"
# above this share of its values computed from imputed ones, a derived variable analysed non-linearly is attenuated
NONLINEAR_SHARE = 0.05
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
    # the declared derived variables (Derived variables tab), computed in each imputed dataset rather than imputed; one
    # whose formula reads a variable that cannot join the model (a date, a free text) is imputed as its own variable
    pairs = [
        (d, c)
        for d, c in recomputed(state.derived_variables, state.derived_check)
        if d.target in candidates
    ]
    found = [
        s
        for s in derivation_steps(
            state.derived_variables, state.derived_check, frame, imputed=state.imputed_cells()
        )
        if s["target"] in candidates
    ]
    steps = [s for s in found if all(i in candidates for i in s["inputs"])]
    beyond = {
        s["target"]: [i for i in s["inputs"] if i not in candidates]
        for s in found
        if s not in steps
    }
    computed = [s["target"] for s in steps] if st.session_state.get(DERIVE_KEY, True) else []
    imputed_candidates = [c for c in candidates if c not in computed]

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
        # the derived variables computed from the others add no imputation uncertainty: not counted
        share = incomplete_share(frame, imputed_candidates)
        hint = suggest_m(frame, imputed_candidates)
        draft = state.session.draft
        if draft is not None and any(decided_on_data(p.params.get("derived")) for p in draft.steps):
            st.warning(
                "The draft holds values filled exactly that are not applied yet: apply the draft first, as the "
                "imputation is computed on the dataset.",
                icon=":material/edit_note:",
            )
        else:
            fill = state.exact_fill_plan()
            if fill.filled and not fill.problem:
                if any(f.inverse for f in fill.fills):
                    st.warning(
                        f"{fill.filled:,} values of the declared derived variables or of their variables can be "
                        "computed exactly (Derived variables, Fill exactly), some by an inverse: fill them and apply "
                        "the draft first. Imputed instead, those variables would no longer match the recorded derived "
                        "values beside them.",
                        icon=":material/calculate:",
                    )
                else:
                    st.info(
                        f"{fill.filled:,} values of the declared derived variables can be computed exactly "
                        "(Derived variables, Fill exactly): fill them and apply the draft first, so that they need "
                        "no imputation.",
                        icon=":material/calculate:",
                    )
        with st.container(horizontal=True):
            st.metric(
                "Analysis population",
                f"{len(frame):,} rows",
                help=population.label if population else None,
            )
            st.metric(
                "Incomplete rows",
                f"{share:.0%}",
                help="Rows with at least one missing value in the model"
                + (
                    ", the derived variables computed from the others left out."
                    if computed
                    else "."
                ),
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
        in_model = [s for s in steps if s["target"] in model]
        derive = False
        if in_model:
            derive = st.checkbox(
                "Compute the declared derived variables in each imputed dataset from its imputed values",
                value=True,
                key=DERIVE_KEY,
                help="A derived variable imputed as a variable of its own no longer follows its formula, and the "
                "chains predict the height from the imputed BMI and back, so that they barely move. Ticked, each is "
                "left out of the chains, neither imputed nor used to predict any variable, and the variables of its "
                "formula join the model, which then carries its information; its missing values are computed by its "
                "formula in each imputed dataset (passive imputation: impute, then derive). Its recorded values are "
                "kept.",
            )
            one = len(in_model) == 1
            names = ", ".join(as_code(s["target"]) for s in in_model)
            st.caption(
                f"{names}: computed in each imputed dataset by {'its formula' if one else 'their formulas'}; a value "
                "computed from imputed ones varies between the imputations as they do."
                if derive
                else f"{names}: imputed as {'a variable of its own' if one else 'variables of their own'}."
            )
            if derive:  # the variables of their formulas carry their information: in the model
                added = [i for s in in_model for i in s["inputs"] if i not in model]
                if added:
                    model = list(dict.fromkeys([*model, *added]))
                    st.caption(
                        "In the model as variables of their formulas: "
                        + ", ".join(as_code(i) for i in dict.fromkeys(added))
                        + "."
                    )
        for target, names in beyond.items():
            if target in model:
                st.caption(
                    f"{as_code(target)}: imputed as a variable of its own, as a variable of its formula "
                    f"({', '.join(as_code(n) for n in names)}) cannot join the imputation model (a date, an "
                    "identifier or a free text)."
                )
        left_out = [s["target"] for s in steps if s["target"] not in model]
        if left_out:
            st.caption(
                "Declared derived variables outside the model are not computed: "
                + ", ".join(as_code(t) for t in left_out)
                + "."
            )
        kept_out = [s["target"] for s in in_model] if derive else []
        table = summary[summary["column"].isin(model) & ~summary["column"].isin(kept_out)].copy()
        if table.empty:
            st.info(
                "Only derived variables are missing: Fill exactly computes those whose variables are recorded "
                "(Derived variables)."
                if kept_out
                else "None of the chosen variables has missing values."
            )
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
            and bool(frame[c].notna().all())
            and 2 <= int(frame[c].astype(str).nunique()) <= 10
        ]
        by = st.selectbox(
            "Impute separately within",
            ["(no strata)", *strata],
            key="mi_by",
            help="For example the trial arm: each level is imputed with its own model.",
        )
        complete_numeric = [
            c for c in frame.columns if is_numeric(frame[c]) and bool(frame[c].notna().all())
        ]
        binary = [
            c
            for c in frame.columns
            if bool(frame[c].notna().all()) and int(frame[c].astype(str).nunique()) == 2
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
        row["column"]: ("" if str(row["method"]) == DO_NOT_IMPUTE else row["method"])
        for _, row in edited.iterrows()
        if str(row["method"]) != default_method(frame[row["column"]])
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
            from manage.compute import mi_workers

            # the worker processes are shared by every session of the server: a lease grants what is free
            budget = mi_workers()
            with budget.lease(min(spec.m, budget.total) if parallel else 1) as workers:
                label = f"Imputing in {workers} processes" if workers > 1 else "Imputing"
                if parallel and workers == 1:
                    st.caption(
                        "This server allows one process for imputations (CMP_MI_MAX_WORKERS)."
                        if budget.total <= 1
                        else "Other imputations use the processor cores of this server: this one runs in one."
                    )
                bar = st.progress(0.0, text=label)
                run = impute_and_derive(
                    frame,
                    spec,
                    in_model if derive else [],
                    declined=[] if derive else [s["record"] for s in in_model],
                    # recorded derived values their imputed variables may no longer match
                    watch=[
                        record(d, [d.target], c.summary()) for d, c in pairs if d.target in model
                    ],
                    progress=lambda f: bar.progress(f, text=f"{label}: {f:.0%}"),
                    workers=workers,
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


def _computed_share(mi, variable: str) -> float:
    """The share of the rows whose derived variable was computed from imputed values (a variable of its formula was
    imputed there), from the formula's variables the imputation recorded."""
    steps = (mi.info.get("derived") or {}).get("steps") or []
    inputs = next((s.get("inputs") or [] for s in steps if s.get("target") == variable), [])
    computed = mi.imputations.get(variable)
    if computed is None or not len(mi.base):
        return 0.0
    rows = computed.index
    from_imputed = np.zeros(len(rows), dtype=bool)
    for name in inputs:
        if name in mi.imputations:
            from_imputed |= rows.isin(mi.imputations[name].index)
    return float(from_imputed.sum()) / len(mi.base)


def _distribution_figure(
    observed: pd.Series, imputed: pd.Series, numeric: bool, *, label: str = "Imputed"
) -> go.Figure:
    fig = go.Figure()
    if numeric:
        fig.add_histogram(
            x=observed.astype(float), name="Observed", histnorm="probability density", opacity=0.6
        )
        fig.add_histogram(
            x=imputed.astype(float), name=label, histnorm="probability density", opacity=0.6
        )
        fig.update_layout(barmode="overlay")
    else:
        for name, values in (("Observed", observed), (label, imputed)):
            share = values.astype(str).value_counts(normalize=True).sort_index()
            fig.add_bar(x=share.index, y=share.to_numpy(), name=name)
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
        derived = stored.mi.methods.get(variable) == DERIVED_METHOD
        traces = diag.get("trace", {})
        names = [k for k in traces if k == variable or k.startswith(f"{variable} [")]
        left, right = st.columns(2)
        with left:
            if derived:
                st.caption(
                    "Computed in each imputed dataset by its formula from the imputed values: there is no chain to "
                    "check."
                )
                share = _computed_share(stored.mi, variable)
                if share > NONLINEAR_SHARE:
                    st.caption(
                        f":material/info: {share:.0%} of its values were computed from imputed ones. Analysed in "
                        "categories, splines or interactions, its association is attenuated (von Hippel 2009); "
                        "substantive-model-compatible imputation (smcfcs; Bartlett et al. 2015) is a sensitivity "
                        "analysis."
                    )
            else:
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
                _distribution_figure(
                    observed, values, numeric, label="Computed" if derived else "Imputed"
                ),
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
        edited = state.mi_derived_edited()
        if edited:
            st.warning(
                "This imputation computed "
                + ", ".join(as_code(t) for t in edited)
                + " with a definition edited or removed since (Derived variables): it keeps the formula it used. "
                "Run it again to use the current one."
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
                help="In Stata: import delimited, then mi import flong, m(_mi_m) id(_mi_id) imputed(...)"
                + (
                    " passive("
                    + " ".join(c for c, m in stored.mi.methods.items() if m == DERIVED_METHOD)
                    + ")"
                    if DERIVED_METHOD in stored.mi.methods.values()
                    else ""
                )
                + ".",
            )
            if st.button("Discard the imputation", icon=":material/delete:", key="mi_discard"):
                state.clear_mi()
                st.rerun(scope="app")
        st.markdown("Methods paragraph")
        st.code(
            methods_paragraph(
                stored.binding,
                n_rows=stored.binding.n_rows,
                # the derived variables computed from the others are not counted, as on the page
                incomplete=incomplete_share(
                    stored.mi.base,
                    [
                        c
                        for c in stored.binding.columns
                        if stored.binding.methods.get(c) != DERIVED_METHOD
                    ],
                ),
                info=dict(stored.mi.info),
            ),
            language=None,
            wrap_lines=True,
        )


__all__ = ["render_multiple_imputation"]
