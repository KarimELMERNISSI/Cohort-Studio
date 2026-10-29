"""
Observed or multiply imputed data on the analysis pages (follow-up programme M2).

The sidebar selector (under the population) appears once an imputation exists; the analysis pages call
``mi_banner`` and get the imputed datasets restricted to the analysis population, or ``None`` to analyse the
observed data. A stale imputation is never used silently: the page says why and analyses the observed data.
The helpers below show a pooled result (estimates with FMI and df, D1/D2 tests, notes, the complete-case
analysis for comparison).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from ui.components.text import plain

MODE_KEY = "analysis_data_mode"
ALLOW_SUBSET_KEY = "mi_allow_subset"
OBSERVED = "observed"
MULTIPLE = "multiple"

STATUS_ICON = {
    "fresh": ":material/check_circle:",
    "by_subset": ":material/check_circle:",
    "subset": ":material/warning:",
    "stale": ":material/error:",
}


def render_mi_selector(state) -> None:
    """Sidebar: analyse the observed data or the multiply imputed datasets (only when an imputation exists)."""
    stored = state.mi
    if stored is None:
        st.session_state.pop(MODE_KEY, None)
        return
    current = state.mi_status()
    observed = "Observed + single-imputed" if state.single_imputed_counts() else "Observed"
    st.segmented_control(
        "Analysis data",
        [OBSERVED, MULTIPLE],
        format_func=lambda v: observed if v == OBSERVED else f"Imputed (m = {stored.mi.m})",
        key=MODE_KEY,
        default=OBSERVED,
        help="Observed: complete cases of each analysis. Imputed: every analysis is run on the m imputed datasets "
        "and pooled with Rubin's rules (Enrich > Handle missing data > Multiple imputation).",
    )
    if current is not None and st.session_state.get(MODE_KEY) == MULTIPLE:
        st.caption(f"{STATUS_ICON[current.kind]} {current.reason}")


def wants_imputed() -> bool:
    return st.session_state.get(MODE_KEY) == MULTIPLE


def mi_banner(state) -> Any | None:
    """
    Top of an analysis page: the imputed datasets to pool (biostats.mi.data.MIData), or None for the observed data.
    """
    if not wants_imputed() or state.mi is None:
        return None
    current = state.mi_status()
    if current is None:
        return None
    if current.kind == "subset":
        allow = st.checkbox(
            "Use the imputation although it was fitted on more rows than the analysis population",
            key=ALLOW_SUBSET_KEY,
            help="The imputation model also learnt from rows outside the current population. For a definitive "
            "analysis, run the imputation again on the current population.",
        )
        if not allow:
            st.warning(
                f"Imputed data selected, but {current.reason}. The analyses below use the observed data.",
                icon=":material/warning:",
            )
            return None
    elif not current.usable:
        st.warning(
            f"Imputed data selected, but the imputation is out of date: {current.reason}. The analyses below use the "
            "observed data; run the imputation again (Enrich > Handle missing data > Multiple imputation).",
            icon=":material/error:",
        )
        return None
    mi = state.analysis_mi(allow_subset=current.kind == "subset")
    if mi is None:
        return None
    st.info(
        f"Analyses pool **{mi.m} imputed datasets** of the {len(mi.base):,} rows of the analysis population "
        "(Rubin's rules, Barnard-Rubin df).",
        icon=":material/stacks:",
    )
    return mi


def mi_token(state) -> str:
    """What identifies the imputation in the settings of a stored pooled result."""
    stored = state.mi
    return "" if stored is None else f"{stored.binding.run_id}:{stored.binding.created_at}"


def pooled_frame(
    result, *, ratio_label: str | None = None, drop: tuple[str, ...] = ("const",)
) -> pd.DataFrame:
    """The pooled terms: estimate (ratio when pooled on the log scale), 95% CI, p, df, FMI and RIV."""
    frame = result.to_frame()
    if frame.empty:
        return frame
    frame = frame[~frame["term"].isin(drop)].reset_index(drop=True)
    on_log = frame["scale"] == "log"
    out = pd.DataFrame({"term": frame["term"]})
    if ratio_label and on_log.any():
        out[ratio_label] = frame["natural_estimate"].where(on_log, frame["estimate"])
        out["ci_low"] = frame["natural_ci_low"].where(on_log, frame["ci_low"])
        out["ci_high"] = frame["natural_ci_high"].where(on_log, frame["ci_high"])
        out["coef"] = frame["estimate"]
    else:
        out["estimate"] = frame["estimate"]
        out["ci_low"] = frame["ci_low"]
        out["ci_high"] = frame["ci_high"]
    out["se"] = frame["se"]
    out["p"] = frame["p_value"]
    out["df"] = frame["df"]
    out["fmi"] = frame["fmi"]
    out["riv"] = frame["riv"]
    return out


def show_pooled_terms(
    result, *, ratio_label: str | None = None, alt: str = "Pooled estimates"
) -> None:
    frame = pooled_frame(result, ratio_label=ratio_label)
    number = st.column_config.NumberColumn
    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            "term": "Term",
            **(
                {ratio_label: number(ratio_label.replace("_", " ").capitalize(), format="%.3f")}
                if ratio_label
                else {}
            ),
            "estimate": number("Estimate", format="%.3f"),
            "coef": number("Coefficient (log)", format="%.3f"),
            "ci_low": number("95% CI low", format="%.3f"),
            "ci_high": number("95% CI high", format="%.3f"),
            "se": number("SE", format="%.3f"),
            "p": number("p", format="%.4f"),
            "df": number("df", format="%.1f", help="Barnard-Rubin degrees of freedom"),
            "fmi": number("FMI", format="%.3f", help="Fraction of missing information"),
            "riv": number(
                "RIV", format="%.3f", help="Relative increase in variance due to the missing data"
            ),
        },
        alt=alt,
    )


def show_tests(result, *, title: str = "Tests of several terms at once") -> None:
    """D1 (factors, covariate blocks) and D2 (chi-square statistics) of a pooled result."""
    if not result.tests:
        return
    rows = [
        {
            "test": name,
            "method": test.method,
            "F": test.statistic,
            "df1": test.df1,
            "df2": test.df2,
            "p": test.p_value,
            "riv": test.riv,
        }
        for name, test in result.tests.items()
    ]
    number = st.column_config.NumberColumn
    st.markdown(f"**{title}**")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            "test": "Variable",
            "method": st.column_config.TextColumn(
                "Method", help="D1: pooled Wald test; D2: pooled chi-square"
            ),
            "F": number("F", format="%.3f"),
            "df1": number("df1", format="%.0f"),
            "df2": number("df2", format="%.1f"),
            "p": number("p", format="%.4f"),
            "riv": number("RIV", format="%.3f"),
        },
        alt="Pooled tests of several terms",
    )
    if any(t.method == "D2" for t in result.tests.values()):
        st.caption(
            "D2 combines chi-square statistics; it is approximate and can be conservative or liberal."
        )


def show_mi_notes(
    result,
    *,
    complete_case: Callable[[Any], None] | None = None,
    label: str = "Complete-case analysis, for comparison",
    caption: bool = True,
    warnings: bool = True,
) -> None:
    """
    Pooling caption, preview note, warnings counted across imputations and the complete-case analysis; ``caption``
    and ``warnings`` False leave the pooling line and the warnings to a caller showing several pooled results.
    """
    if caption:
        st.caption(
            f":material/stacks: Pooled over {result.m} imputed datasets with Rubin's rules (Barnard-Rubin df)."
            + (" Preview: not every imputed dataset was analysed." if result.partial else "")
        )
    for note in result.notes:
        st.caption(note)
    for warning in result.warnings if warnings else ():
        st.warning(plain(warning))  # a model's message can name a level or a term: as text
    if complete_case is not None and result.complete_case is not None:
        with st.expander(label):
            complete_case(result.complete_case)


__all__ = [
    "ALLOW_SUBSET_KEY",
    "MODE_KEY",
    "MULTIPLE",
    "OBSERVED",
    "mi_banner",
    "mi_token",
    "pooled_frame",
    "render_mi_selector",
    "show_mi_notes",
    "show_pooled_terms",
    "show_tests",
    "wants_imputed",
]
