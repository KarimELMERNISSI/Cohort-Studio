"""Multiple imputation with Rubin's rules on the Enrich page (audit S4, stage 2)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from biostats.imputation import multiple_impute, pooled_statistics
from ui.components import error_card, help_popover, long_op


def render_multiple_imputation() -> None:
    df = st.session_state.get("enriched_df")
    if df is None:
        return
    with st.expander("Multiple imputation (Rubin's rules)", expanded=False):
        st.caption(
            "Draws m completed datasets with chained equations (posterior sampling), pools per-variable means with Rubin's rules, "
            "and reports the fraction of missing information. Use it to see how much the imputation uncertainty matters."
        )
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        with_missing = [c for c in numeric if df[c].isna().any()]
        if not with_missing:
            st.info("No numeric column has missing values.")
            return
        c1, c2 = st.columns(2)
        m = int(
            c1.number_input(
                "Number of imputations (m)",
                min_value=2,
                max_value=50,
                value=5,
                key="mi_m",
                help="5 to 20 is usual; more when the fraction of missing information is high.",
            )
        )
        exclude = c2.multiselect(
            "Columns not to impute (e.g. the outcome)",
            numeric,
            key="mi_exclude",
            help="Excluded columns keep their missing values and are not used as predictors.",
        )
        if st.button("Run multiple imputation", key="mi_run", type="primary"):
            try:
                with long_op(f"Drawing {m} completed datasets"):
                    completed = multiple_impute(df, m=m, exclude=exclude)
                    table = pooled_statistics(
                        completed, [c for c in with_missing if c not in exclude]
                    )
                st.session_state["mi_result"] = {"completed": completed, "table": table, "m": m}
            except Exception as exc:
                error_card("Multiple imputation failed.", exc)
        res = st.session_state.get("mi_result")
        if res:
            st.dataframe(
                res["table"].style.format(
                    {
                        "estimate": "{:.4g}",
                        "se": "{:.3g}",
                        "ci_low": "{:.4g}",
                        "ci_high": "{:.4g}",
                        "df": "{:.1f}",
                        "fmi": "{:.3f}",
                        "between_variance": "{:.3g}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            if (res["table"]["fmi"] > 0.3).any():
                st.warning(
                    "The fraction of missing information exceeds 0.3 for some variables: increase m or review the imputation model."
                )
            if st.button(
                "Use the first completed dataset as the working data",
                key="mi_adopt",
                help="Keeps one completed copy for exploration; inference on it ignores the between-imputation variance shown above.",
            ):
                first = res["completed"][0]
                st.session_state["enriched_df"] = first
                st.session_state["data"] = first
                st.session_state["working_df"] = first.copy()
                if "transformation_manager" in st.session_state:
                    st.session_state.transformation_manager.add_step(
                        "multiple_imputation",
                        {
                            "m": res["m"],
                            "adopted": 1,
                            "excluded": list(st.session_state.get("mi_exclude", [])),
                        },
                        f"Multiple imputation (m = {res['m']}); first completed dataset adopted",
                    )
                st.success("First completed dataset adopted as the working data.")
            help_popover(
                "Rubin's rules: the pooled estimate is the mean of the m estimates; its variance adds the within-imputation "
                "variance to (1 + 1/m) times the between-imputation variance. The fraction of missing information tells "
                "how much of the uncertainty comes from the missing data."
            )


__all__ = ["render_multiple_imputation"]
