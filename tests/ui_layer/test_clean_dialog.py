"""
A variable cleaned where it is analysed (app_pages/clean_dialog.py): the warning under a list of groups offers to
clean it in a dialog holding Data > Clean's editor, then to group by its clean copy; the pooled comparison, which reads
the imputed datasets, says that a copy written after the imputation needs it run again; a variable that Clean does not
list is said so. (Apply inside the dialog reruns it as a fragment, which AppTest does not replay: the browser check
covers it.)
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

SITES = ["Metz", "Metz ", "metz", "Esch"] * 10


def _script(
    values: list[str], with_copy: bool = False, pooled: bool = False, direct: bool = False
) -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.analysis_config import render_analysis_configuration
    from app_pages.clean_dialog import group_warning
    from core.state import DatasetRef
    from enrich.recode import recode_column
    from ingest.analyzer import DataAnalyzer
    from ui.state import get_state

    st.session_state.setdefault("username", "dialog_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        frame = pd.DataFrame({"site": values, "value": range(len(values))})
        state.set_dataset(
            frame,
            DatasetRef(name="sites", origin="upload", file_hash="sites"),
            source_bytes=frame.to_csv(index=False).encode("utf-8"),
            source_name="sites.csv",
        )
        if with_copy:  # the copy that Apply writes
            mapping = {"Metz ": "Metz", "metz": "Metz"}
            new, _ = recode_column(state.base, "site", "site_clean", mapping)
            state.commit_change(
                new,
                "recode_levels",
                {"source": "site", "target": "site_clean", "mapping": mapping},
                "Recoded site into site_clean",
            )
    df = state.base
    if direct:  # the helper alone: a page would not offer so many groups
        group_warning("site", sorted(df["site"].unique()), group_key="t_group_col")
        return
    if pooled:  # the imputed datasets hold the variables of the imputation: not the copy
        group_warning(
            "site",
            sorted(df["site"].unique()),
            group_key="mi_cmp_group",
            groupable=["site"],
            imputed=True,
        )
        return
    analyzer = DataAnalyzer(df)
    render_analysis_configuration(
        df,
        analyzer.numeric_cols,
        analyzer.categorical_cols,
        analyzer.binary_cols,
        default_group_col="site",
        key_prefix="t_",
    )


def _open(**kwargs) -> AppTest:
    at = AppTest.from_function(_script, kwargs={"values": SITES, **kwargs}, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def test_a_variable_is_cleaned_where_it_is_analysed_in_a_dialog() -> None:
    """The warning sent the user to another page, which drops this page's choices: it opens Clean's editor here."""
    at = _open()
    assert any("differ only by spaces" in w.value for w in at.warning)
    at.button(key="clean_open::t_group_col").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert any("As on Data > Clean > Categories" in c.value for c in at.caption)
    keys = {b.key for b in at.button}
    assert (
        "clean_apply::dialog::site" in keys
    )  # the editor's own Apply, its keys apart from the page's


def test_once_a_copy_cleans_it_the_warning_offers_to_group_by_the_copy() -> None:
    at = _open(with_copy=True)
    assert "clean_open::t_group_col" not in {
        b.key for b in at.button
    }  # cleaned already: nothing to open
    at.button(key="clean_use::t_group_col").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.selectbox(key="t_group_col").value == "site_clean"
    assert not any(
        "differ only by spaces" in w.value for w in at.warning
    )  # the copy's groups are one each


def test_the_pooled_comparison_says_a_copy_needs_the_imputation_again() -> None:
    at = _open(with_copy=True, pooled=True)
    assert any(
        "`site_clean` cleans `site`, but was written after the imputation: run the imputation again"
        in c.value
        for c in at.caption
    )
    assert "clean_use::mi_cmp_group" not in {b.key for b in at.button}


def test_a_variable_clean_does_not_list_is_said_so() -> None:
    """More than 200 values (free text or an identifier): Clean does not list it, and the dialog would bypass that."""
    names = [f"Person {i}" for i in range(250)] + ["Anna", "anna"] * 3
    at = AppTest.from_function(
        _script, kwargs={"values": names, "direct": True}, default_timeout=60
    )
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert any(
        "lists text variables of 2 to 200 values: `site` is not one" in c.value for c in at.caption
    )
    assert "clean_open::t_group_col" not in {b.key for b in at.button}
