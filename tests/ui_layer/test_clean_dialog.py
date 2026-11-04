"""
A variable cleaned where it is analysed (app_pages/clean_dialog.py): the warning under a list of groups offers to
clean it in a dialog holding Data > Clean's editor, then to group by its clean copy; the pooled comparison, which reads
the imputed datasets, says that a copy written after the imputation needs it run again; a variable that Clean does not
list is said so. (Apply inside the dialog reruns it as a fragment, which AppTest does not replay: the browser check
covers it.)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp

SITES = ["Metz", "Metz ", "metz", "Esch"] * 10


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded dataset writes its user's workspace: in a temporary folder, never in the data folder."""
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))


def _script(
    values: list[str],
    with_copy: bool = False,
    pooled: bool = False,
    direct: bool = False,
    stale: bool = False,
) -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.analysis_config import render_analysis_configuration
    from app_pages.clean_dialog import group_warning
    from core.fingerprint import frame_fingerprint
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
            params = {
                "source": "site",
                "target": "site_clean",
                "mapping": mapping,
                "source_fp": frame_fingerprint(state.base[["site"]]),
            }
            state.commit_change(new, "recode_levels", params, "Recoded site into site_clean")
            if stale:  # a later step changed a row of site that the copy has not had (an imputation, a merge's rows)
                later = state.base.copy()
                later.loc[later.index[-1], "site"] = "metz"  # was Esch: the copy still says Esch
                state.commit_change(later, "external_write", {}, "Changed a site")
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
    at.button(key="clean_open::t_group_col::site").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert any(
        "Your choices are kept until you apply them, here or on Data > Clean > Categories."
        in c.value
        for c in at.caption
    )
    keys = {b.key for b in at.button}
    assert (
        "clean_apply::dialog::site" in keys
    )  # the editor's own Apply, its keys apart from the page's


def test_once_a_copy_cleans_it_the_warning_offers_to_group_by_the_copy() -> None:
    at = _open(with_copy=True)
    assert "clean_open::t_group_col::site" not in {
        b.key for b in at.button
    }  # cleaned already: nothing to open
    at.button(key="clean_use::t_group_col").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.selectbox(key="t_group_col").value == "site_clean"
    assert not any(
        "differ only by spaces" in w.value for w in at.warning
    )  # the copy's groups are one each


def test_the_warning_names_the_copy_and_a_copy_behind_its_variable_is_cleaned_again() -> None:
    """Choosing `site` once `site_clean` was written still said "clean site on Data > Clean > Categories"."""
    at = _open(with_copy=True, direct=True)
    assert any(
        "`site` is cleaned into `site_clean`: group by `site_clean`." in w.value for w in at.warning
    )
    assert "clean_use::t_group_col" in {b.key for b in at.button}
    stale = _open(with_copy=True, direct=True, stale=True)
    assert any("as it was before a later step changed it" in w.value for w in stale.warning)
    keys = {b.key for b in stale.button}
    assert "clean_use::t_group_col" not in keys  # its copy lacks the later change: not offered
    (opener,) = [b for b in stale.button if b.key == "clean_open::t_group_col::site"]
    assert opener.label == "Clean `site` again here"


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
    assert "clean_open::t_group_col::site" not in {b.key for b in at.button}


def _plan_script(with_copy: bool = False, stale: bool = False) -> None:
    import numpy as np
    import pandas as pd
    import streamlit as st

    from app_pages.analysis_survival import _render_plan
    from core.fingerprint import frame_fingerprint
    from core.state import DatasetRef
    from enrich.recode import recode_column
    from ui.guidance.outcomes import plan_time_to_event
    from ui.state import get_state

    st.session_state.setdefault("username", "plan_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        rng = np.random.default_rng(3)
        n = 120
        frame = pd.DataFrame(
            {
                "days": rng.integers(10, 900, n),
                "death": rng.integers(0, 2, n),
                "arm": rng.choice(["A", "B"], n),
                "sex": rng.choice(["F", "F ", "f", "M", "m"], n),
            }
        )
        state.set_dataset(
            frame,
            DatasetRef(name="plan", origin="upload", file_hash="plan"),
            source_bytes=frame.to_csv(index=False).encode("utf-8"),
            source_name="plan.csv",
        )
        if with_copy:
            mapping = {"F ": "F", "f": "F", "m": "M"}
            new, _ = recode_column(state.base, "sex", "sex_clean", mapping)
            params = {
                "source": "sex",
                "target": "sex_clean",
                "mapping": mapping,
                "source_fp": frame_fingerprint(state.base[["sex"]]),
            }
            state.commit_change(new, "recode_levels", params, "Recoded sex into sex_clean")
            if stale:  # a later step changed a row of sex that the copy has not had
                later = state.base.copy()
                first = later.index[later["sex"] == "M"][0]
                later.loc[first, "sex"] = "f"  # the copy still says M
                state.commit_change(later, "external_write", {}, "Changed a sex")
    st.session_state.setdefault("surv_group", "arm")
    st.session_state.setdefault("surv_adjust", ["sex"])
    df = state.base
    covariates = [st.session_state["surv_group"], *st.session_state["surv_adjust"]]
    plan = plan_time_to_event(df, "days", "death", "arm", covariates, group=True)
    _render_plan(plan, keys=("surv_group", "surv_adjust"), candidates=list(df.columns))


def test_the_survival_plan_offers_to_clean_a_covariate_then_to_use_its_copy() -> None:
    """The plan said "Clean it first on Data > Clean" with nothing to click: a covariate written several ways is
    cleaned here, then its copy takes its place in "Adjust for"."""
    at = AppTest.from_function(_plan_script, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert any("`sex` holds one value written several ways" in m.value for m in at.markdown)
    assert "clean_open::surv_adjust::sex" in {b.key for b in at.button}
    with_copy = AppTest.from_function(_plan_script, kwargs={"with_copy": True}, default_timeout=60)
    with_copy.run()
    with_copy.button(key="clean_use::surv_adjust::sex").click().run()
    assert not with_copy.exception, [str(e.value)[:300] for e in with_copy.exception]
    assert with_copy.session_state["surv_adjust"] == ["sex_clean"]
    assert not any("written several ways" in m.value for m in with_copy.markdown)


def test_the_survival_plan_says_a_copy_is_behind_its_variable_and_offers_to_clean_it_again() -> (
    None
):
    """A copy a later step left behind was offered in the variable's place: the rows changed since left the model."""
    at = AppTest.from_function(
        _plan_script, kwargs={"with_copy": True, "stale": True}, default_timeout=60
    )
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert any(
        c.value
        == "`sex_clean` was written before a later step changed `sex`: clean `sex` again, then use the copy."
        for c in at.caption
    )
    keys = {b.key for b in at.button}
    assert "clean_use::surv_adjust::sex" not in keys
    (opener,) = [b for b in at.button if b.key == "clean_open::surv_adjust::sex"]
    assert opener.label == "Clean `sex` again here"


def _two_warnings_script() -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.clean_dialog import group_warning
    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state.setdefault("username", "dialog_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        frame = pd.DataFrame(
            {"site": ["Metz", "Metz ", "metz", "Esch"] * 10, "sex": ["F", "F ", "f", "M"] * 10}
        )
        state.set_dataset(
            frame,
            DatasetRef(name="two", origin="upload", file_hash="two"),
            source_bytes=frame.to_csv(index=False).encode("utf-8"),
            source_name="two.csv",
        )
    for column in ("site", "sex"):
        group_warning(column, sorted(state.base[column].unique()), group_key=f"t_{column}")
        st.markdown(f"end of {column}")


def test_what_apply_did_is_said_under_its_own_variable() -> None:
    """With two variables written several ways on one page, "Added `sex_clean` ..." was said under the first warning
    drawn, `site`'s."""
    at = AppTest.from_function(_two_warnings_script, default_timeout=60)
    at.session_state["clean_done"] = {"sex": "Added `sex_clean` beside `sex`: 1 value changed."}
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    order = [
        "success" if el.type == "success" else el.value
        for el in at.main
        if el.type == "success" or (el.type == "markdown" and el.value.startswith("end of"))
    ]
    assert order == ["end of site", "success", "end of sex"]
    assert "clean_done" not in at.session_state  # said once


def _editor_script(discard: bool = False) -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.clean_categories import (
        _choices,
        category_editor,
        discard_choices,
        unapplied_choices,
    )
    from core.state import DatasetRef
    from enrich.recode import give, level_counts
    from ui.state import get_state

    st.session_state.setdefault("username", "dialog_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        frame = pd.DataFrame({"site": ["Metz", "Metz ", "metz", "Esch", "Ettelbruck"] * 8})
        state.set_dataset(
            frame,
            DatasetRef(name="sites", origin="upload", file_hash="sites"),
            source_bytes=frame.to_csv(index=False).encode("utf-8"),
            source_name="sites.csv",
        )
    base = state.base
    if "chosen" not in st.session_state:  # a choice made in the dialog, not yet applied
        st.session_state["chosen"] = True
        give(_choices("site"), level_counts(base["site"]), "Ettelbruck", "Ettelbruck (Nord)")
    st.session_state["waiting"] = unapplied_choices(state, base, "site")
    if discard:
        discard_choices(state, base, "site")
        st.session_state["after"] = unapplied_choices(state, base, "site")
    category_editor(
        state, base, "site", host="dialog"
    )  # the dialog's editor, drawn here (AppTest replays no dialog)


def test_in_the_dialog_find_values_is_drawn_inline_and_picks_the_values() -> None:
    """A popover closed by the search left the keyboard's focus outside the dialog, whose next keystroke closed it with
    what was being typed in the table: in the dialog, the search is a field and Pick them on one row."""
    at = AppTest.from_function(_editor_script, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    (field,) = [t for t in at.text_input if t.label == "Find values containing"]
    assert not [p for p in at.get("popover") if "Find values" in str(getattr(p, "label", ""))]
    field.input("Ettel")
    (pick,) = [b for b in at.button if b.label == "Pick them"]
    pick.click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    (values,) = [m for m in at.multiselect if m.label == "Values"]
    assert values.value == ["Ettelbruck"]


def test_the_dialog_s_choices_not_yet_applied_are_counted_and_can_be_discarded() -> None:
    """Escape or the cross kept them unannounced: they came back on reopening and in Data > Clean's table, where the
    next Apply wrote them. Close now says they are kept, and Discard drops them."""
    at = AppTest.from_function(_editor_script, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["waiting"] == 1
    gone = AppTest.from_function(_editor_script, kwargs={"discard": True}, default_timeout=60)
    gone.run()
    assert not gone.exception, [str(e.value)[:300] for e in gone.exception]
    assert gone.session_state["waiting"] == 1 and gone.session_state["after"] == 0


def _whole_script() -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.clean_dialog import group_warning
    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state.setdefault("username", "dialog_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        frame = pd.DataFrame({"site": ["Metz", "Esch"] * 10})
        state.set_dataset(
            frame,
            DatasetRef(name="whole", origin="upload", file_hash="whole"),
            source_bytes=frame.to_csv(index=False).encode("utf-8"),
            source_name="whole.csv",
        )
    group_warning("site", sorted(state.base["site"].unique()), group_key="t_group_col")


def test_what_apply_did_is_said_when_it_leaves_no_groups_to_warn_of() -> None:
    """Apply made the copy whole, so the warning went with its actions, and "Wrote `site_clean` ..." was never said."""
    at = AppTest.from_function(_whole_script, default_timeout=60)
    at.session_state["clean_done"] = {
        "site": "Wrote `site_clean` beside `site`: 2 values instead of 3."
    }
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert [s.value for s in at.success] == [
        "Wrote `site_clean` beside `site`: 2 values instead of 3."
    ]
    assert not at.warning and "clean_done" not in at.session_state


def test_a_covariate_whose_copy_is_the_comparison_is_offered_to_leave_the_list() -> None:
    """Compared by `sex_clean` and adjusted for `sex`, the plan offered to put the copy in "Adjust for" too, where it is
    compared already."""
    at = AppTest.from_function(_plan_script, kwargs={"with_copy": True}, default_timeout=60)
    at.session_state["surv_group"] = "sex_clean"
    at.session_state["surv_adjust"] = ["sex"]
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    keys = {b.key for b in at.button}
    assert "clean_use::surv_adjust::sex" not in keys
    at.button(key="clean_drop::surv_adjust::sex").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["surv_adjust"] == []
