"""
A study configuration written under the rules of version 1 (a test written with != or ~ true on a missing value), or
stating no version, says on upload, in one line of the sidebar, that some of its criteria select other rows; the notice
above Validate's tabs says which, with what to write for each, once the data are loaded (a file uploaded before them
is said when they come) and until it is dismissed. The configuration downloaded records the version.
"""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

RULES = {
    "mask_families": {
        "study_inclusion": {
            "operator": "all",
            "Not pregnant": {"expression": "df['pregnant'] != 'Yes'"},
        }
    }
}


def _app(content: str, with_data: bool = True) -> None:
    import io

    import pandas as pd
    import streamlit as st

    from app_pages.data_monitoring import _earlier_rules_notice, read_uploaded_config
    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state.setdefault("username", "rules_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None and (with_data or st.session_state.get("load_now")):
        df = pd.DataFrame({"pregnant": [None, None, None, "Yes", "No", "No"]})
        state.set_dataset(
            df,
            DatasetRef(name="rules", origin="upload", file_hash="r"),
            source_bytes=df.to_csv(index=False).encode("utf-8"),
            source_name="rules.csv",
        )
    if "uploaded" not in st.session_state:  # once, as the uploader's callback does
        config: dict = {"mask_families": {}}
        st.session_state["uploaded"] = read_uploaded_config(config, io.StringIO(content))
        st.session_state["merged"] = config
    _earlier_rules_notice()


def _open(content: dict, with_data: bool = True) -> AppTest:
    at = AppTest.from_function(
        _app, kwargs={"content": json.dumps(content), "with_data": with_data}, default_timeout=60
    )
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def _notices(at: AppTest) -> list[str]:
    return [w.value for w in at.warning if "missing value changed how it counts" in w.value]


def test_a_file_of_the_earlier_rules_says_which_criteria_read_otherwise() -> None:
    at = _open(RULES)
    loaded, messages = at.session_state["uploaded"]
    assert loaded
    assert messages[1:] == [
        "This file states no version of its rules: it may have been written before a missing value changed how it "
        "counts: 1 of its criteria selects other rows of this data, said above Validate's tabs."
    ]
    (notice,) = _notices(at)
    assert notice.startswith("The configuration loaded states no version of its rules")
    assert (
        "`Not pregnant` (inclusion) excludes 3 more rows: to keep the rows whose value is missing, as before, write "
        "it `df['pregnant'].isna() | (df['pregnant'] != 'Yes')`" in notice
    )
    assert notice.endswith(
        "With all of its criteria, the analysis population keeps 2 of the 6 rows now, 5 as the file was written."
    )
    assert "rules_version" not in at.session_state["merged"]  # a file's version, never merged


def test_an_exclusion_criterion_is_told_how_to_exclude_them_again() -> None:
    """The notice told every criterion how to keep the rows whose value is missing: for an exclusion criterion, which
    excludes fewer rows now, the same expression excludes them, as before."""
    rules = {
        "rules_version": 1,
        "mask_families": {
            "clinical_anomalies": {
                "operator": "any",
                "Not negative": {"expression": "~(df['pregnant'] == 'No')"},
            }
        },
    }
    (notice,) = _notices(_open(rules))
    assert notice.startswith(
        "The configuration loaded was written before a missing value changed how it counts"
    )
    assert (
        "`Not negative` (exclusion) excludes 3 fewer rows: to exclude the rows whose value is missing, as before, "
        "write it `df['pregnant'].isna() | (~(df['pregnant'] == 'No'))`" in notice
    )
    assert "to keep the rows" not in notice


def test_a_file_uploaded_before_the_data_is_said_once_they_come_and_until_dismissed() -> None:
    """Uploaded with no data, the file said "loaded" and nothing else, and its version was dropped: once the data came,
    no page said that its criteria read otherwise."""
    at = _open(RULES, with_data=False)
    _, messages = at.session_state["uploaded"]
    assert messages[1].endswith(
        "Once the data are loaded, Validate says above its tabs which of its criteria select other rows."
    )
    assert not _notices(at)
    at.session_state["load_now"] = True
    at.run()
    (notice,) = _notices(at)
    assert "`Not pregnant` (inclusion) excludes 3 more rows" in notice
    at.button(key="rules_earlier_dismiss").click().run()
    assert not _notices(at) and "rules_earlier" not in at.session_state


def test_a_file_of_today_s_rules_says_nothing_of_them() -> None:
    at = _open({**RULES, "rules_version": 2})
    loaded, messages = at.session_state["uploaded"]
    assert loaded and messages[1:] == [] and not _notices(at)
