"""
A study configuration written under the rules of version 1 (a test written with != or ~ true on a missing value), or
before the version was recorded, says on upload which of its criteria select other rows of the loaded data now; the
configuration downloaded records the version.
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


def _app(content: str) -> None:
    import io

    import pandas as pd
    import streamlit as st

    from app_pages.data_monitoring import read_uploaded_config
    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state.setdefault("username", "rules_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        df = pd.DataFrame({"pregnant": [None, None, None, "Yes", "No", "No"]})
        state.set_dataset(
            df,
            DatasetRef(name="rules", origin="upload", file_hash="r"),
            source_bytes=df.to_csv(index=False).encode("utf-8"),
            source_name="rules.csv",
        )
    config: dict = {"mask_families": {}}
    st.session_state["uploaded"] = read_uploaded_config(config, io.StringIO(content))
    st.session_state["merged"] = config


def _read(content: dict) -> tuple[bool, list[str], dict]:
    at = AppTest.from_function(_app, kwargs={"content": json.dumps(content)}, default_timeout=60)
    at.run()
    assert not at.exception
    loaded, messages = at.session_state["uploaded"]
    return loaded, messages, at.session_state["merged"]


EARLIER = "This file was written before a missing value changed how it counts"


def test_a_file_of_the_earlier_rules_says_which_criteria_read_otherwise() -> None:
    loaded, messages, merged = _read(RULES)
    assert loaded
    (notice,) = [m for m in messages if m.startswith(EARLIER)]
    assert (
        "`Not pregnant` (inclusion) excludes 3 more rows: to keep the rows whose value is missing, as before, write "
        "it `df['pregnant'].isna() | (df['pregnant'] != 'Yes')`" in notice
    )
    assert notice.endswith(
        "With all of its criteria, the analysis population keeps 2 of the 6 rows now, 5 as the file was written."
    )
    assert (
        "rules_version" not in merged
    )  # a file's version, never merged into the session's configuration


def test_an_exclusion_criterion_is_told_how_to_exclude_them_again() -> None:
    """The notice told every criterion how to keep the rows whose value is missing: for an exclusion criterion, which
    excludes fewer rows now, the same expression excludes them, as before."""
    rules = {
        "mask_families": {
            "clinical_anomalies": {
                "operator": "any",
                "Not negative": {"expression": "~(df['pregnant'] == 'No')"},
            }
        }
    }
    _, messages, _ = _read(rules)
    (notice,) = [m for m in messages if m.startswith(EARLIER)]
    assert (
        "`Not negative` (exclusion) excludes 3 fewer rows: to exclude the rows whose value is missing, as before, "
        "write it `df['pregnant'].isna() | (~(df['pregnant'] == 'No'))`" in notice
    )
    assert "to keep the rows" not in notice


def test_a_file_of_today_s_rules_says_nothing_of_them() -> None:
    loaded, messages, _ = _read({**RULES, "rules_version": 2})
    assert loaded and not any(m.startswith(EARLIER) for m in messages)
