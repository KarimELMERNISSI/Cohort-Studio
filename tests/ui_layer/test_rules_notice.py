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


def test_a_file_of_the_earlier_rules_says_which_criteria_read_otherwise() -> None:
    loaded, messages, merged = _read(RULES)
    assert loaded
    (notice,) = [m for m in messages if "rules of version" in m]
    assert notice.startswith("This file was written under the rules of version 1")
    assert "`Not pregnant` (inclusion) excludes 3 more rows" in notice
    assert (
        "rules_version" not in merged
    )  # a file's version, never merged into the session's configuration


def test_a_file_of_today_s_rules_says_nothing_of_them() -> None:
    loaded, messages, _ = _read({**RULES, "rules_version": 2})
    assert loaded and not any("rules of version" in m for m in messages)
