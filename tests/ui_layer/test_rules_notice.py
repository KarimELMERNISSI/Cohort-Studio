"""
A study configuration written under the rules of version 1 (a test written with != or ~ true on a missing value), or
stating no version, says on upload, in one line of the sidebar, that some of its criteria select other rows; the notice
above Validate's tabs says which, with what to write for each, once the data are loaded (a file uploaded before them
is said when they come), while the session holds them as loaded, and until it is dismissed. Its population counts are
the ticked criteria's, as the sidebar counts them. The configuration downloaded records the version.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded dataset writes its user's workspace: in a temporary folder, never in the data folder."""
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))


PREGNANT = "study_inclusion::Not pregnant"
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
        st.session_state["uploaded"] = read_uploaded_config(state.config, io.StringIO(content))
    if (second := st.session_state.pop("second", None)) is not None:  # another file, later
        st.session_state["uploaded_second"] = read_uploaded_config(
            state.config, io.StringIO(second)
        )
    if (ticks := st.session_state.pop("tick", None)) is not None:  # the sidebar's ticks
        state.set_ticked(ticks)
    if st.session_state.pop(
        "tutorial", False
    ):  # the tutorial cohort with its configuration, from Start
        state.load_tutorial_cohort()
    if (edit := st.session_state.pop("edit", None)) is not None:  # a criterion edited in its tab
        family, name, definition = edit
        state.config["mask_families"][family][name] = definition
    _earlier_rules_notice()
    if state.base is not None:
        st.session_state["sidebar_keeps"] = state.population.n_included


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
    assert notice.startswith(
        "Criteria loaded from a configuration that states no version of its rules may have been written before"
    )
    assert (
        "`Not pregnant` (inclusion) excludes 3 more rows: to keep the rows whose value is missing, as before, write "
        "it `df['pregnant'].isna() | (df['pregnant'] != 'Yes')`" in notice
    )
    # it said "the analysis population keeps 2 of the 6 rows" with nothing ticked: every row was kept
    assert notice.endswith(
        "None of these is ticked: the analysis population is the same under either reading."
    )
    assert "rules_version" not in at.session_state["config"]  # a file's version, never merged
    at.session_state["tick"] = [PREGNANT]
    at.run()
    (notice,) = _notices(at)
    assert notice.endswith(
        "With the criteria ticked, the analysis population keeps 2 of the 6 rows now, 5 under the earlier reading."
    )
    assert at.session_state["sidebar_keeps"] == 2  # "now" as the sidebar counts


def test_a_criterion_corrected_as_advised_is_no_longer_said() -> None:
    """The notice read a copy of the file: corrected as it advised, the criterion was still said to read otherwise."""
    at = _open(RULES)
    at.session_state["edit"] = (
        "study_inclusion",
        "Not pregnant",
        {"expression": "df['pregnant'].isna() | (df['pregnant'] != 'Yes')"},
    )
    at.run()
    assert not _notices(at) and "rules_earlier" not in at.session_state


def test_the_criteria_of_two_earlier_files_are_both_said() -> None:
    """A second file replaced the first in the notice, whose criteria the session still held as they were loaded."""
    at = _open(RULES)
    second = {
        "mask_families": {
            "clinical_anomalies": {
                "operator": "any",
                "Not negative": {"expression": "~(df['pregnant'] == 'No')"},
            }
        }
    }
    at.session_state["second"] = json.dumps(second)
    at.run()
    (notice,) = _notices(at)
    assert "`Not pregnant` (inclusion)" in notice and "`Not negative` (exclusion)" in notice


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
        "Criteria loaded from a configuration written before a missing value changed how it counts"
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


def test_the_tutorial_s_own_file_is_never_said_to_be_an_earlier_one() -> None:
    """The tutorial's configuration, downloaded from Start and loaded on Validate, stated no version: it was said to
    have maybe been written before a missing value changed how it counts."""
    from ui.state import TUTORIAL_CONFIG_PATH

    at = _open(json.loads(TUTORIAL_CONFIG_PATH.read_text(encoding="utf-8")))
    loaded, messages = at.session_state["uploaded"]
    assert loaded and not any("missing value changed how it counts" in m for m in messages)
    assert "rules_earlier" not in at.session_state


def test_loading_the_tutorial_clears_the_notice_of_an_earlier_file() -> None:
    """Its configuration replaces the session's: the notice still described the earlier file's criteria."""
    at = _open(RULES)
    assert _notices(at)
    at.session_state["tutorial"] = True
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert not _notices(at) and "rules_earlier" not in at.session_state
