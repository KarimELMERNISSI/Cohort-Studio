"""
Clean > Categories (2026-10): a variable's values as written, given clean values or set to missing, and written into a
new column beside the original; spaces and capitals pre-filled, the rest offered, never applied; one traced step that
the replay redoes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from manage.reproduction_manager import apply_steps

SEX = ["F"] * 5 + ["M"] * 5 + ["F ", "f", "Female", "unknown"]


def _app(frame: dict) -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.clean_categories import render_categories
    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state.setdefault("username", "clean_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        df = pd.DataFrame(frame)
        state.set_dataset(
            df,
            DatasetRef(name="clean", origin="upload", file_hash="c"),
            source_bytes=df.to_csv(index=False).encode("utf-8"),
            source_name="clean.csv",
        )
    render_categories()


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))


def _open(frame: dict) -> AppTest:
    at = AppTest.from_function(_app, kwargs={"frame": frame}, default_timeout=120)
    at.run()
    _ok(at)
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def test_values_are_recoded_into_a_copy_and_the_step_replays() -> None:
    at = _open({"sex": SEX, "age": list(range(14))})
    assert at.selectbox(key="clean_variable").value == "sex"
    offers = [c for c in at.checkbox if c.key.startswith("clean_offer::sex::")]
    assert [c.key for c in offers] == ["clean_offer::sex::Female", "clean_offer::sex::unknown"]
    assert not any(c.value for c in offers)  # offered, never applied
    at.checkbox(key="clean_offer::sex::Female").check().run()
    at.multiselect(key="clean_pick::sex").set_value(["unknown"]).run()
    at.button(key="clean_missing::sex").click().run()
    _ok(at)
    assert at.get("plotly_chart")  # the groups, before and after
    before = at.session_state["data"].copy()
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    step = at.session_state["transformation_manager"].history[-1]
    assert step["function"] == "recode_levels"
    params = step["params"]
    assert params["mapping"] == {"F ": "F", "f": "F", "Female": "F", "unknown": None}
    assert params["reasons"] == {
        "F ": "spaces",
        "f": "capitals",
        "Female": "first letter",
        "unknown": "chosen",
    }
    assert (params["values_before"], params["values_after"], params["rows_set_to_missing"]) == (
        6,
        2,
        1,
    )
    data = at.session_state["data"]
    assert list(data.columns) == [
        "sex",
        "sex_clean",
        "age",
    ]  # the copy right after the original, which is kept
    assert data["sex"].tolist() == SEX
    clean = [None if pd.isna(v) else v for v in data["sex_clean"]]
    assert clean == ["F"] * 5 + ["M"] * 5 + ["F", "F", "F", None]
    assert at.success[0].value == "Added `sex_clean`: 6 values to 2, 1 row set to missing."
    replayed = apply_steps(before, [step])
    assert replayed.error is None and replayed.divergence is None and not replayed.warnings


def test_cleaning_again_starts_from_the_last_mapping_and_writes_the_copy_anew() -> None:
    at = _open({"sex": SEX})
    at.button(key="clean_apply::sex").click().run()  # spaces and capitals only
    _ok(at)
    assert at.button(key="clean_apply::sex").label == "Write sex\\_clean again (sex kept)"
    assert at.button(key="clean_apply::sex").disabled is False
    at.checkbox(key="clean_offer::sex::Female").check().run()
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    data = at.session_state["data"]
    assert list(data.columns) == ["sex", "sex_clean"]
    assert data["sex_clean"].tolist() == ["F"] * 5 + ["M"] * 5 + ["F", "F", "F", "unknown"]


def test_a_column_of_that_name_is_never_written_over() -> None:
    at = _open({"sex": SEX, "sex_clean": ["x"] * len(SEX)})
    assert at.button(key="clean_apply::sex").disabled
    assert any("already a column" in c.value for c in at.caption)
    at.text_input(key="clean_target::sex").set_value("sex_fixed").run()
    _ok(at)
    assert not at.button(key="clean_apply::sex").disabled


def test_nothing_to_clean_says_so() -> None:
    at = _open({"site": ["A", "B", "A"], "age": [1, 2, 3]})
    assert at.selectbox(key="clean_variable").value == "site"
    assert at.button(key="clean_apply::site").disabled
    assert any("Nothing to change" in c.value for c in at.caption)
