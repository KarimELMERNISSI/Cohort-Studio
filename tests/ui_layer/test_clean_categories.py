"""
Clean > Categories (2026-10): a variable's values as written, given clean values or set to missing, and written into a
new column beside the original. The fold rule pre-fills spaces and capitals and is recorded as a rule; offers are
ticked, never applied; a value set to missing says what missing means; a grouping says why; the record names only
what the user chose; choices never follow the user to another dataset; one traced step that the replay redoes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from enrich.recode import (
    CAPITALS,
    CHOSEN,
    FIRST_LETTER,
    FOLD,
    GROUPING,
    MISSING_CODE,
    NOT_APPLICABLE,
    SPELLING,
    UNKNOWN,
    report_params,
)
from manage.reproduction_manager import apply_steps

# F␣ folds into F; f, a code's case variant, is offered; n.a. asks what it means; unknown reads as unknown
SEX = ["F"] * 30 + ["M"] * 30 + ["F ", "f", "Female", "unknown", "n.a.", "n.a."]
PARAMS = {
    "source",
    "target",
    "fold",
    "mapping",
    "kinds",
    "kept",
    "missing_meaning",
    "groupings",
    "replaces",
    "values_before",
    "values_after",
    "kept_as_written",
    "rows_changed",
    "rows_missing_before",
    "rows_set_to_missing",
    "changes",
    "values_fp",
    "target_fp",
}


def _app(frame: dict, other: dict | None = None) -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.clean_categories import render_categories
    from core.state import DatasetRef
    from ui.state import get_state

    st.session_state.setdefault("username", "clean_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    switch = (
        other is not None
        and st.session_state.get("load_other")
        and not st.session_state.get("other_in")
    )
    if state.base is None or switch:
        df = pd.DataFrame(other if switch else frame)
        state.set_dataset(
            df,
            DatasetRef(
                name="other" if switch else "clean",
                origin="upload",
                file_hash="o" if switch else "c",
            ),
            source_bytes=df.to_csv(index=False).encode("utf-8"),
            source_name="other.csv" if switch else "clean.csv",
        )
        if switch:
            st.session_state["other_in"] = True
    if st.session_state.pop("change_copy", False):  # a later step that fills a value of the copy
        changed = state.base.copy()
        changed.loc[changed.index[-1], "sex_clean"] = "F"
        state.commit_change(
            changed, "targeted_imputation", {"target_variable": "sex_clean"}, "Filled sex_clean"
        )
    render_categories()


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))


def _open(frame: dict, other: dict | None = None) -> AppTest:
    at = AppTest.from_function(_app, kwargs={"frame": frame, "other": other}, default_timeout=120)
    at.run()
    _ok(at)
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _one(widgets, prefix: str):
    found = [w for w in widgets if str(w.key).startswith(prefix)]
    assert len(found) == 1, [w.key for w in widgets]
    return found[0]


def _group(at: AppTest, name: str):
    """The pills of one group of offers: written another way, one letter apart, codes for missing."""
    found = [
        w
        for w in at.pills
        if str(w.key).startswith("clean_offers::") and f"::{name}::" in str(w.key)
    ]
    assert len(found) == 1, [w.key for w in at.pills]
    return found[0]


def _set_to_missing(at: AppTest, values: list[str], meaning: str) -> None:
    _one(at.multiselect, "clean_pick::").set_value(values).run()
    _one(at.selectbox, "clean_means::").set_value(meaning).run()
    _one(at.button, "clean_missing::").click().run()
    _ok(at)


def _step(at: AppTest) -> dict:
    step = at.session_state["transformation_manager"].history[-1]
    assert step["function"] == "recode_levels"
    return step


def test_the_record_names_what_was_chosen_and_the_fold_as_a_rule() -> None:
    at = _open({"sex": SEX, "age": list(range(len(SEX)))})
    assert _one(at.selectbox, "clean_variable::").value == "sex"
    written, missing = _group(at, "written"), _group(at, "missing")
    assert written.options == [  # the drawn labels
        "`Female` → `F` · first-letter code · 1 row",
        "`f` → `F` · capitals · 1 row",
    ]
    assert missing.options == [  # n.a. names no meaning; unknown reads as unknown
        "`n.a.` → missing · code for missing · 2 rows",
        "`unknown` → missing · code for missing · 1 row",
    ]
    assert not written.value and not missing.value  # offered, never applied
    written.select("Female").select("f").run()
    _group(at, "missing").select("unknown").run()
    _ok(at)
    _set_to_missing(at, ["n.a."], NOT_APPLICABLE)
    assert at.get("plotly_chart")
    before = at.session_state["data"].copy()
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    step = _step(at)
    params = step["params"]
    assert set(params) == PARAMS
    assert params["fold"] == FOLD and params["kept"] == []
    # the fold of "F " is the rule's: never recorded
    assert params["mapping"] == {"Female": "F", "f": "F", "unknown": None, "n.a.": None}
    assert params["kinds"] == {
        "Female": FIRST_LETTER,
        "f": CAPITALS,
        "unknown": MISSING_CODE,
        "n.a.": CHOSEN,
    }
    assert params["missing_meaning"] == {"unknown": UNKNOWN, "n.a.": NOT_APPLICABLE}
    assert (params["values_before"], params["values_after"], params["rows_set_to_missing"]) == (
        7,
        2,
        3,
    )
    data = at.session_state["data"]
    assert list(data.columns) == [
        "sex",
        "sex_clean",
        "age",
    ]  # the copy right after the original, kept
    assert data["sex"].tolist() == SEX
    clean = [None if pd.isna(v) else v for v in data["sex_clean"]]
    assert clean == ["F"] * 30 + ["M"] * 30 + ["F", "F", "F", None, None, None]
    assert any(
        "Added `sex_clean` beside `sex`: 2 values instead of 7" in s.value for s in at.success
    )
    replayed = apply_steps(before, [step])
    assert replayed.error is None and replayed.divergence is None and not replayed.warnings


def test_cleaning_again_starts_from_the_last_choices_with_their_offers_ticked() -> None:
    at = _open({"sex": SEX})
    _group(at, "written").select("Female").run()
    _set_to_missing(at, ["n.a.", "unknown"], UNKNOWN)
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    assert _one(at.selectbox, "clean_variable::").value == "sex"
    assert "cleaned into sex_clean" in _one(at.selectbox, "clean_variable::").format_func("sex")
    assert at.button(key="clean_apply::sex").label == "Write sex\\_clean again (sex kept)"
    assert "Female" in (_group(at, "written").value or [])  # applied, so ticked
    _group(at, "written").unselect("Female").run()  # unticked: back to its spelling
    _ok(at)
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    data = at.session_state["data"]
    assert list(data.columns) == ["sex", "sex_clean"]
    assert data["sex_clean"].tolist()[62] == "Female"
    assert _step(at)["params"]["mapping"] == {"n.a.": None, "unknown": None}


def test_a_value_set_to_missing_says_what_missing_means() -> None:
    at = _open({"sex": SEX})
    _group(at, "missing").select("n.a.").run()  # n.a.: not applicable, an absence, a result?
    _ok(at)
    assert at.button(key="clean_apply::sex").disabled
    assert any("Choose what missing means for `n.a.`" in c.value for c in at.caption)
    assert _one(at.button, "clean_missing::").disabled  # nothing picked, no meaning yet
    _set_to_missing(at, ["n.a."], NOT_APPLICABLE)
    assert not at.button(key="clean_apply::sex").disabled


def test_merging_two_categories_is_a_grouping_that_says_why() -> None:
    nyha = ["I"] * 20 + ["II"] * 40 + ["III"] * 30 + ["IV"] * 10
    at = _open({"nyha": nyha})
    _one(at.multiselect, "clean_pick::").set_value(["IV"]).run()
    _one(at.selectbox, "clean_into::").set_value("III").run()  # (AppTest sets listed options only)
    _one(at.button, "clean_map::").click().run()
    _ok(at)
    assert at.button(key="clean_apply::nyha").disabled
    reason = _one(at.text_input, "clean_reason::")
    reason.input("advanced heart failure, as pre-specified").run()
    _ok(at)
    assert not at.button(key="clean_apply::nyha").disabled
    at.button(key="clean_apply::nyha").click().run()
    _ok(at)
    params = _step(at)["params"]
    assert params["mapping"] == {"IV": "III"}
    assert params["kinds"] == {"IV": GROUPING}
    assert params["groupings"] == {
        "III": {
            "reason": "advanced heart failure, as pre-specified",
            "values": ["III", "IV"],
            "each_5_rows": True,
        }
    }


def test_a_column_of_that_name_is_never_written_over() -> None:
    at = _open({"sex": SEX, "sex_clean": ["x"] * len(SEX)})
    assert at.button(key="clean_apply::sex").disabled
    assert any("already a column" in c.value for c in at.caption)
    _one(at.text_input, "clean_target::").set_value("sex_fixed").run()
    _ok(at)
    assert not at.button(key="clean_apply::sex").disabled


def test_a_copy_a_later_step_changed_is_never_written_again() -> None:
    """sex recoded, then sex_clean filled by a later step: writing it again would discard that without a word."""
    at = _open({"sex": SEX})
    _group(at, "written").select("Female").run()
    _set_to_missing(at, ["n.a.", "unknown"], UNKNOWN)
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    assert at.button(key="clean_apply::sex").disabled  # nothing changed: nothing to write again
    assert any("Nothing changed since `sex_clean` was written." in c.value for c in at.caption)
    at.session_state["change_copy"] = True
    at.run()
    _ok(at)
    assert (
        at.session_state["transformation_manager"].history[-1]["function"] == "targeted_imputation"
    )
    assert at.button(key="clean_apply::sex").disabled
    assert any("was changed by a later step" in c.value for c in at.caption)


def test_nothing_to_clean_says_so() -> None:
    at = _open({"site": ["A", "B", "A"], "age": [1, 2, 3]})
    assert _one(at.selectbox, "clean_variable::").value == "site"
    assert at.button(key="clean_apply::site").disabled
    assert any("Nothing to change" in c.value for c in at.caption)


def test_choices_never_follow_the_user_to_another_dataset() -> None:
    """On dataset A, "M" (manquant) set to missing; on B, where M means male, it must start as written."""
    a = {"status": ["M"] * 3 + ["oui"] * 20 + ["non"] * 20}
    b = {"status": ["M"] * 30 + ["F"] * 30}
    at = _open(a, other=b)
    _set_to_missing(at, ["M"], UNKNOWN)
    assert at.session_state["clean_manual"]["status"]["mapping"] == {"M": None}
    at.session_state["load_other"] = True
    at.run()
    _ok(at)
    assert at.session_state["clean_manual"].get("status", {}).get("mapping", {}) == {}
    assert at.button(key="clean_apply::status").disabled  # nothing chosen on B


def test_values_nobody_chose_reach_no_record_and_markup_is_drawn_as_text() -> None:
    commune = (
        ["Luxembourg"] * 120  # a 1-row near spelling is offered beside 100 rows or more
        + ["Metz"] * 30
        + ["Esch"] * 10
        + [
            "CANARY-SPACE-6E2 ",
            "Canary-Case-8a1",
            "Canary-Case-8a1",
            "canary-case-8a1",
            "CANARY-KEEP-2D4",
        ]
        + ["Luxembourgg", "CANARY-MISS-5B9", '<a href="http://canary.invalid/a">CANARY-LINK</a>']
    )
    at = _open({"commune": commune})
    spec = at.get("plotly_chart")[0].proto.spec
    assert "<a href" not in spec and "&lt;a href" in spec
    offers = _group(at, "letter")
    assert offers.options == ["`Luxembourgg` → `Luxembourg` · one letter apart: check · 1 row"]
    offers.select("Luxembourgg").run()
    _set_to_missing(at, ["CANARY-MISS-5B9"], UNKNOWN)
    at.button(key="clean_apply::commune").click().run()
    _ok(at)
    step = _step(at)
    params = step["params"]
    recorded = repr(params) + step.get("description", "")
    for canary in ("CANARY-SPACE", "Canary-Case", "canary-case", "CANARY-KEEP", "CANARY-LINK"):
        assert canary not in recorded, canary
    assert params["mapping"] == {"Luxembourgg": "Luxembourg", "CANARY-MISS-5B9": None}
    assert params["kinds"]["Luxembourgg"] == SPELLING
    shown = repr(report_params(params))
    assert "CANARY" not in shown and "Luxembourgg" not in shown


def test_tick_all_ticks_every_offer_of_a_group() -> None:
    at = _open({"sex": SEX})
    (button,) = [
        b
        for b in at.button
        if str(b.key).startswith("clean_tick_all::") and str(b.key).endswith("written")
    ]
    assert button.label == "Tick all 2"
    button.click().run()
    _ok(at)
    assert set(_group(at, "written").value or []) == {"Female", "f"}
    assert not _group(at, "missing").value  # the other groups untouched


def test_a_grouping_s_question_draws_its_values_as_text() -> None:
    """A value holding Markdown in the question's label: an image the browser would fetch."""
    level = "![g](http://canary.invalid/g.png)"
    at = _open({"x": ["Metz"] * 40 + [level] * 30 + ["Luxembourg"] * 30})
    _one(at.multiselect, "clean_pick::").set_value([level]).run()
    _one(at.selectbox, "clean_into::").set_value("Metz").run()
    _one(at.button, "clean_map::").click().run()
    _ok(at)
    label = _one(at.text_input, "clean_reason::").label
    assert "`![g](http://canary.invalid/g.png)`" in label
