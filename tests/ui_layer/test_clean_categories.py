"""
Clean > Categories (2026-10): a variable's values as written, given clean values or set to missing, and written into a
new column beside the original. The fold rule pre-fills spaces and is recorded as a rule; offers are ticked, never
applied; a value set to missing says what missing means; a grouping says why; the record names only what the user
chose; choices never follow the user to another dataset; one traced step that the replay redoes. The table's edits
reach the page as the browser sends them: every edit since the table was drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from enrich.recode import (
    CASE,
    CHOSEN,
    FIRST_LETTER,
    FOLD,
    GROUPING,
    MISSING_CODE,
    NOT_APPLICABLE,
    RESULT,
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
    "source_fp",
}


def _app(frame: dict, other: dict | None = None, ai: dict | None = None) -> None:
    import pandas as pd
    import streamlit as st

    if ai is not None and "rag_manager" not in st.session_state:  # a running AI engine, faked

        class _Engine:
            initialized = True

            def __init__(self) -> None:
                self.provider, self.base_url = ai["provider"], ai["base_url"]
                # as the connectors keep it: in their configuration
                config = type("Config", (), {"model_name": ai.get("model", "fake-model")})()
                self.llm = type("Model", (), {"config": config})()
                self.sent: list = []

            def recipients(self) -> frozenset:
                return frozenset(ai["recipients"])

            def suggest_category_groups(self, column, values, max_proposals=30):
                self.sent.append((column, list(values)))
                return ai["proposals"], None

        st.session_state["rag_manager"] = _Engine()

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
    if st.session_state.pop(
        "change_source", False
    ):  # a later step that changes a value of the source
        changed = state.base.copy()
        changed.loc[changed.index[-1], "sex"] = "F"
        state.commit_change(
            changed, "targeted_imputation", {"target_variable": "sex"}, "Filled sex"
        )
    if st.session_state.pop(
        "freeze", False
    ):  # Keep only these rows: two rows leave the source and its copy alike
        state.commit_change(
            state.base.iloc[2:].copy(),
            "freeze_population",
            {"n_before": len(state.base), "n_after": len(state.base) - 2},
            "Kept the analysis population as the dataset",
        )
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


def _open(frame: dict, other: dict | None = None, ai: dict | None = None) -> AppTest:
    at = AppTest.from_function(
        _app, kwargs={"frame": frame, "other": other, "ai": ai}, default_timeout=120
    )
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
    """The pills of one group of offers: written another way, close spellings, codes for missing."""
    found = [
        w
        for w in at.pills
        if str(w.key).startswith("clean_offers::") and f"::{name}::" in str(w.key)
    ]
    assert len(found) == 1, [w.key for w in at.pills]
    return found[0]


def _editor(at: AppTest):
    found = []

    def walk(node) -> None:
        proto = getattr(node, "proto", None)
        if (
            proto is not None
            and getattr(node, "type", None) == "dataframe"
            and "clean_editor" in (proto.id or "")
        ):
            found.append(node)
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(at._tree)
    (editor,) = found
    return editor


def _rows(at: AppTest) -> list[tuple]:
    return [tuple(r) for r in _editor(at).value.itertuples(index=False)]


def _row(at: AppTest, shown: str) -> int:
    return [r[0] for r in _rows(at)].index(shown)


def _edit(at: AppTest, edited_rows: dict) -> str:
    """The table's state as the browser sends it: every edit made since the table was drawn, by row. Returns the
    table's id."""
    editor = _editor(at)
    original = at._tree.get_widget_states

    def sent():
        states = original()
        widget = states.widgets.add()
        widget.id = editor.proto.id
        widget.string_value = json.dumps(
            {"edited_rows": edited_rows, "added_rows": [], "deleted_rows": []}
        )
        return states

    at._tree.get_widget_states = sent  # the next run's tree replaces it
    at.run()
    _ok(at)
    return editor.proto.id


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
    # the drawn labels: the values in bold body text, never markup
    assert written.options == ["**Female** → **F** · first-letter code · 1 row"]
    # a one-letter value: case may name a category (Kell K and k), so it is checked on its own
    assert _group(at, "case").options == ["**f** → **F** · 1 row"]
    assert _tick_all(at, "case") is None
    assert (
        missing.options
        == [  # n.a. names no meaning; unknown reads as unknown; the group's title says why
            "**n\\.a\\.** → missing · 2 rows",
            "**unknown** → missing · 1 row",
        ]
    )
    assert not written.value and not missing.value  # offered, never applied
    written.select("Female").run()
    _group(at, "case").select("f").run()
    _group(at, "missing").select("unknown").run()
    _ok(at)
    _set_to_missing(at, ["n.a."], NOT_APPLICABLE)
    assert at.button(
        key="clean_apply::sex"
    ).disabled  # f and F may be two categories: say why they are one
    _one(at.text_input, "clean_reason::").input("f is the code F written in lower case").run()
    _ok(at)
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
        "f": CASE,
        "unknown": MISSING_CODE,
        "n.a.": CHOSEN,
    }
    assert params["groupings"]["F"]["reason"] == "f is the code F written in lower case"
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
    assert any(
        "Say what missing means for `n.a.` in the table's Missing column." in c.value
        for c in at.caption
    )
    assert _rows(at)[_row(at, "n.a.")][3] == "choose a meaning"
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
    assert (
        reason.label == "Why group `IV` with `III`?"
    )  # the value keeping its name is not among those grouped
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
    assert offers.options == ["**Luxembourgg** → **Luxembourg** · 1 edit apart · 1 → 120 rows"]
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


def _tick_all(at: AppTest, name: str):
    found = [
        b
        for b in at.button
        if str(b.key).startswith("clean_tick_all::") and str(b.key).endswith(name)
    ]
    return found[0] if found else None


def test_tick_all_ticks_every_offer_of_a_group() -> None:
    at = _open({"sex": ["F"] * 30 + ["M"] * 30 + ["Female", "male", "MALE", "unknown"]})
    button = _tick_all(at, "written")
    assert button is not None and button.label == "Tick all 3"
    _group(at, "written").select("MALE").run()
    _ok(at)
    assert _tick_all(at, "written").label == "Tick the other 2"
    _tick_all(at, "written").click().run()
    _ok(at)
    assert set(_group(at, "written").value or []) == {"Female", "male", "MALE"}
    assert not _group(at, "missing").value  # the other groups untouched


def test_no_tick_all_where_two_values_may_be_two_categories() -> None:
    """A close spelling (Dysphasia, Dysphagia) or the AI engine's: each is checked on its own."""
    terms = ["Dysphagia"] * 400 + ["Dysphasia"] * 2 + ["Luxembourg"] * 300 + ["Luxembourgg"] * 2
    at = _open({"term": terms})
    assert len(_group(at, "letter").options) == 2
    assert _tick_all(at, "letter") is None


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


def test_the_ai_engine_proposes_after_a_preview_and_the_record_names_who_proposed() -> None:
    """A cloud provider is sent the values held by 5 rows or more, no count; its proposals are offered, never
    applied; the record names the provider and the model, never the prompt or the answer."""
    frame = {"sex": ["Female"] * 40 + ["Male"] * 40 + ["Femme"] * 6 + ["Homme"] * 3}
    ai = {
        "provider": "anthropic",
        "base_url": None,
        "recipients": ["anthropic"],
        "proposals": [
            {"value": "Femme", "clean": "Female", "reason": "French for female"},
            {"value": "Homme", "clean": "Male", "reason": "French for male"},
        ],
    }
    at = _open(frame, ai=ai)
    assert not [
        c for c in at.caption if "not sent" in c.value
    ]  # the preview is made only while it is open
    popover = "clean_ai_open::" + str(_one(at.button, "clean_map::").key).split("::", 1)[1]
    at.session_state[popover] = True
    at.run()
    _ok(at)
    assert any(
        "1 value held by fewer than 5 rows is not sent; rows are counted, not people" in c.value
        for c in at.caption
    )
    assert any(c.value.startswith("Send to Anthropic") for c in at.caption)
    sent_table = [d for d in at.dataframe if list(d.value.columns) == ["Values sent"]]
    assert sent_table and sent_table[0].value["Values sent"].tolist() == ["Female", "Femme", "Male"]
    _one(at.button, "clean_ai_send::").click().run()
    _ok(at)
    assert at.session_state[popover] is False  # closed by Send
    assert at.session_state["rag_manager"].sent == [
        ("sex", ["Female", "Femme", "Male"])
    ]  # A to Z, no Homme
    # made by the run that followed Send, beside the heading: every proposal accounted for (Homme was not sent)
    assert "clean_ai_pending" not in at.session_state
    assert any(
        c.value == "The AI engine made 1 proposal, below, under Proposed by the AI engine."
        for c in at.caption
    )
    offers = [w for w in at.pills if str(w.key).startswith("clean_offers::")]
    assert "::ai::" in str(
        offers[0].key
    )  # the AI engine's group first: what Send changed shows at the top
    group = _group(at, "ai")
    assert group.options == [
        "**Femme** → **Female** · French for female · 6 → 40 rows"
    ]  # both counts
    assert not group.value  # offered, never applied
    assert _tick_all(at, "ai") is None
    group.select("Femme").run()
    _ok(at)
    # two categories of 6 and 40 rows merged on the model's word: the user says why; its words are a hint only
    assert at.button(key="clean_apply::sex").disabled
    reason = _one(at.text_input, "clean_reason::")
    assert reason.value == "" and "French for female" in (reason.placeholder or "")
    reason.input("Femme is French for Female").run()
    _ok(at)
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    params = _step(at)["params"]
    assert params["kinds"] == {"Femme": "ai"}
    assert params["ai_by"] == {
        "Femme": {
            "provider": "anthropic",
            "model": "fake-model",
            "where": "Anthropic Claude",
            "values_sent": 3,
            "values_held": 1,
        }
    }
    assert params["groupings"]["Female"]["reason"] == "Femme is French for Female"
    # written again in another session: who proposed is kept
    at.session_state["clean_ai"] = {}
    at.session_state["clean_manual"] = {}
    at.run()
    _ok(at)
    assert at.session_state["clean_manual"]["sex"]["ai_by"] == params["ai_by"]


def test_no_ai_engine_no_request() -> None:
    at = _open({"sex": SEX})
    assert not [b for b in at.button if str(b.key).startswith("clean_ai_send::")]


def test_each_edit_of_the_table_is_read_once_and_a_later_one_wins() -> None:
    """The browser sends every edit since the table was drawn: read again in row order against a store the later one
    had changed, an earlier edit undid it (Female ␣ typed F, Female F, then Female W: Female ␣ became W while its
    cell read F)."""
    at = _open({"sex": ["Female"] * 30 + ["Male"] * 30 + ["Female "] * 2})
    a, b = _row(at, "Female"), _row(at, "Female␣")
    _edit(at, {str(b): {"Clean value": "F"}})
    _edit(at, {str(b): {"Clean value": "F"}, str(a): {"Clean value": "F"}})
    _edit(at, {str(b): {"Clean value": "F"}, str(a): {"Clean value": "W"}})
    store = at.session_state["clean_manual"]["sex"]
    assert store["mapping"] == {"Female ": "F", "Female": "W"}
    assert [r[2] for r in _rows(at)][a] == "W" and [r[2] for r in _rows(at)][b] == "F"


def test_the_table_starts_again_when_the_store_and_what_it_shows_differ() -> None:
    """A clean value typed with spaces around it is stored without them: the cell reads what the store holds, not
    what the browser painted."""
    at = _open({"virus": ["detected"] * 40 + ["negative"] * 40 + ["ND"] * 6})
    r = _row(at, "negative")
    first = _edit(at, {str(r): {"Clean value": "  Negative  "}})
    assert _editor(at).proto.id != first  # drawn again from the store
    assert _rows(at)[r][2] == "Negative"
    assert at.session_state["clean_manual"]["virus"]["mapping"] == {"negative": "Negative"}
    second = _edit(at, {str(_row(at, "ND")): {"Missing": "declined"}})
    assert (
        _editor(at).proto.id == second
    )  # the store shows what the browser shows: the table is kept
    assert at.session_state["clean_manual"]["virus"]["meanings"] == {"ND": "declined"}


def test_a_pill_reads_what_the_table_did() -> None:
    """Female → F ticked, Female kept as written in the table, then f → F ticked: the stale pills ticked Female again."""
    at = _open({"sex": ["F"] * 30 + ["M"] * 30 + ["Female", "f"]})
    _group(at, "written").select("Female").run()
    _ok(at)
    _edit(at, {str(_row(at, "Female")): {"Clean value": "Female"}})
    assert "Female" not in (_group(at, "written").value or [])
    _group(at, "case").select("f").run()
    _ok(at)
    assert at.session_state["clean_manual"]["sex"]["mapping"] == {"f": "F"}


def test_apply_never_brings_back_the_browser_s_edits() -> None:
    at = _open({"status": ["yes"] * 40 + ["no"] * 40 + ["n.a."] * 3})
    before = _edit(at, {str(_row(at, "n.a.")): {"Missing": "declined"}})
    at.button(key="clean_apply::status").click().run()
    _ok(at)
    assert _editor(at).proto.id != before


def test_two_values_that_take_each_other_s_name_wait_for_one() -> None:
    at = _open({"answer": ["Yes"] * 40 + ["No"] * 40 + ["Oui"] * 10})
    _edit(at, {str(_row(at, "Oui")): {"Clean value": "Yes"}})
    _edit(
        at,
        {
            str(_row(at, "Oui")): {"Clean value": "Yes"},
            str(_row(at, "Yes")): {"Clean value": "Oui"},
        },
    )
    assert at.button(key="clean_apply::answer").disabled
    assert any("take each other's name" in c.value for c in at.caption)


def test_the_model_is_read_from_a_real_connector_s_configuration() -> None:
    """Every connector keeps its model in its configuration; only one also had model_name, so the record said ""."""
    pytest.importorskip("langchain_core")
    from manage.rag.providers.ollama import OllamaChatConnector
    from manage.rag.schemas import ProviderConfig
    from ui.services import engine_model

    engine = type("Engine", (), {})()
    engine.llm = OllamaChatConnector(config=ProviderConfig(model_name="gpt-oss:120b-cloud"))
    assert engine_model(engine) == "gpt-oss:120b-cloud"


def test_a_corrected_meaning_of_missing_is_recorded_though_the_copy_is_the_same() -> None:
    """Apply wrote nothing when the copy's values were unchanged, yet the page then showed the correction as
    recorded, and Apply stayed offered."""
    at = _open({"status": ["yes"] * 40 + ["no"] * 40 + ["n.a."] * 3})
    _set_to_missing(at, ["n.a."], NOT_APPLICABLE)
    at.button(key="clean_apply::status").click().run()
    _ok(at)
    assert at.button(key="clean_apply::status").disabled  # nothing changed since it was written
    steps = len(at.session_state["transformation_manager"].history)
    _set_to_missing(at, ["n.a."], "declined")
    assert not at.button(key="clean_apply::status").disabled  # the record says another thing
    at.button(key="clean_apply::status").click().run()
    _ok(at)
    history = at.session_state["transformation_manager"].history
    assert len(history) == steps + 1 and history[-1]["params"]["missing_meaning"] == {
        "n.a.": "declined"
    }
    assert any("Recorded the corrected choices" in s.value for s in at.success)
    assert at.button(key="clean_apply::status").disabled
    # no change of the data: the workflow counts no new revision, and the replay checks the step's table
    from core.progress import data_revision

    assert history[-1]["data"]["fp_in"] == history[-1]["data"]["fp_out"]
    assert data_revision(history) == data_revision(history[:-1])
    replayed = apply_steps(at.session_state["data"].drop(columns=["status_clean"]), history[-2:])
    assert replayed.error is None and not replayed.warnings
    assert replayed.verified_steps == 2 and replayed.divergence is None


def _popover(at: AppTest, prefix: str):
    found = []

    def walk(node) -> None:
        proto = getattr(node, "proto", None)
        if getattr(node, "type", None) == "popover" and prefix in str(getattr(proto, "id", "")):
            found.append(proto.popover)
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(at._tree)
    (popover,) = found
    return popover


def test_a_code_answered_a_result_is_kept_and_shown_as_answered() -> None:
    """ND said to be a result: the answer was kept nowhere, so the table started again at its first row, the cell read
    "—", and "Tick the other 1" set ND back to missing."""
    at = _open(
        {
            "virus": ["detected"] * 40
            + ["negative"] * 40
            + ["ND"] * 6
            + ["n.a."] * 3
            + ["unknown"] * 2
        }
    )
    assert _tick_all(at, "missing").label == "Tick all 3"
    r = _row(at, "ND")
    first = _edit(at, {str(r): {"Missing": RESULT}})
    assert (
        _editor(at).proto.id == first
    )  # the store holds the answer the browser shows: the table is kept
    assert _rows(at)[r][3] == RESULT and _rows(at)[r][4] == "kept: a result"
    store = at.session_state["clean_manual"]["virus"]
    assert store["results"] == ["ND"] and "ND" not in store["mapping"]
    assert _tick_all(at, "missing").label == "Tick the other 2"  # an answered code is left as it is
    _tick_all(at, "missing").click().run()
    _ok(at)
    assert at.session_state["clean_manual"]["virus"]["results"] == ["ND"]
    assert _tick_all(at, "missing").label == "All answered"
    _set_to_missing(at, ["n.a."], NOT_APPLICABLE)
    at.button(key="clean_apply::virus").click().run()
    _ok(at)
    params = _step(at)["params"]
    assert params["results"] == ["ND"] and params["mapping"] == {"n.a.": None, "unknown": None}
    shown = report_params(params)
    assert shown["codes kept as results"] == 1 and "ND" not in repr(shown)
    assert (
        _rows(at)[_row(at, "ND")][3] == RESULT
    )  # cleaning again starts from the record, the answer with it
    assert any("Nothing changed since `virus_clean` was written." in c.value for c in at.caption)
    _edit(
        at, {str(_row(at, "ND")): {"Missing": "—"}}
    )  # the answer withdrawn: the record would say another thing
    assert not at.button(key="clean_apply::virus").disabled
    assert not at.session_state["clean_manual"]["virus"]["results"]


def test_a_pill_ticked_keeps_its_key_and_a_pill_the_table_changed_is_drawn_again() -> None:
    """The keyboard's focus stays on a pill that was ticked; a pill whose offer the table changed is new."""
    at = _open({"sex": ["F"] * 30 + ["M"] * 30 + ["Female", "male", "MALE", "unknown"]})
    editor = _editor(at).proto.id
    before = _group(at, "written").key
    _group(at, "written").select("Female").run()
    _ok(at)
    assert _group(at, "written").key == before and "Female" in _group(at, "written").value
    assert _editor(at).proto.id == editor  # the table too keeps its key, its scroll and its focus
    _edit(at, {str(_row(at, "Female")): {"Clean value": "Female"}})  # kept as written in the table
    assert _group(at, "written").key != before and "Female" not in (
        _group(at, "written").value or []
    )


def test_values_sent_show_their_hidden_spaces() -> None:
    """For an Ollama on this machine every value is sent as written: "Luxembourg" and "Luxembourg " read the same."""
    frame = {"site": ["Luxembourg"] * 40 + ["Luxembourg "] * 2 + ["Metz"] * 30 + ["Lux"] * 6}
    ai = {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "recipients": [],
        "proposals": [],
    }
    at = _open(frame, ai=ai)
    popover = "clean_ai_open::" + str(_one(at.button, "clean_map::").key).split("::", 1)[1]
    at.session_state[popover] = True
    at.run()
    _ok(at)
    table = [d for d in at.dataframe if list(d.value.columns) == ["Values sent"]]
    assert table and table[0].value["Values sent"].tolist() == [
        "Lux",
        "Luxembourg",
        "Luxembourg␣",
        "Metz",
    ]
    _one(at.button, "clean_ai_send::").click().run()
    _ok(at)
    assert any(c.value == "The AI engine made no proposal for these values." for c in at.caption)


def test_free_text_keeps_the_ai_button_disabled_and_says_why() -> None:
    """Mostly rare values are never sent: the button vanished without a word."""
    frame = {"note": [f"note {i}" for i in range(60)] + ["usual"] * 40}
    ai = {"provider": "anthropic", "base_url": None, "recipients": ["anthropic"], "proposals": []}
    at = _open(frame, ai=ai)
    popover = _popover(at, "clean_ai_open::")
    assert popover.disabled and "never sent" in popover.help


def test_the_variable_list_counts_extra_spaces_apart() -> None:
    at = _open(
        {"site": ["Metz"] * 20 + ["Metz "] * 2 + ["Luxembourg"] * 20, "age": list(range(42))}
    )
    label = _one(at.selectbox, "clean_variable::").format_func("site")
    assert label == "site · 3 values, 1 with extra spaces"


def test_the_ai_caption_accounts_for_every_proposal() -> None:
    """The caption counted proposals that the rules already offer, or set aside: the AI group held fewer."""
    from app_pages.clean_categories import _ai_said

    assert _ai_said({"shown": 5, "offered": 2, "apart": 1, "reversed": 0}) == (
        "The AI engine made 8 proposals: 5 below, under Proposed by the AI engine; 2 on values the rules already "
        "offer, below; 1 set aside: the rules keep its values apart."
    )
    assert _ai_said({"shown": 0, "offered": 2, "apart": 0, "reversed": 0}) == (
        "The AI engine made 2 proposals, all on values the rules already offer, below."
    )
    assert _ai_said({"shown": 0, "offered": 0, "apart": 0, "reversed": 0}) == (
        "The AI engine made no proposal for these values."
    )


def test_a_dense_variable_s_chart_names_every_bar() -> None:
    """34 bars drew 17 names: the 15 largest are drawn, then the others in one bar, then the missing ones."""
    from app_pages.clean_categories import _figure

    counts = {f"value {i:02d}": 100 - i for i in range(30)}
    figure = _figure(counts, {"value 29": None}, after=True)
    labels = list(figure.data[0].y)
    assert len(labels) == 17
    assert labels[15] == f"14 others ({sum(100 - i for i in range(15, 29)):,})"
    assert labels[16] == "missing (71)"
    assert (
        len(list(_figure(dict(list(counts.items())[:16]), {}, after=True).data[0].y)) == 16
    )  # no "1 others"


def test_the_grouping_question_names_what_joins_which_value() -> None:
    from app_pages.clean_categories import _grouping_question

    question = _grouping_question(
        "Metz", ["Esch", "Luxembourg", "Metz"], ["Esch", "Luxembourg"], "Metz"
    )
    assert question == "Why group `Esch` and `Luxembourg` with `Metz`?"
    # a new name: the most held value is a category of its own, named with the new name, never as a spelling (it read
    # "Why group Stroke as Cardiovascular? Also merged, as its spellings: Hypertension")
    question = _grouping_question(
        "Cardiovascular", ["Hypertension", "Stroke"], ["Stroke"], "Hypertension"
    )
    assert question == "Why group `Stroke` with `Hypertension`, as `Cardiovascular`?"
    question = _grouping_question(
        "Hypotension (low BP)",
        ["Hypertension", "Hypertenson", "Hypotension"],
        ["Hypotension"],
        "Hypertension",
    )
    assert question == (
        "Why group `Hypotension` with `Hypertension`, as `Hypotension (low BP)`? Also merged, as spellings of "
        "`Hypertension`: `Hypertenson`."
    )
    # the neighbouring category first, the spellings apart: one answer is never read as the reason for both
    parts = ["HYPERTENSION", "Hypertension", "Hypertenson", "Hypotension", "hypertension"]
    assert _grouping_question("Hypertension", parts, ["Hypotension"], "Hypertension") == (
        "Why group `Hypotension` with `Hypertension`? Also merged, as its spellings: `HYPERTENSION`, `Hypertenson` "
        "and `hypertension`."
    )
    many = [f"v{i:02d}" for i in range(20)]
    assert _grouping_question("Other", many, many[1:], "v00").startswith(
        "Why group `v01`, `v02`, `v03`, `v04`, `v05`, `v06` and 13 more with `v00`, as `Other`?"
    )


def test_choices_start_again_when_a_step_changes_the_values() -> None:
    """A choice made before a merge covered the rows the merge added, and the cells already read were matched to other
    values once the rows moved: the choices are those of the values they were made on."""
    at = _open({"sex": SEX})
    _set_to_missing(at, ["n.a."], NOT_APPLICABLE)
    revision = at.session_state["clean_manual"]["sex"]["revision"]
    at.session_state["change_source"] = True  # n.a. now holds one row, F one more
    at.run()
    _ok(at)
    store = at.session_state["clean_manual"]["sex"]
    assert store["mapping"] == {} and store["revision"] > revision
    said = "1 choice not yet applied was dropped: Filled sex (a later step) changed the values of `sex`"
    assert any(i.value.startswith(said) for i in at.info)
    at.run()
    _ok(at)
    assert any(i.value.startswith(said) for i in at.info)  # said until the next choice
    _set_to_missing(at, ["unknown"], UNKNOWN)
    assert not any(i.value.startswith(said) for i in at.info)


def test_values_are_picked_by_a_regular_expression_and_mapped_at_once() -> None:
    """A messy column: the values a pattern finds are picked in Values, for the buttons that already map them."""
    at = _open(
        {
            "diagnosis": ["Hypertension"] * 40
            + ["hypertension"] * 3
            + ["HYPERTENSION"] * 2
            + ["Hypertenson"]
            + ["Diabetes"] * 30
        }
    )
    _one(at.checkbox, "clean_regex::").check()
    _one(at.text_input, "clean_pattern::").set_value("^hyp.*ten")
    _one(at.button, "clean_pattern_pick::").click().run()
    _ok(at)
    found = ["HYPERTENSION", "Hypertenson", "Hypertension", "hypertension"]
    assert sorted(_one(at.multiselect, "clean_pick::").value) == sorted(found)
    said = next(c.value for c in at.caption if c.value.startswith("4 values picked, 46 rows: "))
    assert said.startswith("4 values picked, 46 rows: **Hypertension** (40), **hypertension** (3)")
    _one(at.selectbox, "clean_into::").set_value("Hypertension").run()
    # the line stays while Values holds the pick, through the choice of a clean value
    assert any(c.value.startswith("4 values picked") for c in at.caption)
    _one(at.button, "clean_map::").click().run()
    _ok(at)
    assert not any(
        c.value.startswith("4 values picked") for c in at.caption
    )  # Values emptied: the line goes
    full = at.session_state["clean_manual"]["diagnosis"]["mapping"]
    assert {full.get(v, v) for v in found} == {"Hypertension"}
    _one(at.text_input, "clean_pattern::").set_value("(")
    _one(at.button, "clean_pattern_pick::").click().run()
    _ok(at)
    assert any("Not a regular expression" in c.value for c in at.caption)


def test_text_typed_as_it_reads_finds_it_and_nothing_found_leaves_values_as_they_were() -> None:
    """n.a., typed as it reads, picks n.a. only (a dot is a dot until Regular expression is ticked): Set to missing
    says how many rows; a search that finds nothing leaves Values as it was."""
    at = _open(
        {
            "diagnosis": ["Myocardial infarction"] * 30
            + ["n.a."] * 6
            + ["Diabetes"] * 30
            + ["Asthma"] * 5
        }
    )
    _one(at.text_input, "clean_pattern::").set_value("n.a.")
    _one(at.button, "clean_pattern_pick::").click().run()
    _ok(at)
    assert _one(at.multiselect, "clean_pick::").value == ["n.a."]
    missing = _one(at.button, "clean_missing::")
    assert missing.label == "Set 1 value to missing (6 rows)"
    _one(at.text_input, "clean_pattern::").set_value("zzz")
    _one(at.button, "clean_pattern_pick::").click().run()
    _ok(at)
    assert _one(at.multiselect, "clean_pick::").value == ["n.a."]
    assert any(c.value == "Nothing found: Values is as it was." for c in at.caption)


def test_what_a_search_picked_is_named_under_the_row() -> None:
    """The six values holding the most rows are named with their rows, then how many more; nothing found is said."""
    from app_pages.clean_categories import _picked_words

    counts = {f"v{i}": i + 1 for i in range(9)}
    said = _picked_words(list(counts), counts)
    assert (
        said.startswith("9 values picked, 45 rows: **v8** (9), **v7** (8), ")
        and said.count("**") == 12
    )
    assert "**v3** (4) and 3 more. Give them a clean value" in said
    assert _picked_words(["x"], {"x": 1}).endswith(
        "1 row: **x** (1). Give it a clean value and Map, or set it to missing."
    )
    assert _picked_words([], counts) == "Nothing found: Values is as it was."


def test_a_copy_a_later_step_left_behind_its_source_is_said_and_written_again() -> None:
    """sex cleaned into sex_clean, then a later step fills a value of sex: the page never says "Nothing changed" of a
    copy that no longer follows its source, says why, and writes it again; the list says so too."""
    at = _open({"sex": SEX})
    _set_to_missing(at, ["n.a.", "unknown"], UNKNOWN)
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    assert at.button(key="clean_apply::sex").disabled  # written: nothing changed
    assert any("Nothing changed since" in c.value for c in at.caption)
    at.session_state["change_source"] = True  # a later step fills the last n.a. of sex
    at.run()
    _ok(at)
    assert not any("Nothing changed since" in c.value for c in at.caption)
    assert any("changed since `sex_clean` was written (a later step)" in i.value for i in at.info)
    assert "which no longer follows it" in _one(at.selectbox, "clean_variable::").format_func("sex")
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    data = at.session_state["data"]
    assert data["sex_clean"].tolist()[-1] == "F"  # the filled row brought in
    assert "which no longer follows it" not in _one(at.selectbox, "clean_variable::").format_func(
        "sex"
    )


def test_a_freeze_leaves_a_copy_following_its_source() -> None:
    """Keep only these rows removes rows from the source and its copy alike: the page said that sex "changed since
    sex_clean was written", the list that the copy no longer followed it, and writing again recorded "the corrected
    choices" though none was corrected."""
    at = _open({"sex": SEX})
    _set_to_missing(at, ["n.a.", "unknown"], UNKNOWN)
    at.button(key="clean_apply::sex").click().run()
    _ok(at)
    steps = len(at.session_state["transformation_manager"].history)
    at.session_state["freeze"] = True
    at.run()
    _ok(at)
    assert not any("changed since" in i.value for i in at.info)
    assert "which no longer follows it" not in _one(at.selectbox, "clean_variable::").format_func(
        "sex"
    )
    assert at.button(key="clean_apply::sex").disabled
    assert any("Nothing changed since `sex_clean` was written" in c.value for c in at.caption)
    assert len(at.session_state["transformation_manager"].history) == steps + 1  # the freeze only


def test_a_grouping_that_no_longer_groups_is_recorded_as_a_choice() -> None:
    """A record cleaned again keeps its kinds; a grouping whose clean value now groups nothing (Diferdange into
    Differdange: a typo's rows) is recorded as a choice, never as a grouping without one or its reason."""
    from app_pages.clean_categories import _record_of
    from enrich.recode import CHOSEN, GROUPING

    record = ({"Diferdange": "Differdange"}, {"Diferdange": GROUPING}, [], {})
    out = _record_of(record, {}, {"reasons": {}}, {"Differdange": 300, "Diferdange": 2})
    assert out["kinds"] == {"Diferdange": CHOSEN} and out["groupings"] == {}
    grouped = {"Differdange": ["Diferdange", "Differdange"]}
    out = _record_of(
        record,
        grouped,
        {"reasons": {"Differdange": "one town"}},
        {"Differdange": 300, "Diferdange": 200},
    )
    assert (
        out["kinds"] == {"Diferdange": GROUPING}
        and out["groupings"]["Differdange"]["reason"] == "one town"
    )
