"""
How a text file is read (M5): the delimiter choice of Data > Import, Enrich's external file and Validate's Compare.
A rarer delimiter is detected or typed ('other…'); one that cannot separate columns is refused.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from core.state import ParseOptions

STAR = b"id*group*value\n1*A*2.5\n2*B*3.5\n"


def _choice_script() -> None:
    import streamlit as st

    from ui.components.csv_options import delimiter_input

    st.session_state["sep"] = delimiter_input(st.session_state.get("current", ","), key="t")


def _options_script() -> None:
    import streamlit as st

    from ui.components.csv_options import csv_options

    options = csv_options(st.session_state["data"], st.session_state["name"], key="t")
    st.session_state["chosen"] = None if options is None else (options.sep, options.decimal)


def _import_script() -> None:
    import streamlit as st

    from ui.import_flow import _render_options

    st.session_state["options"] = _render_options(st.session_state["draft"])


def _run(script, **state) -> AppTest:
    at = AppTest.from_function(script, default_timeout=60)
    for name, value in state.items():
        at.session_state[name] = value
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def test_the_usual_delimiters_space_and_any_other_character() -> None:
    at = _run(_choice_script)
    assert at.radio(key="t").value == "," and at.session_state["sep"] == ","
    at.radio(key="t").set_value(" ").run()
    assert at.session_state["sep"] == " "
    at.radio(key="t").set_value("other").run()
    assert at.session_state["sep"] == ""  # nothing typed yet
    at.text_input(key="t_other").input("*").run()
    assert at.session_state["sep"] == "*"
    assert any("Read with: '*'" in c.value for c in at.caption)


def test_a_detected_rarer_delimiter_is_shown_in_the_other_box() -> None:
    at = _run(_choice_script, current="^")
    assert at.radio(key="t").value == "other"
    assert at.text_input(key="t_other").value == "^" and at.session_state["sep"] == "^"


def test_a_file_read_outside_import_can_be_read_with_another_delimiter() -> None:
    at = _run(_options_script, data=STAR, name="labs.csv")
    assert at.session_state["chosen"] == (",", ".")  # '*' is not detected: one column
    assert any("single column" in w.value for w in at.warning)
    at.radio(key="t_sep").set_value("other").run()
    at.text_input(key="t_sep_other").input("*").run()
    assert at.session_state["chosen"] == ("*", ".")


def test_a_delimiter_that_cannot_work_keeps_the_detected_options() -> None:
    at = _run(_options_script, data=b"id;bmi\n1;24,5\n2;31,2\n", name="bmi.csv")
    assert at.session_state["chosen"] == (";", ",")
    assert not any("single column" in w.value for w in at.warning)
    at.radio(key="t_sep").set_value("other").run()
    at.text_input(key="t_sep_other").input("7").run()
    assert at.session_state["chosen"] == (";", ",")
    assert any("letter or a digit" in w.value for w in at.warning)


def test_a_workbook_has_no_delimiter_to_choose() -> None:
    at = _run(_options_script, data=b"", name="labs.xlsx")
    assert at.session_state["chosen"] is None and not at.expander


def test_the_import_page_offers_the_same_choice() -> None:
    draft = {
        "options": ParseOptions(sep="^"),
        "kind": "csv",
        "file_id": "f1",
        "sheet_names": [],
        "data": b"id^v\n1^2\n",
    }
    at = _run(_import_script, draft=draft)
    assert at.radio(key="imp_sep_f1").value == "other"
    assert at.text_input(key="imp_sep_f1_other").value == "^"
    assert at.session_state["options"].sep == "^"
