"""
A calculator's result in the study dossier (calculations from numbers typed): the button under the numbers typed, the
basket that keeps them apart from the results of the dataset, the Study dossier page's own list, and the Word
dossier's own section. A calculation read no data: it is never stale, never in the participant flow, and never said
to be "Computed on" the dataset.

Numbers: 15 events over 1,200 person-years against 4 over 1,000 (ratio 3.125, p = 0.051); 25 observed against 18.4
expected (1.359); 30 of 100 against 20% (p = 0.017).
"""

from __future__ import annotations

import json
import logging
import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from docx import Document
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from biostats import plan
from ui import nav
from ui.components.population import _family_key
from ui.guidance import measures as m

ROOT = Path(__file__).resolve().parents[2]
PAIRED = "Paired, rates & stratified"
TWO = {"events": 15, "person_time": 1200.0, "reference_events": 4, "reference_person_time": 1000.0}
RATIO_WORDS = (
    "15 events over 1,200 person-years (group compared) against 4 over 1,000 (reference group), rates per 1,000 "
    "person-years"
)
NOT_DATA = "Not from the dataset: the analysis population and the imputation do not apply."
DIAGNOSTIC = "Diagnostic accuracy & agreement"
TABLE = {"tp": 90, "fp": 30, "fn": 10, "tn": 270}
TABLE_WORDS = "test positive, 90 with the condition and 30 without; test negative, 10 with the condition and 270 without"
# a cut-off found on the data that calls one participant positive: its table is not written, typed back or not
FOUND = {"tp": 1, "fp": 0, "fn": 29, "tn": 370}


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=600)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "typed_dossier_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    at.switch_page(nav.page_path("analyze")).run()
    _ok(at)
    yield at
    dp.DATA_ROOT = previous


def _open(at: AppTest, question: str, **choices: str) -> None:
    at.switch_page(nav.page_path("analyze")).run()
    at.session_state["epi_tab"] = PAIRED
    at.run()
    if at.segmented_control(key="paired_question").value != question:
        at.segmented_control(key="paired_question").set_value(question).run()
    for name, value in choices.items():
        if at.segmented_control(key=name).value != value:
            at.segmented_control(key=name).set_value(value).run()
    _ok(at)


def _type(at: AppTest, key: str, value: float | None) -> None:
    at.number_input(key=key).set_value(value).run()
    _ok(at)


def _texts(at: AppTest) -> str:
    return "\n".join([*(c.value for c in at.caption), *(md.value for md in at.markdown)])


def _buttons(at: AppTest, prefix: str) -> list[str]:
    return [b.key for b in at.button if str(b.key or "").startswith(prefix)]


def _basket(at: AppTest) -> list[dict[str, Any]]:
    return (
        list(at.session_state["_report_typed"].values())
        if "_report_typed" in at.session_state
        else []
    )


def _two_rates(at: AppTest, events: int = 15) -> None:
    _open(at, m.RATES, rate_calc=m.TWO_RATES)
    for key, value in (
        ("rate_e1", events),
        ("rate_t1", 1200.0),
        ("rate_e0", 4),
        ("rate_t0", 1000.0),
    ):
        _type(at, key, value)


def _document(parts: dict[str, Any]) -> tuple[list[str], str]:
    """The Word dossier of these parts, read back: its headings, and its paragraphs with its tables' cells."""
    from app_pages.study_dossier import finish_input
    from manage.dossier import build_dossier

    doc = Document(
        BytesIO(build_dossier(finish_input(parts, title="A study", author="a")).getvalue())
    )
    cells = [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    return headings, "\n".join([*(p.text for p in doc.paragraphs), *cells])


def _parts(typed: list[dict[str, Any]], kept: list | None = None) -> dict[str, Any]:
    """The parts of a made-up session, with these calculations (and these results of the dataset) kept."""
    from core.attrition import FlowBox, ParticipantFlow

    return {
        "flow": ParticipantFlow(main=(FlowBox("loaded", "Loaded: made-up", 400, "loaded"),)),
        "kept": kept or [],
        "omitted": [],
        "population_label": "all 400 rows",
        "provenance": [("Dataset", "made-up")],
        "criteria": None,
        "data_steps": [],
        "missing": None,
        "single_imputed": {},
        "exact_imputed": {},
        "mi": None,
        "typed": typed,
    }


def _stored(kind: str, numbers: dict, note: str = "", **options: Any) -> dict[str, Any]:
    """A calculation as the session keeps it."""
    from manage.report_tables import typed_numbers
    from ui.results import typed_id

    spec = plan.specify(kind, typed=numbers, options=options)
    return {
        "id": typed_id(spec),
        "spec": spec.to_dict(),
        "note": note,
        "words": typed_numbers(spec),
        "added_at": "2026-11-16T16:40:00",
    }


# ----------------------------------------------------------------------------- on the calculator's page


def test_the_button_stands_under_a_result_only(app: AppTest) -> None:
    _open(app, m.RATES, rate_calc=m.TWO_RATES)
    for key in ("rate_e1", "rate_t1", "rate_e0", "rate_t0"):
        _type(app, key, None)
    assert not app.metric and not _buttons(app, "report_typed_")
    # where the numbers come from is asked with the numbers, before any result
    assert app.text_input(key="rate_from").value in ("", None)
    _two_rates(app)
    assert {metric.label for metric in app.metric} >= {"Rate ratio"}
    assert len(_buttons(app, "report_typed_add::typed:rates:")) == 1
    text = _texts(app)
    assert f"From the numbers typed: {RATIO_WORDS}. {NOT_DATA}" in text
    assert not _basket(app)  # nothing is kept until it is added


def test_a_source_typed_just_before_the_click_is_kept_with_the_calculation(app: AppTest) -> None:
    """A button's callback runs with the arguments of the run before: the source typed and the click that follows
    it reach the server together, and the calculation was kept with "Source: not given."; the button then read Add
    again, and a second click kept the calculation twice."""
    _two_rates(app)
    (add,) = _buttons(app, "report_typed_add::")
    app.text_input(key="rate_from").input(
        "Smith 2021, table 2"
    )  # not committed by a run of its own
    app.button(key=add).click().run()
    _ok(app)
    (item,) = _basket(app)
    assert item["note"] == "Smith 2021, table 2" and item["words"] == RATIO_WORDS
    assert item["id"].startswith("typed:rates:") and item["id"] == add.removeprefix(
        "report_typed_add::"
    )
    assert item["spec"]["typed"] == TWO and item["spec"]["kind"] == "rates"
    assert json.loads(json.dumps(item)) == item  # JSON only: no object an update could leave unread
    assert f"From the numbers typed (Smith 2021, table 2): {RATIO_WORDS}." in _texts(app)
    # the button now says where it is; the results of the dataset know nothing of it
    assert _buttons(app, "report_typed_remove::") == [f"report_typed_remove::{item['id']}"]
    assert not _buttons(app, "report_typed_add::") and not _buttons(app, "report_typed_update::")
    assert "rates" not in (app.session_state["_results"] if "_results" in app.session_state else {})
    assert not (app.session_state["_report"] if "_report" in app.session_state else {})


def test_the_same_numbers_are_one_item_whatever_their_source(app: AppTest) -> None:
    """Named by its numbers and its source, a source typed after adding made a second item of one calculation, the
    first left with "Source: not given."."""
    (item,) = _basket(app)
    app.text_input(key="rate_from").input("Smith 2021, table 3").run()
    _ok(app)
    (update,) = _buttons(app, "report_typed_update::")
    assert not _buttons(app, "report_typed_add::") and not _buttons(app, "report_typed_remove::")
    app.button(key=update).click().run()
    _ok(app)
    (again,) = _basket(app)
    assert again["id"] == item["id"] and again["note"] == "Smith 2021, table 3"
    assert _buttons(app, "report_typed_remove::") == [f"report_typed_remove::{item['id']}"]
    # the source emptied: the dossier's can be taken out the same way
    app.text_input(key="rate_from").input("")
    app.button(
        key=f"report_typed_remove::{item['id']}"
    ).click().run()  # drawn for the source before
    assert not _basket(app)
    (add,) = _buttons(app, "report_typed_add::")
    app.text_input(key="rate_from").input("Smith 2021, table 2")
    app.button(key=add).click().run()
    _ok(app)
    # another number: another calculation, to add; the dossier keeps the first as it was
    _type(app, "rate_e1", 16)
    (other,) = _buttons(app, "report_typed_add::")
    assert other != add and len(_basket(app)) == 1
    app.button(key=other).click().run()
    assert [stored["spec"]["typed"]["events"] for stored in _basket(app)] == [15, 16]
    # the first numbers typed again: the item is found, not added twice
    _type(app, "rate_e1", 15)
    assert _buttons(app, "report_typed_remove::") == [f"report_typed_remove::{item['id']}"]
    assert len(_basket(app)) == 2


def test_the_other_calculators_add_their_own_kinds(app: AppTest) -> None:
    _open(app, m.RATES, rate_calc=m.EXPECTED)
    _type(app, "smr_obs", 25)
    _type(app, "smr_exp", 18.4)
    text = _texts(app)
    assert "From the numbers typed: 25 observed deaths against 18.4 expected." in text
    # the test named by its rule, in the words of the Word file
    assert "(exact Poisson test, twice the smaller tail)" in text
    (add,) = _buttons(app, "report_typed_add::typed:standardised_ratio:")
    app.button(key=add).click().run()
    _open(app, m.ONE_GROUP, one_compare=m.SHARE, prop_source=m.TYPED)
    _type(app, "prop_k", 30)
    _type(app, "prop_n", 100)
    _type(app, "prop_ref_typed", 0.2)
    (add,) = _buttons(app, "report_typed_add::typed:proportion:")
    app.button(key=add).click().run()
    _ok(app)
    assert [stored["spec"]["kind"] for stored in _basket(app)] == [
        "rates",
        "rates",
        "standardised_ratio",
        "proportion",
    ]


# ----------------------------------------------------------------------------- on the Study dossier page


def test_the_dossier_lists_calculations_apart_never_stale_and_out_of_the_flow(app: AppTest) -> None:
    app.switch_page(nav.page_path("dossier")).run()
    _ok(app)
    text = _texts(app)
    assert "**From numbers typed by hand**" in text
    assert (
        "not computed on the dataset, so the participant flow does not count them. They stay with this dataset, as "
        "the results do: loading another dataset clears them."
    ) in text
    assert "**Rate ratio (numbers typed)** · " + RATIO_WORDS in text
    assert "**Standardised mortality ratio (numbers typed)**" in text
    assert "**Share against a reference (numbers typed)**" in text
    # each card: its source first, then when it was added; what the heading says is not repeated
    assert text.count(":material/edit_note: Source: Smith 2021, table 2. Added at ") == 2
    assert text.count("Source: not given. Added at ") == 2
    assert NOT_DATA not in text and "Computed on" not in text
    # no result of the dataset yet, and the page says which kind is missing
    assert any("No result of the dataset yet" in info.value for info in app.info)
    # the flow counts no calculation
    flow = app.selectbox(key="dossier_flow::item").options
    assert not any("typed" in option or "ratio" in option.lower() for option in flow)
    # a change of the analysis population: a calculation read no data, and is not stale
    app.checkbox(key=_family_key("study_inclusion")).check().run()
    _ok(app)
    assert len(_basket(app)) == 4 and "stale" not in _texts(app).lower()
    app.checkbox(key=_family_key("study_inclusion")).uncheck().run()
    # Remove takes one calculation out
    second = _basket(app)[1]["id"]
    app.button(key=f"dossier_remove::{second}").click().run()
    _ok(app)
    assert [stored["spec"]["kind"] for stored in _basket(app)] == [
        "rates",
        "standardised_ratio",
        "proportion",
    ]


def test_nothing_typed_reaches_a_log_the_trace_or_the_flow(
    app: AppTest, caplog: pytest.LogCaptureFixture
) -> None:
    """Typed numbers, their source and the name of a calculation stay in the session's list: no log record holds
    one, no step is recorded, the study definition's checksum does not move, and the participant flow is the
    same with and without them."""
    from core.protocol import body_checksum, body_of

    def trace() -> Any:
        if "transformation_manager" not in app.session_state:
            return None
        tm = app.session_state["transformation_manager"]
        return getattr(tm, "head_hash", None), len((tm.get_trace() or {}).get("steps", []))

    def definition() -> str:
        return body_checksum(
            body_of(app.session_state["config"], app.session_state["_data_session"].ticked)
        )

    kept = json.loads(json.dumps(_basket(app)))
    before = trace(), definition()
    with caplog.at_level(logging.DEBUG):
        _two_rates(app, events=17)
        (add,) = _buttons(app, "report_typed_add::")
        app.text_input(key="rate_from").input("Register of Wiltz, 2021")
        app.button(key=add).click().run()
        app.switch_page(nav.page_path("dossier")).run()
        _ok(app)
        _document(_parts(json.loads(json.dumps(_basket(app)))))
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert (
        "Wiltz" not in logged and "typed:rates:" not in logged and "typed:proportion:" not in logged
    )
    assert "17 events" not in logged
    # no step of the trace for a calculation, nothing of it in the study definition, and no box of the flow:
    # the same with and without the list
    assert (trace(), definition()) == before
    flow_with = list(app.selectbox(key="dossier_flow::item").options)
    app.session_state["_report_typed"] = {}
    app.run()
    assert list(app.selectbox(key="dossier_flow::item").options) == flow_with
    app.session_state["_report_typed"] = {stored["id"]: stored for stored in kept}
    app.run()
    _ok(app)


def test_a_calculation_this_version_cannot_compute_keeps_its_place_and_its_numbers(
    app: AppTest,
) -> None:
    """An item written by a later version showed "Type its numbers again" without the numbers; one that read but no
    longer computed was listed without a word while the Word file wrote a fixed line."""
    kept = dict(app.session_state["_report_typed"])
    later = _stored("rates", TWO, note="Smith 2021, table 2")
    later["spec"] = {**later["spec"], "kind_version": 2}
    app.session_state["_report_typed"] = {later["id"]: later}
    app.switch_page(nav.page_path("dossier")).run()
    _ok(app)
    text = _texts(app)
    assert "**Calculation from numbers typed** · " + RATIO_WORDS in text
    assert (
        ":material/warning: This version of CohortStudio cannot compute this calculation again; the Word dossier "
        "writes a line in its place. Type its numbers again on the calculator's page."
    ) in text
    assert (
        text.count(RATIO_WORDS) == 1
    )  # in the card's heading; said again under it, they stood twice
    app.session_state["_report_typed"] = kept
    app.run()
    _ok(app)


def test_the_word_dossier_writes_calculations_in_a_section_of_their_own(app: AppTest) -> None:
    from manage.dossier import CALCULATIONS_INTRO, NO_CALCULATION

    headings, text = _document(_parts(json.loads(json.dumps(_basket(app)))))
    at = headings.index("6. Calculations from numbers typed by hand")
    assert headings[at - 1 : at] == ["5. Results"] and headings[-1] == "7. References"
    assert headings[at + 1 : -1] == [
        "Rate ratio (numbers typed)",
        "Standardised mortality ratio (numbers typed)",
        "Share against a reference (numbers typed)",
    ]
    assert (
        CALCULATIONS_INTRO in text and "No result of the dataset was added to the dossier." in text
    )
    assert (
        "if they were read from it, they are not updated when the data change" in CALCULATIONS_INTRO
    )
    assert f"Numbers typed: {RATIO_WORDS}." in text
    assert "Source: Smith 2021, table 2. Typed by hand, added on " in text
    assert "Source: not given. Typed by hand, added on " in text
    assert "Not computed from the dataset." in text
    # the results as the page gives them, each test named by its rule and each interval by its method
    assert "Rate ratio\n3.125\n0.996 to 12.935" in text and "Measure\nEstimate\n95% CI" in text
    assert "p = 0.051 (exact conditional test, twice the smaller tail)." in text
    assert "Rate difference: 8.5 per 1,000 person-years (no interval is given for it)." in text
    assert (
        "Intervals: exact Poisson (Garwood) for the rates; exact conditional (Clopper-Pearson) for the ratio."
        in text
    )
    assert (
        "Standardised mortality ratio = 1.359 (95% CI 0.879 to 2.006), p = 0.165 (exact Poisson test, twice the "
        "smaller tail); 25 observed, 18.4 expected." in text
    )
    assert (
        "Share: 30.0% (30 of 100; 95% CI 21.9% to 39.6%, Wilson); against a reference of 20.0%: p = 0.017 (exact "
        "binomial test, two-sided); 10.0 points above the reference." in text
    )
    # what the numbers are taken to be: said as an assumption, never as a fact about numbers the software never saw
    assert "Note: The events are taken as first events, one per participant" in text
    assert "First events only" not in text
    # a calculation claims nothing that only a result of the dataset can
    for claim in ("Computed on", "N analysed", "population:", "trace ", "data revision"):
        assert claim not in text[text.index("6. Calculations") :], claim
    # their methods join the references
    references = text[text.index("7. References") :]
    assert "Garwood" in references and "Ulm" in references and "Wilson" in references
    # without a calculation, the section says so
    _, empty = _document(_parts([]))
    assert NO_CALCULATION in empty and CALCULATIONS_INTRO not in empty


# ----------------------------------------------------------------------------- the 2x2 table typed


def _table_form(at: AppTest) -> None:
    """Analyze > Diagnostic accuracy & agreement > Accuracy of a test > Positive or negative > Numbers I type."""
    at.switch_page(nav.page_path("analyze")).run()
    at.session_state["epi_tab"] = DIAGNOSTIC
    at.run()
    for key, value in (
        ("diag_question", m.ACCURACY),
        ("diag_result", m.YES_NO),
        ("diag_source", m.TYPED),
    ):
        if at.segmented_control(key=key).value != value:
            at.segmented_control(key=key).set_value(value).run()
    _ok(at)


def _table(at: AppTest, tp: int | None, fp: int | None, fn: int | None, tn: int | None) -> None:
    for key, value in (("diag_tp", tp), ("diag_fp", fp), ("diag_fn", fn), ("diag_tn", tn)):
        if at.number_input(key=key).value != value:
            _type(at, key, value)


def test_the_table_of_a_paper_is_a_calculator_inside_accuracy_of_a_test(app: AppTest) -> None:
    _table_form(app)
    _table(app, None, None, None, None)
    assert not app.metric and not _buttons(app, "report_typed_")
    assert "Type the four counts of the table: the result appears here." in _texts(app)
    assert not [button for button in app.button if button.label == "Run analysis"]  # a calculator
    _table(app, **TABLE)
    text = _texts(app)
    assert (
        "100 with the condition and 300 without; 120 positive tests and 280 negative; 400 in all."
        in text
    )
    shown = {metric.label: metric.value for metric in app.metric}
    assert (shown["Sensitivity"], shown["Specificity"], shown["Participants"]) == (
        "90.0%",
        "90.0%",
        "400",
    )
    assert f"From the numbers typed: {TABLE_WORDS}. {NOT_DATA}" in text
    # not stated: no predictive value at the table's share, and no Run to "run again"
    assert (
        "No predictive values: if the participants were taken as they came (a series), say so to read them at the "
        "share in this table (25.0%); otherwise give the prevalence where the test will be used."
        in text
    )
    assert "run again" not in text and "these data" not in text
    (add,) = _buttons(app, "report_typed_add::typed:diagnostic:")
    # a series reads them at the table's share, and is another calculation than the same table not stated
    app.radio(key="diag_sampling_typed").set_value("series").run()
    _ok(app)
    text = _texts(app)
    assert (
        "Predictive values at the share with the condition in this table (25.0%, a series): positive 75.0%"
        in text
    )
    assert "In this table: 25.0% (a series)." in text
    (series,) = _buttons(app, "report_typed_add::typed:diagnostic:")
    assert series != add
    app.text_input(key="diag_from").input(
        "Smith 2021, table 2"
    )  # typed, then the click, in one run
    app.button(key=series).click().run()
    _ok(app)
    item = _basket(app)[-1]
    assert item["spec"]["kind"] == "diagnostic" and item["spec"]["typed"] == TABLE
    assert item["spec"]["options"] == {"alpha": 0.05, "prevalence": None, "sampling": "series"}
    assert item["note"] == "Smith 2021, table 2"
    assert item["words"] == f"{TABLE_WORDS}; participants taken as they came (a series)"
    assert _buttons(app, "report_typed_remove::") == [f"report_typed_remove::{item['id']}"]
    # never a result of the dataset: nothing kept under the accuracy analysis, nothing in its list
    assert "diagnostic" not in (
        app.session_state["_results"] if "_results" in app.session_state else {}
    )


def test_a_table_that_cannot_be_one_is_refused_and_one_swapped_is_said(app: AppTest) -> None:
    _table_form(app)
    _table(app, 0, 30, 0, 270)  # no one with the condition
    assert not app.metric and not _buttons(app, "report_typed_")
    assert (
        "Type participants with the condition (the first column) and without it (the second)"
        in _texts(app)
    )
    _table(app, 90, 30, 0, 0)  # every test positive
    assert not app.metric
    assert "Type tests positive and tests negative: with a row of zeros" in _texts(app)
    # the condition typed in the rows: the test reads positive more often without the condition than with it
    _table(app, 10, 270, 90, 30)
    said = "\n".join(warning.value for warning in app.warning).replace("\\", "")  # shown as text
    assert (
        "In this table the test is positive for 10.0% of those with the condition and 90.0% of those without it: "
        "it points the other way. Check that the two rows, or the two columns, are not exchanged."
        in said
    )
    # fewer than 10 on a side: a wide interval, said before the result; the result is shown all the same
    _table(app, 3, 30, 4, 270)
    # and the calculator says what the study dossier will do with it, before it is added
    assert (
        "7 participants with the condition: the interval of the sensitivity is wide; added to the study dossier, "
        "this table is written without its counts and measures." in _texts(app)
    )
    # a side of 10 to the dossier, and enough for the interval: nothing is said
    _table(app, 9, 30, 1, 270)
    assert "the interval of the sensitivity is wide" not in _texts(app)
    # a rare condition: its share is written with the decimals that show it
    _table(app, 20, 500, 4, 49476)
    assert "In this table: 0.048% (a series)." in _texts(app)
    assert "in this table (0.048%, a series)" in _texts(app) and "(0.0%" not in _texts(app)
    assert "Compare these totals with the paper's" in _texts(app)
    assert {metric.label for metric in app.metric} >= {"Sensitivity", "Specificity"}
    _table(app, **TABLE)


def test_the_dossier_page_lists_a_table_typed_and_says_when_it_is_not_written(
    app: AppTest,
) -> None:
    kept = dict(app.session_state["_report_typed"])
    small = _stored("diagnostic", {"tp": 3, "fp": 30, "fn": 4, "tn": 270}, note="A register, 2021")
    whole = _stored("diagnostic", TABLE, sampling="series")
    app.session_state["_report_typed"] = {small["id"]: small, whole["id"]: whole}
    app.switch_page(nav.page_path("dossier")).run()
    _ok(app)
    text = _texts(app)
    assert f"**Diagnostic accuracy of a yes/no test (numbers typed)** · {whole['words']}" in text
    # the table the Word dossier does not write keeps its numbers in the user's own list, with the reason
    assert f"**Diagnostic accuracy of a yes/no test (numbers typed)** · {small['words']}" in text
    assert text.count(":material/visibility_off: ") == 1
    assert (
        ":material/visibility_off: Fewer than 10 participants have the condition in this table: the t"
        "able and its measures read them one by one, so the Word dossier writes neither, as f"
        "or a table of the dataset; the calculator shows them." in text
    )
    app.session_state["_report_typed"] = kept
    app.run()
    _ok(app)


# ----------------------------------------------------------------------------- the words, without a session


def test_a_table_typed_is_written_as_the_table_of_a_yes_no_test_of_the_dataset() -> None:
    """One block for a 2x2 table, whatever its source: the measures, the table and the sampling's line of a table
    typed are those of a yes/no test of the dataset with the same four counts; only what a table cannot name (a
    column, a value) is absent, and its share is "this table"'s."""
    from biostats.diagnostic import diagnostic_accuracy
    from manage.report_tables import TABLE_NOTE, result_blocks

    typed = _stored(
        "diagnostic", TABLE, note="Smith 2021, table 2", sampling="series", prevalence=0.02
    )
    headings, text = _document(_parts([typed]))
    assert "Diagnostic accuracy of a yes/no test (numbers typed)" in headings
    assert (
        f"Numbers typed: {TABLE_WORDS}; participants taken as they came (a series); predictive values at a "
        "prevalence of 2%." in text
    )
    assert "Source: Smith 2021, table 2. Typed by hand, added on 2026-11-16 at 16:40." in text
    counted = result_blocks(
        {
            "acc": diagnostic_accuracy(90, 30, 10, 270, prevalence=0.02, sampling="series"),
            "auc": None,
            "index_form": "yes_no",
            "outcome": "ill",
            "score": "rapid",
            "pos_label": "Yes",
            "absent": "No",
            "positive": "Positive",
            "negative": "Negative",
            "prevalence": 0.02,
        }
    )
    (_, measures), (caption, _) = counted.tables
    for row in measures.itertuples(index=False):
        name = str(row[0]).replace("these data", "this table")
        assert f"{name}\n{row[1]}\n{row[2]}" in text, name
    assert caption in text and "Test positive\n90\n30\n120" in text
    assert "Test negative\n10\n270\n280" in text and "Total\n100\n300\n400" in text
    for line in counted.lines:
        assert line in text
    assert f"Note: {TABLE_NOTE}" in text
    # nothing of a column or of a value, and nothing that only a result of the dataset can claim
    for absent in (
        "rapid",
        "ill",
        "(Yes)",
        "(Positive)",
        "these data",
        "Computed on",
        "N analysed",
    ):
        assert absent not in text[text.index("6. Calculations") :], absent
    references = text[text.index("7. References") :]
    assert "Simel" in references and "Mercaldo" in references


def test_a_table_typed_with_a_side_under_ten_is_not_written_nor_its_numbers() -> None:
    """Typed numbers are counted as people's under the rule of the document they go into: the accuracy block's
    rule for a side of the reference standard reads a table typed as it reads one of the dataset. The line of the
    numbers typed is held back with the table: the four counts are the table."""
    small = _stored("diagnostic", {"tp": 3, "fp": 30, "fn": 4, "tn": 270}, note="A register, 2021")
    few_without = _stored("diagnostic", {"tp": 30, "fp": 3, "fn": 270, "tn": 6})
    whole = _stored("diagnostic", {"tp": 9, "fp": 1, "fn": 1, "tn": 9})  # 10 on each side: written
    headings, text = _document(_parts([small, few_without, whole]))
    assert headings.count("Diagnostic accuracy of a yes/no test (numbers typed)") == 3
    assert (
        "Fewer than 10 participants have the condition in this table: the table and its measures read them on"
        "e by one, so the Word dossier writes neither, as for a table of the dataset; the calculator shows them."
        in text
    )
    assert (
        "Fewer than 10 participants are without the condition in this table: the table and its"
        in text
    )
    assert "Source: A register, 2021. Typed by hand, added on " in text
    # neither the counts, in any writing, nor a measure of the two tables held back, in a paragraph or in a cell
    assert text.count("Numbers typed: ") == 1
    assert "Numbers typed: test positive, 9 with the condition and 1 without" in text
    for written in ("3 with the condition", "270", "0.429", "42.9", "Test positive\n3\n30"):
        assert written not in text, written
    assert text.count("Sensitivity\n") == 1 and "Test positive\n9\n1\n10" in text


def test_a_table_typed_that_cannot_be_computed_writes_none_of_its_counts() -> None:
    """An item this version cannot read keeps the words of its numbers, which the Word file writes in place of its
    result. For a table they are its four counts, and the rule that would hold them back cannot read an item that
    does not compute: nothing of such a table is written, where a share or rates keep their words."""
    from manage.dossier import CALCULATION_NOT_COMPUTED

    small = {"tp": 3, "fp": 30, "fn": 4, "tn": 270}
    later = _stored("diagnostic", small, note="A register, 2021")
    later["spec"] = {**later["spec"], "kind_version": 2}
    broken = _stored("diagnostic", small)
    broken["spec"] = {**broken["spec"], "typed": {**small, "tp": 0, "fn": 0}}
    unread = _stored("diagnostic", small)
    unread["spec"] = "not a spec"
    rates = _stored("rates", TWO)
    rates["spec"] = {**rates["spec"], "kind_version": 2}
    for item in (later, broken, unread):
        assert item["words"].startswith("test positive, 3 with the condition")
    headings, text = _document(_parts([later, broken, unread, rates]))
    assert text.count(CALCULATION_NOT_COMPUTED) == 4
    assert "3 with the condition" not in text and "270" not in text
    assert text.count("Numbers typed: ") == 1 and f"Numbers typed: {RATIO_WORDS}." in text
    assert "Source: A register, 2021. Typed by hand" in text
    # the one that still reads keeps its title
    assert headings.count("Diagnostic accuracy of a yes/no test (numbers typed)") == 1
    assert headings.count("Calculation from numbers typed") == 3


def test_a_table_typed_that_copies_a_table_the_dossier_holds_back_is_not_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plain copy: the four counts of a cut-off found on the data that calls one participant positive, typed
    into the calculator, wrote the positive predictive value "1 of 1" that the dossier holds back with the
    cut-off. In any order: the rows or the columns swapped are the same four counts."""
    import app_pages.study_dossier as page
    from biostats.diagnostic import FOUND_CUTOFF, diagnostic_accuracy
    from manage.report_tables import TABLE_OF_A_RESULT_HELD_BACK, table_held_back

    result = {
        "acc": diagnostic_accuracy(**FOUND, sampling="series"),
        "auc": {"auc": 0.9, "n_positive": 30, "n_negative": 370},
        "mode": FOUND_CUTOFF,
        "direction": "higher",
        "cutoff": 3987.0,
        "outcome": "ill",
        "score": "marker",
        "pos_label": "Yes",
    }
    assert table_held_back(result) == (0, 1, 29, 370)
    # a table the dossier writes is nothing to compare with, nor is a result without a table
    typed_beforehand = {**result, "mode": "Manual"}
    assert table_held_back(typed_beforehand) is None
    assert table_held_back({**result, "acc": None}) is None and table_held_back(object()) is None
    # what the session holds is read once, as counts: the result on the page and those of the dossier's list
    monkeypatch.setattr(page, "stored_value", lambda key: result if key == "diagnostic" else None)
    listed = SimpleNamespace(key="diagnostic", value=typed_beforehand)
    other = SimpleNamespace(key="cox", value=object())
    assert page.held_tables([listed, other]) == [(0, 1, 29, 370)]
    monkeypatch.setattr(page, "stored_value", lambda key: None)
    assert page.held_tables([listed, other]) == []
    assert page.held_tables([SimpleNamespace(key="diagnostic", value=result)]) == [(0, 1, 29, 370)]

    copy = _stored("diagnostic", FOUND, sampling="series")
    swapped = _stored("diagnostic", {"tp": 29, "fp": 370, "fn": 1, "tn": 0})
    another = _stored("diagnostic", {**FOUND, "tn": 371})
    parts = _parts([copy, swapped, another])
    _, free = _document(parts)  # in a session that holds no such result, the three are written
    assert TABLE_OF_A_RESULT_HELD_BACK not in free and free.count("Numbers typed: ") == 3
    _, text = _document({**parts, "held_tables": [(0, 1, 29, 370)]})
    calculations = text[text.index("6. Calculations") : text.index("7. References")]
    assert calculations.count(TABLE_OF_A_RESULT_HELD_BACK) == 2
    assert calculations.count("Numbers typed: ") == 1 and "371 without" in calculations
    assert "29 with the condition" not in calculations.replace(
        "test negative, 29 with the condition and 371 without", ""
    )


def test_a_table_held_back_is_held_back_however_its_counts_are_typed_again() -> None:
    """Typed with the condition in its columns (3 and 30, then 4 and 200), a table has a side of 7 and is held back.
    Typed again with its rows and columns exchanged (3 and 4, then 30 and 200), an ordinary correction of a table
    read the other way round, it has sides of 33 and 204 and was written whole. Whatever the order they were added
    in, the second is held back as the copy of the first."""
    from manage.report_tables import TABLE_OF_A_RESULT_HELD_BACK

    held = _stored("diagnostic", {"tp": 3, "fp": 30, "fn": 4, "tn": 200})
    again = _stored("diagnostic", {"tp": 3, "fp": 4, "fn": 30, "tn": 200})
    other = _stored("diagnostic", {"tp": 30, "fp": 40, "fn": 30, "tn": 200})
    for order in ([held, again, other], [again, other, held]):
        _, text = _document(_parts(order))
        assert text.count(TABLE_OF_A_RESULT_HELD_BACK) == 1
        assert text.count("Fewer than 10 participants have the condition in this table") == 1
        assert text.count("Numbers typed: ") == 1 and "30 with the condition and 40 without" in text
        for written in ("3 with the condition", "Test positive\n3\n4", "Test positive\n3\n30"):
            assert written not in text, written
    # alone, the table typed the other way is a table like another: its own sides decide
    _, alone = _document(_parts([again]))
    assert "Numbers typed: test positive, 3 with the condition and 4 without" in alone


def test_the_table_of_a_result_is_kept_once_the_next_run_has_replaced_it() -> None:
    """The page keeps one accuracy result at a time. A score whose cut-off found on the data calls one participant
    positive was replaced by the next run, and its four counts typed afterwards were written, with the predictive
    value "1 of 1" the dossier holds back. Its counts are kept for the dataset, sorted and without a name."""
    import streamlit as st

    import app_pages.study_dossier as page
    from biostats.diagnostic import FOUND_CUTOFF, diagnostic_accuracy
    from ui import results

    found = {
        "acc": diagnostic_accuracy(**FOUND, sampling="series"),
        "auc": {"auc": 0.9, "n_positive": 30, "n_negative": 370},
        "mode": FOUND_CUTOFF,
    }
    written = {"acc": diagnostic_accuracy(90, 30, 10, 270), "auc": None, "index_form": "yes_no"}
    st.session_state.pop(results.HELD_TABLES_KEY, None)
    try:
        results._keep_held_table(found)
        results._keep_held_table(written)  # a table the dossier writes is nothing to keep
        results._keep_held_table(found)  # once
        results._keep_held_table(object())  # nor is a result of another shape
        assert results.held_tables_kept() == [(0, 1, 29, 370)]
        assert st.session_state[results.HELD_TABLES_KEY] == [
            (0, 1, 29, 370)
        ]  # counts, and nothing else
        # the page now holds another result, and the dossier's list is empty: the copy is still known
        assert page.held_tables([]) == [(0, 1, 29, 370)]
    finally:
        st.session_state.pop(results.HELD_TABLES_KEY, None)


def test_an_interval_ends_at_none_or_all_exactly_and_a_result_is_titled_by_its_form() -> None:
    """For 12 of 12 the Wilson bound was 0.9999999999999999, which the Word block read as a share strictly below 1
    and wrote "1.00000000 (0.75750599 to 1.00000000)"."""
    from biostats.diagnostic import diagnostic_accuracy
    from manage.report_tables import result_title

    all_right = diagnostic_accuracy(12, 0, 0, 12)
    assert all_right.sensitivity_ci.upper == 1.0 and all_right.specificity_ci.upper == 1.0
    none_right = diagnostic_accuracy(0, 5, 12, 12)
    assert none_right.sensitivity_ci.lower == 0.0 and 0 < none_right.sensitivity_ci.upper < 1
    _, text = _document(_parts([_stored("diagnostic", {"tp": 12, "fp": 0, "fn": 0, "tn": 12})]))
    assert "Sensitivity\n100.0%\n75.8% to 100.0%" in text
    assert "1.00000000" not in text and "e-17" not in text
    # the form of the test in the title: two results of the same columns read alike under one title
    title = "Diagnostic accuracy"
    assert (
        result_title("diagnostic", {"index_form": "yes_no"}, title) == f"{title} of a yes/no test"
    )
    assert result_title("diagnostic", {"auc": {}}, title) == f"{title} of a score"
    assert result_title("cox", {"index_form": "yes_no"}, "Cox model") == "Cox model"
    assert plan.KINDS["diagnostic"].typed_title == f"{title} of a yes/no test (numbers typed)"


def test_small_counts_typed_are_written_as_typed() -> None:
    """The dossier's rule for small groups reads the results of the dataset, not numbers typed (a deliberate act of
    the person who types them): a share, rates and a standardised ratio are written exactly."""
    typed = [
        _stored("proportion", {"count": 3, "total": 7}, reference=0.5),
        _stored(
            "rates",
            {"events": 2, "person_time": 3.5, "reference_events": 0, "reference_person_time": 9.25},
        ),
        _stored("standardised_ratio", {"observed": 1, "expected": 0.37}),
    ]
    _, text = _document(_parts(typed))
    assert (
        "Numbers typed: 3 of 7 participants with the characteristic, against a reference share of 50%."
        in text
    )
    assert "Share: 42.9% (3 of 7;" in text
    assert (
        "2 events over 3.5 person-years (group compared) against 0 over 9.25 (reference group)"
        in text
    )
    assert "Rate ratio\nno upper bound\nat least " in text
    # one death is one death, and fewer than 1 expected is said in the document as on the page
    assert "Numbers typed: 1 observed death against 0.37 expected." in text
    assert "Note: Fewer than 1 event expected: the ratio rests on almost nothing." in text
    assert "not written" not in text


def test_a_share_next_to_none_or_all_is_written_with_the_decimals_that_show_it() -> None:
    """24 of 50,000 against 0.02% read "Share: 0.0% (...; 95% CI 0.0% to 0.1%); against a reference of 0.0%: p <
    0.001", and 9,999 of 10,000 "100.0%"."""
    rare = _stored("proportion", {"count": 24, "total": 50000}, reference=0.0002)
    nearly = _stored("proportion", {"count": 9999, "total": 10000}, reference=0.9995)
    _, text = _document(_parts([rare, nearly]))
    assert (
        "Share: 0.048% (24 of 50,000; 95% CI 0.032% to 0.071%, Wilson); against a reference of 0.020%: p < 0.001 "
        "(exact binomial test, two-sided); 0.028 points above the reference." in text
    )
    assert (
        "Share: 99.990% (9,999 of 10,000; 95% CI 99.943% to 99.998%, Wilson); against a reference of 99.950%"
        in text
    )
    assert (
        "against a reference share of 0.02%" in text
        and "against a reference share of 99.95%" in text
    )
    # a reference next to none or all beside a share far from it: "against a reference of 0.0%" stood for 0.04%,
    # a value the page refuses to take
    low = _stored("proportion", {"count": 3, "total": 10}, reference=0.0004)
    high = _stored("proportion", {"count": 9, "total": 10}, reference=0.9996)
    _, text = _document(_parts([low, high]))
    assert "Share: 30.0% (3 of 10;" in text and "against a reference of 0.040%: p < 0.001" in text
    assert "Share: 90.0% (9 of 10;" in text and "against a reference of 99.960%: p" in text


def test_a_share_counted_in_the_dataset_reads_as_a_share_of_numbers_typed() -> None:
    """One Word dossier wrote the same share two ways: "observed proportion (reference 0.2) = 0.3 (95% CI 0.219 to
    0.396); number counted = 30 ... n = sample: 100" among the results, "Share: 30.0% (30 of 100; ...)" among the
    calculations."""
    from app_pages.study_dossier import _result_blocks
    from biostats.paired import one_sample_proportion
    from ui.results import CURRENT

    item = SimpleNamespace(
        key="proportion",
        value=one_sample_proportion(30, 100, 0.2),
        title="Share against a reference",
        settings="diabetes, Yes, 0.2",
        stamp=SimpleNamespace(caption="computed on made-up rows", single_imputed=()),
    )
    typed = _stored("proportion", {"count": 30, "total": 100}, reference=0.2)
    line = (
        "Share: 30.0% (30 of 100; 95% CI 21.9% to 39.6%, Wilson); against a reference of 20.0%: p = 0.017 (exact "
        "binomial test, two-sided); 10.0 points above the reference."
    )
    assert _result_blocks(item).lines == [line]
    _, text = _document(_parts([typed], kept=[(item, CURRENT)]))
    assert text.count(line) == 2 and "observed proportion" not in text and "n = sample" not in text
    # a kind without a record block keeps the block of its type
    other = SimpleNamespace(key="paired", value=item.value)
    assert _result_blocks(other).lines != [line]


def test_a_source_is_written_as_typed_and_a_calculation_unread_keeps_its_place() -> None:
    """The source is the user's text: markdown or HTML in it stays text, and a full stop closes it only when it has
    none. A calculation this version cannot read or compute again keeps its place with a fixed line and its
    numbers, and the others are written."""
    from manage.dossier import CALCULATION_NOT_COMPUTED

    marked = "**Smith** <b>2021</b> [table 2](http://example.org)"
    good = _stored("rates", TWO, note=marked)
    closed = _stored(
        "standardised_ratio",
        {"observed": 25, "expected": 18.4},
        note="Smith J. Lancet 2021;398:1-10.",
    )
    unread = _stored("rates", {**TWO, "events": 16})
    unread["spec"] = {**unread["spec"], "kind_version": 2}
    fails = _stored("rates", {**TWO, "events": 17})
    fails["spec"] = {**fails["spec"], "typed": {**fails["spec"]["typed"], "person_time": 0.0}}
    huge = _stored("proportion", {"count": 3, "total": 10}, reference=0.2)
    huge["spec"] = {**huge["spec"], "typed": {"count": 10**400, "total": 10}}
    headings, text = _document(_parts([good, closed, unread, fails, huge]))
    assert f"Source: {marked}. Typed by hand, added on 2026-11-16 at 16:40." in text
    assert "Source: Smith J. Lancet 2021;398:1-10. Typed by hand" in text and "1-10.." not in text
    assert text.count(CALCULATION_NOT_COMPUTED) == 3 and "Rate ratio\n3.125" in text
    # one that reads keeps its own title and numbers; one that does not, the words kept when it was added
    assert headings.count("Calculation from numbers typed") == 2
    assert "Numbers typed: 16 events over 1,200 person-years (group compared)" in text
    assert "Numbers typed: 17 events over 0 person-years (group compared)" in text
    assert "Numbers typed: 3 of 10 participants with the characteristic" in text


def test_the_source_keeps_only_what_is_seen() -> None:
    """A source pasted from a PDF can hold a soft hyphen or a zero-width space inside a name: the page shows the
    name whole, and a scan of rare values would read two words. A right-to-left override shows another text than
    the one kept."""
    from ui.results import NOTE_MAX, clean_note, typed_id

    for hidden in ("­", "​", "⁠", "‮", "\x07"):
        assert clean_note(f"Register of Wil{hidden}tz, 2021") == "Register of Wiltz, 2021"
    assert clean_note("  Smith\n2021 \t table 2 ") == "Smith 2021 table 2"
    assert clean_note("Smith 2021, table 2") == "Smith 2021, table 2"
    assert len(clean_note("x" * 500)) == NOTE_MAX and clean_note(None) == ""
    # an item is named by its kind and its numbers; 1200 and 1200.0 are one
    spec = plan.specify("rates", typed=TWO)
    assert typed_id(spec) == typed_id(plan.specify("rates", typed={**TWO, "person_time": 1200}))
    assert typed_id(spec) != typed_id(plan.specify("rates", typed={**TWO, "events": 16}))
    assert (
        typed_id(spec).startswith("typed:rates:")
        and len(typed_id(spec)) == len("typed:rates:") + 16
    )


def test_a_calculation_is_never_stored_as_a_result_of_the_dataset() -> None:
    """Stamped, it would read "Computed on <dataset>, N analysed" for numbers that are not the dataset's, and go
    stale with data it never read. Nor is it disclosed as one: its request holds the numbers typed."""
    from core.disclosure import disclose_record, disclose_spec
    from ui.results import remember

    spec = plan.specify("rates", typed=TWO)
    with pytest.raises(ValueError, match="not a result of the dataset"):
        remember("rates", object(), config=(), spec=spec)
    done = plan.run(spec)
    record = plan.record_of(done.spec, done.result)
    with pytest.raises(ValueError, match="not disclosed as a result of the dataset"):
        disclose_record(record, None, spec=spec, analysis=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not disclosed as a result of the dataset"):
        disclose_spec(spec, None, analysis=None)  # type: ignore[arg-type]


def test_a_rate_is_not_written_under_a_unit_nobody_named() -> None:
    """A rate computed without the unit of its person-time took "person-years", and a standardised ratio "deaths":
    a caller that had not said had its result written under a word of the function's own (15 events over 1,200
    person-days read "12.50 per 1,000 person-years"). No unit is assumed: the record holds none, and no block
    is written."""
    from biostats.rates import rates, standardised_ratio
    from manage.report_tables import record_blocks

    bare = rates(15, 1200)
    record = plan.record("rates", bare)
    assert bare.unit is None and record is not None
    assert "unit" not in {number.name for number in record.parts[0].numbers}
    with pytest.raises(ValueError, match="without the unit of its person-time"):
        record_blocks(record)
    unnamed = standardised_ratio(25, 18.4)
    record = plan.record("standardised_ratio", unnamed)
    assert unnamed.events_are is None and record is not None
    assert "events_are" not in {number.name for number in record.part("ratio").numbers}
    with pytest.raises(ValueError, match="without what its events are"):
        record_blocks(record)
    # named, they are written as before
    days = plan.record("rates", rates(15, 1200, unit="person-days", per=100000))
    assert days is not None
    assert "Rate, per 100,000 person-days" in str(record_blocks(days).tables[0][1])


def test_every_kind_computed_from_numbers_typed_has_its_words_and_its_block() -> None:
    """A kind given a typed runner without its words would stop the calculator's page and the dossier's: the
    catalogue, the words and the blocks are held together here."""
    from biostats.rates import has_ratio
    from manage.report_tables import (
        CALCULATION_KINDS,
        RECORD_BLOCKS,
        TABLE_KIND,
        calculation_blocks,
        record_blocks,
        typed_numbers,
        typed_title,
    )

    samples = {
        "proportion": ({"count": 30, "total": 100}, {"reference": 0.2}),
        "rates": (TWO, {}),
        "standardised_ratio": ({"observed": 25, "expected": 18.4}, {}),
        "diagnostic": (TABLE, {}),
    }
    typed = [kind.id for kind in plan.KINDS.values() if kind.typed_run is not None]
    assert sorted(typed) == sorted(samples) and set(RECORD_BLOCKS) <= set(plan.KINDS)
    # the kinds the dossier writes a calculation of are those computed from numbers typed, no more and no fewer
    assert set(typed) == CALCULATION_KINDS == {*RECORD_BLOCKS, TABLE_KIND}
    for kind in typed:
        numbers, options = samples[kind]
        spec = plan.specify(kind, typed=numbers, options=options)
        assert typed_numbers(spec) and typed_title(spec).endswith("(numbers typed)")
        done = plan.run(spec)
        record = plan.record_of(done.spec, done.result)
        blocks = calculation_blocks(done.spec, done.result, record)
        assert blocks.lines and blocks.references and blocks.warnings and not blocks.held_numbers
        if kind in RECORD_BLOCKS:  # written from its record alone
            alone = record_blocks(record)
            assert (blocks.lines, blocks.warnings, blocks.references) == (
                alone.lines,
                alone.warnings,
                alone.references,
            )
    # the page offers what the kinds know, and a rate's title follows the rule the rates are computed by
    choices = plan.KINDS["rates"].choices
    assert tuple(m.PERSON_TIME) == choices["unit"]
    assert (m.DEATHS, m.CASES) == plan.KINDS["standardised_ratio"].choices["events_are"]
    for events in range(3):
        for reference in range(3):
            spec = plan.specify(
                "rates", typed={**TWO, "events": events, "reference_events": reference}
            )
            assert (typed_title(spec) == "Rate ratio (numbers typed)") == has_ratio(
                events, reference
            )
            assert (plan.run(spec).result.ratio is not None) == has_ratio(events, reference)


def test_the_basket_is_declared_with_the_dossier_s_lifetime_and_no_new_word_is_refused() -> None:
    from manage import dossier, report_tables
    from ui import keys
    from ui.results import TYPED_KEY

    declared = {key.name: key for key in keys.KEYS}
    assert declared[TYPED_KEY].lifetime == declared["_report"].lifetime == "dataset"
    words = " ".join(
        [
            dossier.CALCULATIONS_INTRO,
            dossier.CALCULATION_NOT_COMPUTED,
            dossier.CALCULATION_NOT_WRITTEN,
            dossier.NO_CALCULATION,
            report_tables.TYPED_SOURCE,
            report_tables.SHARE_NOTE,
            report_tables.RATES_NOTE,
            report_tables.STANDARDISED_NOTE,
            m.HELP["numbers_source"],
        ]
    ).lower()
    for refused in (
        "validated",
        "verified",
        "confirmed",
        "compliant",
        "guarantee",
        "signed",
        "certificate",
    ):
        assert refused not in words
    assert "patients" not in words and "subjects" not in words
