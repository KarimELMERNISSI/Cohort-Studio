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
    assert "(exact Poisson, twice the smaller tail)" in text  # the test named by its rule
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
    one, no step is recorded, and the participant flow is the same with and without them."""

    def trace() -> Any:
        if "transformation_manager" not in app.session_state:
            return None
        tm = app.session_state["transformation_manager"]
        return getattr(tm, "head_hash", None), len((tm.get_trace() or {}).get("steps", []))

    kept = json.loads(json.dumps(_basket(app)))
    before = trace()
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
    # no step of the trace for a calculation, and no box of the flow: the same with and without the list
    assert trace() == before
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
        f"writes a line in its place. Its numbers, as added: {RATIO_WORDS}. Type them again on the calculator's page."
    ) in text
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


# ----------------------------------------------------------------------------- the words, without a session


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


def test_every_kind_computed_from_numbers_typed_has_its_words_and_its_block() -> None:
    """A kind given a typed runner without its words would stop the calculator's page and the dossier's: the
    catalogue, the words and the blocks are held together here."""
    from biostats.rates import has_ratio
    from manage.report_tables import RECORD_BLOCKS, record_blocks, typed_numbers, typed_title

    samples = {
        "proportion": ({"count": 30, "total": 100}, {"reference": 0.2}),
        "rates": (TWO, {}),
        "standardised_ratio": ({"observed": 25, "expected": 18.4}, {}),
    }
    typed = [kind.id for kind in plan.KINDS.values() if kind.typed_run is not None]
    assert sorted(typed) == sorted(samples) and set(RECORD_BLOCKS) <= set(plan.KINDS)
    for kind in typed:
        numbers, options = samples[kind]
        spec = plan.specify(kind, typed=numbers, options=options)
        assert typed_numbers(spec) and typed_title(spec).endswith("(numbers typed)")
        done = plan.run(spec)
        blocks = record_blocks(plan.record_of(done.spec, done.result))
        assert blocks.lines and blocks.references and blocks.warnings
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
