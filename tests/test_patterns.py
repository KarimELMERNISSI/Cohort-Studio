"""
Values picked by a search (enrich.patterns, Data > Clean > Categories): text found anywhere in a value as written (a
dot is a dot, * any text), or a regular expression on request, case ignored unless asked; a search that cannot be made
says why; one that backtracks without end stops at its time limit, counted once its process is ready, in that process,
never holding the server.
"""

from __future__ import annotations

import subprocess
import time

import pytest

from enrich.patterns import MAX_LENGTH, PatternError, matching

VALUES = [
    "Hypertension",
    "hypertension",
    "HYPERTENSION",
    "Hypertenson",
    "Hypotension",
    "Pétange",
    "it's",
    "n.a.",
    "Myocardial infarction",
]


def test_text_is_found_anywhere_as_typed_a_dot_being_a_dot() -> None:
    """n.a. typed as it reads finds n.a., never "infa" of Myocardial infarction (the dot of a regular expression)."""
    assert matching("n.a.", VALUES) == ["n.a."]
    assert matching("HYPERT", VALUES) == [
        "Hypertension",
        "hypertension",
        "HYPERTENSION",
        "Hypertenson",
    ]
    assert matching("HYPERT", VALUES, ignore_case=False) == ["HYPERTENSION"]
    assert matching("*tension", VALUES) == [
        "Hypertension",
        "hypertension",
        "HYPERTENSION",
        "Hypotension",
    ]
    assert matching("hyp*ension", VALUES) == [
        "Hypertension",
        "hypertension",
        "HYPERTENSION",
        "Hypotension",
    ]


def test_a_regular_expression_picks_the_values_it_finds_in_their_order() -> None:
    assert matching("^hyper", VALUES, regex=True) == [
        "Hypertension",
        "hypertension",
        "HYPERTENSION",
        "Hypertenson",
    ]
    assert matching("ten.?s?ion$", VALUES, regex=True, ignore_case=False) == [
        "Hypertension",
        "hypertension",
        "Hypotension",
    ]


def test_any_text_travels_whole() -> None:
    assert matching("p[eé]tange", VALUES, regex=True) == ["Pétange"]
    assert matching("pétange", VALUES) == ["Pétange"]
    assert matching("'s", VALUES) == ["it's"]
    assert matching("x", []) == []


@pytest.mark.parametrize(
    ("pattern", "regex", "why"),
    [
        ("", False, "Type what to look for"),
        ("   ", False, "Type what to look for"),
        (
            "(",
            True,
            r"Not a regular expression: missing \), unterminated subpattern \(at character 1\)\.",
        ),
        ("*kidney*", True, "to find values holding kidney, type kidney"),
        ("x" * (MAX_LENGTH + 1), False, "longer than"),
    ],
)
def test_a_search_that_cannot_be_made_says_why(pattern: str, regex: bool, why: str) -> None:
    with pytest.raises(PatternError, match=why):
        matching(pattern, VALUES, regex=regex)


def test_a_pattern_that_backtracks_without_end_stops_at_its_time_limit() -> None:
    started = time.perf_counter()
    with pytest.raises(
        PatternError, match="took more than 0.5 seconds on these values: write it without"
    ):
        matching("(a+)+$", ["a" * 40 + "b"], regex=True, timeout=0.5)
    assert time.perf_counter() - started < 10


def test_a_slow_start_is_the_server_s_not_the_pattern_s(monkeypatch: pytest.MonkeyPatch) -> None:
    """The time limit counts from the process's start; a start that does not come is said apart, and a process that
    cannot start at all is a refusal, never an error of the page."""
    with pytest.raises(PatternError, match="busy"):
        matching("hyper", VALUES, start=0.000001)

    def refused(*args, **kwargs):
        raise OSError("no interpreter")

    monkeypatch.setattr("enrich.patterns.subprocess.Popen", refused)
    with pytest.raises(PatternError, match="could not start"):
        matching("hyper", VALUES)


def test_a_process_that_ends_before_it_is_ready_is_the_server_s_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that ended before its ready byte (an interpreter that cannot start) was answered "could not be made on
    these values": the values were never read."""
    monkeypatch.setattr("enrich.patterns._MATCHER", "import sys; sys.exit(3)")
    with pytest.raises(PatternError, match="could not start on this server"):
        matching("hyper", VALUES)


def test_the_search_needs_no_imputation_engine() -> None:
    """The first search of a fresh server imported the multiple-imputation engine (biostats.mi) to find its
    interpreter: several hundred modules and seconds before the search started."""
    import sys

    code = (
        "import sys\n"
        "from enrich.patterns import matching\n"
        "assert matching('hyper', ['Hypertension', 'Diabetes']) == ['Hypertension']\n"
        "print('biostats.mi' in sys.modules)\n"
    )
    found = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert found.stdout.strip() == "False", found.stderr[-500:]


def test_the_search_process_gets_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server's settings (its keys among them) stay out of the process: only what it needs to start."""
    seen: dict = {}
    real = subprocess.Popen

    def watched(*args, **kwargs):
        seen.update(kwargs.get("env") or {})
        return real(*args, **kwargs)

    monkeypatch.setenv("CMP_TEST_SECRET", "never-there")
    monkeypatch.setattr("enrich.patterns.subprocess.Popen", watched)
    assert matching("hyper", VALUES)
    assert "CMP_TEST_SECRET" not in seen and set(seen) <= {"__PYVENV_LAUNCHER__"}
