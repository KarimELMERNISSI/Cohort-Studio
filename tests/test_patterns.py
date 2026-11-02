"""
Values picked by a regular expression (enrich.patterns, Data > Clean > Categories): found anywhere in a value as
written, case ignored unless asked; a pattern that cannot be used says why; one that backtracks without end stops at
its time limit, in its own process, never holding the server.
"""

from __future__ import annotations

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
]


def test_a_pattern_picks_the_values_it_finds_in_their_order() -> None:
    assert matching("^hyper", VALUES) == [
        "Hypertension",
        "hypertension",
        "HYPERTENSION",
        "Hypertenson",
    ]
    assert matching("ten.?s?ion$", VALUES, ignore_case=False) == [
        "Hypertension",
        "hypertension",
        "Hypotension",
    ]


def test_any_text_travels_whole() -> None:
    assert matching("p[eé]tange", VALUES) == ["Pétange"]
    assert matching("'s$", VALUES) == ["it's"]
    assert matching("x", []) == []


@pytest.mark.parametrize(
    ("pattern", "why"),
    [
        ("", "Type a pattern"),
        ("   ", "Type a pattern"),
        ("(", "Not a regular expression"),
        ("x" * (MAX_LENGTH + 1), "longer than"),
    ],
)
def test_a_pattern_that_cannot_be_used_says_why(pattern: str, why: str) -> None:
    with pytest.raises(PatternError, match=why):
        matching(pattern, VALUES)


def test_a_pattern_that_backtracks_without_end_stops_at_its_time_limit() -> None:
    started = time.perf_counter()
    with pytest.raises(PatternError, match="took more than"):
        matching("(a+)+$", ["a" * 40 + "b"], timeout=0.5)
    assert time.perf_counter() - started < 5
