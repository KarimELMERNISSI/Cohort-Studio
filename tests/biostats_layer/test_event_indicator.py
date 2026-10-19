"""The event indicator of the survival analyses: 0 and 1 only, never a guess (M5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biostats.survival import compare_survival, event_indicator


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (pd.Series([0, 1, 1, 0]), [0, 1, 1, 0]),
        (pd.Series([0.0, 1.0, np.nan]), [0, 1, np.nan]),
        (pd.Series([0, 1, None], dtype="Int64"), [0, 1, np.nan]),
        (pd.Series([True, None, False], dtype="boolean"), [1, np.nan, 0]),
        (pd.Series(["Yes", "No", None]), [1, 0, np.nan]),
        (pd.Series(["Dead", "Alive"], dtype="category"), [1, 0]),
        (pd.Series(["death", "survived"], dtype="string"), [1, 0]),
        (pd.Series(["1", "0"]), [1, 0]),
    ],
)
def test_the_usual_codings_are_read(values: pd.Series, expected: list[float]) -> None:
    np.testing.assert_array_equal(event_indicator(values), np.asarray(expected, dtype=float))


def test_a_1_2_coding_is_refused_with_how_to_recode_it() -> None:
    # R's survival package codes 1 = censored and 2 = event: reading 2 as an event code counted it twice
    with pytest.raises(ValueError, match=r"also holds 2 \(2 rows\).*recode 1 to 0 and 2 to 1"):
        event_indicator(pd.Series([1, 2, 2, 1]))
    with pytest.raises(ValueError, match="recode 1 to 0 and 2 to 1"):
        event_indicator(pd.Series(["1", "2"]))


def test_a_competing_event_or_an_unknown_word_is_refused() -> None:
    with pytest.raises(ValueError, match=r"also holds 2 \(1 row\).*competing event"):
        event_indicator(pd.Series([0, 1, 2]))
    # an unknown word was read as censored without a word
    with pytest.raises(ValueError, match="'relapse' \\(1 row\\)"):
        event_indicator(pd.Series(["yes", "no", "relapse"]))


def test_kaplan_meier_refuses_a_code_of_2() -> None:
    df = pd.DataFrame({"time": [5, 8, 12, 20], "status": [1, 2, 2, 1]})
    with pytest.raises(ValueError, match="event indicator must be 1"):
        compare_survival(df, "time", "status")
