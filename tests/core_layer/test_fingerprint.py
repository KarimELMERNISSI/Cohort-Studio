"""Content fingerprints identify the data a result was computed on (audit E8)."""

import numpy as np
import pandas as pd

from core.fingerprint import frame_fingerprint


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"age": [61, 72, 55], "arm": ["A", "B", "A"], "bmi": [24.1, np.nan, 31.0]})


def test_equal_frames_have_equal_fingerprints() -> None:
    assert frame_fingerprint(_frame()) == frame_fingerprint(_frame())
    assert len(frame_fingerprint(_frame())) == 16
    assert len(frame_fingerprint(_frame(), length=12)) == 12


def test_a_changed_value_changes_the_fingerprint() -> None:
    imputed = _frame()
    imputed.loc[1, "bmi"] = 27.5  # same shape and row count, different content
    assert frame_fingerprint(imputed) != frame_fingerprint(_frame())


def test_names_order_dtypes_and_row_labels_matter() -> None:
    base = frame_fingerprint(_frame())
    assert frame_fingerprint(_frame().rename(columns={"age": "age_years"})) != base
    assert frame_fingerprint(_frame()[["arm", "age", "bmi"]]) != base
    assert frame_fingerprint(_frame().astype({"age": "float64"})) != base
    relabelled = _frame().set_axis([10, 11, 12])
    assert frame_fingerprint(relabelled) != base
    assert frame_fingerprint(relabelled, index=False) == frame_fingerprint(_frame(), index=False)


def test_unhashable_cells_are_fingerprinted_through_their_text() -> None:
    flagged = _frame().assign(fired=[["age"], [], ["age", "bmi"]])
    other = _frame().assign(fired=[["age"], [], ["bmi"]])
    assert frame_fingerprint(flagged) == frame_fingerprint(flagged.copy())
    assert frame_fingerprint(flagged) != frame_fingerprint(other)


def test_an_empty_frame_has_a_fingerprint() -> None:
    assert frame_fingerprint(pd.DataFrame()) != frame_fingerprint(pd.DataFrame({"a": []}))
