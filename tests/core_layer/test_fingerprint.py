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


def test_row_ids_live_in_the_index() -> None:
    from core.fingerprint import ROW_ID, ensure_row_ids

    loaded = ensure_row_ids(pd.DataFrame({"a": [5, 6, 7]}))
    assert loaded.index.name == ROW_ID and list(loaded.index) == [0, 1, 2]
    filtered = ensure_row_ids(loaded.iloc[[0, 2]])
    assert list(filtered.index) == [0, 2]  # rows keep the id of the original row
    persisted = ensure_row_ids(filtered.reset_index())  # a saved file carries the column
    assert list(persisted.index) == [0, 2] and list(persisted.columns) == ["a"]
    assert frame_fingerprint(persisted) == frame_fingerprint(filtered)
    text_index = ensure_row_ids(pd.DataFrame({"a": [1, 2]}, index=["x", "y"]))
    assert list(text_index.index) == [0, 1]


def test_rows_added_by_a_join_get_fresh_ids() -> None:
    from core.fingerprint import ROW_ID, extend_row_ids

    left = pd.DataFrame({"id": [1, 2], ROW_ID: [10, 11]})
    merged = left.merge(pd.DataFrame({"id": [2, 2, 3], "x": [7, 8, 9]}), on="id", how="outer")
    out = extend_row_ids(merged, used=[10, 11, 12])
    assert out.index.is_unique and out.index.name == ROW_ID
    # 11 is kept once; the repeated row and the row added by the join get fresh ids
    assert sorted(out.index) == [10, 11, 13, 14]
    assert ROW_ID not in out.columns


def test_id_ranges_round_trip() -> None:
    from core.fingerprint import ids_fingerprint, ids_to_ranges, ranges_to_ids

    ids = [9, 1, 2, 3, 7, 8, 20]
    ranges = ids_to_ranges(ids)
    assert ranges == [[1, 3], [7, 9], [20, 20]]
    assert sorted(ranges_to_ids(ranges).tolist()) == sorted(ids)
    assert ids_to_ranges([]) == [] and len(ranges_to_ids([])) == 0
    assert ids_fingerprint([3, 1, 2]) == ids_fingerprint([1, 2, 3])


def test_the_sentinel_notices_structure_changes() -> None:
    from core.fingerprint import frame_sentinel

    frame = _frame()
    assert frame_sentinel(frame) == frame_sentinel(frame.copy())
    assert frame_sentinel(frame) != frame_sentinel(frame.assign(flag=True))
    assert frame_sentinel(frame) != frame_sentinel(frame.astype({"age": "float64"}))
