"""Merges keep the row ids of the main dataset and count what they did (enrich.external_data.add_data)."""

import pandas as pd
import pytest

from core.fingerprint import ROW_ID, ensure_row_ids
from enrich.external_data import add_data

MAIN = ensure_row_ids(pd.DataFrame({"pid": [1, 2, 3], "age": [50, 60, 70]})).iloc[[0, 2]].copy()
LABS = pd.DataFrame({"pid": [3, 3, 4], "hb": [11.0, 12.0, 13.0], ROW_ID: [7, 8, 9]})  # ids of another table


@pytest.mark.parametrize(
    ("strategy", "ids", "counts"),
    [
        ("left", [0, 2, 3], {"rows_from_main": 3, "rows_added": 0, "main_rows_dropped": 0, "main_rows_repeated": 1}),
        ("inner", [2, 3], {"rows_from_main": 2, "rows_added": 0, "main_rows_dropped": 1, "main_rows_repeated": 1}),
        ("outer", [0, 2, 3, 4], {"rows_from_main": 3, "rows_added": 1, "main_rows_dropped": 0, "main_rows_repeated": 1}),
    ],
)
def test_main_rows_keep_their_ids(strategy: str, ids: list[int], counts: dict) -> None:
    merged = add_data(MAIN, LABS, ["pid"], ["pid"], ["hb"], strategy=strategy, conflict_resolution="keep")
    assert merged.index.name == ROW_ID and merged.index.is_unique
    assert sorted(merged.index) == ids  # 0 and 2 kept; the repeat of 2 and the added row get 3, 4
    assert ROW_ID not in merged.columns
    assert {k: merged.attrs["join_counts"][k] for k in counts} == counts
    kept = merged.loc[[i for i in (0, 2) if i in merged.index], "age"].tolist()
    assert kept == [age for i, age in ((0, 50), (2, 70)) if i in merged.index]


def test_a_failed_merge_returns_the_main_dataset_unchanged() -> None:
    merged = add_data(MAIN, LABS, ["missing"], ["pid"], ["hb"], strategy="left")
    assert merged is MAIN
