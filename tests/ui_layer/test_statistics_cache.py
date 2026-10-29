"""
The cached statistics and correlation workbooks of Statistics & correlations are keyed by every row of the table and
by the signed-in user's workspace.

Streamlit hashes a sample of 10,000 rows of a table of 50,000 or more, so two versions of a cohort that differ outside
the sample got each other's results. The workbooks are cached for every session of the server, and the statistics for
every user signed in to one browser session in turn: neither may serve one user's results to another.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


def _two_versions(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two versions of one table that differ in one value, outside the rows Streamlit samples to hash it."""
    from streamlit.runtime.caching.hashing import (
        _PANDAS_ROWS_LARGE,
        _PANDAS_SAMPLE_SIZE,
        _sample_seed,
    )

    rng = np.random.default_rng(seed)
    first = pd.DataFrame(
        {
            "age": rng.integers(18, 90, _PANDAS_ROWS_LARGE).astype(float),
            "sbp": rng.normal(130.0, 15.0, _PANDAS_ROWS_LARGE),
        }
    )
    sampled = set(first.sample(n=_PANDAS_SAMPLE_SIZE, random_state=_sample_seed()).index)
    outside = next(i for i in first.index if i not in sampled)
    second = first.copy()
    second.loc[outside, "sbp"] = 10_000.0
    # what Streamlit hashes of each (their shape and their sample) is the same: its own key would serve one for the other
    pd.testing.assert_frame_equal(
        first.sample(n=_PANDAS_SAMPLE_SIZE, random_state=_sample_seed()),
        second.sample(n=_PANDAS_SAMPLE_SIZE, random_state=_sample_seed()),
    )
    return first, second


def _results() -> None:
    import streamlit as st

    from app_pages import home
    from ingest.analyzer import DataAnalyzer

    if "as_user" not in st.session_state:  # the session was ended: nothing of it is left
        return
    st.session_state["username"] = st.session_state["as_user"]
    table = st.session_state["table"]
    columns = DataAnalyzer(table)
    numerical, _, _ = home.get_statistics_dataframe(
        table, columns, dataset_name="cohort", table_key=home._table_key(table), owner=home._owner()
    )
    st.session_state["statistics"] = numerical.to_csv()
    workbook = home._cached_corr_excel_bytes(
        table,
        ["sbp"],
        ["age"],
        None,
        "pearson",
        table_key=home._table_key(table),
        owner=home._owner(),
    )
    st.session_state["workbook"] = workbook


@pytest.fixture
def counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """How often each result is computed rather than served from its cache; a workbook holds the largest blood
    pressure of the table it was built from."""
    import explore.corr_matrix as ecm
    from app_pages import home

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    from streamlit.runtime.caching import cache_data_api

    home._cached_corr_excel_bytes.clear()
    cache_data_api.clear_session_cache("test session id")  # the session of every AppTest
    computed = {"statistics": 0, "workbook": 0}
    fingerprint = home._frame_fingerprint

    def statistics(df: pd.DataFrame) -> str:
        computed["statistics"] += 1
        return fingerprint(df)

    def workbook(*, data: pd.DataFrame, file_name, **_: object) -> None:
        computed["workbook"] += 1
        file_name.write(f"{data['sbp'].max():.1f}".encode())

    monkeypatch.setattr(home, "_frame_fingerprint", statistics)
    monkeypatch.setattr(ecm, "custom_corr_mat_to_excel", workbook)
    return computed


def test_two_versions_that_differ_outside_streamlits_sample_get_their_own_results(
    counted: dict[str, int],
) -> None:
    first, second = _two_versions(7)
    at = AppTest.from_function(_results, default_timeout=120)
    at.session_state["as_user"] = "alice"
    at.session_state["table"] = first
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    before = at.session_state["statistics"], at.session_state["workbook"]
    at.session_state["table"] = second
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert counted == {"statistics": 2, "workbook": 2}
    assert at.session_state["statistics"] != before[0]
    assert at.session_state["workbook"] == b"10000.0" != before[1]


def test_one_users_cached_results_never_reach_another(counted: dict[str, int]) -> None:
    """The same table for alice, then for bob: both results are computed again for bob. AppTest's sessions share the
    session-scoped cache, as one browser session does for the users signed in to it in turn."""
    first, _ = _two_versions(11)  # its own data: no other test's entry can be found
    for user, computed in (("alice", 1), ("bob", 2)):
        at = AppTest.from_function(_results, default_timeout=120)
        at.session_state["as_user"] = user
        at.session_state["table"] = first
        at.run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]
        assert counted == {"statistics": computed, "workbook": computed}, user
    at.run()  # bob again: his own results, served from the caches
    assert counted == {"statistics": 2, "workbook": 2}


class _Store:
    """A workspace's saved statistics, by name."""

    def __init__(self) -> None:
        self.saved: dict[tuple[str, str], pd.DataFrame] = {}

    def load_stats(self, name: str, kind: str):
        return self.saved.get((name, kind)), ""

    def save_stats(self, df: pd.DataFrame, name: str, kind: str) -> None:
        self.saved[(name, kind)] = df


def test_the_saved_statistics_follow_the_number_of_top_values() -> None:
    """A saved dataset's statistics were saved under its name and content only: the number of top values asked
    after the first had no effect."""
    from app_pages import home
    from ingest.analyzer import DataAnalyzer

    table = pd.DataFrame({"site": list("ABCDEFGH") * 5, "age": np.arange(40.0)})
    store = _Store()
    compute = home.get_statistics_dataframe.__wrapped__
    for top in (3, 5, 3):
        compute(
            table,
            DataAnalyzer(table),
            nb_top_categories=top,
            dataset_name="cohort",
            _db_manager=store,
            table_key="",
            owner=None,
        )
    names = {name for name, kind in store.saved if kind == "categorical"}
    assert len(names) == 2  # one per number of top values; the third call read the first's


def test_the_key_of_a_table_sees_what_the_statistics_see() -> None:
    from app_pages.home import _table_key

    base = pd.DataFrame({"x": [1.0, 2.0]})
    assert _table_key(base) == _table_key(base.copy())
    assert _table_key(base) != _table_key(base.assign(x=[1.0, 3.0]))
    # an unused category, and the order of the categories, change the counts and the top values shown
    plain = pd.DataFrame({"c": pd.Categorical(["a", "b"])})
    unused = pd.DataFrame({"c": pd.Categorical(["a", "b"], categories=["a", "b", "z"])})
    ordered = pd.DataFrame({"c": pd.Categorical(["a", "b"], ordered=True)})
    assert len({_table_key(plain), _table_key(unused), _table_key(ordered)}) == 3
    # 1 and "1" are two values for the statistics, and pandas hashes them alike
    assert _table_key(pd.DataFrame({"v": [1, 2]}, dtype=object)) != _table_key(
        pd.DataFrame({"v": ["1", 2]}, dtype=object)
    )
    # a column of lists is hashed as text, on its own
    lists = pd.DataFrame({"v": [[1, 2], [3]], "w": [1.0, 2.0]})
    assert _table_key(lists) != _table_key(lists.assign(v=[[1, 2], [4]]))
