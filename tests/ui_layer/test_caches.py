"""
A cache is cleared on its own (M5): Reset Cache on Home ran st.cache_data.clear(), which empties
every cached function of every user of the server. The statistics are now kept per session, and
the reset clears that session's only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_PACKAGES = ("app_pages", "ui", "utils", "explore", "graph", "manage", "monitor", "enrich")


class _Workspace:
    """Stands for the user's statistics files: counts the computations that reach them."""

    def __init__(self) -> None:
        self.computed = 0

    def load_stats(self, name: str, kind: str):
        if kind == "numerical":
            self.computed += 1
        return None, None

    def save_stats(self, *args, **kwargs) -> None:
        pass


def test_a_reset_clears_the_statistics_of_this_session_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.runtime.caching import cache_data_api

    import app_pages.home as home
    from utils.data_analyzer import DataAnalyzer

    session = {"id": "alice"}
    monkeypatch.setattr(cache_data_api, "get_session_id_or_throw", lambda: session["id"])
    df = pd.DataFrame({"age": [30.0, 41.0, 52.0, 63.0, 70.0], "sex": ["F", "M", "F", "M", "F"]})
    workspace = _Workspace()

    def statistics(user: str) -> None:
        session["id"] = user
        home.get_statistics_dataframe(df, DataAnalyzer(df), dataset_name="t", _db_manager=workspace)

    try:
        statistics("alice")
        statistics("bob")
        assert workspace.computed == 2  # each session computes its own
        statistics("bob")
        assert workspace.computed == 2  # then reads its cache

        session["id"] = "bob"
        home.get_statistics_dataframe.clear()  # what Reset Cache runs, in Bob's session
        statistics("alice")
        assert workspace.computed == 2  # Alice's statistics are still cached
        statistics("bob")
        assert workspace.computed == 3  # Bob's are computed again
    finally:
        for user in ("alice", "bob"):
            session["id"] = user
            home.get_statistics_dataframe.clear()


def test_the_correlation_workbooks_kept_are_bounded() -> None:
    import app_pages.home as home

    assert home._cached_corr_excel_bytes._info.max_entries == 8


def test_no_page_or_service_clears_every_cache_of_every_user() -> None:
    offenders = []
    for package in APP_PACKAGES:
        for file in (ROOT / package).rglob("*.py"):
            text = file.read_text(encoding="utf-8")
            if re.search(r"st\.cache_(data|resource)\.clear\(", text):
                offenders.append(str(file.relative_to(ROOT)))
    assert not offenders, (
        f"clear the cached function itself (its .clear()), not every cache: {offenders}"
    )
