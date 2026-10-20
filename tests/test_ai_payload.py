"""What the AI features send about a dataset: a summary of each column, never a row or an extreme value (M5)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from manage.ai_payload import (
    MIN_COUNT,
    TRACING_VARIABLES,
    column_profile,
    disable_remote_tracing,
    safe_stats,
    taxonomy_for_prompt,
)


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame(
        {
            "creatinine": rng.normal(90, 20, n).round(1),
            "nyha": rng.choice(["I", "II", "III"], n),
            "rare_note": ["common"] * (n - 2) + ["Mrs Dupont, room 12", "unique entry"],
            "visit": pd.date_range("2019-03-01", periods=n, freq="7D"),
            "visit_text": [d.strftime("%d/%m/%Y") for d in pd.date_range("2020-01-01", periods=n, freq="3D")],
            "smoker": rng.integers(0, 2, n),
        }
    )
    df.loc[17, "creatinine"] = 912.0  # one extreme value
    return df


def test_no_row_and_no_extreme_value_is_sent() -> None:
    df = _cohort()
    sent = json.dumps(column_profile(df, date_columns=["visit_text"]))
    assert "912" not in sent  # the maximum of creatinine singles out one participant
    assert "Mrs Dupont" not in sent and "unique entry" not in sent  # values of one row
    for value in df.iloc[3].astype(str):
        if value not in ("I", "II", "III", "0", "1", "common"):  # frequent values may be sent
            assert value not in sent, value


def test_each_kind_of_column_is_summarised() -> None:
    profile = {p["name"]: p for p in column_profile(_cohort(), date_columns=["visit_text"])}
    assert profile["creatinine"]["type"] == "Numeric"
    assert set(profile["creatinine"]["stats"]) == {"p5", "p25", "p50", "p75", "p95"}
    assert profile["nyha"]["type"] == "Categorical" and set(profile["nyha"]["top_values"]) == {"I", "II", "III"}
    assert all(n >= MIN_COUNT for n in profile["rare_note"]["top_values"].values())
    assert profile["visit"]["range"] == "2019 to 2022" and profile["visit_text"]["range"] == "2020 to 2021"
    assert profile["smoker"]["type"] == "Categorical"  # 0/1: described by its values


def test_statistics_outside_the_notice_are_dropped() -> None:
    # an imported variable map, or one built before the rule, may hold the extremes of a column
    assert safe_stats({"min": 31.5, "max": 912.0, "mean": 88.4}) == {}
    kept = safe_stats({"p5": 60.0, "p50": 88.0, "p95": 130.0, "max": 912.0})
    assert kept == {"p5": 60.0, "p50": 88.0, "p95": 130.0}
    # values shared by fewer than 5 rows are not sent
    assert safe_stats({"II": 80, "III": 40, "Mrs Dupont": 1}) == {"II": 80, "III": 40}
    assert safe_stats("min 31.5, max 912") == {} and safe_stats(None) == {}
    taxonomy = {"creatinine": {"unit": "umol/L", "stats": {"min": 31.5, "p50": 88.0}}, "note": "text"}
    assert taxonomy_for_prompt(taxonomy) == {
        "creatinine": {"unit": "umol/L", "stats": {"p50": 88.0}},
        "note": "text",
    }
    assert taxonomy["creatinine"]["stats"]["min"] == 31.5  # the user's map itself is unchanged


def test_remote_tracing_of_the_prompts_is_switched_off(monkeypatch) -> None:
    # a LangSmith switch left in .env would send every prompt to a tracing service
    import os

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert disable_remote_tracing() == ["LANGCHAIN_TRACING_V2"]
    assert all(os.environ[name] == "false" for name in TRACING_VARIABLES)
