"""What the AI features send about a dataset: a summary of each column, never a row or an extreme value (M5)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from manage.ai_payload import MIN_COUNT, column_profile


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
