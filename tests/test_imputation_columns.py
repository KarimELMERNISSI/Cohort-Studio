"""
Imputation keeps every column and names it correctly (regression: "Shape of passed values is (500, 20),
indices imply (500, 19)" after adding a computed variable to a cohort with a date stored as text).
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app_pages.data_enrichment import apply_imputer
from enrich.data_imputation import detect_remainder_columns, partition_columns
from ingest.loaders import read_path

SAMPLE = Path(__file__).resolve().parents[1] / "assets" / "sample_cohort.csv"


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    df = read_path(str(SAMPLE)).df  # enrol_date is a day-first date stored as text; patient_id is an identifier
    df["bmi_x2"] = df["bmi"] * 2  # computed numeric variable
    df["elderly"] = df["age"] > 65  # computed yes/no variable (bool dtype)
    return df


def _impute(df: pd.DataFrame, num: str, cat: str, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return apply_imputer(df, num, cat, remainder_columns="auto", remainder_threshold=0.3, **kwargs)


def test_every_column_is_assigned_to_exactly_one_group(cohort: pd.DataFrame) -> None:
    remainder = detect_remainder_columns(cohort, threshold=0.3, date_like_columns=["enrol_date"])
    assert remainder == ["patient_id", "enrol_date"]  # text identifier and text date (pandas 3 'str' dtype)
    numerical, categorical, passthrough = partition_columns(cohort, remainder)
    groups = [set(numerical), set(categorical), set(passthrough)]
    assert set().union(*groups) == set(cohort.columns)
    assert sum(len(g) for g in groups) == cohort.shape[1]
    assert "elderly" in set(categorical) and "bmi_x2" in set(numerical)
    assert passthrough == ["patient_id", "enrol_date"]


@pytest.mark.parametrize(("num", "cat"), [("Mean", "Most_Frequent"), ("Median", "Most_Frequent"), ("KNN", "Most_Frequent"),
                                          ("MICE", "Most_Frequent")])
def test_imputation_keeps_and_aligns_all_columns(cohort: pd.DataFrame, num: str, cat: str) -> None:
    out, mask = _impute(cohort, num, cat)
    assert set(out.columns) == set(cohort.columns) and out.shape == cohort.shape
    assert out[["biomarker", "lab_mcar"]].isna().sum().sum() == 0  # the gaps were filled
    # names line up with values: untouched columns are identical to the input
    assert np.allclose(out["age"].astype(float), cohort["age"].astype(float))
    assert np.allclose(out["bmi_x2"].astype(float), cohort["bmi_x2"].astype(float))
    assert (out["enrol_date"].astype(str) == cohort["enrol_date"].astype(str)).all()
    assert (out["patient_id"].astype(str) == cohort["patient_id"].astype(str)).all()
    assert (out["arm"].astype(str) == cohort["arm"].astype(str)).all()
    assert mask["biomarker"].sum() == cohort["biomarker"].isna().sum()


def test_one_hot_encoding_returns_a_dense_named_table(cohort: pd.DataFrame) -> None:
    out, _ = _impute(cohort, "Mean", "Most_Frequent", cat_encoder=True)
    assert {"arm_Placebo", "arm_Treatment", "sex_F", "sex_M"} <= set(out.columns)
    assert "patient_id" in out.columns and not any(c.startswith("patient_id_") for c in out.columns)  # identifiers not encoded
    assert np.allclose(out["age"].astype(float), cohort["age"].astype(float))
    assert ((out["arm_Placebo"].astype(float) == 1) == (cohort["arm"] == "Placebo")).all()


def test_drop_strategy_removes_only_passthrough_columns(cohort: pd.DataFrame) -> None:
    out, _ = _impute(cohort, "Mean", "Most_Frequent", remainder_strategy="drop")
    assert set(out.columns) == set(cohort.columns) - {"patient_id", "enrol_date"}


def test_missforest_for_both_groups(cohort: pd.DataFrame) -> None:
    small = cohort.head(120).copy()
    out, _ = _impute(small, "MissForest", "MissForest")
    assert set(out.columns) == set(small.columns)
    assert out[["biomarker", "lab_mcar"]].isna().sum().sum() == 0
    assert (out["enrol_date"].astype(str) == small["enrol_date"].astype(str)).all()
