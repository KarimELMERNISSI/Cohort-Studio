"""
Exports that held every row without saying so (the data-protection specialist's PS2-L1 to PS2-L3).

- Epidemiology > Z-scores wrote every column of every row (an identifier, initials, a birth date) for the z-score of
  one variable: the file now holds what was asked, the row number, each variable with its z-score and the reference
  grouping, and its button says it holds one value per participant.
- Validate's comparison of two datasets writes both whole, beside the summary: the page now says so before the click.
- A clustermap with its labels hidden wrote every sample's identifier in its file (its axis and its heatmap): hidden
  labels are now positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app_pages.data_monitoring import COMPARISON_EXPORT_CAPTION
from app_pages.epidemiology import ZSCORE_FILE_HELP, zscore_file
from biostats.zscores import reference_zscores
from utils.visualization_utils import create_clustermap_streamlit


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = 40
    return pd.DataFrame(
        {
            "patient_id": [f"PT-CANARY-{i:03d}" for i in range(rows)],
            "initials": ["JD", "AB", "CK", "MZ"] * 10,
            "birth_date": pd.date_range("1950-01-01", periods=rows, freq="200D"),
            "arm": ["control", "treated"] * 20,
            "crp": rng.gamma(2.0, 3.0, rows).round(2),
            "bmi": rng.normal(26, 4, rows).round(1),
        }
    )


def test_the_z_score_file_holds_what_was_asked() -> None:
    df = _cohort()
    z_df, _ = reference_zscores(df, ["crp"], "control", ref_col="arm", ref_value="control")
    held = zscore_file(z_df, ["crp"], ["crp_Z"], "arm")
    assert list(held.columns) == ["Row", "arm", "crp", "crp_Z"]
    assert held["Row"].tolist() == df.index.tolist() and len(held) == len(df)
    assert not {"patient_id", "initials", "birth_date", "bmi"} & set(held.columns)
    # standardised against the whole cohort: no grouping to carry
    z_all, _ = reference_zscores(df, ["crp", "bmi"], "cohort")
    assert list(zscore_file(z_all, ["crp", "bmi"], ["crp_Z", "bmi_Z"], None).columns) == [
        "Row",
        "crp",
        "crp_Z",
        "bmi",
        "bmi_Z",
    ]
    assert "One row per participant" in ZSCORE_FILE_HELP
    assert "only where the dataset itself may go" in ZSCORE_FILE_HELP


def test_the_comparison_s_export_says_it_holds_both_datasets() -> None:
    assert "both datasets whole, every row and every column" in COMPARISON_EXPORT_CAPTION
    assert "only where both datasets may go" in COMPARISON_EXPORT_CAPTION


def test_a_clustermap_with_its_labels_hidden_writes_no_identifier() -> None:
    df = _cohort()
    for dendrograms in (False, True):
        fig, error = create_clustermap_streamlit(
            df,
            ["crp", "bmi"],
            sample_col="patient_id",
            transformation="zscore",
            show_row_labels=False,
            show_col_labels=False,
            show_row_dendrogram=dendrograms,
            show_col_dendrogram=dendrograms,
        )
        assert error is None and fig is not None
        written = fig.to_json()
        assert "CANARY" not in written, dendrograms
    # shown, they are written, as before
    fig, _ = create_clustermap_streamlit(
        df, ["crp", "bmi"], sample_col="patient_id", transformation="zscore"
    )
    assert "PT-CANARY-000" in fig.to_json()
