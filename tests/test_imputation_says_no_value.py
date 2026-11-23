"""
A single imputation says nothing of the data: no value reaches the process's output (stdout, stderr, at the file
descriptor, where a server's or a container's log takes it) nor the log at DEBUG, whatever the pair of methods, on
the Enrich page's path (``impute_rows``) and on a replay (``apply_steps``). MissForest printed the whole imputed
table to stdout until this was tested.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import pytest

from enrich import imputation_service as service
from manage.replay.engine import apply_steps

NUMERICAL = ("Mean", "Median", "KNN", "MICE", "MissForest")
CATEGORICAL = ("Most_Frequent", "MissForest")
MARK_NUMBER = 4242.4242  # an observed value, as a table printed whole would show it
MARK_TEXT = "zq-marker-7731"
# the marked number as a table prints it, and as numpy writes it among values of other scales (4.2424242e+03)
MARKS = ("4242.4", "4.24242", MARK_TEXT)


def _frame() -> pd.DataFrame:
    """Two numbers and two texts with missing cells (two texts: MissForest on one text column alone fails), and a
    marked value in each kind of column."""
    rng = np.random.default_rng(3)
    n = 40
    frame = pd.DataFrame(
        {
            "age": rng.normal(60, 10, n).round(1),
            "weight": rng.normal(75, 12, n).round(1),
            "arm": rng.choice(["a", "b"], n).astype(object),
            "site": rng.choice(["north", "south", "east"], n).astype(object),
        }
    )
    frame.loc[0, "age"] = MARK_NUMBER
    frame.loc[1, "site"] = MARK_TEXT
    frame.loc[[3, 7, 11], "age"] = np.nan
    frame.loc[[4, 9], "weight"] = np.nan
    frame.loc[[5, 13], "arm"] = np.nan
    frame.loc[[6, 15], "site"] = np.nan
    return frame


@pytest.mark.parametrize(
    ("numerical", "categorical"), list(itertools.product(NUMERICAL, CATEGORICAL))
)
def test_an_imputation_shows_no_value_on_the_output_or_in_the_log(
    numerical: str,
    categorical: str,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    recwarn: pytest.WarningsRecorder,
) -> None:
    if "MissForest" in (numerical, categorical):
        pytest.importorskip(
            "lightgbm", reason="MissForest needs LightGBM, in the optional 'ml' extra"
        )
    caplog.set_level(logging.DEBUG)
    frame = _frame()
    outcome = service.impute_rows(frame, numerical=numerical, categorical=categorical)
    assert outcome.frame[["age", "weight"]].notna().all().all()
    params = {
        "numerical_imputation_method": numerical,
        "categorical_imputation_method": categorical,
    }
    if "random_state" in outcome.params:
        params["random_state"] = outcome.params["random_state"]
    replayed = apply_steps(frame, [{"function": "imputation", "params": params}])
    assert replayed.error is None, replayed.error
    # the replay imputed (a step it skipped would leave the output silent too); the replay names its index
    pd.testing.assert_frame_equal(replayed.df, outcome.frame, check_names=False)
    out, err = capfd.readouterr()
    # a warning is recorded by pytest before it reaches stderr; a server's stderr shows it
    warned = "\n".join(str(w.message) for w in recwarn)
    for text in (out, err, caplog.text, warned):
        assert not any(mark in text for mark in MARKS)
