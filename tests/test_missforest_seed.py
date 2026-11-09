"""
MissForest gives the same values on any machine (M8, the study package): its LightGBM learners run in their
deterministic mode, whatever the number of threads (they were built for every processor without it), and the seed it
draws with is recorded with the imputation step and used by the replay, so that a recipient with more cores, or
another CMP_RANDOM_SEED, imputes the same values.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm", reason="MissForest needs LightGBM, in the optional 'ml' extra")


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 160
    frame = pd.DataFrame(
        {
            "age": rng.normal(60, 10, n).round(),
            "sbp": rng.normal(135, 15, n).round(),
            "bmi": rng.normal(27, 4, n).round(1),
            "arm": rng.choice(["A", "B"], n).astype(object),
        }
    )
    for column, share in (("sbp", 0.2), ("bmi", 0.15), ("arm", 0.1)):
        frame.loc[rng.random(n) < share, column] = np.nan
    return frame


def test_one_thread_and_four_impute_the_same_values() -> None:
    from enrich.data_imputation import MissForestTransformer

    frame = _frame()
    one = MissForestTransformer(categorical_cols=["arm"], n_jobs=1, random_state=11).fit_transform(
        frame.copy()
    )
    four = MissForestTransformer(categorical_cols=["arm"], n_jobs=4, random_state=11).fit_transform(
        frame.copy()
    )
    pd.testing.assert_frame_equal(one, four)
    assert not one.isna().any().any()


def test_the_step_records_the_seed_and_the_replay_draws_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another process seed (CMP_RANDOM_SEED) at the replay: the recorded one is used, and the values are the same."""
    import enrich.data_imputation as edi
    import enrich.imputation_service as service

    frame = _frame()
    recorded = service.impute_rows(frame, numerical="MissForest", categorical="MissForest")
    assert recorded.params["random_state"] == service.RANDOM_STATE
    monkeypatch.setattr(edi, "RANDOM_STATE", 12345)  # another machine's process seed
    monkeypatch.setattr(service, "RANDOM_STATE", 12345)
    replayed = service.impute_rows(
        frame,
        numerical="MissForest",
        categorical="MissForest",
        random_state=recorded.params["random_state"],
    )
    pd.testing.assert_frame_equal(recorded.frame, replayed.frame)


def test_the_replay_passes_the_recorded_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from manage import reproduction_manager as engine

    seen: dict = {}

    def impute(frame, rows, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("enough")

    monkeypatch.setattr(engine, "impute_rows", impute)
    step = {
        "function": "imputation",
        "params": {"numerical_imputation_method": "MissForest", "random_state": 77},
    }
    with pytest.raises(RuntimeError, match="enough"):
        engine._imputation(_frame(), step, 0, None, [])
    assert seen["random_state"] == 77
    seen.clear()
    with pytest.raises(
        RuntimeError
    ):  # a step recorded before the seed was kept: the process's, as before
        engine._imputation(_frame(), {"function": "imputation", "params": {}}, 0, None, [])
    assert seen["random_state"] is None


def test_no_seed_is_recorded_without_missforest() -> None:
    from enrich.imputation_service import impute_rows

    assert (
        "random_state"
        not in impute_rows(_frame(), numerical="Median", categorical="Most_Frequent").params
    )
