"""The chained-equations engine (M2): conditional methods, chains, options and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biostats.linear import linear_regression
from biostats.mi import analyze_mi
from biostats.mi import methods as fcs
from biostats.mi.engine import HAZARD, impute, nelson_aalen
from biostats.mi.spec import ImputationSpec, default_method, suggest_m


def _cohort(n: int = 600, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A cohort and its copy with MAR missingness in a number, a binary and a three-level variable."""
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 10, n)
    site = rng.choice(["A", "B", "C"], n)
    smoker = np.where(rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.04 * (age - 60)))), "yes", "no")
    sbp = 100 + 0.5 * age + 6 * (smoker == "yes") + 4 * (site == "B") + rng.normal(0, 8, n)
    full = pd.DataFrame({"age": age, "site": site, "smoker": smoker, "sbp": sbp})
    df = full.copy()
    df.loc[rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.04 * (sbp - 130)))), "age"] = np.nan
    df.loc[rng.random(n) < 0.15, "smoker"] = np.nan
    df.loc[rng.random(n) < 0.10, "site"] = np.nan
    return full, df


SPEC = ImputationSpec(columns=["age", "site", "smoker", "sbp"], m=4, maxit=5, seed=11)


# --- methods ----------------------------------------------------------------------------------------


def test_match_donors_draws_among_the_nearest_predictions() -> None:
    rng = np.random.default_rng(0)
    obs = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    nearest = fcs.match_donors(obs, np.array([2.1, 10.4]), 1, rng)
    assert list(obs[nearest]) == [2.0, 10.0]
    picks = {
        int(fcs.match_donors(obs, np.array([11.0]), 3, np.random.default_rng(s))[0])
        for s in range(40)
    }
    assert picks == {4, 5, 6}  # one of the three closest, never a far one


def test_pmm_imputes_observed_values_and_norm_draws_new_ones() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 2))
    y = np.round(x @ [1.0, -2.0] + rng.normal(size=200), 1)
    out = fcs.pmm(y[:150], x[:150], x[150:], np.random.default_rng(2))
    assert set(out) <= set(y[:150])
    again = fcs.pmm(y[:150], x[:150], x[150:], np.random.default_rng(2))
    assert (out == again).all()
    drawn = fcs.norm(y[:150], x[:150], x[150:], np.random.default_rng(3))
    assert not set(drawn) <= set(y[:150]) and np.corrcoef(drawn, x[150:] @ [1.0, -2.0])[0, 1] > 0.8


def test_augmentation_adds_weighted_pseudo_observations() -> None:
    x = np.random.default_rng(4).normal(size=(50, 3))
    xa, ya, wa = fcs.augment(np.zeros(40, dtype=int), x[:40], x, 2)
    assert xa.shape == (40 + 2 * 3 * 2, 3) and set(ya[40:]) == {0, 1}
    assert wa[40:].sum() == pytest.approx(3 + 1)  # total weight p + 1
    assert (xa[40:] >= x.min(axis=0)).all() and (xa[40:] <= x.max(axis=0)).all()


def test_logreg_survives_perfect_prediction() -> None:
    x = np.linspace(-3, 3, 100)[:, None]
    y = (x[:, 0] > 0).astype(int)  # x separates the classes
    draws = np.array(
        [fcs.logreg(y, x, np.array([[-2.5], [2.5]]), np.random.default_rng(s)) for s in range(50)]
    )
    assert draws[:, 0].mean() < 0.2 and draws[:, 1].mean() > 0.8


def test_polyreg_and_rf_return_valid_values() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 2))
    codes = np.digitize(x[:, 0], [-0.5, 0.5])  # three categories driven by x
    out = fcs.polyreg(codes[:250], 3, x[:250], x[250:], rng)
    assert set(out) <= {0, 1, 2} and np.mean(out == codes[250:]) > 0.5
    numeric = fcs.rf(x[:250, 1], x[:250], x[250:], rng, categorical=False)
    assert set(numeric) <= set(x[:250, 1])
    category = fcs.rf(codes[:250], x[:250], x[250:], rng, categorical=True)
    assert set(category) <= {0, 1, 2} and np.mean(category == codes[250:]) > 0.5
    alone = fcs.rf(codes[:250], np.zeros((250, 0)), np.zeros((50, 0)), rng, categorical=True)
    assert set(alone) <= {0, 1, 2}


# --- engine -----------------------------------------------------------------------------------------


def test_engine_fills_only_the_missing_cells_with_valid_values() -> None:
    _, df = _cohort()
    run = impute(df, SPEC)
    assert run.methods == {"age": "pmm", "site": "polyreg", "smoker": "logreg"}
    mi = run.mi
    assert mi.m == 4 and mi.n_imputed() == {
        c: int(df[c].isna().sum()) for c in ["age", "site", "smoker"]
    }
    for _, completed in mi.completed():
        assert not completed[["age", "site", "smoker"]].isna().any().any()
        observed = df["age"].notna()
        assert (completed.loc[observed, "age"] == df.loc[observed, "age"]).all()
        assert set(completed["site"]) <= {"A", "B", "C"} and set(completed["smoker"]) <= {
            "no",
            "yes",
        }
    assert set(mi.imputations["age"].stack()) <= set(
        df["age"].dropna()
    )  # pmm: observed values only
    assert run.trace["age"]["mean"].shape == (5, 4) and np.isfinite(run.rhat["age"])
    assert run.mi.info["engine"].startswith("cohortstudio-fcs")


def test_chains_are_reproducible_and_independent_of_m() -> None:
    _, df = _cohort()
    five = impute(df, SPEC.model_copy(update={"m": 5}))
    three = impute(df, SPEC.model_copy(update={"m": 3}))
    for col in ["age", "site", "smoker"]:
        for i in (1, 2, 3):
            assert (five.mi.imputations[col][i] == three.mi.imputations[col][i]).all()
    other = impute(df, SPEC.model_copy(update={"seed": 12}))
    assert not (other.mi.imputations["age"][1] == five.mi.imputations["age"][1]).all()


def test_mar_imputation_recovers_the_regression() -> None:
    full, df = _cohort(n=1500, seed=8)
    truth = linear_regression(full, "sbp", ["age", "smoker", "site"]).params
    run = impute(df, SPEC.model_copy(update={"m": 10, "maxit": 8}))
    levels = run.mi.levels(["smoker", "site"])
    pooled = analyze_mi(
        lambda d: linear_regression(d, "sbp", ["age", "smoker", "site"], levels=levels), run.mi
    )
    for term in ["age", "smoker_yes", "site_B"]:
        assert pooled.term(term).ci.contains(truth[term]), term


def test_method_overrides_and_validation() -> None:
    _, df = _cohort()
    run = impute(df, SPEC.model_copy(update={"methods": {"age": "rf", "site": "rf", "smoker": ""}}))
    assert run.methods == {"age": "rf", "site": "rf"} and "smoker" not in run.mi.imputations
    assert any("smoker is incomplete and not imputed" in w for w in run.warnings)
    assert all("smoker" not in preds for preds in run.predictors.values())
    norm = impute(df, SPEC.model_copy(update={"methods": {"age": "norm"}}))
    assert not set(norm.mi.imputations["age"].stack()) <= set(df["age"].dropna())
    bad = [
        ({"columns": ["age", "bmi"]}, "unknown column"),
        ({"methods": {"age": "logreg"}}, "numeric: use pmm"),
        ({"methods": {"site": "pmm"}}, "categorical: use logreg"),
        ({"methods": {"site": "logreg"}}, "more than two categories"),
        ({"methods": {"age": "cart"}}, "unknown method"),
        ({"predictors": {"age": ["weight"]}}, "not usable"),
    ]
    for update, message in bad:
        with pytest.raises(ValueError, match=message):
            impute(df, SPEC.model_copy(update=update))
    with pytest.raises(ValueError, match="dates"):
        impute(
            df.assign(seen=pd.Timestamp("2026-01-01")),
            SPEC.model_copy(update={"columns": [*SPEC.columns, "seen"]}),
        )
    with pytest.raises(ValueError, match="unique"):
        impute(df.set_axis([0] * len(df)), SPEC)
    with pytest.raises(ValueError, match="no observed value"):
        impute(df.assign(age=np.nan), SPEC)


def test_predictor_selection_drops_constant_and_collinear_columns() -> None:
    _, df = _cohort()
    df = df.assign(constant=1.0, age_twice=df["age"] * 2, sbp_copy=df["sbp"] + 1e-9)
    run = impute(df, SPEC.model_copy(update={"columns": [*SPEC.columns, "constant", "sbp_copy"]}))
    assert "constant" in run.dropped["age"]
    assert {"sbp", "sbp_copy"} & set(run.dropped["age"])  # one of the duplicates is left out
    chosen = impute(df, SPEC.model_copy(update={"predictors": {"age": ["sbp"]}}))
    assert chosen.predictors["age"] == ["sbp"]


def test_imputation_within_strata_uses_donors_of_the_same_stratum() -> None:
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "arm": ["a"] * 100 + ["b"] * 100,
            "x": np.r_[rng.normal(0, 1, 100), rng.normal(100, 1, 100)],
        }
    )
    df["y"] = df["x"] + rng.normal(0, 1, 200)
    df.loc[[5, 6, 7, 105, 106], "x"] = np.nan
    run = impute(df, ImputationSpec(columns=["x", "y"], m=3, maxit=3, by="arm"))
    values = run.mi.imputations["x"]
    assert (values.loc[[5, 6, 7]] < 50).all().all() and (values.loc[[105, 106]] > 50).all().all()
    assert set(run.trace) == {"x [arm = a]", "x [arm = b]"}
    with pytest.raises(ValueError, match="must be complete"):
        impute(
            df.assign(arm=df["arm"].where(df.index > 0)),
            ImputationSpec(columns=["x", "y"], by="arm"),
        )


def test_survival_auxiliary_replaces_the_time_by_the_cumulative_hazard() -> None:
    times = np.array([2.0, 3.0, 3.0, 5.0, 8.0])
    events = np.array([1, 1, 0, 1, 0])
    # at risk 5, 4, 2, 1 at times 2, 3, 5, 8: H = 1/5, 1/5 + 1/4, ..., + 1/2, + 0
    expected = np.array([0.2, 0.45, 0.45, 0.95, 0.95])
    assert np.allclose(nelson_aalen(times, events), expected)
    rng = np.random.default_rng(10)
    df = pd.DataFrame(
        {
            "time": rng.exponential(5, 200),
            "event": rng.integers(0, 2, 200),
            "x": rng.normal(size=200),
        }
    )
    df.loc[:20, "x"] = np.nan
    run = impute(
        df, ImputationSpec(columns=["time", "event", "x"], m=2, maxit=2, survival=("time", "event"))
    )
    assert set(run.predictors["x"]) == {"event", HAZARD}
    with pytest.raises(ValueError, match="must be complete"):
        impute(
            df.assign(time=df["time"].where(df.index > 3)),
            ImputationSpec(columns=["x"], survival=("time", "event")),
        )


def test_single_level_progress_and_suggested_m() -> None:
    df = pd.DataFrame({"g": ["a"] * 20 + [None] * 5, "x": np.arange(25.0)})
    ticks: list[float] = []
    run = impute(df, ImputationSpec(columns=["g", "x"], m=2, maxit=3), progress=ticks.append)
    assert set(run.mi.imputations["g"].stack()) == {"a"} and ticks[-1] == pytest.approx(1.0)
    assert default_method(pd.Series([1.0, None])) == "pmm"
    assert default_method(pd.Series(["x", "y", None])) == "logreg"
    assert default_method(pd.Series([True, False, None])) == "logreg"
    assert default_method(pd.Series(["x", "y", "z"])) == "polyreg"
    _, cohort = _cohort()
    share = cohort[["age", "site", "smoker"]].isna().any(axis=1).mean()
    assert suggest_m(cohort, ["age", "site", "smoker"]) == int(np.ceil(100 * share))
    assert suggest_m(cohort[["sbp"]], ["sbp"]) == 5


def test_parallel_chains_give_the_same_imputations() -> None:
    _, df = _cohort(n=200)
    spec = SPEC.model_copy(update={"m": 3, "maxit": 2})
    sequential = impute(df, spec)
    parallel = impute(df, spec, workers=2)
    for col in sequential.methods:
        assert sequential.mi.imputations[col].equals(parallel.mi.imputations[col])
    assert all(
        np.array_equal(sequential.trace[k]["mean"], parallel.trace[k]["mean"], equal_nan=True)
        for k in sequential.trace
    )


def test_a_failed_parallel_run_falls_back_to_one_process(monkeypatch: pytest.MonkeyPatch) -> None:
    import biostats.mi.parallel as parallel

    def refuse(*args, **kwargs):
        raise RuntimeError("the parallel imputation failed: exit code 3")

    monkeypatch.setattr(parallel, "run_in_subprocess", refuse)
    _, df = _cohort(n=200)
    spec = SPEC.model_copy(update={"m": 3, "maxit": 2})
    fallback = impute(df, spec, workers=2)
    sequential = impute(df, spec)
    assert fallback.mi.fingerprint() == sequential.mi.fingerprint()
    assert any("the parallel run failed" in w and "exit code 3" in w for w in fallback.warnings)
    assert not any("parallel" in w for w in sequential.warnings)


def test_an_interrupted_parallel_run_stops_its_worker_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a click reruns the page: Streamlit raises a BaseException in the progress callback, which skipped the
    # clean-up, so the worker process ran on and kept the files of its temporary folder
    from pathlib import Path

    from biostats.mi import parallel

    class Worker:
        def __init__(self, args, **kwargs) -> None:
            Path(args[-1]).write_text("0.5", encoding="utf-8")  # its progress file
            self.running, self.waited = True, False
            self.returncode = None

        def poll(self):
            return None if self.running else self.returncode

        def terminate(self) -> None:
            self.running, self.returncode = False, -15

        def wait(self, timeout=None):
            self.waited = True
            return self.returncode

    started: list[Worker] = []

    def popen(args, **kwargs):
        started.append(Worker(args, **kwargs))
        return started[-1]

    class Rerun(BaseException):
        pass

    def progress(fraction: float) -> None:
        raise Rerun

    monkeypatch.setattr(parallel.subprocess, "Popen", popen)
    with pytest.raises(Rerun):
        parallel.run_in_subprocess(pd.DataFrame({"x": [1.0]}), object(), 2, progress)
    assert not started[0].running and started[0].waited
