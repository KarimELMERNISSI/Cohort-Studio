"""The chained-equations engine (M2): conditional methods, chains, options and diagnostics."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

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


def _long(n: int = 12_000) -> pd.DataFrame:
    """Two complete numbers and two with missing values, observed on more than 10,000 rows: where a BLAS of several
    threads adds a product up in another order than one thread does."""
    rng = np.random.default_rng(7)
    x, z = rng.normal(size=n), rng.normal(size=n)
    y = 1.5 * x - 0.7 * z + rng.normal(size=n)
    out = pd.DataFrame({"x": x, "z": z, "y": y, "w": 0.4 * x + rng.normal(size=n)})
    out.loc[rng.random(n) < 0.08, "y"] = np.nan
    out.loc[rng.random(n) < 0.08, "w"] = np.nan
    return out


LONG = ImputationSpec(
    columns=["x", "z", "y", "w"], methods={"y": "norm", "w": "norm"}, m=2, maxit=3, seed=11
)


def _on_its_own_threads() -> bool:
    """Puts this process's linear algebra on several threads, as a process that never set one is; False where the
    machine has one only."""
    from threadpoolctl import threadpool_limits

    from biostats.mi.engine import blas_threads

    threadpool_limits(limits=max(os.cpu_count() or 1, 2), user_api="blas")
    return max(blas_threads(), default=1) > 1


def _threads_change_a_long_product() -> bool:
    """Whether this machine's BLAS adds a product of 12,000 numbers up otherwise on its own threads than on one:
    where it does not (one core, a BLAS built for one thread), the cells cannot differ and a test of them shows
    nothing. Twenty vectors: two orders of adding can give the same number for one of them. Leaves the process on
    its own threads."""
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(1)
    vectors = [rng.normal(size=12_000) for _ in range(20)]
    if not _on_its_own_threads():
        return False
    many = [float(v @ v) for v in vectors]
    with threadpool_limits(limits=1, user_api="blas"):
        one = [float(v @ v) for v in vectors]
    return many != one


def test_the_cells_are_the_same_in_this_process_and_in_workers_above_ten_thousand_rows(
    tmp_path: Path,
) -> None:
    """Every chain is drawn on one BLAS thread, here as in a worker process. Drawn with this process's own threads,
    a ``norm`` variable observed on more than 10,000 rows came out with other last digits than in workers (on 14
    threads, this very table gave another fingerprint): whether a faithful imputation read as reproduced then
    depended on where it was drawn again."""
    from biostats.mi.engine import blas_threads

    df = _long()
    shows = _threads_change_a_long_product()
    here = impute(df, LONG)
    assert set(blas_threads()) <= {1}  # set by the draw, and kept
    assert here.blas_threads == 1 and not any("threads" in w for w in here.warnings)
    workers = impute(df, LONG, workers=2, scratch=tmp_path)
    assert workers.blas_threads == 1
    assert here.mi.fingerprint() == workers.mi.fingerprint()
    for column in here.methods:
        assert here.mi.imputations[column].equals(workers.mi.imputations[column])
    # the parallel run's copy of the table was written under the folder given, and is gone
    assert not list(tmp_path.iterdir())
    if not shows:
        pytest.skip(
            "this machine's BLAS adds a long product up the same way on one thread and on its own: the cells could "
            "not have differed here (the test of a chain's threads holds on any machine)"
        )


def test_a_chain_runs_on_one_blas_thread_in_this_process_and_in_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the library reports inside a chain, whatever the machine shows of it in the cells: one thread, in this
    process and in the run that follows a failed parallel one. With the limit taken out of this process and of the
    workers together, their cells would agree again, on as many threads as the machine has cores."""
    import biostats.mi.parallel as parallel
    from biostats.mi.engine import blas_threads

    seen: list[list[int]] = []
    drawn = fcs.norm

    def noting(*args, **kwargs):
        seen.append(blas_threads())
        return drawn(*args, **kwargs)

    def refuse(*args, **kwargs):
        raise RuntimeError("the parallel imputation failed: exit code 3")

    monkeypatch.setattr(fcs, "norm", noting)
    df = _long(n=400)
    several = _on_its_own_threads()
    impute(df, LONG)
    assert len(seen) == 12 and all(
        set(threads) <= {1} for threads in seen
    )  # 2 variables, 2 chains, 3 passes
    seen.clear()
    assert _on_its_own_threads() == several
    monkeypatch.setattr(parallel, "run_in_subprocess", refuse)
    fallback = impute(df, LONG, workers=2)
    assert any("the parallel run failed" in w for w in fallback.warnings)
    assert len(seen) == 12 and all(set(threads) <= {1} for threads in seen)


def test_one_blas_thread_is_set_once_and_never_given_back() -> None:
    from biostats.mi.engine import blas_threads, one_blas_thread

    _on_its_own_threads()
    one_blas_thread()
    assert set(blas_threads()) <= {1}
    one_blas_thread()
    assert set(blas_threads()) <= {1}


class _HeldLimit:
    """Another session's thread that sets one BLAS thread and later restores what it found, as scikit-learn does
    around every run of KMeans."""

    def __init__(self) -> None:
        import threading

        self.held, self.leave = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._run)

    def _run(self) -> None:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="blas"):
            self.held.set()
            self.leave.wait(60)

    def __enter__(self) -> _HeldLimit:
        self.thread.start()
        assert self.held.wait(60)
        return self

    def release(self, *_: object) -> None:
        """Called from the draw's progress: the other thread restores the threads it found, mid-draw."""
        self.leave.set()
        self.thread.join(60)

    def __exit__(self, *_: object) -> None:
        self.release()


def test_another_sessions_limit_does_not_move_a_draw_on_a_server() -> None:
    """The server's process is on one BLAS thread from its start: a KMeans of another session that began before a
    draw saved one thread, and restores one when it returns in the middle of the draw."""
    from biostats.mi.engine import blas_threads, one_blas_thread

    df = _long()
    one_blas_thread()  # as the server sets it when it starts
    alone = impute(df, LONG)
    with _HeldLimit() as other:
        beside = impute(df, LONG, progress=other.release)
    assert set(blas_threads()) <= {1} and beside.blas_threads == 1
    assert not any("threads" in w for w in beside.warnings)
    assert beside.mi.fingerprint() == alone.mi.fingerprint()


def test_a_draw_whose_threads_were_moved_says_so_and_records_them() -> None:
    """A process that never set one thread, where another thread's limit began before the draw and restored the
    process's own threads during it: the draw cannot prevent it, so it says it and the step records the threads."""
    from biostats.mi.engine import blas_threads, drawn_with, one_blas_thread

    if not _on_its_own_threads():
        pytest.skip("one processor: the linear algebra has one thread whatever is asked")
    df = _long()
    try:
        with _HeldLimit() as other:  # it found several threads and holds one
            run = impute(df, LONG, progress=other.release)
        assert run.blas_threads > 1 and max(blas_threads()) > 1
        (said,) = [w for w in run.warnings if "threads during the draw" in w]
        assert f"up to {run.blas_threads} threads" in said
        recorded = drawn_with(run.blas_threads)["platform"]["blas"]
        assert recorded and {pool["threads"] for pool in recorded} == {run.blas_threads}
    finally:
        one_blas_thread()  # the suite's own state


def test_threads_moved_and_put_back_between_two_chains_are_seen() -> None:
    """The draw reads the threads after each chain, not only as it begins and ends: moved during the first chain
    and put back during the second, they are one again when the draw ends."""
    from threadpoolctl import threadpool_limits

    from biostats.mi.engine import blas_threads, one_blas_thread

    if not _on_its_own_threads():
        pytest.skip("one processor: the linear algebra has one thread whatever is asked")
    one_blas_thread()
    df = _long(n=400)
    passes = 0

    def meanwhile(_: float) -> None:  # LONG: two chains of three passes each
        nonlocal passes
        passes += 1
        if passes == 1:
            threadpool_limits(limits=max(os.cpu_count() or 1, 2), user_api="blas")
        elif passes == 5:
            threadpool_limits(limits=1, user_api="blas")

    run = impute(df, LONG, progress=meanwhile)
    assert passes == 6 and set(blas_threads()) <= {1}
    assert run.blas_threads > 1 and any("threads during the draw" in w for w in run.warnings)


def test_a_failed_parallel_run_falls_back_to_one_process(monkeypatch: pytest.MonkeyPatch) -> None:
    import biostats.mi.parallel as parallel

    def refuse(*args, **kwargs):
        raise RuntimeError("the parallel imputation failed: exit code 3")

    def no_pool(*args, **kwargs):
        # inside Streamlit the page script is __main__: a pool started here re-ran it in each worker
        raise AssertionError("a process pool was started in the calling process")

    monkeypatch.setattr(parallel, "run_in_subprocess", refuse)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", no_pool)
    _, df = _cohort(n=200)
    spec = SPEC.model_copy(update={"m": 3, "maxit": 2})
    fallback = impute(df, spec, workers=2)
    sequential = impute(df, spec)
    assert fallback.mi.fingerprint() == sequential.mi.fingerprint()
    assert any("the parallel run failed (RuntimeError)" in w for w in fallback.warnings)
    # the child's last line can quote a cell
    assert not any("exit code" in w for w in fallback.warnings)
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


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes.wintypes

        wintypes = ctypes.wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_an_interrupted_parallel_run_leaves_no_process_and_no_copy_of_the_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping the run's interpreter left its pool workers computing, unseen, with the data (review, M5b)."""
    import tempfile

    from biostats.mi import parallel

    # every interpreter the run starts notes its process id (a test-only sitecustomize on PYTHONPATH)
    hook, seen, temp = tmp_path / "hook", tmp_path / "pids", tmp_path / "temp"
    for folder in (hook, seen, temp):
        folder.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "folder = os.environ.get('CMP_TEST_PIDS')\n"
        "if folder:\n"
        "    open(os.path.join(folder, str(os.getpid())), 'w').close()\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(hook))
    monkeypatch.setenv("CMP_TEST_PIDS", str(seen))
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    class Rerun(BaseException):
        pass

    def progress(fraction: float) -> None:
        if fraction > 0:  # the first chains are done, the others are being computed
            raise Rerun

    _, df = _cohort(n=3000)
    with pytest.raises(Rerun):
        parallel.run_in_subprocess(df, SPEC.model_copy(update={"m": 4, "maxit": 30}), 2, progress)
    assert list(temp.iterdir()) == []  # the copy of the data is deleted
    pids = [int(p.name) for p in seen.iterdir()]
    assert len(pids) >= 3  # the run's interpreter and its two workers
    deadline = time.monotonic() + 15
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not [pid for pid in pids if _alive(pid)]
