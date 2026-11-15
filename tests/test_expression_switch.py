"""
Formulas and criteria computed in the worker (manage.workers), as the application meets them: a stop of the worker is
the expression's own failure, in its own words; "no worker" is nobody's, goes through every handler that catches
everything, and is asked again at the next run; the pool is installed from the settings, unless the development switch
keeps expressions in this process (the test suite's case), which the log says.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from core.criteria import (
    CriterionError,
    criterion_lacking,
    criterion_mask,
    criterion_unknown,
    earlier_mask,
    rule_checks,
)
from core.expressions import ExpressionStopped, WorkerUnavailable
from core.expressions.runner import running
from enrich.formulas import evaluate_formula_safely, try_formula
from manage.workers import install_for
from manage.workers.pool import BUSY, NOT_STARTED
from manage.workers.wire import UNREAD, round_trip, stopped
from ui import nav

ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "main.py")
DEFINITION = {"expression": "df['age'] > 60"}


def _raising(error: BaseException):
    def runner(*args: Any, **kwargs: Any) -> Any:
        raise error

    return runner


def _cohort() -> pd.DataFrame:
    return pd.DataFrame({"age": [40.0, 70.0], "site": ["Alpha", "Beta"]})


def test_no_worker_goes_through_every_handler_of_the_criteria() -> None:
    """Nothing is known of the criterion: none of them reads it as the criterion's failure, or as no row."""
    df = _cohort()
    with running(_raising(WorkerUnavailable(BUSY))):
        for call in (
            lambda: criterion_mask(df, "older", DEFINITION),
            lambda: earlier_mask(df, "older", DEFINITION),
            lambda: criterion_unknown(df, "older", DEFINITION),
            lambda: criterion_lacking(df, "older", DEFINITION),
            lambda: rule_checks(df, "older", DEFINITION),
        ):
            with pytest.raises(WorkerUnavailable):
                call()


def test_a_stop_is_the_criterion_s_own_failure_in_its_own_words() -> None:
    with running(_raising(stopped("time", 30))), pytest.raises(CriterionError) as caught:
        criterion_mask(_cohort(), "older", DEFINITION)
    said = str(caught.value)
    assert said.startswith("criterion 'older': It took longer than 30 seconds")
    assert "expression rejected" not in said


def test_a_formula_stopped_by_a_limit_says_so_without_examples_of_the_data() -> None:
    stop = stopped("memory", 2048)
    with running(_raising(stop)):
        ok, said, _ = evaluate_formula_safely("age * 2", "double", _cohort())
        trial = try_formula("site + age", _cohort())
    assert not ok and isinstance(said, ExpressionStopped) and "2,048 MB" in str(said)
    assert trial.problem == str(stop)  # whole, and no "site holds text (such as ...)"
    assert trial.stopped and trial.tried  # shown as the application's sentence, not as code
    with running(_raising(WorkerUnavailable(BUSY))), pytest.raises(WorkerUnavailable):
        evaluate_formula_safely("age * 2", "double", _cohort())


def test_the_pool_is_installed_from_the_settings_unless_the_switch_keeps_expressions_here(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.expressions import runner

    here = SimpleNamespace(
        development=SimpleNamespace(formula_in_process=True),
        formulas=SimpleNamespace(seconds=30, memory_mb=2048, workers=2),
    )
    before = runner.installed()
    with caplog.at_level(logging.WARNING, logger="manage.workers"):
        assert install_for(here) is False
    assert runner.installed() is before
    assert any("CMP_FORMULA_IN_PROCESS is on" in r.getMessage() for r in caplog.records)

    made: list[dict[str, Any]] = []

    class Pool:
        def __init__(self, **limits: Any) -> None:
            made.append(limits)

        def warm(self) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("not computed in this test")

    monkeypatch.setattr("manage.workers.pool.ExpressionWorkers", Pool)
    monkeypatch.setattr("manage.workers._pool", None)
    there = SimpleNamespace(
        development=SimpleNamespace(formula_in_process=False),
        formulas=SimpleNamespace(seconds=12, memory_mb=1024, workers=3),
    )
    try:
        with caplog.at_level(logging.INFO, logger="manage.workers"):
            assert install_for(there) is True
            assert isinstance(runner.installed(), Pool)
            install_for(there)  # once per process
    finally:
        runner.install(before)
    assert made == [{"workers": 3, "seconds": 12, "memory_mb": 1024}]
    assert any("each stopped after 12 seconds or 1024 MB" in r.getMessage() for r in caplog.records)


# --- the pages ------------------------------------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path: Path):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path)
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "worker_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    yield at
    dp.DATA_ROOT = previous


def test_enrich_says_why_a_formula_was_stopped(app: AppTest) -> None:
    app.switch_page(nav.page_path("enrich")).run()
    app.text_input(key="variable_name_input").input("double_age").run()
    app.text_area(key="formula_input").input("age * 2").run()
    with running(_raising(stopped("time", 30))):
        next(b for b in app.button if b.label == "Compute & Preview").click().run()
    assert not app.exception
    shown = [e.value for e in app.error]
    assert any("It took longer than 30 seconds" in text for text in shown), shown
    assert not any("Make sure your formula" in i.value for i in app.info)


def test_no_worker_is_said_by_the_page_and_asked_again_at_the_next_run(app: AppTest) -> None:
    """Validate checks the study's criteria when it first opens on a table (then keeps them by the table's
    fingerprint): without a worker, the page says so, and the next run checks them again (the failure was not kept)."""
    with running(_raising(WorkerUnavailable(BUSY))):
        app.switch_page(nav.page_path("validate")).run()
    assert not app.exception
    assert any(BUSY in e.value for e in app.error), [e.value for e in app.error]
    computed: list[int] = []

    def counting(*args: Any, **kwargs: Any) -> Any:
        computed.append(1)
        return round_trip(*args, **kwargs)

    with running(counting):
        app.run()
    assert not app.exception and not any(BUSY in e.value for e in app.error)
    assert computed, "the criteria were not checked again"


def _expression_criterion(app: AppTest) -> str:
    """The sidebar box of one of the tutorial's expression criteria (none is ticked when the tutorial loads)."""
    from core.population import catalogue
    from ui.components.population import _widget_key

    for criterion in catalogue(app.session_state["config"]):
        if "expression" in criterion.definition:
            return _widget_key(criterion.id)
    raise AssertionError("the tutorial holds no expression criterion")


@pytest.mark.parametrize("why", [BUSY, NOT_STARTED, UNREAD])
def test_no_worker_leaves_the_sidebar_and_the_page_usable(app: AppTest, why: str) -> None:
    """A criterion ticked while no worker can compute: the sidebar says the population was not computed and goes on
    (Log out is there), the page says why where the waiting line was, with Try again; no traceback."""
    app.switch_page(nav.page_path("settings")).run()
    box = _expression_criterion(app)
    with running(_raising(WorkerUnavailable(why))):
        app.sidebar.checkbox(key=box).check().run()
    assert not app.exception, [str(e.value)[:200] for e in app.exception]
    assert any("Population not computed" in c.value for c in app.sidebar.caption)
    assert any(b.label == "Log out" for b in app.sidebar.button)
    app.run()  # a worker again: the population is computed, nothing of the failure kept
    assert not app.exception
    assert not any("Population not computed" in c.value for c in app.sidebar.caption)


def test_a_ticked_criterion_is_computed_in_the_run_under_its_waiting_line(app: AppTest) -> None:
    """The tick's callback records the tick only; the run that follows computes the population, where the waiting
    line and its hook are set (a click elsewhere or Stop then ends the wait)."""
    from manage.workers import pool

    hooks: list[object] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        hooks.append(pool._waiting.get())
        return round_trip(*args, **kwargs)

    box = _expression_criterion(app)
    with running(recording):
        app.sidebar.checkbox(key=box).check().run()
    assert not app.exception
    assert hooks and all(hook is not None for hook in hooks), hooks


def test_validate_s_form_says_a_stop_before_the_criterion_is_added() -> None:
    from app_pages.data_monitoring import unknown_advice

    with running(_raising(stopped("time", 30))):
        with pytest.raises(CriterionError, match="It took longer than 30 seconds"):
            unknown_advice(_cohort(), "older", DEFINITION)
        assert not criterion_unknown(
            _cohort(), "older", DEFINITION
        ).any()  # elsewhere: no row, as before


def test_no_worker_at_a_run_s_start_is_kept_for_the_run_and_no_page_reads_excluded_rows(
    app: AppTest,
) -> None:
    """A worker busy at the start of the run and free a moment later: the run asks the pool once (each ask can wait
    a minute and more), the analyses see no row rather than the rows the ticked criterion excludes, and the analysis
    page says why; the next run computes the population, and the page reads its rows only."""
    app.switch_page(nav.page_path("survival")).run()
    rows = len(app.session_state["data"])
    calls: list[int] = []

    def busy_once(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise WorkerUnavailable(BUSY)
        return round_trip(*args, **kwargs)

    with running(busy_once):
        app.sidebar.checkbox(key=_expression_criterion(app)).check().run()
    assert not app.exception, [str(e.value)[:200] for e in app.exception]
    assert len(calls) == 1, len(calls)
    assert len(app.session_state["data"]) == 0
    assert any(BUSY in e.value for e in app.error)
    assert any("Population not computed" in c.value for c in app.sidebar.caption)
    app.run()
    assert not app.exception and not any(BUSY in e.value for e in app.error)
    assert (
        0 < len(app.session_state["data"]) < rows
    )  # the population's rows, the excluded ones left out
