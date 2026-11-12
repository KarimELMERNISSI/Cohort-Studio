"""
The Survival & regression page, organised by the outcome, and the Tier C tab run end to end on the sample cohort
(AppTest): each outcome reads its comparison's effect first, one Run fits the curves and the Cox model, each panel
keeps its choices while another is open, and only the open one runs.
"""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from app_pages import analysis_survival as page
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "tierc_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_sample").click().run()
    assert at.session_state["data"].shape == (500, 19)
    yield at
    dp.DATA_ROOT = previous


def _stored(at: AppTest, key: str):
    """A result kept by the result store (ui/results.py)."""
    return at.session_state["_results"][key].value


def _labels(at: AppTest) -> list[str]:
    return [m.label for m in at.metric]


def _outcome(at: AppTest, outcome: str) -> None:
    at.session_state[page.OUTCOME_KEY] = outcome
    at.run()
    _ok(at)


def test_one_run_draws_the_curves_and_fits_the_cox_model(app: AppTest) -> None:
    app.switch_page(nav.page_path("survival")).run()
    _ok(app)
    assert any(
        "Before you run: the plan" in m.value for m in app.markdown
    )  # Guidance is on by default
    app.selectbox(key="surv_time").select("fu_days").run()
    app.selectbox(key="surv_event").select("died").run()
    app.selectbox(key="surv_group").select("arm").run()
    app.multiselect(key="surv_adjust").select("age").run()
    assert any("A Cox model comparing `arm`, adjusted for `age`" in m.value for m in app.markdown)
    app.button(key="surv_run").click().run()
    _ok(app)
    curves = _stored(app, "survival")
    assert {c.label for c in curves.curves} == {"Placebo", "Treatment"}
    assert (
        curves.logrank_p is not None and curves.logrank_p < 0.05
    )  # placebo shorter by construction
    table = _stored(app, "cox").table
    assert set(table["term"]) == {"arm_Treatment", "age"}  # the comparison and the adjustment
    assert float(table.loc[table["term"] == "arm_Treatment", "hr"].iloc[0]) < 1
    labels = _labels(app)
    assert labels[0] == "Hazard ratio · Treatment vs Placebo"  # the effect first
    assert "Log-rank test (unadjusted)" in labels  # beside an adjusted ratio
    assert any("rows in the Cox model" in c.value for c in app.caption)  # the ratio's own rows
    assert any("Adjusted for `age`" in c.value for c in app.caption)
    spec = app.get("plotly_chart")[0].proto.spec
    assert "At risk" in spec  # the numbers at risk under the curves
    assert (
        sum("Computed on" in c.value for c in app.caption) == 2
    )  # provenance under each stored result
    assert {"report_add::survival", "report_add::cox"} <= {b.key for b in app.button}
    assert {"**Kaplan-Meier curves**", "**Cox model**"} <= {
        c.value for c in app.caption
    }  # each button's result


def test_the_binary_outcome_reads_its_odds_ratio_first(app: AppTest) -> None:
    _outcome(app, "binary")
    app.selectbox(key="logit_outcome").select("event").run()
    app.selectbox(key="logit_compare").select("arm").run()
    app.multiselect(key="logit_adjust").select("age").select("sex").run()
    app.button(key="logit_run").click().run()
    _ok(app)
    fitted = _stored(app, "logistic")
    assert {"arm_Treatment", "age", "sex_M"} == set(fitted.table["term"])
    assert _labels(app)[0] == "Odds ratio · Treatment vs Placebo"
    assert any("Adjusted for `age` and `sex`" in c.value for c in app.caption)
    assert any(
        c.value.startswith("Odds of `event` = `") for c in app.caption
    )  # what the odds are of
    assert any(e.label == "Model details: every term" for e in app.expander)


def test_the_count_outcome_compares_rates_per_person_time(app: AppTest) -> None:
    _outcome(app, "count")
    app.selectbox(key="count_outcome").select("sbp").run()
    app.selectbox(key="count_time").select("fu_days").run()
    app.selectbox(key="count_compare").select("arm").run()
    assert any(
        "A Poisson model of `sbp` per `fu_days` of person-time" in m.value for m in app.markdown
    )
    app.button(key="count_run").click().run()
    _ok(app)
    assert _stored(app, "count").kind == "poisson"
    assert _labels(app)[0] == "Incidence-rate ratio · Treatment vs Placebo"
    app.selectbox(key="count_time").select("(none)").run()
    app.button(key="count_run").click().run()
    _ok(app)
    assert (
        _labels(app)[0] == "Ratio of mean counts · Treatment vs Placebo"
    )  # no person-time: no rates
    app.selectbox(key="count_time").select("fu_days").run()


def test_each_stored_result_keeps_the_spec_that_computes_it_again(app: AppTest) -> None:
    """After the three runs above: each result is stored with its analysis spec (biostats.plan), and the spec run on
    the analysis population gives the same numbers, which is what a study package's check does."""
    from biostats import plan

    frame = app.session_state["working_df"]
    store = app.session_state["_results"]
    trace = app.session_state["transformation_manager"]
    for key in ("survival", "cox", "logistic", "count"):
        stored = store[key]
        spec = stored.spec
        assert spec is not None and spec.kind == key and not spec.pooled, key
        again = plan.run(spec, frame)
        assert (
            plan.record(key, again.result).to_json() == plan.record(key, stored.value).to_json()
        ), key
        assert again.spec.decided == spec.decided and again.spec.backend == spec.backend, key
        assert dict(again.derived) == dict(spec.decided), (
            key
        )  # the data still give what was recorded
        on = spec.data
        assert on.mode == "observed" and on.imputation is None
        assert on.n_analysed == stored.stamp.n_analysed and on.n_population == len(frame)
        assert on.data_fp == stored.stamp.data_fp and len(on.rows_fp) >= 16
        assert len(on.protocol_hash) == 64 and on.rules_version >= 2
        assert on.trace_head == stored.stamp.trace_head and 0 < on.trace_steps <= len(trace.history)
        # what a package meant to travel may say of it: no fingerprint of rows, no head of the trace
        shared = on.shareable()
        assert set(shared) == {
            "mode",
            "data_fp",
            "n_population",
            "n_analysed",
            "rules_version",
            "trace_steps",
        }
    cox = store["cox"].spec
    assert cox.variables == {
        "time": "fu_days",
        "event": "died",
        "compare": "arm",
        "adjust": ["age"],
    }
    assert cox.options == {"alpha": 0.05, "ties": "efron"}
    assert cox.decided["levels"] == {"arm": ["Placebo", "Treatment"]}
    assert store["logistic"].spec.variables["adjust"] == ["age", "sex"]
    assert store["count"].spec.options["family"] == "poisson"
    assert store["count"].spec.variables["person_time"] is None  # the last run had no person-time
    # the settings name the comparison and the adjustment apart: two requests of one model are two results
    assert store["cox"].config == ("fu_days", "died", "arm", ("age",))


def test_each_outcome_keeps_its_choices_and_only_the_open_one_runs(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    _outcome(app, "time_to_event")
    assert app.selectbox(key="surv_group").value == "arm"
    assert app.multiselect(key="surv_adjust").value == ["age"]
    assert (
        _labels(app)[0] == "Hazard ratio · Treatment vs Placebo"
    )  # the stored result, shown again
    calls: list[str] = []
    for name in ("_time_to_event", "_binary", "_count"):
        original = getattr(page, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(page, name, counted)
    _outcome(app, "binary")
    assert calls == ["_binary"]
    assert app.selectbox(key="logit_compare").value == "arm"


def test_guidance_can_be_turned_off(app: AppTest) -> None:
    app.session_state[page.GUIDANCE_KEY] = False
    app.run()
    _ok(app)
    assert not any("Before you run: the plan" in m.value for m in app.markdown)
    app.session_state[page.GUIDANCE_KEY] = True
    app.run()


def test_the_plan_is_kept_until_a_choice_changes(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Computed on every rerun, a plan took 0.5 to 0.7 s on 100,000 rows at each click."""
    calls: list[int] = []
    original = page.plan_time_to_event

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(page, "plan_time_to_event", counted)
    _outcome(app, "time_to_event")
    before = len(calls)
    app.run()
    assert len(calls) == before  # nothing changed: the plan kept
    app.multiselect(key="surv_adjust").set_value(["age", "sex"]).run()
    assert len(calls) == before + 1  # a choice changed: planned again
    app.multiselect(key="surv_adjust").set_value(["age"]).run()


def test_paired_rates_tab_and_multiple_imputation_render(app: AppTest) -> None:
    app.switch_page(nav.page_path("analyze")).run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    app.button(key="mh_run").click().run() if any(b.key == "mh_run" for b in app.button) else None
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    app.switch_page(nav.page_path("enrich")).run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    assert any(b.key == "mi_run" for b in app.button), (
        "multiple imputation control missing on the Enrich page"
    )


def test_with_imputed_data_only_its_variables_are_offered() -> None:
    """A variable created after the imputation is not in it: comparing by it stopped the page (KeyError)."""
    from types import SimpleNamespace

    import pandas as pd

    df = pd.DataFrame(
        {
            "t": [1.0, 2.0, 3.0],
            "e": [1, 0, 1],
            "arm": ["a", "b", "a"],
            "made_after": ["x", "y", "x"],
        }
    )
    analyzer = SimpleNamespace(numeric_cols=["t"], binary_cols=["e", "arm", "made_after"])
    mi = SimpleNamespace(base=df.drop(columns=["made_after"]))
    numeric, binary, groups = page._variables(df, analyzer, mi)
    assert "made_after" not in [*numeric, *binary, *groups] and "arm" in groups
    assert "made_after" in page._variables(df, analyzer, None)[2]


@pytest.mark.parametrize(
    ("levels", "event"),
    [
        (["Died", "Survived"], "Died"),
        (["No", "Yes"], "Yes"),
        (["Event", "No event"], "Event"),
        (["A", "B"], "B"),
    ],
)
def test_the_event_level_is_the_one_that_reads_as_an_event(levels, event) -> None:
    """The last sorted level took "Survived" for Died/Survived and "No event" for Event/No event."""
    assert levels[page._event_index(levels)] == event


def _plan_script() -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.analysis_survival import _render_plan
    from core.state import DatasetRef
    from ui.guidance.outcomes import plan_time_to_event
    from ui.state import TUTORIAL_COHORT_PATH, get_state

    st.session_state.setdefault("username", "plan_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        frame = pd.read_csv(TUTORIAL_COHORT_PATH)
        state.set_dataset(
            frame,
            DatasetRef(name="tutorial", origin="upload", file_hash="tutorial"),
            source_bytes=TUTORIAL_COHORT_PATH.read_bytes(),
            source_name="tutorial_cohort.csv",
        )
    st.session_state.setdefault("surv_group", "site")
    st.session_state.setdefault("surv_adjust", ["hba1c_pct"])
    df = state.base
    plan = plan_time_to_event(df, "fu_days", "death", "site", ["hba1c_pct"], group=True)
    _render_plan(plan, keys=("surv_group", "surv_adjust"), candidates=list(df.columns))


def test_the_plan_offers_to_clean_a_variable_once_whichever_steps_read_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tutorial cohort compared by `site` and adjusted for `hba1c_pct`: the page stopped with a duplicate key."""
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    at = AppTest.from_function(_plan_script, default_timeout=120)
    at.run()
    _ok(at)
    opened = [b for b in at.button if str(b.key).startswith("clean_open::")]
    assert [b.key for b in opened] == ["clean_open::surv_group::site"]


def test_a_text_variable_written_several_ways_is_offered_by_its_values_set_aside_spaces_and_capitals() -> (
    None
):
    """The tutorial's `sex` holds 10 spellings of 4 values, more than the 8 a model offers: it was in neither "Compare
    by" nor "Adjust for", and its plan could never offer to clean it. A column of names is still left out."""
    import pandas as pd

    from ingest.analyzer import DataAnalyzer
    from ui.state import TUTORIAL_COHORT_PATH

    df = pd.read_csv(TUTORIAL_COHORT_PATH)
    df["name"] = [f"Person {i}" for i in range(len(df))]
    offered = page._group_candidates(df, DataAnalyzer(df))
    assert {"sex", "site"} <= set(offered) and "name" not in offered


def test_rows_with_a_follow_up_time_below_zero_are_no_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They were counted with the rows that lack a value ("2 not used: missing values"), which a shared flow gives
    exactly: they are a count of participants with a characteristic, like a group too small to draw."""
    import numpy as np
    import pandas as pd

    from biostats.survival import compare_survival, cox_regression
    from core.attrition import not_used_words

    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "days": rng.exponential(300, 200).round() + 1,
            "died": rng.integers(0, 2, 200),
            "arm": ["control", "drug"] * 100,
            "age": rng.normal(60, 8, 200).round(1),
        }
    )
    df.loc[[3, 8], "days"] = -5.0  # two follow-up times below zero
    df.loc[11, "age"] = np.nan  # one missing value
    df.loc[20, "arm"] = "Luxembourgg"  # a group of one row
    curves = compare_survival(df, "days", "died", "arm")
    assert page.left_out_of(curves, pooled=False) == (
        3,
        "in a group too small to draw or with a follow-up time below zero",
    )
    analysed = sum(c.n for c in curves.curves)
    assert not_used_words(200, analysed, n_left_out=3, left_out="left out") == ("3 left out",)
    cox = cox_regression(df, "days", "died", ["arm", "age"])
    n_left_out, words = page.left_out_of(cox, pooled=False)
    assert (n_left_out, words) == (2, "with a follow-up time below zero")
    assert not_used_words(200, cox.n, n_left_out=n_left_out, left_out=words) == (
        "2 with a follow-up time below zero",
        "1 not used: missing values",
    )
    # a pooled result counts what its imputed datasets left out (it counted none: the curves then held fewer rows
    # than the flow said were analysed)
    from biostats.mi import MIData
    from biostats.plan import run, specify

    filled = df.copy()
    filled.loc[11, "age"] = 61.0
    mi = MIData.from_completed(df, [filled.copy() for _ in range(5)], methods={"age": "pmm"})
    pooled = run(specify("survival", pooled=True, time="days", event="died", group="arm"), mi=mi)
    assert page.left_out_of(pooled.result, pooled=True) == (
        3,
        "in a group too small to draw or with a follow-up time below zero",
    )
    assert page.left_out_of(curves, pooled=True) == (0, "")  # not a pooled result: nothing to read
    # and the number it analysed is every row but those: the page read 200 beside "3 in a group too small to
    # draw or with a follow-up time below zero"
    kept: dict = {}
    monkeypatch.setattr(page, "remember", lambda key, result, **settings: kept.update(settings))
    monkeypatch.setattr(page, "mark_step", lambda *args, **kwargs: None)
    page._fit_curves(df, mi, "survival_mi", {}, "days", "died", "arm")
    assert (kept["n_analysed"], kept["n_left_out"]) == (197, 3)
    page._fit_cox(df, mi, "cox_mi", {}, "days", "died", "arm", ["age"])
    assert (kept["n_analysed"], kept["n_left_out"]) == (198, 2)
    assert (
        page.analysed_by(curves, pooled=False) == 197
        and page.analysed_by(cox, pooled=False) == cox.n
    )
    assert page.analysed_by(pooled.result, pooled=True) == 197
    # the event is missing for 20 rows, and no imputation fills it: every dataset reads 20 rows fewer, which the
    # page counted as analysed (they are missing values)
    lacking = df.copy()
    lacking["died"] = lacking["died"].astype(float)
    lacking.loc[30:49, "died"] = np.nan
    filled = lacking.copy()
    filled.loc[11, "age"] = 61.0
    mi = MIData.from_completed(lacking, [filled.copy() for _ in range(5)], methods={"age": "pmm"})
    page._fit_curves(lacking, mi, "survival_mi", {}, "days", "died", "arm")
    assert (kept["n_analysed"], kept["n_left_out"]) == (177, 3)
    page._fit_cox(lacking, mi, "cox_mi", {}, "days", "died", "arm", ["age"])
    assert (kept["n_analysed"], kept["n_left_out"]) == (178, 2)
    assert not_used_words(200, 177, n_left_out=3, left_out="left out") == (
        "3 left out",
        "20 not used: missing values",
    )
