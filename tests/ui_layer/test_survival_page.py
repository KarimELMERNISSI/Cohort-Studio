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
