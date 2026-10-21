"""The Enrich draft: apply a current draft, redo or discard one built on older data or another population (AppTest)."""

import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def _banner_script(change: str) -> None:
    import pandas as pd
    import streamlit as st

    from core.state import DatasetRef
    from ui.components.staging import render_draft_banner
    from ui.state import get_state

    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        state.set_dataset(
            pd.DataFrame({"a": [1, 2, 3]}), DatasetRef(name="d", origin="upload", file_hash="h")
        )
        frame = state.working_frame()
        state.stage(
            frame.assign(b=frame["a"] * 2),
            "variable_computation",
            {"variable_name": "b", "formula": "a * 2"},
            "Computed new variable: b",
        )
        if change == "rename":  # the formula's column is renamed after the draft was started
            state.commit_change(
                state.base.rename(columns={"a": "A"}), "rename_columns", {}, "Renamed 1 column"
            )
        elif change == "added":  # the dataset gains a column after the draft was started
            state.commit_change(state.base.assign(c=0), "external_write", {}, "Added c")
    render_draft_banner()
    draft = state.working_frame()
    st.session_state["probe"] = {
        "columns": list(state.base.columns),
        "draft_columns": list(draft.columns),
        "status": state.staging_status().state,
        "steps": [s["function"] for s in state.transformation_manager.history],
    }


def _labels(at: AppTest) -> list[str]:
    return [b.label for b in at.button]


def _run(change: str) -> AppTest:
    at = AppTest.from_function(_banner_script, kwargs={"change": change}, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_a_current_draft_is_applied_as_traced_steps(data_root: Path) -> None:
    at = _run("none")
    assert at.session_state["probe"]["status"] == "ok"
    assert "Computed new variable: b" in " ".join(m.value for m in at.markdown)
    assert _labels(at) == ["Apply to the dataset", "Discard the draft"]
    at.button(key="draft_apply").click().run()
    probe = at.session_state["probe"]
    assert probe["columns"] == ["a", "b"] and probe["status"] == "empty"
    assert probe["steps"] == ["initial_load", "variable_computation"]


def test_a_draft_built_on_older_data_is_redone_on_the_current_data(data_root: Path) -> None:
    at = _run("added")
    assert at.session_state["probe"]["status"] == "base_changed"
    assert any("cannot be applied as it is" in w.value for w in at.warning)
    assert _labels(at) == [
        "Redo my 1 change on the current data",
        "Discard the draft and restart from the current data",
    ]
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    assert probe["status"] == "ok" and probe["draft_columns"] == ["a", "c", "b"]
    at.button(key="draft_apply").click().run()
    assert at.session_state["probe"]["columns"] == ["a", "c", "b"]


def test_a_change_that_no_longer_fits_the_data_can_only_be_discarded(data_root: Path) -> None:
    at = _run("rename")
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    assert probe["status"] == "base_changed"  # the formula refers to a column that was renamed
    assert any("could not" in w.value or "failed" in w.value for w in at.warning)
    at.button(key="draft_discard_stale").click().run()
    probe = at.session_state["probe"]
    # the rename is kept and the draft is gone
    assert probe["columns"] == ["A"] and probe["status"] == "empty"
    assert probe["steps"] == ["initial_load", "rename_columns"]


def _population_script(kind: str = "imputation") -> None:
    import pandas as pd
    import streamlit as st

    from core.population import criterion_id
    from core.state import DatasetRef
    from enrich.imputation_service import describe_formula_fill, fill_with_formula, impute_rows
    from ui.components.population import population_banner
    from ui.components.staging import render_draft_banner
    from ui.state import get_state

    state = get_state()
    adult = criterion_id("study_inclusion", "Adult")
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        st.session_state["config"] = {
            "mask_families": {
                "study_inclusion": {"operator": "all", "Adult": {"expression": "df['age'] >= 18"}}
            }
        }
        state.set_dataset(
            pd.DataFrame({"age": [10, 30, 50, 70], "bmi": [None, 20.0, None, 30.0]}),
            DatasetRef(name="d", origin="upload", file_hash="h"),
        )
        # unscoped and freeze: staged with no criterion ticked (the population is recorded all the same)
        if kind not in ("unscoped", "freeze"):
            state.set_ticked([adult])
        population = state.population
        if kind in (
            "formula",
            "two",
            "freeze",
        ):  # the mean of the rows used, 25, for each missing bmi
            formula = "bmi.mean() + 0 * age"
            filled, n = fill_with_formula(
                state.working_frame(), "bmi", formula, population.included
            )
            scoped = population.n_included if population.criteria else None
            state.stage(
                filled,
                "targeted_imputation",
                {"target_variable": "bmi", "formula": formula, "filled_count": n, "exact": False},
                describe_formula_fill("bmi", n, exact=False, n_rows=scoped),
                population=population,
            )
        else:
            outcome = impute_rows(state.working_frame(), population.included, numerical="Mean")
            state.stage(
                outcome.frame,
                "imputation",
                outcome.params,
                "Mean imputation",
                population=population,
            )
        if kind == "unscoped":  # a criterion ticked after the imputation was staged
            state.set_ticked([adult])
        elif (
            kind == "freeze"
        ):  # the population kept as the dataset: the draft was built on older data
            state.set_ticked([adult])
            state.freeze_population()
        else:  # the population changes after the imputation was fitted
            state.set_ticked([])
        if kind == "two":  # a second imputation, on the population of the time (every row)
            outcome = impute_rows(state.working_frame(), None, numerical="Mean")
            state.stage(
                outcome.frame, "imputation", outcome.params, "Mean", population=state.population
            )
    if st.session_state.get("apply_now"):
        st.session_state["apply_now"] = False
        state.commit_staging()
    render_draft_banner()
    population_banner()
    draft = state.session.draft
    step = draft.steps[-1] if draft is not None and draft.steps else None
    st.session_state["probe"] = {
        "status": state.staging_status().state,
        "draft_bmi": state.working_frame()["bmi"].tolist(),
        "params": dict(step.params) if step else None,
        "description": step.description if step else None,
    }


def test_an_imputation_fitted_on_another_population_is_refitted(data_root: Path) -> None:
    at = AppTest.from_function(_population_script, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"]["status"] == "population_changed"
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    assert probe["status"] == "ok"
    bmi = probe["draft_bmi"]
    # refitted on every row: the child (excluded before) is imputed too, like the other missing value
    assert not any(math.isnan(v) for v in bmi) and bmi[0] == bmi[2] and bmi[1::2] == [20.0, 30.0]
    # the trace will record what the refit did
    params = probe["params"]
    assert params["rows"] == "all" and params["n_rows_fitted"] == 4
    assert params["imputed_values"] == {"bmi": 2}
    assert probe["description"] == "Applied data imputation (Mean / Most_Frequent)"


def test_a_targeted_imputation_on_another_population_is_recomputed(data_root: Path) -> None:
    at = AppTest.from_function(_population_script, kwargs={"kind": "formula"}, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    probe = at.session_state["probe"]
    assert probe["status"] == "population_changed" and probe["params"]["filled_count"] == 1
    assert math.isnan(probe["draft_bmi"][0]) and probe["draft_bmi"][2] == 25.0
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    assert probe["status"] == "ok" and probe["draft_bmi"] == [25.0, 20.0, 25.0, 30.0]
    # its count, which the replay compares, and its description are those of the recomputation
    assert probe["params"]["filled_count"] == 2
    assert probe["description"] == "Imputed 2 values in bmi using a formula"


def test_an_imputation_staged_before_a_criterion_was_ticked_is_refitted(data_root: Path) -> None:
    # biostatistics review: staged with no criterion ticked, it was applied on every row without a word
    at = AppTest.from_function(_population_script, kwargs={"kind": "unscoped"}, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"]["status"] == "population_changed"
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    # refitted on the adults: the child keeps its missing value, the adult's is filled
    bmi = probe["draft_bmi"]
    assert probe["status"] == "ok" and math.isnan(bmi[0]) and not math.isnan(bmi[2])
    assert bmi[1::2] == [20.0, 30.0]


def test_each_imputation_of_the_draft_must_be_on_the_current_population(data_root: Path) -> None:
    # biostatistics review: the draft remembered the last population only
    at = AppTest.from_function(_population_script, kwargs={"kind": "two"}, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # the second imputation is on the current population, the first is not
    assert at.session_state["probe"]["status"] == "population_changed"


def test_a_draft_redone_on_new_data_is_recorded_as_redone(data_root: Path) -> None:
    # architecture review: the redo kept the first fill's count and population (a second click, a replay warning)
    at = AppTest.from_function(_population_script, kwargs={"kind": "freeze"}, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    probe = at.session_state["probe"]
    assert probe["status"] == "base_changed" and probe["params"]["filled_count"] == 2
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    # the three adults kept: one missing value left, filled with their mean
    assert probe["status"] == "ok" and probe["draft_bmi"] == [20.0, 25.0, 30.0]
    assert probe["params"]["filled_count"] == 1
    assert probe["description"] == "Imputed 1 value in bmi using a formula"


def _applied_script(change: bool, scoped: bool = True, kind: str = "statistical") -> None:
    import pandas as pd
    import streamlit as st

    from core.population import criterion_id
    from core.state import DatasetRef
    from enrich.formulas import row_wise
    from enrich.imputation_service import fill_with_formula, impute_rows
    from ui.components.population import population_banner
    from ui.state import get_state

    state = get_state()
    adult = criterion_id("study_inclusion", "Adult")
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        st.session_state["config"] = {
            "mask_families": {
                "study_inclusion": {"operator": "all", "Adult": {"expression": "df['age'] >= 18"}}
            }
        }
        state.set_dataset(
            pd.DataFrame({"age": [10, 30, 50, 70], "bmi": [None, 20.0, None, 30.0]}),
            DatasetRef(name="d", origin="upload", file_hash="h"),
        )
        if scoped:
            state.set_ticked([adult])
        population = state.population
        if kind == "statistical":
            outcome = impute_rows(state.working_frame(), population.included, numerical="Mean")
            state.stage(
                outcome.frame, "imputation", outcome.params, "Imputation", population=population
            )
        else:  # a formula of each row's own values, or one of other rows (a mean)
            formula = "age / 2" if kind == "row-wise" else "bmi.mean() + 0 * age"
            filled, n = fill_with_formula(
                state.working_frame(), "bmi", formula, population.included
            )
            params = {"target_variable": "bmi", "formula": formula, "row_wise": row_wise(formula)}
            state.stage(filled, "targeted_imputation", params, "Imputation", population=population)
        state.commit_staging()
        # on the adults: the analyses now include the child, whose missing value was not imputed;
        # on every row: the analyses leave out the child, who was among the rows computed on
        if change:
            state.set_ticked([] if scoped else [adult])
    population_banner()


@pytest.mark.parametrize("kind", ["statistical", "aggregate", "row-wise"])
@pytest.mark.parametrize("scoped", [True, False])
@pytest.mark.parametrize("change", [False, True])
def test_imputed_values_from_another_population_are_flagged_on_analysis_pages(
    data_root: Path, change: bool, scoped: bool, kind: str
) -> None:
    at = AppTest.from_function(
        _applied_script,
        kwargs={"change": change, "scoped": scoped, "kind": kind},
        default_timeout=120,
    ).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    flagged = [w.value for w in at.warning if "computed on other rows" in w.value]
    # a formula of each row's own values matters only for rows gained since, which it did not fill
    assert bool(flagged) is (change and not (kind == "row-wise" and not scoped))
    if flagged and not scoped:  # an imputation on every row names its rows
        assert "Imputation (all 4 rows)" in flagged[0]


def _left_out_script() -> None:
    import pandas as pd
    import streamlit as st

    from core.population import criterion_id
    from core.state import DatasetRef
    from enrich.imputation_service import describe_formula_fill, fill_with_formula
    from ui.components.staging import render_draft_banner
    from ui.state import get_state

    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        st.session_state["config"] = {
            "mask_families": {
                "study_inclusion": {"operator": "all", "Adult": {"expression": "df['age'] >= 18"}}
            }
        }
        state.set_dataset(
            pd.DataFrame({"age": [10, 30, 50, 70], "bmi": [None, 20.0, 25.0, 30.0]}),
            DatasetRef(name="d", origin="upload", file_hash="h"),
        )
        population = state.population  # no criterion ticked: the child's bmi is the one missing
        formula = "18 + 0 * age"
        filled, n = fill_with_formula(state.working_frame(), "bmi", formula, population.included)
        state.stage(
            filled,
            "targeted_imputation",
            {"target_variable": "bmi", "formula": formula, "filled_count": n, "exact": True},
            describe_formula_fill("bmi", n, exact=True),
            population=population,
        )
        state.set_ticked([criterion_id("study_inclusion", "Adult")])
    render_draft_banner()
    st.session_state["probe"] = {"status": state.staging_status().state}


def test_a_refit_that_changes_nothing_says_what_it_left_out(data_root: Path) -> None:
    # architecture review: the step was dropped, and the banner said "1 change redone"
    at = AppTest.from_function(_left_out_script, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"]["status"] == "population_changed"
    at.button(key="draft_rebase").click().run()
    assert at.session_state["probe"]["status"] == "empty"
    shown = " ".join(s.value for s in at.success)
    assert (
        "The draft is empty: 'Imputed 1 value in bmi using an exact formula' changes nothing on "
        "the current data." in shown
    )


def _enrich_page_script() -> None:
    import pandas as pd
    import streamlit as st

    from core.state import DatasetRef
    from ui.state import get_state

    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        state.set_dataset(
            pd.DataFrame({"a": [1, 2, 3]}), DatasetRef(name="d", origin="upload", file_hash="h")
        )
        frame = state.working_frame()
        state.stage(
            frame.assign(b=frame["a"] * 2),
            "variable_computation",
            {"variable_name": "b", "formula": "a * 2"},
            "Computed new variable: b",
        )
    from app_pages.data_enrichment import app

    app()


def test_the_draft_banner_is_at_the_top_of_the_enrich_page(data_root: Path) -> None:
    """It used to be rendered after the tabs, so it appeared at the bottom of the page, under every tab."""
    at = AppTest.from_function(_enrich_page_script, default_timeout=180)
    at.session_state["username"] = "draft_top"
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    first, second = list(at.main.children.values())[:2]
    assert "draft_apply" in [b.key for b in first.button]  # Apply to the dataset, above the tabs
    assert second.type == "tab_container"


def _snapshots_script(action: str) -> None:
    from pathlib import Path

    import pandas as pd
    import streamlit as st

    from core.state import DatasetRef
    from ui.state import get_state
    from utils.data_paths import get_traces_dir

    state = get_state()
    # no user is signed in: the snapshots folder of the data folder (save_snapshot)
    folder = Path(get_traces_dir()) / "snapshots"
    folder.mkdir(exist_ok=True)

    def snapshot(name: str, frame: pd.DataFrame) -> str:
        path = folder / f"{name}.parquet"
        frame.to_parquet(path)
        return str(path)

    state.set_dataset(
        pd.DataFrame({"a": [1, 2, 3]}), DatasetRef(name="d", origin="upload", file_hash="h")
    )
    frame = state.working_frame()
    staged = frame.assign(b=frame["a"] * 2)
    params = {"variable_name": "b", "formula": "a * 2"}
    state.stage(
        staged,
        "variable_computation",
        params,
        "Computed new variable: b",
        output_dataset_path=snapshot("staged", staged),
    )
    if action == "no_change":  # the same table again: the change is not added to the draft
        again = state.working_frame()
        state.stage(
            again, "variable_computation", params, "b", output_dataset_path=snapshot("same", again)
        )
    elif action == "same_file":  # a change that changes nothing names a file the draft still uses
        state.stage(
            state.working_frame(),
            "variable_computation",
            params,
            "b",
            output_dataset_path=str(folder / "staged.parquet"),
        )
    elif action == "discard":
        state.discard_staging()
    elif action == "rebase":  # the dataset changes underneath: the change is redone on the new data
        state.commit_change(state.base.assign(c=0), "external_write", {}, "Added c")
        assert state.rebase_staging()[0]
    elif action == "load":  # another dataset while the draft is pending
        state.set_dataset(
            pd.DataFrame({"x": [1]}), DatasetRef(name="e", origin="upload", file_hash="h2")
        )
    elif action == "clear":
        state.clear_dataset()
    elif action == "apply":
        state.commit_staging()
    elif action == "error":  # the change fails before it joins the draft

        def fail(*args, **kwargs):
            raise RuntimeError("the change failed")

        state.session.stage = fail
        try:
            state.stage(
                staged,
                "variable_computation",
                params,
                "b",
                output_dataset_path=snapshot("x", staged),
            )
        except RuntimeError:
            pass
    st.session_state["probe"] = {
        "left": sorted(p.name for p in folder.glob("*.parquet")),
        "traced": [s.get("output_dataset_path") for s in state.transformation_manager.history],
    }


@pytest.mark.parametrize(
    ("action", "left"),
    [
        ("none", ["staged.parquet"]),
        ("no_change", ["staged.parquet"]),
        ("same_file", ["staged.parquet"]),
        ("discard", []),
        ("rebase", []),
        ("load", []),
        ("clear", []),
        ("apply", ["staged.parquet"]),
        ("error", ["staged.parquet"]),
    ],
)
def test_snapshots_of_changes_that_never_reach_the_trace_are_deleted(
    data_root: Path, action: str, left: list[str]
) -> None:
    """Each change of the draft writes a copy of the table; one that is never applied left it behind."""
    at = AppTest.from_function(
        _snapshots_script, kwargs={"action": action}, default_timeout=120
    ).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    probe = at.session_state["probe"]
    assert probe["left"] == left
    if action == "apply":  # the trace names the snapshot it keeps
        assert probe["traced"][-1] == "traces/snapshots/staged.parquet"


def _signed_in_script() -> None:
    import pandas as pd
    import streamlit as st

    from core.state import DatasetRef
    from ui.shell import _user_menu
    from ui.state import get_state
    from utils.session_scope import get_current_scope

    if not st.session_state.get("username"):
        return  # signed out: the session was reset
    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        state.set_dataset(
            pd.DataFrame({"a": [1, 2, 3]}), DatasetRef(name="d", origin="upload", file_hash="h")
        )
        frame = state.working_frame()
        staged = frame.assign(b=frame["a"] * 2)
        scope = get_current_scope()
        assert scope is not None
        path = scope.snapshots_dir() / "staged.parquet"
        staged.to_parquet(path)
        state.stage(staged, "variable_computation", {}, "b", output_dataset_path=str(path))
    _user_menu(state)


def test_signing_out_deletes_the_snapshots_of_a_pending_draft(data_root: Path) -> None:
    at = AppTest.from_function(_signed_in_script, default_timeout=120)
    at.session_state["username"] = "alice"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    folder = data_root / "users" / "alice" / "traces" / "snapshots"
    assert [p.name for p in folder.glob("*.parquet")] == ["staged.parquet"]
    at.button(key="shell_logout").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert list(folder.glob("*.parquet")) == []


def _fill_rule_script(derived: bool = False) -> None:
    import pandas as pd
    import streamlit as st

    from core.derived import DerivedVariable
    from core.population import criterion_id
    from core.state import DatasetRef
    from enrich.imputation_service import describe_formula_fill, fill_with_formula
    from ui.components.staging import render_draft_banner
    from ui.state import get_state

    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        rule = {"expression": "df['weight_kg'] > 300"}
        st.session_state["config"] = {
            "mask_families": {"clinical_anomalies": {"operator": "any", "Implausible weight": rule}}
        }
        state.set_dataset(
            pd.DataFrame(
                {
                    "weight_kg": [70.0, None, 80.0, 60.0],
                    "height_cm": [170.2, 150.0, 180.4, 160.1],  # not codes: whole numbers would be
                    "bmi": [
                        24.2,
                        160.0,
                        24.7,
                        None,
                    ],  # a BMI typed 160: its weight comes out at 360 kg
                    "bmi_x2": [48.4, None, 49.4, None],
                }
            ),
            DatasetRef(name="d", origin="upload", file_hash="h"),
        )
        state.set_ticked([criterion_id("clinical_anomalies", "Implausible weight")])
        population = state.population
        if derived:  # four rows: too few to check, so accepted, as a user may
            state.set_derived_variables(
                [
                    DerivedVariable("weight_kg", "bmi * (height_cm / 100) ** 2", accepted=True),
                    DerivedVariable("bmi_x2", "bmi * 2", accepted=True),
                ]
            )
            assert state.fill_exactly() == 2
        else:
            frame = state.working_frame()
            for target, formula in (
                ("weight_kg", "bmi * (height_cm / 100) ** 2"),
                ("bmi_x2", "bmi * 2"),
            ):
                frame, n = fill_with_formula(
                    frame, target, formula, population.included, complete_inputs=True
                )
                params = {
                    "target_variable": target,
                    "formula": formula,
                    "filled_count": n,
                    "exact": True,
                    "complete_inputs": True,
                }
                state.stage(
                    frame,
                    "targeted_imputation",
                    params,
                    describe_formula_fill(target, n, exact=True),
                    population=population,
                )
        # the dataset changes underneath: the draft was built on older data
        state.commit_change(state.base.assign(c=0), "external_write", {}, "Added c")
    render_draft_banner()
    draft = state.session.draft
    st.session_state["probe"] = {
        "status": state.staging_status().state,
        "bmi_x2": state.working_frame()["bmi_x2"].tolist(),
        "steps": len(draft.steps) if draft is not None else 0,
    }


def test_a_redone_draft_fills_the_rows_it_filled_before(data_root: Path) -> None:
    # the first fill makes a ticked criterion fire (360 kg): the second still fills on the population the draft was
    # built on, as on the page, instead of on the table the first fill changed
    at = AppTest.from_function(_fill_rule_script, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"]["status"] == "base_changed"
    at.button(key="draft_rebase").click().run()
    probe = at.session_state["probe"]
    assert probe["status"] == "ok" and probe["steps"] == 2
    assert probe["bmi_x2"][1] == 320.0


def test_a_draft_filled_exactly_is_filled_again_not_redone(data_root: Path) -> None:
    at = AppTest.from_function(
        _fill_rule_script, kwargs={"derived": True}, default_timeout=120
    ).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"]["status"] == "base_changed"
    assert _labels(at) == ["Discard the draft and restart from the current data"]


def _declined_script() -> None:
    import pandas as pd
    import streamlit as st

    from core.population import criterion_id
    from core.state import DatasetRef
    from enrich.imputation_service import impute_rows
    from ui.components.staging import render_draft_banner
    from ui.state import get_state

    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        adult = {"expression": "df['age'] >= 18"}
        st.session_state["config"] = {
            "mask_families": {"study_inclusion": {"operator": "all", "Adult": adult}}
        }
        state.set_dataset(
            pd.DataFrame({"age": [10, 30, 50, 70], "bmi": [None, 20.0, None, 30.0]}),
            DatasetRef(name="d", origin="upload", file_hash="h"),
        )
        state.set_ticked([criterion_id("study_inclusion", "Adult")])
        outcome = impute_rows(state.working_frame(), state.population.included, numerical="Mean")
        # derived variables declared, their computation declined: a plain imputation
        params = {**outcome.params, "derived": {"recompute": False, "definitions": []}}
        state.stage(
            outcome.frame, "imputation", params, "Mean imputation", population=state.population
        )
        state.set_ticked([])  # the population changes after the imputation was fitted
    render_draft_banner()
    st.session_state["probe"] = {"status": state.staging_status().state}


def test_an_imputation_declining_the_derived_variables_can_be_refitted(data_root: Path) -> None:
    at = AppTest.from_function(_declined_script, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["probe"]["status"] == "population_changed"
    assert "Refit on the current population" in _labels(at)
    at.button(key="draft_rebase").click().run()
    assert at.session_state["probe"]["status"] == "ok"
