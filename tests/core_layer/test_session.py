"""The session's data: one write path, an Enrich draft that cannot overwrite newer data, toggleable populations."""

import pandas as pd
import pytest

from core.population import criterion_id
from core.session import (
    BASE_CHANGED,
    EMPTY,
    OK,
    POPULATION_CHANGED,
    DataSession,
    PopulationError,
    StagingError,
    StaleBaseError,
    is_exact_imputation,
)

CONFIG = {
    "mask_families": {
        "clinical_anomalies": {
            "operator": "any",
            "Implausible age": {"expression": "df['age'] > 110"},
        },
        "study_inclusion": {"operator": "all", "Adult": {"expression": "df['age'] >= 18"}},
    }
}
ANOMALY = criterion_id("clinical_anomalies", "Implausible age")
ADULT = criterion_id("study_inclusion", "Adult")


class Sink:
    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add_step(self, function_name, params, description, output_dataset_path=None, **extra):
        self.steps.append({"function": function_name, "params": params, **extra})


@pytest.fixture
def session() -> DataSession:
    data = DataSession(trace=Sink())
    data.install(
        pd.DataFrame({"age": [12, 40, 150, 60, 30], "bmi": [20.0, None, 25.0, None, 22.0]}),
        params={"source": "test"},
    )
    return data


def steps(session: DataSession) -> list[dict]:
    return session.trace.steps  # type: ignore[union-attr]


def test_install_records_the_load_with_what_it_contains(session: DataSession) -> None:
    (load,) = steps(session)
    assert load["function"] == "initial_load" and load["data"]["rows_out"] == 5
    assert session.base.index.name == "__row_id" and session.revision == 0


def test_a_commit_is_one_traced_step_and_a_no_op_is_not_recorded(session: DataSession) -> None:
    fp = session.data_fp
    assert session.commit(session.base.copy(), "noop", {}, "nothing") is None
    renamed = session.base.rename(columns={"bmi": "bmi_kg_m2"})
    change = session.commit(renamed, "rename_columns", {"mapping": {"bmi": "bmi_kg_m2"}}, "Renamed")
    assert change is not None and change.fp_in == fp and change.columns_added == ("bmi_kg_m2",)
    assert [s["function"] for s in steps(session)] == ["initial_load", "rename_columns"]
    assert session.revision == 1
    with pytest.raises(StaleBaseError):
        session.commit(renamed.assign(x=1), "late", {}, "late", expected_fp=fp)


def test_criteria_filter_without_deleting_rows(session: DataSession) -> None:
    session.set_ticked([ANOMALY, ADULT])
    population = session.population(CONFIG)
    assert [(s.n_in, s.n_out) for s in population.stages] == [(5, 4), (4, 3)]
    assert list(session.analysis_frame(CONFIG).index) == [1, 3, 4]
    assert len(session.base) == 5  # nothing deleted
    session.tick(ADULT, False)
    assert list(session.analysis_frame(CONFIG).index) == [0, 1, 3, 4]


def test_the_identity_follows_rows_and_data(session: DataSession) -> None:
    before = session.identity(CONFIG)
    session.set_ticked([ANOMALY])
    after = session.identity(CONFIG)
    assert after.population_fp != before.population_fp and after.data_fp == before.data_fp
    assert after.population_label == "4 of 5 rows (1 criterion)"
    session.set_ticked([])
    assert session.identity(CONFIG) == before


def test_a_broken_ticked_criterion_blocks_the_analysis(session: DataSession) -> None:
    broken = {"mask_families": {"x": {"operator": "any", "Typo": {"expression": "df['agee'] > 1"}}}}
    session.set_ticked([criterion_id("x", "Typo")])
    with pytest.raises(PopulationError):
        session.analysis_frame(broken)


def test_the_draft_reaches_the_base_only_when_applied(session: DataSession) -> None:
    start = session.data_fp
    session.stage(
        session.draft_frame().assign(bmi_sq=1.0), "variable_computation", {"v": 1}, "BMI squared"
    )
    assert session.data_fp == start and "bmi_sq" not in session.base.columns
    assert session.staging_status(CONFIG).state == OK
    assert [s["function"] for s in steps(session)] == ["initial_load"]  # not traced yet
    session.commit_staging(CONFIG)
    assert "bmi_sq" in session.base.columns and session.draft is None
    applied = steps(session)[-1]
    assert applied["function"] == "variable_computation" and applied["data"]["fp_in"] == start
    assert applied["recorded_at"]
    assert session.staging_status(CONFIG).state == EMPTY


def test_a_draft_built_on_older_data_cannot_overwrite_it(session: DataSession) -> None:
    session.stage(session.draft_frame().assign(x=1), "variable_computation", {}, "x")
    session.commit(
        session.base.rename(columns={"age": "age_years"}), "rename_columns", {}, "Renamed"
    )
    status = session.staging_status(CONFIG)
    assert status.state == BASE_CHANGED and not status.can_apply
    with pytest.raises(StagingError):
        session.commit_staging(CONFIG)
    assert "age_years" in session.base.columns and "x" not in session.base.columns
    session.discard_staging()
    assert session.staging_status(CONFIG).state == EMPTY


def test_an_imputation_fitted_on_another_population_cannot_be_applied(session: DataSession) -> None:
    session.set_ticked([ADULT])
    population = session.population(CONFIG)
    imputed = session.draft_frame().copy()
    imputed.loc[population.included, "bmi"] = imputed.loc[population.included, "bmi"].fillna(21.0)
    session.stage(imputed, "imputation", {"numerical": "Median"}, "Median", population=population)
    assert session.staging_status(CONFIG).state == OK
    session.set_ticked([ADULT, ANOMALY])
    assert session.staging_status(CONFIG).state == POPULATION_CHANGED


def test_the_population_is_traced_once_when_used(session: DataSession) -> None:
    session.record_population_use(CONFIG)  # no criterion ticked: nothing to record
    session.set_ticked([ANOMALY])
    session.record_population_use(CONFIG)
    session.record_population_use(CONFIG)
    recorded = [s for s in steps(session) if s["function"] == "population_definition"]
    assert len(recorded) == 1 and recorded[0]["population"]["n_included"] == 4


def test_cells_filled_by_a_single_imputation_are_remembered(session: DataSession) -> None:
    session.stage(session.draft_frame().assign(bmi_sq=1.0), "variable_computation", {}, "x")
    assert session.draft is not None and session.draft.single_imputed == {}  # not an imputation
    imputed = session.draft_frame().assign(bmi=lambda f: f["bmi"].fillna(21.0))
    session.stage(imputed, "imputation", {"numerical": "Median"}, "Median")
    assert session.single_imputed == {}  # still a draft
    session.commit_staging(CONFIG)
    assert {col: ids.tolist() for col, ids in session.single_imputed.items()} == {"bmi": [1, 3]}
    assert session.single_imputed_counts(session.base.index) == {"bmi": 2}
    assert session.single_imputed_counts(session.base.index[:2], ["bmi"]) == {"bmi": 1}
    assert session.single_imputed_counts(session.base.index, ["age"]) == {}
    session.set_ticked([ANOMALY, ADULT])
    session.freeze_population(CONFIG)  # rows 1, 3 and 4 stay
    assert session.single_imputed["bmi"].tolist() == [1, 3]
    session.commit(session.base.drop(columns="bmi"), "drop_columns", {}, "Dropped")
    assert session.single_imputed == {}
    session.restore_single_imputed({"age": [1, 99]})
    assert session.single_imputed["age"].tolist() == [1]  # row ids not in the data are dropped
    session.install(pd.DataFrame({"age": [1]}))
    assert session.single_imputed == {}


def test_values_imputed_by_an_exact_formula_are_kept_apart(session: DataSession) -> None:
    frame = session.draft_frame()
    exact = frame.assign(
        bmi=frame["bmi"].where(frame.index != 1, 21.0)
    )  # row 1, by an exact formula
    params = {"target_variable": "bmi", "formula": "21"}
    session.stage(exact, "targeted_imputation", {**params, "exact": True}, "Exact formula")
    approximate = session.draft_frame().assign(bmi=lambda f: f["bmi"].fillna(23.0))  # row 3
    session.stage(approximate, "targeted_imputation", {**params, "exact": False}, "Approximation")
    assert session.draft is not None and session.draft.exact_imputed["bmi"].tolist() == [1]
    session.commit_staging(CONFIG)
    assert {col: ids.tolist() for col, ids in session.exact_imputed.items()} == {"bmi": [1]}
    assert {col: ids.tolist() for col, ids in session.single_imputed.items()} == {"bmi": [3]}
    assert session.exact_imputed_counts(session.base.index) == {"bmi": 1}
    assert session.exact_imputed_counts(session.base.index, ["age"]) == {}
    session.restore_exact_imputed({"bmi": [1, 99]})
    assert session.exact_imputed["bmi"].tolist() == [1]  # row ids not in the data are dropped
    session.commit(session.base.drop(columns="bmi"), "drop_columns", {}, "Dropped")
    assert session.exact_imputed == {} and session.single_imputed == {}


def test_only_a_targeted_imputation_marked_exact_is_exact() -> None:
    assert is_exact_imputation("targeted_imputation", {"exact": True})
    assert not is_exact_imputation(
        "targeted_imputation", {"formula": "x"}
    )  # recorded before the option
    assert not is_exact_imputation("imputation", {"exact": True})


def test_freezing_keeps_only_the_population(session: DataSession) -> None:
    session.set_ticked([ANOMALY, ADULT])
    change = session.freeze_population(CONFIG)
    assert change is not None and change.rows_removed == 2
    assert list(session.base.index) == [1, 3, 4] and session.ticked == ()
    frozen = steps(session)[-1]
    assert frozen["function"] == "freeze_population" and frozen["population"]["n_included"] == 3


def test_a_memo_is_kept_until_the_data_or_its_key_changes(session: DataSession) -> None:
    calls: list[str] = []

    def profile(label: str) -> str:
        calls.append(label)
        return label

    assert session.memo("profile", "all rows", lambda: profile("a")) == "a"
    assert session.memo("profile", "all rows", lambda: profile("b")) == "a"  # kept: not recomputed
    assert session.memo("audit", "all rows", lambda: profile("c")) == "c"  # one slot per name
    assert (
        session.memo("profile", "population", lambda: profile("d")) == "d"
    )  # another key replaces it
    assert session.memo("profile", "all rows", lambda: profile("e")) == "e"
    session.commit(session.base.assign(bmi=0.0), "external_write", {}, "bmi set")
    assert session.memo("audit", "all rows", lambda: profile("f")) == "f"  # a change of the data
    session.install(pd.DataFrame({"age": [1]}))
    assert session.memo("audit", "all rows", lambda: profile("g")) == "g"  # another dataset
    assert calls == ["a", "c", "d", "e", "f", "g"]
