"""
The targeted imputation (a formula fills the missing values of one variable) follows the analysis population, as the
global imputation does: the formula is computed on the included rows only, an aggregate such as a mean sees no excluded
patient, the excluded rows keep their missing values, and the replay does the same (biostatistics review, 2026-10-21).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "assets" / "tutorial"


def test_a_formula_fills_the_population_only_and_its_aggregates_see_no_one_else() -> None:
    from enrich.imputation_service import fill_with_formula

    frame = pd.DataFrame({"sbp": [None, 120.0, None, None], "base": [118.0, 119.0, 120.0, 303.0]})
    population = pd.Index([0, 1, 2])  # row 3 is excluded
    out, filled = fill_with_formula(frame, "sbp", "base.mean() + 0 * base", rows=population)
    # the mean of rows 0-2 only
    assert filled == 2 and out.loc[[0, 2], "sbp"].tolist() == [119.0, 119.0]
    assert pd.isna(out.loc[3, "sbp"])  # the excluded row keeps its missing value
    everyone, n = fill_with_formula(frame, "sbp", "base.mean() + 0 * base")
    assert n == 3 and everyone.loc[0, "sbp"] == 165.0  # without a population: every row, as before


def test_the_replay_reports_what_an_imputation_did() -> None:
    # a redone Enrich draft step is recorded with what it did on the current data (architecture review)
    from manage.reproduction_manager import apply_steps

    frame = pd.DataFrame({"sbp": [None, 120.0, None, None], "base": [1.0, 2.0, 3.0, 4.0]})
    params = {"target_variable": "sbp", "formula": "base * 10", "filled_count": 9, "exact": True}
    formula = {"function": "targeted_imputation", "params": params, "description": "nine"}
    result = apply_steps(frame, [formula])
    assert result.records[0].params == {**params, "filled_count": 3}
    assert result.records[0].description == "Imputed 3 values in sbp using an exact formula"
    assert result.warnings == ["step 1: 3 values filled on replay, 9 when recorded"]

    methods = {
        "numerical_imputation_method": "Median",
        "categorical_imputation_method": "Most_Frequent",
    }
    imputer = {"function": "imputation", "params": methods, "description": "old"}
    record = apply_steps(frame, [imputer]).records[0]
    assert record.params["rows"] == "all" and record.params["imputed_values"] == {"sbp": 3}
    assert record.description == "Applied data imputation (Median / Most_Frequent)"
    renamed = {"function": "rename_columns", "params": {"mapping": {"base": "b"}}}
    assert (
        apply_steps(frame, [renamed]).records == {}
    )  # a step whose record does not depend on the data


def test_the_targeted_imputation_replays_on_the_population_it_was_recorded_with(
    tmp_path: Path,
) -> None:
    from core.lineage import verify_trace
    from core.population import catalogue
    from core.session import DataSession
    from core.state import ParseOptions
    from enrich.imputation_service import fill_with_formula
    from ingest.loaders import read_table
    from manage.reproduction_manager import run_trace
    from manage.transformation_manager import TransformationManager

    source = tmp_path / "tutorial_cohort.csv"
    source.write_bytes((TUTORIAL / "tutorial_cohort.csv").read_bytes())
    options = ParseOptions(dayfirst=True)
    config = json.loads((TUTORIAL / "tutorial_config.json").read_text(encoding="utf-8"))

    tm = TransformationManager(trace_dir=str(tmp_path / "traces"))
    tm.initialize_session("tutorial_cohort.csv")
    tm.source_dataset = str(source)
    session = DataSession(trace=tm)
    session.install(
        read_table(source.read_bytes(), source.name, options).df,
        params={"source_path": str(source), "parse_options": options.to_dict()},
        description="Loaded the tutorial cohort",
    )
    original = session.base.copy()
    session.set_ticked([c.id for c in catalogue(config)])
    population = session.population(config)
    included = population.included
    excluded = original.index.difference(included)
    assert len(included) == 928 and original.loc[excluded, "sbp_6m"].isna().any()

    formula = "sbp_baseline + np.where(nyha_class == 'III', 5.0, -2.0)"
    filled_frame, filled = fill_with_formula(
        session.working_frame(), "sbp_6m", formula, rows=included
    )
    assert filled == int(original.loc[included, "sbp_6m"].isna().sum()) == 150
    session.stage(
        filled_frame,
        "targeted_imputation",
        {"target_variable": "sbp_6m", "formula": formula, "filled_count": filled, "exact": False},
        f"Imputed {filled} values in sbp_6m using a formula",
        population=population,
    )
    session.commit_staging()
    # the excluded patients (no consent among them) keep their missing values
    kept = session.base.loc[excluded, "sbp_6m"].isna()
    assert kept.equals(original.loc[excluded, "sbp_6m"].isna())

    replay = run_trace(tm.get_trace())
    assert replay.error is None and not replay.warnings, replay.warnings
    assert replay.df["sbp_6m"].isna().sum() == session.base["sbp_6m"].isna().sum()
    assert verify_trace(tm.get_trace()).ok
    (record,) = replay.records.values()
    assert record.params["filled_count"] == 150
    assert record.description == (
        "Imputed 150 values in sbp_6m using a formula, computed on the analysis population of 928 rows"
    )
