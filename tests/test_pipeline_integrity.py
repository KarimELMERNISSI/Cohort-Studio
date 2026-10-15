"""
End-to-end integrity of the data path (follow-up programme M1).

The criteria a user ticks define the analysed rows; an imputation is fitted on, and written into, those
rows only; the trace replays to exactly the table that was analysed and its hash chain verifies. This is
the scenario of the October 2026 architecture review ("change an exclusion upstream, analyse a stale state
downstream"), written before the code that makes it pass.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "assets" / "tutorial"


def test_ticked_criteria_drive_imputation_and_the_trace_replays_the_analysed_table(
    tmp_path: Path,
) -> None:
    from core.fingerprint import frame_fingerprint
    from core.lineage import verify_trace
    from core.population import active_criteria, catalogue, evaluate_population
    from core.session import DataSession
    from core.state import ParseOptions
    from enrich.imputation_service import impute_rows
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
    loaded = read_table(source.read_bytes(), source.name, options).df
    session.install(
        loaded,
        params={"source_path": str(source), "parse_options": options.to_dict()},
        description="Loaded the tutorial cohort",
    )
    original = session.base.copy()

    session.set_ticked([c.id for c in catalogue(config)])
    population = session.population(config)
    assert [s.n_out for s in population.stages] == [1190, 928]

    imputed = impute_rows(
        session.working_frame(),
        population.included,
        numerical="Median",
        categorical="Most_Frequent",
    )
    session.stage(
        imputed.frame, "imputation", imputed.params, "Median imputation", population=population
    )
    session.commit_staging()

    analysed = session.analysis_frame(config)
    excluded = original.index.difference(population.included)
    assert len(analysed) == 928
    assert not analysed[list(imputed.columns)].isna().any().any()
    # the excluded rows are untouched: their missing values are still missing
    assert session.base.loc[excluded].isna().equals(original.loc[excluded].isna())

    replay = run_trace(tm.get_trace())
    assert replay.error is None and replay.reproducible and replay.divergence is None
    replayed = evaluate_population(replay.df, active_criteria(config, session.ticked))
    assert frame_fingerprint(replay.df.loc[replayed.included]) == frame_fingerprint(analysed)
    assert verify_trace(tm.get_trace()).ok

    # the cells the single imputation filled are known, in the session and after the replay (M2, C10)
    filled = {col: set(ids.tolist()) for col, ids in session.single_imputed.items()}
    assert {col: len(ids) for col, ids in filled.items()} == imputed.params["imputed_values"]
    assert all(ids <= set(population.included) for ids in filled.values())
    assert {col: set(ids.tolist()) for col, ids in replay.single_imputed.items()} == filled
