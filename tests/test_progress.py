"""Workflow progress rules (core.progress): what is done, what is stale after a data change, what comes next."""

from core.progress import (
    DONE,
    STALE,
    TODO,
    StepMark,
    data_revision,
    evaluate,
    initial_progress,
    trace_marks,
)


def step(function: str, description: str = "", at: str = "2026-10-13T10:00:00") -> dict:
    return {"function": function, "description": description, "timestamp": at}


LOAD = step("initial_load", "Loaded dataset: cohort")


def test_without_data_load_is_next_and_nothing_is_done() -> None:
    progress = initial_progress(False)
    assert progress.next_step == "load"
    assert all(
        progress.state(s) == TODO for s in ("load", "validate", "enrich", "analyse", "report")
    )
    assert progress.text("load") == "No dataset loaded yet."
    assert progress.text("validate") == "Load a dataset first."


def test_fresh_dataset_suggests_validation() -> None:
    progress = evaluate([LOAD], {}, has_data=True, dataset_label="cohort (500 rows, 19 columns)")
    assert progress.state("load") == DONE and "cohort (500 rows" in progress.text("load")
    assert progress.next_step == "validate"
    assert progress.state("enrich") == TODO and progress.text("enrich").startswith("Optional")


def test_core_path_advances_as_steps_are_done() -> None:
    marks = {"validate": StepMark(0, detail="quality score 91/100")}
    progress = evaluate([LOAD], marks, has_data=True)
    assert progress.state("validate") == DONE and "91/100" in progress.text("validate")
    assert progress.next_step == "analyse"

    marks["analyse"] = StepMark(0, detail="descriptive statistics")
    progress = evaluate([LOAD], marks, has_data=True)
    assert progress.next_step == "report"

    saved = [LOAD, step("save_version", "Saved version: cohort_v2")]
    progress = evaluate(saved, marks, has_data=True)
    assert progress.state("report") == DONE and "cohort_v2" in progress.text("report")
    assert progress.next_step is None


def test_enrichment_loop_makes_validation_analysis_and_report_stale() -> None:
    marks = {
        "validate": StepMark(0, detail="quality score 91/100"),
        "analyse": StepMark(0, detail="tests"),
    }
    steps = [
        LOAD,
        step("save_version", "Saved version: cohort_v2"),
        step("variable_computation", "Computed bmi"),
        step("imputation", "Imputation (KNN)"),
    ]
    progress = evaluate(steps, marks, has_data=True)
    assert progress.revision == 2
    assert progress.state("enrich") == DONE
    assert progress.text("enrich") == "2 changes applied, last: Imputation (KNN)."
    for name in ("validate", "analyse", "report"):
        assert progress.state(name) == STALE
        assert progress.statuses[name].changes_since == 2
    assert "2 changes to the data" in progress.text("validate")
    assert progress.next_step == "validate"  # back to validation after enriching

    marks["validate"] = StepMark(2, detail="quality score 93/100")
    assert evaluate(steps, marks, has_data=True).next_step == "analyse"
    marks["analyse"] = StepMark(2, detail="tests")
    assert evaluate(steps, marks, has_data=True).next_step == "report"
    resaved = [*steps, step("save_version", "Saved version: cohort_v3")]
    assert evaluate(resaved, marks, has_data=True).next_step is None


def test_only_data_changes_move_the_revision() -> None:
    steps = [
        LOAD,
        step("parse_report"),
        step("rename_columns"),
        step("apply_filter", "Kept 480 of 500 rows"),
    ]
    assert data_revision(steps) == 1
    marks = trace_marks(steps)
    assert "enrich" not in marks  # filtering belongs to validation, not to the Enrich path
    progress = evaluate(steps, {"validate": StepMark(0)}, has_data=True)
    assert progress.state("validate") == STALE and progress.statuses["validate"].changes_since == 1
    assert "1 change to the data" in progress.text("validate")


def test_the_latest_evidence_wins() -> None:
    steps = [
        LOAD,
        step("imputation", "Imputation"),
        step("save_version", "Saved version: cohort_v2"),
    ]
    page_mark = {"report": StepMark(0, at="2026-10-13T09:00:00", detail="Word report exported")}
    progress = evaluate(steps, page_mark, has_data=True)
    assert progress.state("report") == DONE and "cohort_v2" in progress.text("report")


def test_knowledge_paths_follow_the_workspace_and_do_not_block_the_core_path() -> None:
    knowledge = {
        "docs": (True, "2 PDFs in your library, 1 knowledge graph saved."),
        "varmap": (False, "No map."),
    }
    progress = evaluate([LOAD], {}, has_data=True, knowledge=knowledge)
    assert progress.state("docs") == DONE and progress.text("docs").startswith("2 PDFs")
    assert progress.state("varmap") == TODO
    assert progress.next_step == "validate"
