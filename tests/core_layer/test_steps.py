"""
The catalogue of trace steps (core/steps.py, M5) agrees with what writes, replays and reports the steps.

Before, seven files kept their own sets of step names. Five steps had no title in the trace report, and a step
written under a new name was silently unknown to the replay.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from core import steps
from core.registry import UnknownIdError
from core.steps import DOCUMENT, RECOMPUTE, SNAPSHOT, STEPS

ROOT = Path(__file__).resolve().parents[2]
# the methods that write a trace step, and the position of the step's name among their arguments
WRITERS = {"commit_change": 1, "stage": 1, "commit": 1, "note": 0, "add_step": 0, "_record": 0}
NAME_ARGUMENTS = ("function", "function_name")  # the step's name passed by keyword
# every id a trace may hold: an id is never removed from the catalogue (a retired kind is marked legacy)
STORED_IDS = {
    "initial_load",
    "parse_report",
    "data_type_conversion",
    "outlier_clipping",
    "enrichment",
    "variable_computation",
    "variable_transformation",
    "imputation",
    "targeted_imputation",
    "mi_impute",
    "population_definition",
    "freeze_population",
    "rename_columns",
    "save_version",
    "external_write",
    "apply_filter",
    "missing_value_handling",
    "multiple_imputation",
    "outlier_handling",
}


def _packages() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]


def _literal(node: ast.AST | None) -> str | None:
    is_text = isinstance(node, ast.Constant) and isinstance(node.value, str)
    return node.value if is_text else None  # type: ignore[union-attr]


def _written_step_names() -> dict[str, str]:
    """Step name -> where the application writes it (literal names only)."""
    found: dict[str, str] = {}
    for package in _packages():
        for file in sorted((ROOT / package).rglob("*.py")):
            tree = ast.parse(file.read_text(encoding="utf-8-sig"))  # a few pages start with a BOM
            for node in ast.walk(tree):
                names: list[str | None] = []
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    position = WRITERS.get(node.func.attr)
                    if position is not None and len(node.args) > position:
                        names.append(_literal(node.args[position]))
                    if position is not None:
                        names += [
                            _literal(k.value) for k in node.keywords if k.arg in NAME_ARGUMENTS
                        ]
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    # DataSession names the step of a method `function` (set_dataset: "initial_load")
                    arguments = node.args.args + node.args.kwonlyargs
                    defaults = [None] * (len(node.args.args) - len(node.args.defaults))
                    defaults += node.args.defaults + node.args.kw_defaults
                    names += [
                        _literal(d) for a, d in zip(arguments, defaults) if a.arg == "function"
                    ]
                for name in names:
                    if name and name.isidentifier() and name.islower():
                        found.setdefault(name, f"{file.relative_to(ROOT)}:{node.lineno}")
    return found


def test_every_step_the_application_writes_is_registered() -> None:
    written = _written_step_names()
    # the scan finds the names, whichever way they are written
    assert {"initial_load", "rename_columns", "mi_impute", "save_version"} <= set(written)
    unregistered = {name: where for name, where in written.items() if name not in STEPS}
    assert not unregistered, f"register these steps in core/steps.py: {unregistered}"


def test_no_stored_id_is_ever_removed() -> None:
    assert STORED_IDS <= set(STEPS.ids())
    legacy = {i for i in STEPS.ids() if STEPS.is_legacy(i)}
    assert legacy == {
        "apply_filter",
        "missing_value_handling",
        "multiple_imputation",
        "outlier_handling",
    }
    assert legacy.isdisjoint(_written_step_names())  # nothing records a legacy step any more


def test_the_trace_report_titles_every_step() -> None:
    from manage.trace_documenter import TraceDocumenter

    report = TraceDocumenter({})
    titles = [report._get_readable_function_name(kind.id) for kind in STEPS]
    assert titles == [kind.title for kind in STEPS]
    assert len(set(titles)) == len(titles)  # two kinds never share a heading
    # the five steps the report showed as their id in words
    untitled = (
        "rename_columns",
        "outlier_clipping",
        "parse_report",
        "save_version",
        "external_write",
    )
    assert [STEPS.get(i).title for i in untitled] == [
        "Columns Renamed",
        "Outliers Clipped to Their Bounds",
        "Import Parse Report",
        "Version Saved",
        "Data Replaced by a Page (Not Replayable)",
    ]
    assert report._get_readable_function_name("a_future_step") == "A Future Step"


def test_the_replay_handles_every_kind_by_its_declared_mode() -> None:
    import manage.reproduction_manager as rm

    assert set(rm._HANDLED) == steps.ids(replay=RECOMPUTE)
    assert (
        rm.SKIPPED_STEPS
        == steps.ids(replay=DOCUMENT)
        == {
            "initial_load",
            "parse_report",
            "save_version",
        }
    )
    assert (
        rm.NOT_REPLAYABLE
        == steps.ids(replay=SNAPSHOT)
        == {
            "multiple_imputation",
            "external_write",
            "outlier_handling",
        }
    )


def test_an_old_trace_with_legacy_steps_replays_without_an_unknown_step() -> None:
    from manage.reproduction_manager import apply_steps

    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": ["x", "y", "z", "w"]})
    old = [
        {"function": "initial_load", "params": {"source_path": "cohort.csv"}},
        {"function": "apply_filter", "params": {"kept_index": [0, 1, 2]}},
        {
            "function": "missing_value_handling",
            "params": {"strategy": "Drop rows", "columns": ["a"]},
        },
        {"function": "multiple_imputation", "params": {}},
        {"function": "outlier_handling", "params": {}},
        {"function": "save_version", "params": {"dataset_name": "v1"}},
    ]
    result = apply_steps(df, old)
    assert result.error is None
    assert not [w for w in result.warnings if "unknown step" in w], result.warnings
    assert list(result.df["b"]) == ["x", "z"]  # the filter, then the dropped row


def test_the_sets_the_modules_kept_are_derived_unchanged() -> None:
    from app_pages.study_dossier import NOT_DATA_HANDLING
    from core.attrition import flow_name
    from core.progress import DATA_CHANGES, ENRICH_CHANGES, SAVE_STEPS
    from core.session import SINGLE_IMPUTATION_STEPS

    # judged by kind for a step without fingerprints: renaming or retyping columns is not a change. Added to the
    # former list: three kinds always recorded with fingerprints, and two legacy kinds
    assert DATA_CHANGES == {
        "enrichment",
        "variable_computation",
        "variable_transformation",
        "targeted_imputation",
        "imputation",
        "multiple_imputation",
        "apply_filter",
        "outlier_clipping",
        "freeze_population",
        "external_write",
        "missing_value_handling",
        "outlier_handling",
    }
    assert SINGLE_IMPUTATION_STEPS == {"imputation", "targeted_imputation", "multiple_imputation"}
    assert NOT_DATA_HANDLING == {
        "initial_load",
        "parse_report",
        "population_definition",
        "save_version",
        "mi_impute",
        "freeze_population",
    }
    assert SAVE_STEPS == {"save_version"}
    # the legacy manual missing-value handling now counts as an Enrich change too
    assert ENRICH_CHANGES == {
        "enrichment",
        "variable_computation",
        "variable_transformation",
        "targeted_imputation",
        "imputation",
        "multiple_imputation",
        "missing_value_handling",
    }
    assert [flow_name(i) for i in ("enrichment", "apply_filter", "freeze_population")] == [
        "merge",
        "filter",
        "population kept as the dataset",
    ]
    assert flow_name("outlier_clipping") == "outlier clipping"


def test_loading_is_never_a_change_and_fingerprints_win_over_the_kind() -> None:
    from core.progress import changes_data

    assert not changes_data({"function": "initial_load", "data": {"fp_in": "", "fp_out": "a"}})
    assert not changes_data({"function": "parse_report"})
    assert changes_data({"function": "rename_columns", "data": {"fp_in": "a", "fp_out": "b"}})
    assert not changes_data({"function": "rename_columns", "data": {"fp_in": "a", "fp_out": "a"}})
    assert changes_data({"function": "imputation"})  # an old step: judged by its kind
    assert not changes_data({"function": "mi_impute"}) and not changes_data({"function": "unknown"})


def test_an_unregistered_step_is_refused_in_strict_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from manage.transformation_manager import TransformationManager

    tm = TransformationManager(trace_dir=str(tmp_path))
    tm.initialize_session("cohort.csv")
    monkeypatch.setenv("CMP_STRICT_STATE", "1")  # as the test suite runs
    with pytest.raises(UnknownIdError, match="unknown trace step 'renamed_columns'"):
        tm.add_step("renamed_columns", {}, "A typo in a step name")
    assert tm.history == []
    monkeypatch.setenv("CMP_STRICT_STATE", "0")  # in production the step is kept, with a warning
    tm.add_step("renamed_columns", {}, "A typo in a step name")
    assert len(tm.history) == 1 and "not registered" in caplog.text
