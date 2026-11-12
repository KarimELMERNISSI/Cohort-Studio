"""
What today's documents leave out (M8, the study package's first steps): a stratum's value in a warning (a site, a
clinic, copied into the dossier), the values a replay error quotes (in the result and in the log), and the
Kaplan-Meier steps of a group whose median the dossier withholds (they are its participants' event times).
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

CANARY = "CANARY-SITE-41"


def test_a_small_stratum_is_counted_never_named() -> None:
    from biostats.rates import mantel_haenszel

    rng = np.random.default_rng(3)
    rows = []
    for site, n in (("Site A", 120), ("Site B", 120), (CANARY, 6)):
        rows += [
            {"exposed": rng.choice(["no", "yes"]), "event": rng.choice(["no", "yes"]), "site": site}
            for _ in range(n)
        ]
    result = mantel_haenszel(pd.DataFrame(rows), "exposed", "event", "site")
    said = " ".join(result.warnings)
    assert "1 of 3 strata has fewer than 5 participants in a group" in said
    assert CANARY not in said


def test_a_replay_error_quotes_no_value_of_the_data(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from manage.replay import engine

    def failing(*args, **kwargs):
        raise ValueError(f"could not convert string to float: '{CANARY}'")

    monkeypatch.setattr(engine, "_apply_step", failing)
    caplog.set_level(logging.DEBUG)
    result = engine.apply_steps(
        pd.DataFrame({"x": [1, 2]}), [{"function": "variable_computation", "params": {}}]
    )
    assert result.error and "could not convert string to float" in result.error
    assert CANARY not in result.error and CANARY not in caplog.text


def _curve(label: str, times, events):
    from biostats.survival import kaplan_meier

    return kaplan_meier(times, events, label=label)


def test_the_dossier_draws_no_group_whose_median_it_withholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import manage.report_tables as tables

    drawn: list[list[str]] = []
    monkeypatch.setattr(
        tables, "km_png", lambda curves: drawn.append([label for label, *_ in curves]) or b"png"
    )
    large = _curve("Large", np.arange(1, 13), np.ones(12))
    small = _curve(CANARY, [3, 5, 8], [1, 1, 0])
    blocks = tables._survival_blocks(
        SimpleNamespace(curves=[large, small], references=[], logrank_statistic=None, warnings=[])
    )
    assert drawn == [["Large"]]
    assert "1 group of fewer than 10 is not drawn." in blocks.lines

    drawn.clear()
    curve = pd.DataFrame({"time": [0.0, 3.0, 5.0], "survival": [1.0, 0.67, 0.33]})
    pooled = SimpleNamespace(
        m=5,
        kind="survival",
        tests={},
        warnings=[],
        references=[],
        notes=[],
        complete_case=None,
        extra={
            "curves": {"Large": curve, CANARY: curve},
            "medians": {"Large": 4.0, CANARY: 4.0},
            "sizes": {"Large": 40, CANARY: 3},
        },
    )
    blocks = tables._pooled_blocks(pooled)
    assert drawn == [["Large"]] and "1 group of fewer than 10 is not drawn." in blocks.lines
    (_, table), *_ = blocks.tables
    small_row = table[table["Group"] == CANARY].iloc[0]
    assert small_row["Survival at last time"] == tables.SMALL  # its last step is a participant's


def _encoding_trace() -> dict:
    one_hot = {
        "function": "variable_transformation",
        "description": "One-Hot Encoding of site",
        "params": {
            "transformation_type": "Encoding",
            "transformation": "One-Hot Encoding",
            "columns": ["site"],
            "naming_pattern": "{initial_variable}_{category}",
        },
        "data": {
            "fp_in": "a" * 16,
            "fp_out": "b" * 16,
            "rows_in": 4,
            "rows_out": 4,
            "cols_in": 2,
            "cols_out": 5,
            "columns_added": ["site_Metz", f"site_{CANARY}", "site_Luxembourg"],
        },
    }
    ordinal = {
        "function": "variable_transformation",
        "description": "Ordinal Encoding of referrer",
        "params": {
            "transformation_type": "Encoding",
            "transformation": "Ordinal Encoding",
            "columns": ["referrer"],
            "naming_pattern": "{initial_variable}_{method_applied}",
            "category_orders": {"referrer": ["Dr Martin", CANARY, "Dr Weber"]},
            "a key of another version": CANARY,
        },
        "data": {
            "fp_in": "b" * 16,
            "fp_out": "c" * 16,
            "columns_added": ["referrer_Ordinal_Encoding"],
        },
    }
    return {"session_id": "s_enc", "source_dataset": "cohort.csv", "steps": [one_hot, ordinal]}


def test_an_encoding_names_no_category_in_the_report_nor_the_dossier() -> None:
    """One-hot columns are named after the categories, and an ordinal encoding records every value in its order: a
    value held by one row (a name) travelled with the Transformation Report and the dossier."""
    import io

    from docx import Document

    from app_pages.study_dossier import _data_steps
    from enrich.transformation_logic import added_columns, report_params
    from manage.trace_documenter import TraceDocumenter

    trace = _encoding_trace()
    one_hot, ordinal = trace["steps"]
    assert added_columns(one_hot) == "3 columns, one per category of site"
    assert added_columns(ordinal) == "referrer_Ordinal_Encoding"
    shown = report_params(ordinal["params"])
    assert shown["categories"] == {"referrer": 3} and CANARY not in repr(shown)
    document = Document(io.BytesIO(TraceDocumenter(trace).generate_report().getvalue()))
    text = "\n".join(p.text for p in document.paragraphs) + "\n".join(
        c.text for table in document.tables for row in table.rows for c in row.cells
    )
    assert "3 columns, one per category of site" in text and CANARY not in text
    dossier = " ".join(line for _, line in _data_steps(trace))
    assert "added: 3 columns, one per category of site" in dossier and CANARY not in dossier
