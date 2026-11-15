"""A 2x2 table with a zero cell in some imputed datasets: the ratios are not pooled, and the summary and the dossier
give the pooled chi-square test, which every study design supports; the detail shows the risk difference only for a
design that supports it (a cohort, a trial, a cross-sectional sample)."""

from __future__ import annotations

from app_pages.analysis_pooled import Pooled, _summary_row, _twobytwo_frame
from biostats.mi import analyze_mi
from manage.report_tables import _pooled_comparison_blocks
from tests.biostats_layer.test_mi_analyze import arm_table, zero_cell_mi


def test_the_summary_and_dossier_fall_back_to_the_test_and_the_detail_follows_the_design() -> None:
    item = Pooled("twobytwo", analyze_mi(arm_table, zero_cell_mi()), "A", "B", "1")
    row = _summary_row("y", item, "arm")
    assert row["analysis"] == "2x2: chi-square test (D2; odds ratio not pooled: a zero cell)"
    assert "estimate" not in row and 0 <= row["p"] <= 1
    assert _twobytwo_frame(item.result)["measure"].tolist() == [
        "Odds ratio: not pooled (a zero cell)"
    ]
    for design, measures in (
        ("case-control", ["Odds ratio: not pooled (a zero cell)"]),
        (
            "cohort",
            [
                "Odds ratio: not pooled (a zero cell)",
                "Risk ratio: not pooled (a zero cell)",
                "Risk difference (percentage points)",
            ],
        ),
        (
            "cross-sectional",
            [
                "Odds ratio: not pooled (a zero cell)",
                "Prevalence ratio: not pooled (a zero cell)",
                "Prevalence difference (percentage points)",
            ],
        ),
    ):
        assert _twobytwo_frame(item.result, design)["measure"].tolist() == measures, design
    blocks = _pooled_comparison_blocks({"results": {"y": item}, "errors": {}})
    (_, table), *_ = blocks.tables
    assert table.loc[0, "Analysis"] == "Chi-square test (D2; odds ratio not pooled: a zero cell)"
    assert table.loc[0, "Result"].startswith("F = ") and "RD" not in table.loc[0, "Result"]
    assert any(w.startswith("The ratios are not pooled") for w in blocks.warnings)
    assert not any(
        "risk ratio" in w for w in blocks.warnings
    )  # the pooled comparison knows no design


def test_the_sensitivity_analysis_is_labelled() -> None:
    item = Pooled(
        "twobytwo",
        analyze_mi(lambda d: arm_table(d, "always"), zero_cell_mi()),
        "A",
        "B",
        "1",
        haldane="always",
    )
    assert _summary_row("y", item, "arm")["analysis"].endswith("(sensitivity analysis)")
    blocks = _pooled_comparison_blocks({"results": {"y": item}, "errors": {}})
    assert blocks.tables[0][1].loc[0, "Analysis"].endswith("(sensitivity analysis)")
