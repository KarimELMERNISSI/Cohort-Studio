"""A 2x2 table with a zero cell in some imputed datasets: the page and the dossier show the pooled risk difference
and say why the ratios are not pooled (M5)."""

from __future__ import annotations

from app_pages.analysis_pooled import Pooled, _summary_row, _twobytwo_frame
from biostats.mi import analyze_mi
from manage.report_tables import _pooled_comparison_blocks
from tests.biostats_layer.test_mi_analyze import arm_table, zero_cell_mi


def test_the_summary_detail_and_dossier_fall_back_to_the_risk_difference() -> None:
    item = Pooled("twobytwo", analyze_mi(arm_table, zero_cell_mi()), "A", "B", "1")
    row = _summary_row("y", item, "arm")
    assert row["analysis"] == "2x2: risk difference (odds ratio not pooled: a zero cell)"
    assert (
        row["estimate_label"].startswith("risk difference")
        and row["ci_low"] < row["estimate"] < row["ci_high"]
    )
    detail = _twobytwo_frame(item.result)
    assert detail["measure"].tolist() == [
        "Odds ratio: not pooled (a zero cell)",
        "Risk ratio: not pooled (a zero cell)",
        "Risk difference",
    ]
    blocks = _pooled_comparison_blocks({"results": {"y": item}, "errors": {}})
    (_, table), *_ = blocks.tables
    assert table.loc[0, "Analysis"] == "Risk difference (odds ratio not pooled: a zero cell)"
    assert table.loc[0, "Result"].startswith("RD = ")
    assert any(w.startswith("Odds ratio and risk ratio not pooled") for w in blocks.warnings)


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
