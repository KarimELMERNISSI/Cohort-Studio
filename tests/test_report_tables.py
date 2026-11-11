"""The dossier's report blocks of single and pooled results (manage.report_tables)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from statsmodels.datasets import spector

from biostats.mi.adapters import SurvivalAdapter
from biostats.regression import APPARENT_AUC, APPARENT_AUC_NOTE, logistic_regression
from biostats.survival import compare_survival
from manage.report_tables import SMALL, _pooled_blocks, result_blocks


def test_the_logistic_auc_is_reported_as_apparent() -> None:
    # computed on the data the model was fitted to, the AUC overstates the discrimination in new patients (M5)
    fitted = logistic_regression(spector.load_pandas().data, "GRADE", ["GPA", "TUCE", "PSI"], event_level="1.0")
    blocks = result_blocks(fitted)
    assert any(f"{APPARENT_AUC} = " in line for line in blocks.lines)
    assert APPARENT_AUC_NOTE in blocks.lines
    assert not any(line.startswith("AUC") or ", AUC =" in line for line in blocks.lines)


def _two_groups() -> pd.DataFrame:
    """Twelve participants in one arm; three in the other, every one an event, at 2, 5 and 9."""
    return pd.DataFrame(
        {
            "time": [*range(1, 13), 2, 5, 9],
            "event": [1, 0] * 6 + [1, 1, 1],
            "arm": ["large"] * 12 + ["small"] * 3,
        }
    )


def test_a_small_group_s_median_stays_out_of_the_dossier() -> None:
    """In a group of three, every one an event, the median is the second time and its interval starts at the first
    (5, and 2 to not reached): participants' durations in a document that travels. The page keeps them."""
    res = compare_survival(_two_groups(), "time", "event", "arm")
    table = result_blocks(res).tables[0][1].set_index("Group")
    assert table.loc["small", "Median survival"] == SMALL
    assert table.loc["small", "Median 95% CI"] == SMALL
    assert not any(time in " ".join(map(str, table.loc["small"])) for time in ("2", "5", "9"))
    assert table.loc["large", "Median survival"] == "9"
    small = next(c for c in res.curves if c.label == "small")
    assert (small.median, small.median_lower) == (5.0, 2.0)


def test_a_small_group_s_pooled_median_stays_out_of_the_dossier() -> None:
    adapter = SurvivalAdapter()
    extracts = [adapter.extract(compare_survival(_two_groups(), "time", "event", "arm")) for _ in range(3)]
    extra = adapter.finish(extracts, None, alpha=0.05)
    assert extra["sizes"] == {"large": 12, "small": 3}
    pooled = SimpleNamespace(kind="survival", m=3, extra=extra, tests={}, warnings=[], notes=[], complete_case=None)
    table = _pooled_blocks(pooled).tables[0][1].set_index("Group")
    assert table.loc["small", "Median survival (pooled curve)"] == SMALL
    assert table.loc["small", "Median 95% CI (pooled band)"] == SMALL
    assert table.loc["large", "Median survival (pooled curve)"] != SMALL
    del extra["sizes"]  # a result pooled before the sizes were kept still gives its table
    table = _pooled_blocks(pooled).tables[0][1].set_index("Group")
    assert table.loc["small", "Median survival (pooled curve)"] != SMALL


def test_a_group_comparison_gives_the_number_analysed_of_each_variable() -> None:
    """STROBE item 12: each variable's own number analysed, not the rows of the selected groups."""
    rows = [
        {"Variable": "sbp_6m", "N": 778, "N per group": {"Control": 382, "Intervention": 396},
         "Test Used": "Welch's t-test", "P-Value": 0.01, "Estimate": 6.64,
         "Estimate CI": "4.01 to 9.26", "Estimate of": "mean difference (Intervention - Control)"},
        {"Variable": "age", "N": 1928, "N per group": {"Control": 964, "Intervention": 964},
         "Test Used": "Welch's t-test", "P-Value": 0.4, "Estimate": 0.2,
         "Estimate CI": "", "Estimate of": ""},
    ]  # fmt: skip
    blocks = result_blocks({"results": rows, "group": "arm", "correction": "None"})
    title, table = blocks.tables[0]
    assert title == "Comparison by arm"
    assert list(table.columns)[:4] == ["Variable", "n", "n by group", "Test"]
    assert list(table["n"]) == ["778", "1,928"]
    assert list(table["n by group"]) == ["Control 382, Intervention 396", "Control 964, Intervention 964"]
    # a result stored before they were kept
    before = [{k: v for k, v in row.items() if k not in ("N", "N per group")} for row in rows]
    earlier = result_blocks({"results": before, "group": "arm"}).tables[0][1]
    assert list(earlier["n"]) == ["", ""] and list(earlier["n by group"]) == ["", ""]


def test_a_stamp_says_the_range_of_rows_of_a_result_of_several_analyses() -> None:
    from ui.results import ResultStamp

    common = dict(
        data_fp="d", population_fp="p", config_hash="c", revision=3, dataset_label="cohort",
        population_label="928 of 1,206 rows (2 criteria)", n_population=928, not_used="missing values",
        computed_at="2026-11-10T16:30:00", trace_head=None,
    )  # fmt: skip
    several = ResultStamp(n_analysed=778, n_most=928, **common)
    assert "N analysed 778 to 928 by variable (up to 150 not used: missing values)" in several.caption
    assert "N analysed 778 (150 not used: missing values)" in ResultStamp(n_analysed=778, **common).caption
    assert "N analysed 928 ·" in ResultStamp(n_analysed=928, n_most=928, **common).caption
    # rows left out for another reason than a missing value are said apart
    apart = ResultStamp(n_analysed=900, n_left_out=3, left_out="not in the compared groups", **common)
    assert "N analysed 900 (3 not in the compared groups; 25 not used: missing values)" in apart.caption
    whole = ResultStamp(n_analysed=925, n_left_out=3, left_out="not in the compared groups", **common)
    assert "N analysed 925 (3 not in the compared groups) ·" in whole.caption
