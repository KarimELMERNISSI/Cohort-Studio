"""
The Word dossier and a cut-off found on the data (the data-protection specialist's N4-L2 and N8-L1).

The dossier's rule of 10 keeps out of it what is one participant's value, or a few participants': a small group's
median and curve. A cut-off found on the data by Youden's J that calls 1 to 9 participants positive is the score of the
one called positive, or of a few, and every measure at it reads the small row of the 2x2 table: with the outcome's two
totals, sensitivity says whether they had the condition. The dossier wrote it all ("positive at least 412", with "1 of
928 is called positive" beside it). Such a cut-off, its range and its measures stay on the page; the AUC is written,
and a cut-off typed beforehand is written with its measures, as before.

The same holds for a side of the reference standard: with one participant with the condition among 400, the dossier
wrote "AUC 1 (1 to 1)", and the participant with the largest score had it. With 1 to 9 participants on a side, the AUC
and the measures at any cut-off stay on the page.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app_pages.analysis_diagnostic import accuracy_of
from manage.report_tables import DOSSIER_MIN_N, dossier_min_n, result_blocks
from ui.guidance import measures as m


def _scored(labels: list[str], seed: int = 3) -> pd.DataFrame:
    """Participants with distinct scores, in increasing order, each with its label ("Yes": the condition)."""
    rng = np.random.default_rng(seed)
    rows = len(labels)
    scores = np.sort(rng.choice(np.arange(1, 10 * rows), rows, replace=False)).astype(float)
    return pd.DataFrame({"ill": labels, "score": scores})


def _top(called: int, ill: int = 30, rows: int = 400) -> pd.DataFrame:
    """``ill`` participants with the condition among ``rows``: ``called`` of them hold the largest scores and the
    others the smallest, so that the cut-off with the largest Youden's J is the smallest of the ``called`` largest
    scores, and calls exactly ``called`` positive."""
    return _scored(["Yes"] * (ill - called) + ["No"] * (rows - ill) + ["Yes"] * called)


def _found(frame: pd.DataFrame, direction: str = m.HIGHER) -> dict[str, Any]:
    return accuracy_of(frame, "ill", "Yes", "score", direction, m.YOUDEN, None, None)


def _typed(frame: pd.DataFrame, cut: float) -> dict[str, Any]:
    return accuracy_of(frame, "ill", "Yes", "score", m.HIGHER, m.FIXED, cut, None)


def _written(value: dict[str, Any]) -> str:
    blocks = result_blocks(value)
    tables = " ".join(f"{title} {table.to_string()}" for title, table in blocks.tables)
    return " ".join([tables, *blocks.lines, *blocks.warnings])


def _measures(value: dict[str, Any]) -> list[str]:
    ((_, table),) = result_blocks(value).tables
    return list(table["Measure"])


def _auc_cell(value: dict[str, Any]) -> str:
    ((_, table),) = result_blocks(value).tables
    return str(table.loc[table["Measure"] == "AUC (DeLong)", "Estimate"].iloc[0])


def test_the_threshold_of_the_dossier_is_one_number() -> None:
    assert dossier_min_n() == DOSSIER_MIN_N == 10


def test_a_cut_off_found_on_the_data_that_calls_one_participant_positive_is_not_written() -> None:
    frame = _top(1)
    value = _found(frame)
    acc = value["acc"]
    largest = frame["score"].max()
    # the one participant with the largest score, who has the condition
    assert value["cutoff"] == largest and (acc.tp, acc.fp) == (1, 0)
    title = result_blocks(value).tables[0][0]
    assert title.endswith("(a cut-off found on these data, not written)")
    assert _measures(value) == ["AUC (DeLong)"]  # the AUC has no cut-off: written
    assert _auc_cell(value) != "not shown: fewer than 10"
    said = _written(value)
    assert f"{largest:.0f}" not in said and f"{largest:,.0f}" not in said
    assert (
        "The cut-off found on these data calls fewer than 10 participants positive: its value, its range and the "
        "measures at it are not written in the dossier; the page shows them."
    ) in said
    assert "called positive" not in said  # the engine's caution gives the count: not copied
    assert "Sensitivity" not in said and "PPV" not in said and "LR+" not in said
    # the page keeps them: the result itself is unchanged
    assert value["youden"]["cutoff"] == largest


def test_nine_called_positive_are_too_few_and_ten_are_not() -> None:
    nine, ten = _found(_top(9)), _found(_top(10))
    assert nine["acc"].tp + nine["acc"].fp == 9 and ten["acc"].tp + ten["acc"].fp == 10
    assert _measures(nine) == ["AUC (DeLong)"]
    assert "Sensitivity" in _measures(ten) and "LR-" in _measures(ten)
    assert result_blocks(ten).tables[0][0].endswith(f"(positive at least {ten['cutoff']:,.0f})")


def test_a_range_end_with_fewer_than_ten_beyond_it_is_not_written() -> None:
    """Ten participants with the condition hold the ten largest scores: the cut-offs found in resamples are their
    scores, and the range's upper end is a score with fewer than 10 participants at or above it."""
    frame = _top(10, ill=10)
    value = _found(frame)
    sides = value["youden"]["ci_sides"]
    ci = value["youden"]["ci"]
    assert ci is not None and sides is not None
    (low_below, low_above), (high_below, high_above) = sides
    scores = frame["score"]
    assert (low_below, low_above) == (
        int((scores < ci.lower).sum()),
        int((scores >= ci.lower).sum()),
    )
    assert high_above < 10  # the upper end is one of the few largest scores
    line = next(text for text in result_blocks(value).lines if text.startswith("Cut-off chosen"))
    assert "the largest observed scores" in line
    assert "(an end with fewer than 10 participants beyond it is not written)" in line
    assert f"{ci.upper:,.0f}" not in line and f"{ci.upper:.0f}" not in line
    # the lower end has enough on each side: written
    if min(low_below, low_above) >= 10:
        assert f"{ci.lower:,.0f}" in line


def test_a_falling_marker_counts_the_side_it_calls_positive() -> None:
    """Lower values pointing to the condition: three participants with the smallest scores are called positive (27
    others with the condition hold the largest)."""
    frame = _top(27, rows=300)
    value = _found(frame, m.LOWER)
    acc = value["acc"]
    assert value["cutoff"] == frame["score"].iloc[2] and (acc.tp, acc.fp) == (3, 0)
    assert _measures(value) == ["AUC (DeLong)"]
    assert "calls fewer than 10 participants positive" in _written(value)


def test_a_cut_off_that_calls_few_participants_negative_is_not_written_either() -> None:
    """Four participants without the condition hold the four smallest scores (26 others the largest): the cut-off
    calls everyone above them positive, and the four negative."""
    frame = _scored(["No"] * 4 + ["Yes"] * 270 + ["No"] * 26)
    value = _found(frame)
    acc = value["acc"]
    assert (acc.fn + acc.tn, acc.tn) == (4, 4)
    assert _measures(value) == ["AUC (DeLong)"]
    assert "calls fewer than 10 participants negative" in _written(value)


def test_a_cut_off_typed_beforehand_is_written_with_its_measures() -> None:
    """A cut-off fixed before the analysis is no participant's value: it is written, and so are its measures, even
    when it calls two participants positive."""
    frame = _top(2)
    cut = float(frame["score"].iloc[-2])
    value = _typed(frame, cut)
    assert value["acc"].tp + value["acc"].fp == 2
    assert "Sensitivity" in _measures(value)
    assert result_blocks(value).tables[0][0].endswith(f"(positive at least {cut:,.0f})")


def test_a_result_kept_before_its_range_s_ends_were_counted_writes_no_range() -> None:
    value = _found(_top(30))
    value["youden"] = {k: v for k, v in value["youden"].items() if k != "ci_sides"}
    line = next(text for text in result_blocks(value).lines if text.startswith("Cut-off chosen"))
    assert line.endswith("the cut-offs found have a range the page shows; it is not written here.")


# --- a side of the reference standard of fewer than 10 (N8-L1)

THIN = (
    "participants {side}: the AUC and the measures at a cut-off rank their scores among the others one by one, and "
    "are not written in the dossier; the page shows them."
)


def test_the_auc_is_not_written_beside_one_participant_with_the_condition() -> None:
    """One participant with the condition among 400, with the largest score: "AUC 1 (1 to 1)" said so."""
    value = _found(_top(1, ill=1))
    assert value["auc"]["auc"] == 1.0 and value["auc"]["n_positive"] == 1
    assert _measures(value) == ["AUC (DeLong)"]
    assert _auc_cell(value) == "not shown: fewer than 10"
    said = _written(value)
    assert "Fewer than 10 " + THIN.format(side="have the condition") in said
    assert "(1 to 1)" not in said and "calls fewer than" not in said
    assert (
        result_blocks(value).tables[0][0].endswith("(a cut-off found on these data, not written)")
    )
    assert result_blocks(value).references == []


def test_a_side_of_nine_is_too_small_and_one_of_ten_is_not() -> None:
    nine, ten = _found(_top(9, ill=9)), _found(_top(10, ill=10))
    assert _auc_cell(nine) == "not shown: fewer than 10"
    assert _auc_cell(ten) == "1" and "Sensitivity" in _measures(ten)
    # four without the condition, among 300: the same, on the other side
    four = _found(_scored(["No"] * 4 + ["Yes"] * 296))
    assert four["auc"]["n_negative"] == 4 and _auc_cell(four) == "not shown: fewer than 10"
    assert "Fewer than 10 " + THIN.format(side="are without the condition") in _written(four)


def test_a_typed_cut_off_beside_a_small_side_is_written_without_its_measures() -> None:
    """The cut-off typed beforehand is no participant's value; sensitivity at it, with three participants with the
    condition, says which of them it calls positive."""
    frame = _top(2, ill=3)
    cut = float(frame["score"].iloc[-2])
    value = _typed(frame, cut)
    assert _measures(value) == ["AUC (DeLong)"] and _auc_cell(value) == "not shown: fewer than 10"
    assert result_blocks(value).tables[0][0].endswith(f"(positive at least {cut:,.0f})")
    assert "Sensitivity" not in _written(value)


def test_the_direction_s_caution_is_not_copied_beside_a_small_side() -> None:
    """The caution of a marker read the wrong way gives the AUC ("the area under the curve is 0.000"): with one
    participant with the condition, at the smallest score, that is their rank."""
    frame = _scored(["Yes"] + ["No"] * 399)
    value = _found(frame)
    assert value["cautions"] and "below 0.5" in value["cautions"][0]
    said = _written(value)
    assert "below 0.5" not in said and "0.000" not in said
    assert "Fewer than 10 " + THIN.format(side="have the condition") in said
