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
