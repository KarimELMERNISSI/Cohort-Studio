"""The dossier's report blocks of single results (manage.report_tables)."""

from __future__ import annotations

from statsmodels.datasets import spector

from biostats.regression import APPARENT_AUC, APPARENT_AUC_NOTE, logistic_regression
from manage.report_tables import result_blocks


def test_the_logistic_auc_is_reported_as_apparent() -> None:
    # computed on the data the model was fitted to, the AUC overstates the discrimination in new patients (M5)
    fitted = logistic_regression(spector.load_pandas().data, "GRADE", ["GPA", "TUCE", "PSI"], event_level="1.0")
    blocks = result_blocks(fitted)
    assert any(f"{APPARENT_AUC} = " in line for line in blocks.lines)
    assert APPARENT_AUC_NOTE in blocks.lines
    assert not any(line.startswith("AUC") or ", AUC =" in line for line in blocks.lines)
