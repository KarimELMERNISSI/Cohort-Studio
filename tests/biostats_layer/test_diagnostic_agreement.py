"""Diagnostic accuracy (Wilson, Simel, DeLong, Youden) and Bland-Altman agreement."""

import numpy as np
import pytest
from statsmodels.stats.proportion import proportion_confint

from biostats.agreement import bland_altman
from biostats.diagnostic import confusion_at, delong_auc, diagnostic_accuracy, youden_cutoff


def test_diagnostic_accuracy_reference_values() -> None:
    acc = diagnostic_accuracy(tp=90, fp=10, fn=20, tn=80, prevalence=0.10)
    assert acc.sensitivity == pytest.approx(90 / 110)
    assert acc.specificity == pytest.approx(80 / 90)
    assert acc.ppv == pytest.approx(0.9) and acc.npv == pytest.approx(0.8)
    lo, hi = proportion_confint(90, 110, method="wilson")
    assert acc.sensitivity_ci.lower == pytest.approx(lo) and acc.sensitivity_ci.upper == pytest.approx(hi)
    assert acc.lr_positive == pytest.approx((90 / 110) / (1 - 80 / 90))
    assert acc.lr_negative == pytest.approx((1 - 90 / 110) / (80 / 90))
    assert acc.lr_positive_ci is not None and acc.lr_positive_ci.lower < acc.lr_positive < acc.lr_positive_ci.upper
    assert acc.dor == pytest.approx(36.0)
    assert acc.dor_ci is not None and acc.dor_ci.lower < 36 < acc.dor_ci.upper
    assert acc.youden == pytest.approx(90 / 110 + 80 / 90 - 1)
    assert acc.ppv_at_prevalence == pytest.approx(0.45, abs=1e-3)
    assert acc.npv_at_prevalence is not None and acc.npv_at_prevalence > 0.97
    assert not acc.warnings


def test_diagnostic_accuracy_zero_cells_are_flagged_not_crashed() -> None:
    acc = diagnostic_accuracy(tp=10, fp=0, fn=2, tn=8)
    assert acc.lr_positive is None and any("undefined" in w for w in acc.warnings)
    assert acc.dor is not None and any("Haldane" in w for w in acc.warnings)


def test_delong_auc_toy_reference() -> None:
    y = ["pos"] * 4 + ["neg"] * 4
    s = [0.9, 0.8, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1]
    res = delong_auc(y, s, pos_label="pos")
    assert res["auc"] == pytest.approx(0.875)
    assert res["variance"] == pytest.approx(0.020833, abs=1e-6)
    assert res["ci"].lower < 0.875 < res["ci"].upper <= 1.0
    with pytest.raises(ValueError):
        delong_auc(["a", "b", "c"], [1, 2, 3])


def test_youden_cutoff_and_confusion() -> None:
    rng = np.random.default_rng(5)
    neg = rng.normal(0, 1, 200)
    pos = rng.normal(2, 1, 200)
    y = ["No"] * 200 + ["Yes"] * 200
    s = np.concatenate([neg, pos])
    res = youden_cutoff(y, s, n_boot=100)
    assert res["pos_label"] == "Yes"
    assert 0.3 < res["cutoff"] < 1.7
    assert res["ci"] is not None and res["ci"].lower <= res["cutoff"] <= res["ci"].upper
    assert "optimistic" in res["note"]
    tp, fp, fn, tn = confusion_at(y, s, res["cutoff"], pos_label="Yes")
    assert tp + fn == 200 and fp + tn == 200
    assert tp / 200 == pytest.approx(res["sensitivity"], abs=1e-9)


def test_bland_altman_with_cis() -> None:
    rng = np.random.default_rng(9)
    b = rng.normal(100, 15, 60)
    a = b + 1.5 + rng.normal(0, 4, 60)
    ba = bland_altman(a, b)
    diff = a - b
    assert ba.n == 60
    assert ba.bias == pytest.approx(diff.mean())
    assert ba.sd == pytest.approx(diff.std(ddof=1))  # ddof = 1 (audit S10)
    assert ba.loa_upper == pytest.approx(ba.bias + 1.959964 * ba.sd, abs=1e-6)
    assert ba.bias_ci.lower < 1.5 < ba.bias_ci.upper
    assert ba.loa_lower_ci.lower < ba.loa_lower < ba.loa_lower_ci.upper
    assert ba.within_loa_pct >= 90
    assert not ba.warnings  # no proportional bias by construction
    with pytest.raises(ValueError):
        bland_altman([1, 2], [1, 2])
