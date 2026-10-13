"""Effect sizes: sign conventions, exact Hedges correction, noncentral CIs, bias-corrected Cramer's V."""

import numpy as np
import pytest

from biostats.effect_sizes import (
    cohens_d,
    cramers_v,
    eta_squared,
    eta_squared_h,
    hedges_j,
    rank_biserial,
    vargha_delaney_a,
)

REFERENCE = [1, 2, 3, 4, 5]
COMPARISON = [3, 4, 5, 6, 7]
CTRL = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
TRT1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
TRT2 = [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]


def test_hedges_j_is_exact() -> None:
    assert hedges_j(8) == pytest.approx(0.902703, abs=1e-6)
    assert hedges_j(1) == 1.0
    assert 0.99 < hedges_j(1000) < 1.0


def test_cohens_d_and_hedges_g_with_noncentral_ci() -> None:
    d = cohens_d(COMPARISON, REFERENCE, hedges=False)
    assert d.estimate == pytest.approx(1.264911, abs=1e-5)
    g = cohens_d(COMPARISON, REFERENCE, hedges=True)
    assert g.estimate == pytest.approx(0.902703 * 1.264911, abs=1e-4)
    assert g.ci is not None and g.ci.method.startswith("noncentral t")
    assert g.ci.lower == pytest.approx(-0.1331, abs=2e-3)
    assert g.ci.upper == pytest.approx(2.3607, abs=2e-3)
    assert g.magnitude() == "large"
    # comparison minus reference: swapping the groups flips the sign
    assert cohens_d(REFERENCE, COMPARISON).estimate == pytest.approx(-g.estimate)


def test_rank_based_effects_share_the_sign_of_d() -> None:
    assert vargha_delaney_a(COMPARISON, REFERENCE) == pytest.approx(0.82)
    r = rank_biserial(COMPARISON, REFERENCE, n_boot=300)
    assert r.estimate == pytest.approx(0.64)
    assert r.ci is not None and r.ci.lower <= 0.64 <= r.ci.upper
    assert rank_biserial(REFERENCE, COMPARISON, n_boot=0).estimate == pytest.approx(-0.64)


def test_cramers_v_bias_corrected_reference_value() -> None:
    v = cramers_v([[30, 10], [20, 40]], n_boot=0)
    assert v.estimate == pytest.approx(0.3977, abs=1e-3)
    assert v.ci is None
    with_ci = cramers_v([[30, 10], [20, 40]], n_boot=200)
    assert with_ci.ci is not None and with_ci.ci.lower <= with_ci.estimate <= with_ci.ci.upper
    # independence: the bias correction pulls V to zero instead of a spurious positive value
    assert cramers_v([[25, 25], [25, 25]], n_boot=0).estimate == 0.0


def test_eta_and_omega_squared_plantgrowth() -> None:
    eta, omega = eta_squared([CTRL, TRT1, TRT2])
    assert eta.estimate == pytest.approx(0.2641, abs=1e-3)
    assert omega.estimate == pytest.approx(0.2041, abs=1e-3)
    assert eta.ci is not None and 0.0 <= eta.ci.lower <= eta.estimate <= eta.ci.upper < 1.0
    assert eta.ci.method.startswith("noncentral F")
    h = eta_squared_h(7.988, 3, 30)
    assert h.estimate == pytest.approx((7.988 - 2) / 27, abs=1e-6)


def test_degenerate_inputs_do_not_raise() -> None:
    assert np.isnan(cohens_d([1.0], [2.0, 3.0]).estimate)
    assert cohens_d([2.0, 2.0], [2.0, 2.0]).estimate == 0.0
    assert np.isnan(rank_biserial([], [1, 2]).estimate)
    eta, _ = eta_squared([[1.0, 2.0]])
    assert np.isnan(eta.estimate)
