"""
The figures of a group comparison (app_pages/group_figures.py, M5): one colour per group in every figure, one
height for a two-column grid, histograms as the percentage of each group, and a Q-Q plot of each group standardised
within the group. Before, they stretched to the full width with fixed heights, overlaid opaque histograms of raw
counts, and drew one Q-Q plot of the pooled residuals beside per-group normality checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app_pages.group_figures as gf


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "diabetes": ["No"] * 300 + ["Yes"] * 100,
            "hba1c": np.r_[rng.normal(5.4, 0.4, 300), rng.normal(7.5, 1.1, 100)],
        }
    )


def test_every_figure_gives_a_group_the_same_colour_and_the_same_height() -> None:
    df = _cohort()
    colors = gf.group_colors(gf.group_order(df, "diabetes"))
    box = gf.box_figure(df, "diabetes", "hba1c", significant_pairs=None)
    means = gf.means_figure(df, "diabetes", "hba1c")
    hist = gf.histogram_figure(df, "diabetes", "hba1c")
    qq = gf.qq_figure(df, "diabetes", "hba1c")
    assert box is not None
    assert [t.marker.color for t in box.data] == [colors["No"], colors["Yes"]]
    assert list(means.data[0].marker.color) == [colors["No"], colors["Yes"]]
    assert {t.name: t.marker.color for t in hist.data} == colors
    assert [t.marker.color for t in qq.data[:2]] == [colors["No"], colors["Yes"]]
    assert {f.layout.height for f in (box, means, hist, qq)} == {gf.HEIGHT}


def test_the_histograms_compare_groups_of_different_sizes() -> None:
    hist = gf.histogram_figure(_cohort(), "diabetes", "hba1c")
    assert all(t.histnorm == "percent" for t in hist.data) and hist.layout.barmode == "overlay"


def test_the_means_carry_their_95_percent_confidence_intervals() -> None:
    from scipy import stats

    df = _cohort()
    means = gf.means_figure(df, "diabetes", "hba1c")
    yes = df.loc[df["diabetes"] == "Yes", "hba1c"]
    half = stats.t.ppf(0.975, len(yes) - 1) * yes.std(ddof=1) / np.sqrt(len(yes))
    assert means.data[0].error_y.array[1] == pytest.approx(half)
    assert means.data[0].x[1] == "Yes<br>n = 100"


def test_the_q_q_plot_standardises_each_group() -> None:
    qq = gf.qq_figure(_cohort(), "diabetes", "hba1c")
    for trace in qq.data[:2]:
        values = np.asarray(trace.y)
        assert values.mean() == pytest.approx(0, abs=1e-9) and values.std(ddof=1) == pytest.approx(
            1
        )
    assert qq.data[2].name == "normal"


def _card_script() -> None:
    import numpy as np

    from biostats.inference import compare_groups
    from ui.components import result_card

    rng = np.random.default_rng(2)
    groups = {"No": rng.normal(5.5, 0.4, 60), "Yes": rng.lognormal(2.0, 0.5, 45)}
    result_card(compare_groups(groups, kind="welch"), group_variable="diabetes")


def test_the_result_card_says_which_group_and_what_each_check_means() -> None:
    at = AppTest.from_function(_card_script, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    text = " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
    assert (
        "Normality, diabetes = No (n = 60)" in text and "Normality, diabetes = Yes (n = 45)" in text
    )
    assert "Not required: Welch's test allows the groups different variances." in text
