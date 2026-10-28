"""The nullity matrix of Validate's dashboard labels the rows its height can show, each label at its own row."""

import numpy as np
import pandas as pd
import pytest

from utils.visualization_utils import plot_nullity_matrix


def _frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(n, 4))
    values[rng.random(size=(n, 4)) < 0.2] = np.nan
    frame = pd.DataFrame(values, columns=list("abcd"))
    frame["patient_id"] = [f"P{i:04d}" for i in range(n)]
    return frame


def _labels(n: int, **options):
    figure = plot_nullity_matrix(_frame(n), row_id_col="patient_id", figsize=(18, 12), **options)
    axis = figure.layout.yaxis
    # the row drawn at each height, as the hover names it
    drawn = {y: row[0] for y, row in zip(figure.data[0].y, figure.data[0].text)}
    return list(axis.tickvals), list(axis.ticktext), drawn


@pytest.mark.parametrize("cluster_rows", [False, True])
def test_a_long_cohort_gets_the_labels_that_fit(cluster_rows: bool) -> None:
    """1,206 labels in 600 pixels could not be read, and drawing them kept the browser busy for a minute each time
    the dashboard was opened."""
    ticks, texts, drawn = _labels(1206, cluster_rows=cluster_rows)
    assert 50 <= len(ticks) <= 80
    assert all(drawn[tick] == text for tick, text in zip(ticks, texts))
    rows = sorted(drawn).index  # a dendrogram lists its rows from the top
    gaps = {abs(rows(b) - rows(a)) for a, b in zip(ticks, ticks[1:])}
    # evenly spaced, the 6-pixel font apart in the 500 pixels inside the margins
    assert len(gaps) == 1, gaps
    (gap,) = gaps
    assert gap * 500 / 1206 >= 8


def test_a_short_cohort_gets_every_label() -> None:
    ticks, texts, drawn = _labels(20)
    assert len(ticks) == 20
    assert all(drawn[tick] == text for tick, text in zip(ticks, texts))
