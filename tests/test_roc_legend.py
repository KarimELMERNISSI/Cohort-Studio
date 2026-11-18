"""The legend of Visualize's ROC curve writes the area with its DeLong interval by the rule of the accuracy of a
test (``biostats.diagnostic.auc_interval_missing``): none for a group of one participant, none of no width."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.visualization_utils import create_roc_curve


def _legend(frame: pd.DataFrame) -> str:
    figure, error = create_roc_curve(frame, "ill", "score", "Yes")
    assert error is None
    return str(figure.data[0].name)


def test_the_legend_writes_no_interval_that_the_accuracy_page_does_not() -> None:
    """The legend read "AUC = 1.000, 95% CI 1.000 to 1.000, DeLong" for two groups that do not overlap, and "AUC =
    0.400, 95% CI 0.222 to 0.578, DeLong" for one participant with the condition among 30: the two intervals the
    accuracy page no longer writes."""
    apart = pd.DataFrame(
        {"ill": ["Yes"] * 40 + ["No"] * 40, "score": [*range(100, 140), *range(40)]}
    )
    # the legend says so: the area alone is also what an interval that could not be computed leaves
    assert _legend(apart) == "ROC curve (AUC = 1.000, no DeLong interval for these data)"
    alone = pd.DataFrame({"ill": ["Yes"] + ["No"] * 30, "score": [11.5, *range(30)]})
    assert _legend(alone) == "ROC curve (AUC = 0.400, no DeLong interval for these data)"
    one_value = pd.DataFrame({"ill": ["Yes"] * 15 + ["No"] * 16, "score": [7.0] * 31})
    assert _legend(one_value) == "ROC curve (AUC = 0.500, no DeLong interval for these data)"
    # an ordinary score keeps its interval and the method
    rng = np.random.default_rng(3)
    ill = rng.random(400) < 0.25
    usual = pd.DataFrame(
        {
            "ill": np.where(ill, "Yes", "No"),
            "score": np.where(ill, rng.normal(60, 15, 400), rng.normal(40, 15, 400)),
        }
    )
    legend = _legend(usual)
    assert (
        legend.startswith("ROC curve (AUC = 0.")
        and ", 95% CI 0." in legend
        and legend.endswith(", DeLong)")
    )
