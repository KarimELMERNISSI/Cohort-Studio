"""
A rule that cannot be evaluated is shown on Validate with its reason, which can quote a cell of the data: as text,
never as Markdown that the browser would follow (a cell such as ![x](http://...) would have been fetched).
"""

from __future__ import annotations

import pandas as pd

from app_pages import data_monitoring as dm


class _Shown:
    """What the check draws: its errors, and nothing else."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args, **kwargs) -> None:
        self.errors.append(message)

    def dataframe(self, *args, **kwargs) -> None:
        pass


def test_a_rule_that_cannot_be_evaluated_is_told_as_text() -> None:
    df = pd.DataFrame({"x": ["1", "![x](http://canary.invalid/p.png)"]})
    family = {"operator": "any", "r": {"expression": "pd.to_numeric(df['x']) > 1"}}
    shown = _Shown()
    dm.apply_family_mask(df, "clinical_anomalies", family, shown)
    (message,) = shown.errors
    assert "canary" in message  # the reason quotes the cell, escaped
    assert "![x](" not in message and "](http" not in message
    assert "`r`" in message and "(implausible values)" in message
