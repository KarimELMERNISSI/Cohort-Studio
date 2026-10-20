"""
The session keys that one part of the application writes and another reads (M5b, runtime).

Once only the open tab of a page runs (``st.tabs`` with ``on_change="rerun"``), a key that one tab writes and another
tab, or a later run, reads is how they share state. Each such key is declared here with its owner (the module that
writes it), its lifetime and the other modules that read it, so that the sharing stays visible. The lifetimes are
those ``ui.state`` gives today (``DERIVED_KEYS``, ``KEPT_ON_DATA_CHANGE``); a test checks that they agree, until
M5b step 7 derives those lists from this file.

Lifetimes:

- ``session``: kept for the whole session;
- ``dataset``: cleared when another dataset is loaded, kept when the same dataset changes;
- ``data_change``: cleared on any change of the data, and when another dataset is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Lifetime = Literal["session", "dataset", "data_change"]


@dataclass(frozen=True)
class Key:
    name: str
    owner: str  # the module that writes it
    lifetime: Lifetime
    readers: tuple[str, ...] = ()  # the other modules that read it
    note: str = ""


KEYS: tuple[Key, ...] = (
    Key(
        "config",
        "ui.state",
        "session",
        ("app_pages.data_monitoring", "app_pages.data_enrichment"),
        "the study configuration: the criteria definitions (edited on Validate), the enrichments, the folders",
    ),
    Key(
        "anomaly_check_done",
        "app_pages.data_monitoring",
        "dataset",
        note="Validate: the anomaly check's results stay shown after the click that ran it",
    ),
    Key(
        "inclusion_check_done",
        "app_pages.data_monitoring",
        "dataset",
        note="Validate: the same, for the inclusion criteria",
    ),
    Key(
        "validity_outlier_method",
        "app_pages.data_quality_dashboard",
        "data_change",
        note="the outlier detection settings, read by the scorecard above them",
    ),
    Key(
        "validity_outlier_params",
        "app_pages.data_quality_dashboard",
        "data_change",
        note="the parameters of that method",
    ),
    Key(
        "rename_dict",
        "app_pages.home",
        "data_change",
        ("app_pages.data_insight",),
        "the renames pending in the Renaming tab, which the AI suggestions add to",
    ),
    Key(
        "epidemiology_results",
        "app_pages.epidemiology",
        "data_change",
        note="written by the Analysis tab, read by the Power tab to prefill it",
    ),
    Key(
        "epidemiology_group_col",
        "app_pages.epidemiology",
        "data_change",
        note="the grouping variable of those results",
    ),
)

BY_NAME: dict[str, Key] = {key.name: key for key in KEYS}

__all__ = ["BY_NAME", "KEYS", "Key", "Lifetime"]
