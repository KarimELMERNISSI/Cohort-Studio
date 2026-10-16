"""The participant flow as a static PNG for the Word dossier."""

from __future__ import annotations

import pandas as pd

from core.attrition import AnalysisCount, participant_flow
from core.population import catalogue
from core.session import DataSession
from manage.flow_figure import flow_png

CONFIG = {
    "mask_families": {
        "clinical_anomalies": {"operator": "any", "Implausible age": {"expression": "df['age'] > 110"}},
        "study_inclusion": {"operator": "all", "Adult": {"expression": "df['age'] >= 18"}},
    }
}


def test_the_flow_is_drawn_as_a_png() -> None:
    session = DataSession()
    session.install(pd.DataFrame({"age": [12, 40, 150, 60, 30]}), record=False)
    session.set_ticked([c.id for c in catalogue(CONFIG)])
    flow = participant_flow(
        session.population(CONFIG), dataset="cohort", analyses=[AnalysisCount("t-test", 3, 2)]
    )
    png = flow_png(flow, dpi=80)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5_000
