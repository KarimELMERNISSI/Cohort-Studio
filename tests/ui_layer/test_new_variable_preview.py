"""The preview of a new variable on Enrich: statistics on the analysis population, entry errors flagged."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav
from ui.components.population import _family_key

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture
def app(tmp_path: Path):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path)
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "variable_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    yield at
    dp.DATA_ROOT = previous


def _statistics(at: AppTest) -> dict[str, str]:
    table = next(d.value for d in at.dataframe if list(d.value.columns) == ["Statistic", "Value"])
    return dict(table.itertuples(index=False))


def test_bmi_preview_shows_typical_values_and_the_entry_errors(app: AppTest) -> None:
    app.switch_page(nav.page_path("enrich")).run()
    app.text_input(key="variable_name_input").input("bmi").run()
    app.text_area(key="formula_input").input("weight_kg / ( height_cm /100 ) **2").run()
    next(b for b in app.button if b.label == "Compute & Preview").click().run()
    _ok(app)
    stats = _statistics(app)
    assert 24 < float(stats["Median"]) < 30 and float(stats["Max"]) > 10_000  # heights in metres
    assert any("lie far outside the others" in w.value for w in app.warning)
    assert any("No inclusion or exclusion criterion is ticked" in c.value for c in app.caption)

    app.checkbox(
        key=_family_key("clinical_anomalies")
    ).check().run()  # the entry errors leave the population
    _ok(app)
    assert any("on the analysis population (1,190 of 1,206 rows" in m.value for m in app.markdown)
    stats = _statistics(app)
    assert float(stats["Max"]) < 80 and float(stats["Min"]) > 10
