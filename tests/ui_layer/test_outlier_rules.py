"""A rule frozen from Validate's outliers records the columns the method analysed, which its words name (M5b step 6)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from core.criteria import describe_definition
from tests.ui_layer.test_rules_written import rules_not_read_as_their_kinds
from ui import nav

MAIN = str(Path(__file__).resolve().parents[2] / "main.py")


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "outlier_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    return at


def test_a_rule_frozen_from_outliers_names_the_columns_analysed(app: AppTest) -> None:
    """An identifier chosen for a multivariate method is not analysed (one value per row says nothing of a row's
    values), and it is among the columns offered by default: the rule named it as if it had been used."""
    from app_pages import data_monitoring as dm

    app.switch_page(nav.page_path("validate")).run()
    app.session_state[dm.TAB_KEY] = "Handle Outliers"
    app.run()
    app.selectbox(key="outlier_detection_method").set_value("Isolation Forest").run()
    app.selectbox(key="outlier_handling_strategy").set_value("remove").run()
    columns = ["age", "sbp_baseline", "patient_id"]
    app.multiselect(key="outlier_columns_multivariate").set_value(columns).run()
    _ok(app)
    app.button(key="outliers_data_apply_criteria").click().run()
    _ok(app)
    rules = app.session_state["config"]["mask_families"]["outliers"]
    ((name, rule),) = [(n, r) for n, r in rules.items() if n != "operator"]
    assert rule["columns"] == ["age", "sbp_baseline"] and "patient_id" not in name
    assert rules_not_read_as_their_kinds(app.session_state["config"]) == []
    assert describe_definition(name, rule).endswith(
        "(flagged by Isolation Forest on age and sbp_baseline together)"
    )
