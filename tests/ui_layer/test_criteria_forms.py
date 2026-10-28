"""
Validate's criteria forms keep what the configuration holds (M5b step 6).

The anomalies form set its family's operator to "any", and the inclusion form set theirs to "all", on every run. An
uploaded family that keeps "at least k rules" lost it when its tab opened: the criterion the user had ticked no longer
existed, so the analysis population dropped it without a word.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def _form() -> None:
    import streamlit as st

    from app_pages.data_monitoring import (
        add_domain_expert_based_anomaly_mask,
        add_study_inclusion_mask,
    )

    config = st.session_state["config"]
    form = (
        add_domain_expert_based_anomaly_mask
        if st.session_state["form"] == "anomalies"
        else add_study_inclusion_mask
    )
    form(config, st)
    st.session_state["operators"] = {
        name: family.get("operator") for name, family in config["mask_families"].items()
    }


@pytest.mark.parametrize(
    ("form", "family", "operator"),
    [
        ("anomalies", "clinical_anomalies", {"threshold": 2}),
        ("inclusion", "study_inclusion", {"threshold": 2}),
        ("anomalies", "clinical_anomalies", "All"),  # read as "all", as an upload may spell it
    ],
)
def test_opening_a_form_keeps_its_familys_operator(
    form: str, family: str, operator: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import utils.data_paths as dp
    from core.population import active_criteria, catalogue

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    config = {
        "mask_families": {
            family: {
                "operator": operator,
                "old": {"numeric": False, "expression": "df['age'] > 80"},
                "age": {
                    "numeric": True,
                    "lower_bound": 18,
                    "upper_bound": 85,
                    "strategy": "exclude",
                },
            }
        }
    }
    ticked = [criterion.id for criterion in catalogue(config)]
    at = AppTest.from_function(_form, default_timeout=60)
    at.session_state["config"] = config
    at.session_state["form"] = form
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["operators"][family] == operator
    # the criteria the user ticked still exist: the population keeps them
    assert [criterion.id for criterion in active_criteria(config, ticked)] == ticked


@pytest.mark.parametrize(
    ("form", "family", "operator"),
    [("anomalies", "clinical_anomalies", "any"), ("inclusion", "study_inclusion", "all")],
)
def test_a_family_a_form_creates_gets_its_operator(
    form: str, family: str, operator: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import utils.data_paths as dp

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    at = AppTest.from_function(_form, default_timeout=60)
    at.session_state["config"] = {"mask_families": {}}
    at.session_state["form"] = form
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["operators"] == {family: operator}


def test_a_family_without_an_operator_keeps_how_it_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family without an operator reads "any": the inclusion tab gave such a loaded family "all", and its criteria
    changed role (their fingerprints with them)."""
    import utils.data_paths as dp
    from core.population import catalogue

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    config = {
        "mask_families": {
            "study_inclusion": {"old": {"numeric": False, "expression": "df['age'] > 80"}}
        }
    }
    before = [criterion.fingerprint for criterion in catalogue(config)]
    at = AppTest.from_function(_form, default_timeout=60)
    at.session_state["config"] = config
    at.session_state["form"] = "inclusion"
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert "operator" not in config["mask_families"]["study_inclusion"]
    assert [criterion.fingerprint for criterion in catalogue(config)] == before
