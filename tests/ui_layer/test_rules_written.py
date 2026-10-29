"""
Every rule the pages write reads as its kind, holds only its kind's keys and passes the checks of a loaded file
(M5b step 6): Validate's two forms here, the outliers frozen as rows (test_outlier_rules.py) and the AI assistant's
"Add" (test_ai_criteria_add.py). The gate of tests/test_quality_gates.py reads how the code is spelt; these read what
it wrote.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from streamlit.testing.v1 import AppTest

from core.criteria import criterion_problems, rule_kind


def rules_not_read_as_their_kinds(config: Mapping[str, Any]) -> list[str]:
    """The rules of ``config`` that do not read as a kind, hold a key their kind does not, or fail its checks."""
    wrong = []
    for family, members in config["mask_families"].items():
        for name, definition in members.items():
            if name == "operator":
                continue
            kind = rule_kind(name, definition)
            stated = ("kind",) if kind.recognise is None else ()
            if set(definition) - {*kind.keys, kind.members_key, *stated}:
                wrong.append(f"{family}: {name} holds another key")
            if criterion_problems(name, definition):
                wrong.append(f"{family}: {name} fails the checks")
    return wrong


def _forms() -> None:
    import streamlit as st

    from app_pages.data_monitoring import (
        add_domain_expert_based_anomaly_mask,
        add_study_inclusion_mask,
    )

    st.session_state.setdefault("config", {"mask_families": {}})
    add_domain_expert_based_anomaly_mask(st.session_state["config"], st)
    add_study_inclusion_mask(st.session_state["config"], st)


def _click(at: AppTest, label: str) -> None:
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def test_the_forms_write_rules_that_read_as_their_kinds() -> None:
    at = AppTest.from_function(_forms, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    # an expression on the anomaly form
    at.text_input(key="anomaly_mask_name").input("Implausible age").run()
    at.text_area(key="clinical_anomalies_txt_area").input("df['age'] > 110").run()
    _click(at, "Add Plausibility Rule")
    # a numeric range on the inclusion form
    at.text_input(key="inclusion_mask_name").input("age").run()
    at.checkbox(key="is_num_inclusion").check().run()
    at.checkbox(key="study_inclusion_lower_bound").check().run()
    at.number_input(key="study_inclusion_lower_bound_input").set_value(18.0).run()
    _click(at, "Add Inclusion Criterion")
    families = at.session_state["config"]["mask_families"]
    assert families["clinical_anomalies"]["Implausible age"] == {
        "numeric": False,
        "expression": "df['age'] > 110",
    }
    assert rule_kind("age", families["study_inclusion"]["age"]).id == "numeric"
    assert rules_not_read_as_their_kinds(at.session_state["config"]) == []
