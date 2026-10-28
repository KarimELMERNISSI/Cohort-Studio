"""
The AI assistant's "Add" replaces a rule of the same name only when asked, as Validate's form asks before it overwrites
one (M5b step 6). It replaced it without a word, and a ticked criterion could change under the user.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

OLD = {"numeric": False, "expression": "df['age'] > 80"}


def _script() -> None:
    import types

    import streamlit as st

    from app_pages.data_monitoring import render_ai_criteria_assistant

    ss = st.session_state
    if "config" not in ss:
        ss["rag_manager"] = types.SimpleNamespace(
            initialized=True
        )  # an AI engine that has answered
        family = {"operator": "any", "old": dict(ss["old"])}
        ss["config"] = {"mask_families": {"clinical_anomalies": family}}
        ss["ai_suggestions_clinical_anomalies"] = [
            {"name": "old", "expression": "df['age'] > 90", "missing_variables": []},
            {"name": "young", "expression": "df['age'] < 18", "missing_variables": []},
        ]
    render_ai_criteria_assistant(st, ss["config"], "clinical_anomalies", "anomaly")


def test_a_suggestion_replaces_a_rule_of_its_name_only_when_asked() -> None:
    at = AppTest.from_function(_script, default_timeout=60)
    at.session_state["old"] = OLD
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    # a new name is added at once; the name of a rule the family holds needs Replace ticked first
    assert not at.button(key="ai_add_clinical_anomalies_1").disabled
    assert at.button(key="ai_add_clinical_anomalies_0").disabled
    family = at.session_state["config"]["mask_families"]["clinical_anomalies"]
    assert family["old"] == OLD
    at.checkbox(key="ai_replace_clinical_anomalies_0").check().run()
    at.button(key="ai_add_clinical_anomalies_0").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    family = at.session_state["config"]["mask_families"]["clinical_anomalies"]
    assert family["old"] == {"numeric": False, "expression": "df['age'] > 90"}
