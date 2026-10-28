"""
An uploaded configuration's criteria are added to the session's whole, never merged into them key by key
(core.population.merge_families, and Validate's upload, app_pages.data_monitoring._merge_uploaded).

The upload merged a rule of the same name key by key: a numeric rule "age" and an uploaded expression "age" became one
rule holding both, which was refused ("holds both a numeric range and an expression") while the page said the
configuration was loaded. A rule the session holds now keeps the session's definition, and is named when the file's
differs.
"""

from __future__ import annotations

from core.population import configuration_problems, merge_families

NUMERIC_AGE = {
    "numeric": True,
    "lower_bound": 18,
    "upper_bound": 85,
    "strategy": "include",
    "description": "Adults.",
}
EXPRESSION_AGE = {"numeric": False, "expression": "df['age'] >= 21"}
OLD = {"numeric": False, "expression": "df['age'] > 110"}


def test_a_rule_the_session_holds_keeps_its_definition_whole() -> None:
    merged, differing = merge_families(
        {"study_inclusion": {"operator": "all", "age": NUMERIC_AGE}},
        {"study_inclusion": {"operator": "all", "age": EXPRESSION_AGE}},
    )
    assert merged["study_inclusion"]["age"] == NUMERIC_AGE
    assert differing == ["study_inclusion: age"]
    assert configuration_problems({"mask_families": merged}) == []
    # the other way round, an expression rule kept the file's bounds as dead keys
    merged, _ = merge_families(
        {"study_inclusion": {"age": EXPRESSION_AGE}}, {"study_inclusion": {"age": NUMERIC_AGE}}
    )
    assert merged["study_inclusion"]["age"] == EXPRESSION_AGE


def test_what_the_session_lacks_is_added_whole() -> None:
    outliers = {
        "operator": "any",
        "Flagged rows": {"row_ids": [[2, 3]], "method": "zscore", "columns": ["x"]},
    }
    merged, differing = merge_families(
        {"clinical_anomalies": {"operator": "any", "old": OLD}},
        {
            "clinical_anomalies": {
                "operator": "any",
                "young": {"numeric": False, "expression": "df['age'] < 0"},
            },
            "outliers": outliers,
        },
    )
    assert merged["outliers"] == outliers
    assert set(merged["clinical_anomalies"]) == {"operator", "old", "young"}
    assert differing == []


def test_the_sessions_operator_is_kept_and_named_when_the_file_differs() -> None:
    merged, differing = merge_families(
        {"clinical_anomalies": {"operator": {"threshold": 2}, "old": OLD}},
        {"clinical_anomalies": {"operator": "any", "old": OLD}},
    )
    assert merged["clinical_anomalies"]["operator"] == {"threshold": 2}
    assert differing == ["clinical_anomalies: the operator"]
    # equal values are equal, however written: 18 and 18.0 differ in nothing
    _, differing = merge_families(
        {"study_inclusion": {"age": {**NUMERIC_AGE, "lower_bound": 18}}},
        {"study_inclusion": {"age": {**NUMERIC_AGE, "lower_bound": 18.0}}},
    )
    assert differing == []


def test_the_merged_families_share_nothing_with_either_side() -> None:
    uploaded = {"f": {"operator": "any", "a": {"numeric": False, "expression": "df['x'] > 1"}}}
    session = {"f": {"operator": "any"}}
    merged, _ = merge_families(session, uploaded)
    merged["f"]["a"]["expression"] = "changed"
    assert uploaded["f"]["a"]["expression"] == "df['x'] > 1"
    assert (
        "a" not in session["f"]
    )  # the session's configuration is replaced, never changed in place


def test_validates_upload_names_the_criteria_that_keep_the_sessions_definition() -> None:
    from app_pages.data_monitoring import _merge_uploaded

    config = {
        "description": "mine",
        "mask_families": {"study_inclusion": {"operator": "all", "age": NUMERIC_AGE}},
    }
    uploaded = {
        "description": "theirs",
        "folders": {"x": 1},
        "mask_families": {
            "study_inclusion": {"operator": "all", "age": EXPRESSION_AGE},
            "clinical_anomalies": {"operator": "any", "old": OLD},
        },
    }
    notices = _merge_uploaded(config, uploaded)
    assert config["mask_families"]["study_inclusion"]["age"] == NUMERIC_AGE
    assert config["mask_families"]["clinical_anomalies"] == {"operator": "any", "old": OLD}
    assert config["description"] == "mine" and config["folders"] == {
        "x": 1
    }  # added when the session lacks them
    assert len(notices) == 1 and "study_inclusion: age" in notices[0]
    assert configuration_problems(config) == []
