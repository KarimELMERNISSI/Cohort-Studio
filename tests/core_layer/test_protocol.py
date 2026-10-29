"""
The study definition (core.protocol, M6): its fingerprint covers the decisions of a study and nothing else, its file
form reads back only as written, and a study is named by a slug of its dataset.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.protocol import (
    ProtocolMeta,
    StudyProtocol,
    body_of,
    diff_lines,
    hashed_view,
    is_study_id,
    next_revision,
    protocol_hash,
    study_id_for,
    with_body,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "assets" / "tutorial" / "tutorial_config.json").read_text(encoding="utf-8")
)
# pinned: a change to the view, or to how a configuration is read, would change every study's fingerprint
TUTORIAL_HASH = "77ff3d46b546f328c7a65a27fa8784e1fe420a6b56b1eaa29185a17da44c79e3"
TICKED = ["study_inclusion::Written consent", "clinical_anomalies::Implausible age"]
UNTICKED = "study_inclusion::age"  # a criterion of the configuration, not ticked
DERIVED = {
    "bmi_kg_m2": {
        "formula": "weight_kg / (height_cm / 100) ** 2",
        "inverses": {"weight_kg": "bmi_kg_m2 * (height_cm / 100) ** 2"},
        "kind": "ratio",
        "origin": "ai",
        "note": "WHO",
        "tolerance": 0.2,
    }
}
FOLLOW_UP = {"index": "enrol_date", "end": "last_contact_date", "note": "as the protocol says"}


def _body() -> dict:
    return {
        **body_of(CONFIG, TICKED),
        "follow_up": copy.deepcopy(FOLLOW_UP),
        "derived_variables": copy.deepcopy(DERIVED),
    }


BASE = protocol_hash(_body())


def _families(body: dict) -> dict:
    return body["mask_families"]


def _reversed_keys(value):
    if isinstance(value, dict):
        return {k: _reversed_keys(v) for k, v in reversed(list(value.items()))}
    return value


def test_the_tutorial_configuration_has_a_pinned_fingerprint() -> None:
    assert "age" in CONFIG["mask_families"]["study_inclusion"] and UNTICKED not in TICKED
    body = body_of(CONFIG)
    assert list(body) == ["mask_families"]  # its description is a label, outside the body
    assert protocol_hash(body) == TUTORIAL_HASH
    assert len(TUTORIAL_HASH) == 64


def _labels_and_order(body: dict) -> None:
    for family, members in _families(body).items():
        _families(body)[family] = {
            name: _reversed_keys(spec) if isinstance(spec, dict) else spec
            for name, spec in members.items()
        }
        for spec in members.values():
            if isinstance(spec, dict):
                spec["description"] = "reworded"
    body["population"]["ticked"].reverse()
    body["derived_variables"]["bmi_kg_m2"].update(note="other", origin="user", kind="")
    body["follow_up"]["note"] = "reworded"


@pytest.mark.parametrize(
    "change",
    [
        _labels_and_order,
        lambda body: _families(body)["study_inclusion"].update(operator="All"),
        lambda body: body["population"]["ticked"].append("clinical_anomalies::No such rule"),
    ],
    ids=["labels and order of keys and ticks", "operator case", "a tick naming nothing"],
)
def test_the_fingerprint_ignores_what_decides_nothing(change) -> None:
    body = _body()
    change(body)
    assert protocol_hash(body) == BASE


def _swap_families(body: dict) -> None:
    body["mask_families"] = dict(reversed(list(_families(body).items())))


def _swap_rules(body: dict) -> None:
    members = _families(body)["clinical_anomalies"]
    rules = [(k, v) for k, v in members.items() if k != "operator"]
    _families(body)["clinical_anomalies"] = {
        "operator": members["operator"],
        **dict([rules[1], rules[0], *rules[2:]]),
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda body: _families(body)["clinical_anomalies"]["Implausible age"].update(
            expression="(df['age'] < 0) | (df['age'] > 120)"
        ),
        lambda body: _families(body)["clinical_anomalies"].update(operator="all"),
        _swap_families,
        _swap_rules,
        lambda body: body["population"]["ticked"].append(UNTICKED),
        lambda body: body["derived_variables"]["bmi_kg_m2"].update(formula="weight_kg / height_cm"),
        lambda body: body["derived_variables"]["bmi_kg_m2"]["inverses"].update(
            weight_kg="bmi_kg_m2"
        ),
        lambda body: body["derived_variables"]["bmi_kg_m2"].update(tolerance=0.3),
        lambda body: body["derived_variables"]["bmi_kg_m2"].update(accepted=True),
        lambda body: body["follow_up"].update(end="death_date"),
    ],
    ids=[
        "a rule",
        "an operator",
        "the order of families",
        "the order of rules",
        "a tick",
        "a formula",
        "an inverse",
        "a tolerance",
        "an acceptance",
        "the follow-up",
    ],
)
def test_the_fingerprint_changes_with_each_decision(change) -> None:
    body = _body()
    change(body)
    assert protocol_hash(body) != BASE


def test_a_configuration_and_its_body_round_trip() -> None:
    body = _body()
    config = {"description": "kept", "folder_names": {"root": "kept"}, "follow_up": {"old": 1}}
    merged = with_body(config, body)
    assert merged["description"] == "kept" and merged["folder_names"] == {"root": "kept"}
    assert body_of(merged) == body
    # a section the body lacks leaves the configuration; the criteria stay, empty
    emptied = with_body(merged, {})
    assert emptied["mask_families"] == {} and "follow_up" not in emptied
    assert body_of(emptied) == {} == body_of({"mask_families": {}, "population": {"ticked": []}})


def test_the_file_form_reads_back_as_written_and_refuses_an_edited_body() -> None:
    meta = ProtocolMeta("tutorial_cohort", title="Tutorial cohort", revision=2, parent="a" * 64)
    protocol = StudyProtocol(_body(), meta)
    data = json.loads(json.dumps(protocol.to_file()))
    read, problems = StudyProtocol.from_file(data)
    assert read is not None and problems == []
    assert read.hash == protocol.hash and read.meta == meta

    edited = copy.deepcopy(data)
    edited["body"]["mask_families"]["clinical_anomalies"]["P-0042 Jane"] = {"expression": "1"}
    read, problems = StudyProtocol.from_file(edited)
    assert read is None and "changed outside CohortStudio" in problems[0]
    assert "P-0042" not in " ".join(problems)  # the reasons quote nothing from the file

    for change, reason in (
        (
            lambda d: d["body"].update(analysis_plan={"x": 1}),
            "later version: it holds analysis_plan",
        ),
        (lambda d: d.update(schema=2), "another version"),
        (lambda d: d["meta"].update(study_id="../x"), "study's id"),
        (lambda d: d["meta"].update(revision=0), "revision"),
        (lambda d: d["meta"].update(parent="P-0042"), "parent"),
    ):
        other = copy.deepcopy(data)
        change(other)
        read, problems = StudyProtocol.from_file(other)
        assert read is None and reason in problems[0], problems
    assert StudyProtocol.from_file([1, 2]) == (None, ["not a study definition"])


def test_a_body_cannot_change_once_read() -> None:
    protocol = StudyProtocol(_body(), ProtocolMeta("cohort"))
    with pytest.raises(TypeError):
        protocol.body["mask_families"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        protocol.body["mask_families"]["clinical_anomalies"]["x"] = {}  # type: ignore[index]
    assert protocol.hash == BASE
    assert protocol.body["population"]["ticked"] == tuple(sorted(TICKED))


def test_the_diff_shows_each_change_labels_included() -> None:
    old, new = _body(), _body()
    _families(new)["clinical_anomalies"]["Implausible age"]["description"] = "Ages out of range."
    _families(new)["clinical_anomalies"]["Implausible age"]["expression"] = "df['age'] > 120"
    lines = diff_lines(old, new)
    assert any(line.startswith("-") and "df['age'] > 110" in line for line in lines)
    assert any(line.startswith("+") and "df['age'] > 120" in line for line in lines)
    assert any(line.startswith("+") and "Ages out of range." in line for line in lines)
    assert diff_lines(old, copy.deepcopy(old)) == []
    assert "description" not in json.dumps(hashed_view(new))


@pytest.mark.parametrize(
    ("name", "study_id"),
    [
        ("tutorial_cohort", "tutorial_cohort"),
        ("Tutorial Cohort (v2)", "tutorial_cohort_v2"),
        ("Données été 2024", "donnees_ete_2024"),
        ("../../etc/passwd", "etc_passwd"),
        ("CON", "con_study"),
        ("", "study"),
        (None, "study"),
        ("%%%", "study"),
        ("x" * 100, "x" * 64),
    ],
)
def test_a_study_is_named_by_a_slug_of_its_dataset(name, study_id) -> None:
    assert study_id_for(name) == study_id and is_study_id(study_id)


def test_revisions_chain_by_fingerprint() -> None:
    meta = ProtocolMeta("cohort", title="Cohort")
    first = next_revision(None, _body(), meta)
    assert (first.meta.revision, first.meta.parent, first.meta.parent_revision) == (1, None, None)
    changed = _body()
    changed["follow_up"]["end"] = "death_date"
    second = next_revision(first, changed, meta)
    assert (second.meta.revision, second.meta.parent, second.meta.parent_revision) == (
        2,
        first.hash,
        1,
    )
    assert next_revision(second, _body(), meta, number=7).meta.revision == 7
