"""
The study definition (core.protocol, M6): its fingerprint covers the decisions of a study and nothing else, read as the
rules select rows; its file is checked by a checksum that never changes, so that a study saved by an earlier version is
always read and a body changed outside the application never is; a study is named by a slug of its dataset.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import core.protocol as protocol_module
from core.lineage import canonical_json
from core.protocol import (
    ProtocolMeta,
    StudyProtocol,
    body_checksum,
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
# pinned: a change to the view, or to how a configuration is read, changes every study's fingerprint, which results
# record; it must be deliberate (a new RULES_VERSION when what the rules select changes)
TUTORIAL_HASH = "8cb610cbdde7b8c6cd25febb37db9c8e5b10c791838f0b7fa59efdec19f3a2d6"
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
    },
    "map_mmhg": {"formula": "(sbp + 2 * dbp) / 3"},
}
FOLLOW_UP = {"index": "enrol_date", "end": "last_contact_date", "note": "as the protocol says"}

# every part of a body: a threshold family, its criterion ticked, a tick naming nothing, rules written loosely (spaces,
# a decimal bound, "numeric": false), derived variables with a decimal, a whole and a text tolerance, a follow-up note
GOLDEN = {
    "mask_families": {
        "clinical_anomalies": {
            "operator": "any",
            "Implausible age": {"expression": " df['age'] > 110 ", "description": "Typing errors."},
            "potassium_mmol_l": {
                "numeric": True,
                "lower_bound": 2.5,
                "upper_bound": 7.0,
                "strategy": "exclude",
            },
        },
        "signs": {
            "operator": {"threshold": 2},
            "fever": {"expression": "df['temp'] > 38", "numeric": False},
            "tachycardia": {"numeric": True, "column": "hr", "lower_bound": 100.0},
        },
    },
    "population": {
        "ticked": [
            "signs::At least 2 rules of signs",
            "clinical_anomalies::potassium_mmol_l",
            "nowhere::nothing",
        ]
    },
    "follow_up": {
        "index": "enrol_date",
        "end": "last_contact_date",
        "note": "as the protocol says",
    },
    "derived_variables": {
        "bmi_kg_m2": {
            "formula": " weight_kg / (height_cm / 100) ** 2 ",
            "inverses": {"weight_kg": "bmi_kg_m2 * (height_cm / 100) ** 2 "},
            "tolerance": 0.2,
            "accepted": True,
            "note": "WHO",
            "origin": "ai",
            "kind": "ratio",
        },
        "map_mmhg": {"formula": "(sbp + 2 * dbp) / 3", "tolerance": 1.0},
        "fever_flag": {"formula": "temp > 38", "tolerance": "0.5"},
    },
}
GOLDEN_VIEW = (
    '{"derived_variables":[["bmi_kg_m2",{"accepted":true,"formula":"weight_kg / (height_cm / 100) ** 2",'
    '"inverses":{"weight_kg":"bmi_kg_m2 * (height_cm / 100) ** 2"},"tolerance":0.2}],["map_mmhg",'
    '{"formula":"(sbp + 2 * dbp) / 3","tolerance":1}],["fever_flag",{"formula":"temp > 38","tolerance":"0.5"}]],'
    '"follow_up":{"end":"last_contact_date","index":"enrol_date"},"mask_families":[["clinical_anomalies",'
    '["any",null],[["Implausible age",{"expression":"df[\'age\'] > 110"}],["potassium_mmol_l",{"lower_bound":2.5,'
    '"numeric":true,"strategy":"exclude","upper_bound":7}]]],["signs",["threshold",2],[["fever",'
    '{"expression":"df[\'temp\'] > 38"}],["tachycardia",{"column":"hr","lower_bound":100,"numeric":true}]]]],'
    '"population":{"ticked":["clinical_anomalies::potassium_mmol_l","signs::At least 2 rules of signs"]},'
    '"rules_version":1}'
)
GOLDEN_HASH = "03d288298e227e0687ee0389da0ed0476379ce0d499c0ef55c24af2e4c6eb092"
# how every saved file is checked: this must never change
GOLDEN_CHECKSUM = "fea1c5d89a8197bddeaefd66eaeda6edcb78bc5bb820b50619c05d0e6a0e0842"


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


def test_every_part_of_a_body_has_a_pinned_view_fingerprint_and_checksum() -> None:
    assert canonical_json(hashed_view(GOLDEN)) == GOLDEN_VIEW
    assert protocol_hash(GOLDEN) == GOLDEN_HASH
    assert body_checksum(body_of(GOLDEN)) == GOLDEN_CHECKSUM


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


def _as_written_loosely(body: dict) -> None:
    anomalies = _families(body)["clinical_anomalies"]
    anomalies["lvef_pct"]["upper_bound"] = 85.0
    anomalies["Implausible age"]["expression"] = f"  {anomalies['Implausible age']['expression']} "
    anomalies["Implausible age"]["numeric"] = False


@pytest.mark.parametrize(
    "change",
    [
        _labels_and_order,
        _as_written_loosely,
        lambda body: _families(body)["study_inclusion"].update(operator="All"),
        lambda body: body["population"]["ticked"].append("clinical_anomalies::No such rule"),
    ],
    ids=[
        "labels and order of keys and ticks",
        "a decimal bound, spaces, numeric false",
        "operator case",
        "a tick naming nothing",
    ],
)
def test_the_fingerprint_ignores_what_selects_nothing_else(change) -> None:
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


def _swap_derived(body: dict) -> None:
    body["derived_variables"] = dict(reversed(list(body["derived_variables"].items())))


@pytest.mark.parametrize(
    "change",
    [
        lambda body: _families(body)["clinical_anomalies"]["Implausible age"].update(
            expression="(df['age'] < 0) | (df['age'] > 120)"
        ),
        lambda body: _families(body)["clinical_anomalies"]["lvef_pct"].update(upper_bound=85.5),
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
        _swap_derived,
        lambda body: body["follow_up"].update(end="death_date"),
    ],
    ids=[
        "a rule",
        "a bound",
        "an operator",
        "the order of families",
        "the order of rules",
        "a tick",
        "a formula",
        "an inverse",
        "a tolerance",
        "an acceptance",
        "the order of derived variables",
        "the follow-up",
    ],
)
def test_the_fingerprint_changes_with_each_decision(change) -> None:
    body = _body()
    change(body)
    assert protocol_hash(body) != BASE


def test_a_new_reading_of_the_rules_changes_the_fingerprint_and_reads_the_old_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the rules select may change (how a missing value meets "!=", say): the studies with criteria get another
    fingerprint, and their files, saved before, are still read."""
    saved = json.loads(json.dumps(StudyProtocol(_body(), ProtocolMeta("cohort")).to_file()))
    derived_only = {"derived_variables": copy.deepcopy(DERIVED)}
    before = protocol_hash(derived_only)
    monkeypatch.setattr(protocol_module, "RULES_VERSION", 2)
    read, problems = StudyProtocol.from_file(saved)
    assert read is not None and problems == []
    assert read.hash != BASE and protocol_hash(derived_only) == before


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


def test_the_population_holds_its_ticks_and_nothing_else() -> None:
    """An uploaded configuration is merged into the session's: a count or a note in its population would be a
    finding in the study definition, outside its fingerprint."""
    config = {"population": {"ticked": ["b::y", "a::x"], "rows_seen": 1206, "note": "P-0042"}}
    assert body_of(config) == {"population": {"ticked": ["a::x", "b::y"]}}
    assert body_of(config, ticked=iter(["c::z"])) == {"population": {"ticked": ["c::z"]}}
    assert body_of({"population": {"rows_seen": 1206}}) == {}
    ticks_in_a_set = {**_body(), "population": {"ticked": set(TICKED)}}
    assert protocol_hash(ticks_in_a_set) == BASE


def test_the_file_form_reads_back_as_written_and_refuses_a_changed_body() -> None:
    meta = ProtocolMeta("tutorial_cohort", title="Tutorial cohort", revision=2, parent="a" * 64)
    protocol = StudyProtocol(_body(), meta)
    data = json.loads(json.dumps(protocol.to_file()))
    assert data["checksum"] == protocol.checksum == body_checksum(data["body"])
    read, problems = StudyProtocol.from_file(data)
    assert read is not None and problems == []
    assert read.hash == protocol.hash and read.meta == meta

    reordered = copy.deepcopy(data)  # the order of the rules is part of the study
    _swap_rules(reordered["body"])
    edited = copy.deepcopy(data)
    edited["body"]["mask_families"]["clinical_anomalies"]["P-0042 Jane"] = {"expression": "1"}
    for changed in (reordered, edited):
        read, problems = StudyProtocol.from_file(changed)
        assert read is None and "changed outside CohortStudio" in problems[0]
        assert "P-0042" not in " ".join(problems)  # the reasons quote nothing from the file

    for change, reason in (
        (
            lambda d: d["body"].update(analysis_plan={"x": 1}),
            "later version: it holds 1 section this version does not read",
        ),
        (  # a key of the file is counted, never quoted: it could hold a name, or forge a log line
            lambda d: d["body"].update({"P-0042 Jane\nINFO forged": 1, "concepts": {}}),
            "later version: it holds 2 sections this version does not read",
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
        assert "P-0042" not in " ".join(problems)
    assert StudyProtocol.from_file([1, 2]) == (None, ["not a study definition"])


def test_a_file_crafted_to_break_the_reading_gives_none() -> None:
    deep: dict = {}
    level = deep
    for _ in range(5000):
        level["x"] = {}
        level = level["x"]
    body = {"mask_families": {"f": {"operator": "any", "r": {"expression": "1", "x": deep}}}}
    data = {"schema": 1, "checksum": "0" * 64, "meta": {"study_id": "cohort"}, "body": body}
    assert StudyProtocol.from_file(data) == (None, ["not a study definition this version can read"])
    # a number too large to be a float: read as written, never converted
    huge = {"derived_variables": {"x": {"formula": "y", "tolerance": 10**400}}}
    saved = json.loads(json.dumps(StudyProtocol(huge, ProtocolMeta("cohort")).to_file()))
    read, _ = StudyProtocol.from_file(saved)
    assert read is not None and len(read.hash) == 64


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


def test_revisions_chain_by_checksum() -> None:
    meta = ProtocolMeta("cohort", title="Cohort")
    first = next_revision(None, _body(), meta)
    assert (first.meta.revision, first.meta.parent, first.meta.parent_revision) == (1, None, None)
    changed = _body()
    changed["follow_up"]["end"] = "death_date"
    second = next_revision(first, changed, meta)
    assert (second.meta.revision, second.meta.parent, second.meta.parent_revision) == (
        2,
        first.checksum,
        1,
    )
    assert next_revision(second, _body(), meta, number=7).meta.revision == 7
