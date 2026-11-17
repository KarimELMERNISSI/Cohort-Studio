"""
A study's privacy choices (core.privacy), the section ``privacy`` of its definition: a file format with one writing
per meaning, a closed reader whose reasons quote nothing of a file, outside the study's fingerprint and inside its
checksum, kept in its revisions and never shared.

    "privacy": {"k": 20, "followed_to_the_event": {"death": {}, "visits": {"beyond_the_cap": true}},
                "rows_not_people": "animal_data", "rows_word": "animals"}
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.disclosure import DEFAULT_K, FLOOR_K, DisclosurePolicy, FollowedToTheEvent
from core.population import configuration_for_sharing
from core.privacy import (
    BELOW_THE_FLOOR,
    LATER_VALUE,
    NOT_PEOPLE_REASONS,
    NOT_READ,
    PRIVACY_KEY,
    ROWS_WORDS,
    SECTION_KEYS,
    Declared,
    StudyPrivacy,
    study_privacy,
    with_privacy,
)
from core.protocol import (
    BODY_KEYS,
    ProtocolMeta,
    StudyProtocol,
    body_checksum,
    body_of,
    diff_lines,
    hashed_view,
    protocol_for_sharing,
    protocol_hash,
    snapshot_params,
    with_body,
)
from manage.protocol_store import ProtocolUnchanged, WorkspaceProtocolStore

PLANTED = "P-0042 Jane"
K_ONLY = {"k": 20}
FOLLOWED = {"followed_to_the_event": {"death": {}, "visits": {"beyond_the_cap": True}}}
NOT_PEOPLE = {"rows_not_people": "animal_data", "rows_word": "animals"}
EVERYTHING = {**K_ONLY, **FOLLOWED, **NOT_PEOPLE}
RULE = {"expression": "df['age'] > 110", "description": "Ages out of range."}
FAMILIES = {"clinical_anomalies": {"operator": "any", "Implausible age": RULE}}


def _body(section: dict | None = None) -> dict:
    body: dict = {"mask_families": copy.deepcopy(FAMILIES)}
    if section is not None:
        body[PRIVACY_KEY] = copy.deepcopy(section)
    return body


def _file(section: object) -> dict:
    """A study file as written, holding ``section`` whatever it is (its checksum is the body's as written)."""
    body = {"mask_families": copy.deepcopy(FAMILIES), PRIVACY_KEY: section}
    return {
        "schema": 1,
        "checksum": body_checksum(body),
        "meta": {"study_id": "cohort"},
        "body": body,
    }


# ----------------------------------------------------------------------------- the format


@pytest.mark.parametrize("section", [K_ONLY, FOLLOWED, NOT_PEOPLE, EVERYTHING])
def test_a_section_reads_back_as_written(section: dict) -> None:
    choices, reasons = StudyPrivacy.from_section(section)
    assert choices is not None and reasons == []
    assert choices.to_section() == section
    assert list(choices.to_section()) == [key for key in SECTION_KEYS if key in section]
    # through a study's file: the section is part of the body, and of its checksum
    protocol = StudyProtocol(_body(section), ProtocolMeta("cohort", revision=1))
    data = json.loads(json.dumps(protocol.to_file()))
    read, problems = StudyProtocol.from_file(data)
    assert read is not None and problems == []
    assert study_privacy(read.body) == choices and read.checksum == protocol.checksum
    assert json.loads(json.dumps(data["body"][PRIVACY_KEY])) == section


def test_the_choices_are_what_the_study_asked() -> None:
    choices = study_privacy(_body(EVERYTHING))
    assert choices.k == 20 and choices.rows_not_people == "animal_data"
    assert choices.followed_to_the_event == (Declared("death"), Declared("visits", True))
    assert choices.word == "animals" and StudyPrivacy().word == "participants"
    # a study that asks nothing holds no section, and is no choice at all
    assert study_privacy({}) == study_privacy(None) == StudyPrivacy() and not StudyPrivacy()
    assert StudyPrivacy().to_section() == {} and StudyPrivacy.from_section({}) == (
        StudyPrivacy(),
        [],
    )
    # a k of 10 is the study asking 10: not the same as asking nothing, and written
    assert StudyPrivacy(k=DEFAULT_K).to_section() == {"k": 10} and StudyPrivacy(k=DEFAULT_K)
    assert StudyPrivacy(k=FLOOR_K).k == 5
    # the closed lists, animal data first
    assert list(NOT_PEOPLE_REASONS)[0] == "animal_data" and ROWS_WORDS == ("animals", "subjects")
    assert SECTION_KEYS == ("k", "followed_to_the_event", "rows_not_people", "rows_word")


def test_equal_choices_have_one_writing_and_one_checksum() -> None:
    """One function writes the section, in one order, and writes no default: the same choices built in another
    order give the same file, and an unchanged choice never makes a revision."""
    first = StudyPrivacy(
        k=20,
        followed_to_the_event=(Declared("visits", True), Declared("death")),
        rows_not_people="animal_data",
        rows_word="animals",
    )
    second = StudyPrivacy(
        rows_word="animals",
        rows_not_people="animal_data",
        followed_to_the_event=(Declared("death"), Declared("visits", beyond_the_cap=True)),
        k=20,
    )
    assert first == second and first.to_section() == second.to_section() == EVERYTHING
    assert json.dumps(first.to_section()) == json.dumps(
        EVERYTHING
    )  # the keys and the outcomes in their order
    assert body_checksum(_body(first.to_section())) == body_checksum(_body(second.to_section()))
    # the default word has one writing: none
    assert StudyPrivacy(rows_word="participants") == StudyPrivacy()
    assert StudyPrivacy(rows_word="participants").to_section() == {}
    # into a configuration and out of it, by the one writer; nothing set removes the section
    config = with_privacy({"description": "d", PRIVACY_KEY: {"k": 5}}, first)
    assert config == {"description": "d", PRIVACY_KEY: EVERYTHING}
    assert with_privacy(config, StudyPrivacy()) == {"description": "d"}


@pytest.mark.parametrize(
    "bad",
    [
        {"k": 4},
        {"k": 20.0},
        {"k": True},
        {"k": "20"},
        {"followed_to_the_event": (Declared("death"), Declared("death", True))},
        {"followed_to_the_event": ("death",)},
        {"rows_not_people": "my mice"},
        {"rows_word": "patients"},
    ],
)
def test_choices_that_no_file_could_hold_are_not_built(bad: dict) -> None:
    with pytest.raises(ValueError):
        StudyPrivacy(**bad)
    for outcome in ("", "   ", "x" * 257, "dea​th", "dea\nth", 3):
        with pytest.raises(ValueError):
            Declared(outcome)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Declared("death", beyond_the_cap="yes")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- the reader


@pytest.mark.parametrize(
    ("section", "reason"),
    [
        # not a section
        ([1, 2], NOT_READ),
        (PLANTED, NOT_READ),
        # a key or a closed value this version does not know: written by a later one, counted and never named
        ({"k": 20, PLANTED: 1}, "written by a later version: its privacy section holds 1 key this"),
        ({PLANTED: 1, "retention": 30}, "holds 2 keys this version does not read"),
        (
            {"followed_to_the_event": {"death": {PLANTED: True}}},
            "holds 1 key this version does not",
        ),
        ({"rows_not_people": PLANTED}, LATER_VALUE),
        ({"rows_not_people": "my mice"}, LATER_VALUE),
        ({"rows_word": "patients"}, LATER_VALUE),
        # a value of another type: never ignored (a k of 20 written "20" and ignored would apply 10)
        ({"k": "20"}, NOT_READ),
        ({"k": 20.0}, NOT_READ),
        ({"k": True}, NOT_READ),
        ({"k": None}, NOT_READ),
        ({"followed_to_the_event": ["death"]}, NOT_READ),
        ({"followed_to_the_event": {}}, NOT_READ),
        ({"followed_to_the_event": {"death": True}}, NOT_READ),
        ({"followed_to_the_event": {"": {}}}, NOT_READ),
        ({"followed_to_the_event": {"x" * 257: {}}}, NOT_READ),
        ({"followed_to_the_event": {"dea​th": {}}}, NOT_READ),
        ({"followed_to_the_event": {"death": {"beyond_the_cap": "yes"}}}, NOT_READ),
        ({"rows_not_people": 3}, NOT_READ),
        ({"rows_word": ["animals"]}, NOT_READ),
        # one writing per meaning: a default written out is another writing of its absence
        ({"followed_to_the_event": {"death": {"beyond_the_cap": False}}}, NOT_READ),
        ({"rows_word": "participants"}, NOT_READ),
        # and so are the same choices in another order, of the keys or of the outcomes: they had another checksum,
        # and saved again unchanged they read as changed
        ({"rows_word": "animals", "k": 20}, NOT_READ),
        ({"rows_not_people": "animal_data", "followed_to_the_event": {"death": {}}}, NOT_READ),
        ({"followed_to_the_event": {"visits": {}, "death": {}}}, NOT_READ),
        # below the floor of every version
        ({"k": 4}, BELOW_THE_FLOOR),
        ({"k": 0}, BELOW_THE_FLOOR),
        ({"k": -3}, BELOW_THE_FLOOR),
    ],
)
def test_a_section_this_version_cannot_read_refuses_the_whole_file(
    section: object, reason: str
) -> None:
    """Read without its section, a study would lose the k it asked for and its outputs would be looser because a
    file was misread: the whole file is refused, in words that quote nothing of it."""
    choices, reasons = StudyPrivacy.from_section(section)
    assert choices is None and len(reasons) == 1 and reason in reasons[0]
    read, problems = StudyProtocol.from_file(_file(section))
    assert read is None and problems == reasons
    said = " ".join(problems)
    for quoted in (PLANTED, "my mice", "patients", "death", "retention", "yes", "20"):
        assert quoted not in said, quoted
    # what is about to be a file is read the same way: such a body is never written
    with pytest.raises(ValueError) as refused:
        StudyProtocol({"mask_families": FAMILIES, PRIVACY_KEY: section}, ProtocolMeta("cohort"))
    assert str(refused.value) == reasons[0]
    with pytest.raises(ValueError):
        study_privacy({PRIVACY_KEY: section})


def test_a_k_looser_than_an_installation_s_floor_is_read_as_asked() -> None:
    """The floor belongs to the installation and the file to the study, which moves between installations: the
    reader reads no setting, and the larger of the two is applied where an output is made, with both recorded."""
    read, problems = StudyProtocol.from_file(_file({"k": 5}))
    assert read is not None and problems == [] and study_privacy(read.body).k == 5
    policy = DisclosurePolicy.for_study(study_privacy(read.body).k, floor=10)
    assert (policy.k, policy.asked, policy.floor) == (10, 5, 10)
    stricter = DisclosurePolicy.for_study(study_privacy(_body(K_ONLY)).k, floor=10)
    assert (stricter.k, stricter.asked, stricter.floor) == (20, 20, 10)
    # a study that asks nothing: the default, raised by the floor
    assert DisclosurePolicy.for_study(study_privacy(_body()).k, floor=15).k == 15
    # no maximum in the format: a huge k only withholds more, and is carried in whole numbers
    huge = 10**400
    data = json.loads(json.dumps(_file({"k": huge})))
    read, problems = StudyProtocol.from_file(data)
    assert read is not None and study_privacy(read.body).k == huge
    assert DisclosurePolicy.for_study(huge, floor=10).k == huge


def test_the_declarations_reach_the_policy_and_the_reserved_keys_change_nothing_of_it() -> None:
    """This version applies the study's k and its declarations where a package is made. The declaration that the
    rows are not people and the word for the rows are read and kept, and applied by nothing: the stricter reading."""
    choices = study_privacy(_body(EVERYTHING))
    declared = tuple(
        FollowedToTheEvent(one.outcome, beyond_the_cap=one.beyond_the_cap)
        for one in choices.followed_to_the_event
    )
    policy = DisclosurePolicy.for_study(choices.k, floor=10, declared=declared)
    assert policy.followed("death") and policy.followed("visits") and not policy.followed("stroke")
    assert [one.beyond_the_cap for one in policy.declared] == [False, True]
    # nothing typed reaches the policy from the study: no reason, no role, no date
    assert all((one.reason, one.role, one.date) == ("", "", "") for one in policy.declared)
    without = study_privacy(_body({**K_ONLY, **FOLLOWED}))
    assert (without.k, without.followed_to_the_event) == (choices.k, choices.followed_to_the_event)
    # kept when a revision is written, as asked
    saved = StudyProtocol(_body(EVERYTHING), ProtocolMeta("cohort", revision=1)).to_file()
    assert saved["body"][PRIVACY_KEY]["rows_not_people"] == "animal_data"
    assert saved["body"][PRIVACY_KEY]["rows_word"] == "animals"


# ----------------------------------------------------------------------------- fingerprint, checksum, diff


def test_the_choices_are_outside_the_fingerprint_and_inside_the_checksum() -> None:
    """A privacy choice decides what a document writes, never which rows are selected nor what is computed: no
    result goes stale for one. It is part of the body as written: a change of it is another file."""
    plain, chosen = _body(), _body(EVERYTHING)
    assert BODY_KEYS[-1] == PRIVACY_KEY  # last: every body written before it keeps its checksum
    assert hashed_view(chosen) == hashed_view(plain) and PRIVACY_KEY not in hashed_view(chosen)
    assert protocol_hash(chosen) == protocol_hash(plain)
    assert body_checksum(chosen) != body_checksum(plain)
    assert body_checksum(_body(K_ONLY)) != body_checksum(_body({"k": 30}))
    assert body_of(chosen)[PRIVACY_KEY] == EVERYTHING and PRIVACY_KEY not in body_of(_body({}))
    # a configuration holds it as a body does; loading a revision brings its section, or takes the session's away
    config = with_body({"description": "d", PRIVACY_KEY: {"k": 5}}, chosen)
    assert config[PRIVACY_KEY] == EVERYTHING and config["description"] == "d"
    assert PRIVACY_KEY not in with_body({PRIVACY_KEY: {"k": 5}}, plain)
    # the trace's step of a saved revision: the number and the fingerprint, never the section
    params = snapshot_params(StudyProtocol(chosen, ProtocolMeta("cohort", revision=3)))
    assert params["protocol_hash"] == protocol_hash(plain) and params["revision"] == 3
    assert "animal" not in json.dumps(params) and "20" not in json.dumps(params["protocol_hash"])
    assert set(params) == {"study_id", "revision", "parent_revision", "protocol_hash", "schema"}


def test_the_diff_of_two_revisions_shows_a_change_of_the_choices() -> None:
    lines = diff_lines(_body(K_ONLY), _body({"k": 30}))
    assert any(line.startswith("-") and '"k": 20' in line for line in lines)
    assert any(line.startswith("+") and '"k": 30' in line for line in lines)
    declared = "\n".join(diff_lines(_body(), _body(FOLLOWED)))
    assert '"followed_to_the_event"' in declared and '"death"' in declared
    assert diff_lines(_body(EVERYTHING), _body(EVERYTHING)) == []


# ----------------------------------------------------------------------------- it never travels


def test_a_shared_definition_and_the_configuration_files_leave_the_choices_out() -> None:
    """Whoever opens a study chooses for their own data: no k and no declaration of another person's arrive with a
    study. The choices are outside the fingerprint, so the shared body is still the definition the results were
    computed under."""
    shared = protocol_for_sharing(_body(EVERYTHING))
    assert PRIVACY_KEY not in shared.body and shared.complete
    assert shared.protocol_hash == shared.shared_hash == protocol_hash(_body())
    assert "animal" not in json.dumps(shared.body) and "visits" not in json.dumps(shared.body)
    # the configuration for someone else: left out, said apart, and not counted as a setting of the installation
    config = {"description": "d", "mask_families": FAMILIES, PRIVACY_KEY: EVERYTHING}
    for_others = configuration_for_sharing(config)
    assert PRIVACY_KEY not in for_others.config and for_others.privacy
    assert for_others.settings == ()
    plain = configuration_for_sharing(
        {"description": "d", "mask_families": FAMILIES, "folder": "x"}
    )
    assert not plain.privacy and plain.settings == ("folder",)


# ----------------------------------------------------------------------------- the revisions


def test_a_change_of_the_choices_alone_is_the_next_revision(tmp_path: Path) -> None:
    """The revisions are then the history of the study's choices. The store compares checksums: the fingerprint is
    the same, and nothing selects differently."""
    store = WorkspaceProtocolStore(None, root=tmp_path / "protocols")
    first = store.save_revision(StudyProtocol(_body(), ProtocolMeta("cohort")))
    second = store.save_revision(StudyProtocol(_body(K_ONLY), ProtocolMeta("cohort")))
    assert (first.meta.revision, second.meta.revision) == (1, 2)
    assert second.hash == first.hash and second.checksum != first.checksum
    assert second.meta.parent == first.checksum
    with pytest.raises(ProtocolUnchanged):
        store.save_revision(StudyProtocol(_body(K_ONLY), ProtocolMeta("cohort")))
    third = store.save_revision(StudyProtocol(_body(EVERYTHING), ProtocolMeta("cohort")))
    assert third.meta.revision == 3
    # read back as written, the choices with it
    read = store.revision("cohort", 3)
    assert read is not None and study_privacy(read.body) == study_privacy(_body(EVERYTHING))
    assert study_privacy(store.revision("cohort", 1).body) == StudyPrivacy()  # type: ignore[union-attr]


def test_a_revision_whose_section_is_refused_is_listed_as_unreadable(tmp_path: Path) -> None:
    store = WorkspaceProtocolStore(None, root=tmp_path / "protocols")
    saved = store.save_revision(StudyProtocol(_body(K_ONLY), ProtocolMeta("cohort")))
    folder = tmp_path / "protocols" / "cohort"
    (path,) = [p for p in folder.rglob("*.json") if "working" not in p.name and "1" in p.name]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["body"][PRIVACY_KEY] = {
        "k": 20,
        PLANTED: 1,
    }  # as a later version would write a key of its own
    data["checksum"] = body_checksum(data["body"])
    path.write_text(json.dumps(data), encoding="utf-8")
    assert store.unreadable("cohort") == (saved.meta.revision,) or list(
        store.unreadable("cohort")
    ) == [saved.meta.revision]
    assert store.revision("cohort", 1) is None


def test_an_older_version_refuses_a_study_that_holds_the_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A version that does not know the section reads the file as written by a later one, counted and never named:
    it does not open the study without the k it asked for."""
    import core.protocol as protocol

    data = json.loads(
        json.dumps(StudyProtocol(_body(K_ONLY), ProtocolMeta("cohort", revision=1)).to_file())
    )
    monkeypatch.setattr(protocol, "BODY_KEYS", tuple(k for k in BODY_KEYS if k != PRIVACY_KEY))
    assert StudyProtocol.from_file(data) == (
        None,
        ["written by a later version: it holds 1 section this version does not read"],
    )
