"""
The study package's formats (core.package): the members and their names, the one writer of a JSON member and its
strict reader, the manifest and the package's hash, the rule for a text of a package, and the policy block's closed
shape. Nothing here builds a package: the writer and the reader come after.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core import package as pk
from core.privacy import FLOOR_K, NOT_PEOPLE_REASONS, ROWS_WORDS

ROOT = Path(__file__).resolve().parents[2]
PLANTED = "P-0042 Jane"
RLO = chr(0x202E)  # right-to-left override: what follows reads backwards
ZWSP = chr(0x200B)  # zero width space


def _on(**changes: object) -> dict:
    block = {
        "rules": "on",
        "version": 2,
        "k": 10,
        "k_asked": None,
        "k_default": 10,
        "floor": 10,
        "floor_set_by": "deployment",
        "cap": 0.05,
        "cap_set_by": "code",
        "followed_to_the_event": [{"outcome": "death", "beyond_the_cap": False}],
        "rows_not_people": None,
        "rows_word": "participants",
        "overrides": [],
    }
    return {**block, **changes}


OFF = {
    **{key: None for key in pk.POLICY_KEYS},
    "rules": "off",
    "version": 2,
    "followed_to_the_event": [],
    "rows_not_people": {"reason": "animal_data", "level": "study"},
    "rows_word": "animals",
    "overrides": [],
}
NONE = {
    **{key: None for key in pk.POLICY_KEYS},
    "rules": "none",
    "followed_to_the_event": [],
    "rows_word": "participants",
    "overrides": [],
}


# ----------------------------------------------------------------------------- names and stored ids


def test_the_tiers_the_members_and_the_names_are_stored_ids() -> None:
    from core import verification
    from core.protocol import STUDY_ID

    assert pk.TIERS == ("shareable", "rules_off", "reproduction")
    assert (pk.SHAREABLE, pk.REPRODUCTION) == (verification.SHAREABLE, verification.REPRODUCTION)
    assert pk.MEMBERS == (
        "package.json",
        "protocol.json",
        "results.json",
        "README.md",
        "uv.lock",
        "MANIFEST.sha256",
    )
    assert pk.LISTED == ("README.md", "package.json", "protocol.json", "results.json", "uv.lock")
    assert set(pk.CAPS) == set(pk.MEMBERS) and sum(pk.CAPS.values()) <= pk.PACKAGE_CAP
    assert (
        pk.STUDY_ID.pattern == STUDY_ID.pattern
    )  # restated: this module does not import core.protocol
    assert pk.RULES == ("on", "off", "none") and pk.SET_BY[:2] == ("code", "deployment")
    # each tier has its own file name
    assert pk.package_name("cohort", 3) == "cohort_r3.cstudio.zip"
    assert pk.package_name("cohort", 3, pk.RULES_OFF) == "cohort_r3_rules_off.cstudio.zip"
    assert pk.package_name("cohort", 3, pk.REPRODUCTION) == "cohort_r3_with_data.cstudio.zip"
    for study, revision, tier in (
        ("../x", 1, pk.SHAREABLE),
        ("cohort", 0, pk.SHAREABLE),
        ("c", True, "x"),
    ):
        with pytest.raises(ValueError):
            pk.package_name(study, revision, tier)  # type: ignore[arg-type]


def test_the_module_reads_a_package_without_the_engine() -> None:
    """``inspect`` reads a package through it: no numpy, pandas, Streamlit, biostats, nor the modules that import
    them."""
    code = (
        "import sys, core.package; "
        "heavy = [m for m in ('numpy', 'pandas', 'streamlit', 'biostats', 'core.protocol', 'core.disclosure', "
        "'core.lineage', 'scipy') if m in sys.modules]; print(heavy)"
    )
    done = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.strip() == "[]"


# ----------------------------------------------------------------------------- the one writer


def test_a_member_has_one_writing() -> None:
    """Two packages of one study, made on two machines, are the same bytes: keys sorted whatever the order given,
    two spaces, line feeds, a final line feed, UTF-8 as it is, the envelope first by its names."""
    payload = {"tier": "shareable", "study": {"revision": 3, "id": "cohort"}, "k": 10, "share": 0.5}
    written = pk.dumps(pk.PACKAGE, payload)
    assert written == (
        b"{\n"
        b'  "format": "cohortstudio.study-package",\n'
        b'  "k": 10,\n'
        b'  "schema": 1,\n'
        b'  "share": 0.5,\n'
        b'  "study": {\n'
        b'    "id": "cohort",\n'
        b'    "revision": 3\n'
        b"  },\n"
        b'  "tier": "shareable"\n'
        b"}\n"
    )
    assert pk.dumps(pk.PACKAGE, dict(reversed(list(payload.items())))) == written
    assert b"\r" not in written and not written.startswith(b"\xef\xbb\xbf")
    accents = pk.dumps(pk.RESULTS, {"title": "Fièvre à l'entrée", "lines": "one\ntwo"})
    assert "Fièvre à l'entrée".encode() in accents  # not escaped
    assert json.loads(accents)["lines"] == "one\ntwo"  # a line feed inside a text is written
    assert pk.loads(accents, pk.CAPS[pk.RESULTS])[0]["title"] == "Fièvre à l'entrée"


def test_the_shared_definition_s_body_keeps_its_order() -> None:
    """The order of the families and of their rules is part of the study: the body is written as given, at every
    depth, and the rest of the member sorted."""
    body = {"mask_families": {"b_family": {"z": 1, "a": 2}, "a_family": {}}, "follow_up": {}}
    written = json.loads(pk.dumps(pk.PROTOCOL, {"study_id": "cohort", "body": body, "revision": 1}))
    assert list(written) == ["body", "format", "revision", "schema", "study_id"]
    assert list(written["body"]) == ["mask_families", "follow_up"]
    assert list(written["body"]["mask_families"]) == ["b_family", "a_family"]
    assert list(written["body"]["mask_families"]["b_family"]) == ["z", "a"]
    # the same key in another member is sorted: only the shared definition's body is a study's order
    assert list(json.loads(pk.dumps(pk.RESULTS, {"body": {"z": 1, "a": 2}}))["body"]) == ["a", "z"]


@pytest.mark.parametrize(
    "payload",
    [
        {"format": "x"},
        {"schema": 2},
        {"share": float("nan")},
        {"share": float("inf")},
        {"nested": {"deep": [1.0, float("-inf")]}},
        {"name": f"file{RLO}txt.exe"},
        {"name": f"dea{ZWSP}th"},
        {"name": "two\r\nlines"},
        {f"key{ZWSP}": 1},
        {"nested": {f"key{RLO}": 1}},
        {"nested": {1: "a key that is no text"}},
        {"when": object()},
        {"values": {1, 2}},
        {"bytes": b"raw"},
    ],
)
def test_the_writer_refuses_what_no_member_holds(payload: dict) -> None:
    with pytest.raises(ValueError) as refused:
        pk.dumps(pk.PACKAGE, payload)
    assert RLO not in str(refused.value) and ZWSP not in str(refused.value)
    with pytest.raises(ValueError):
        pk.dumps(pk.README, {})  # not a JSON member


# ----------------------------------------------------------------------------- the strict reader


def test_what_the_writer_writes_reads_back() -> None:
    payload = {
        "study": {"id": "cohort", "revision": 3},
        "counts": [1, 2, 3],
        "none": None,
        "yes": True,
    }
    value, reasons = pk.loads(pk.dumps(pk.PACKAGE, payload), pk.CAPS[pk.PACKAGE])
    assert reasons == [] and value == {"format": pk.FORMATS[pk.PACKAGE], "schema": 1, **payload}
    assert pk.read_envelope(value, pk.PACKAGE) == []


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"\xef\xbb\xbf{}", pk.NOT_JSON),  # a byte order mark
        (b"\xff\xfe{\x00}\x00", pk.NOT_JSON),  # not UTF-8
        (b'{"a": "\xe9"}', pk.NOT_JSON),  # Latin-1
        (b"{", pk.NOT_JSON),
        (b"", pk.NOT_JSON),
        (b'{"k": 5, "k": 20}', pk.KEY_TWICE),  # which of the two would a reader take?
        (b'{"a": {"k": 5, "k": 5}}', pk.KEY_TWICE),
        (b'{"share": NaN}', pk.NOT_A_NUMBER),
        (b'{"share": Infinity}', pk.NOT_A_NUMBER),
        (b'{"share": -Infinity}', pk.NOT_A_NUMBER),
        (b'{"share": 1e999}', pk.NOT_A_NUMBER),  # read as infinite without a word
        (b"[" * 100_000 + b"]" * 100_000, pk.NOT_JSON),  # nested too deep for the interpreter
        (b"[" * (pk.DEPTH + 1) + b"]" * (pk.DEPTH + 1), pk.NOT_JSON),  # and for the format
        (b'{"a": ' * pk.DEPTH + b"[]" + b"}" * pk.DEPTH, pk.NOT_JSON),
        (b'{"k": ' + b"9" * 5000 + b"}", pk.NOT_JSON),  # a number of too many digits
        (b" " * 200 + b"{}", pk.TOO_LARGE),
    ],
)
def test_the_reader_refuses_what_the_writer_never_writes(data: bytes, reason: str) -> None:
    value, reasons = pk.loads(data, 100 if reason == pk.TOO_LARGE else 10**7)
    assert value is None and reasons == [reason]


def test_a_text_that_hides_or_turns_is_refused_and_never_shown() -> None:
    """A character that shows nothing, or that changes the direction of what follows, makes a name read as
    another: "file[RLO]txt.exe" shows as "fileexe.txt". The reason counts them and quotes nothing."""
    for text in (f"file{RLO}txt.exe", f"{PLANTED}{ZWSP}", "bell\x07", "two\r\nlines", "tab\there"):
        for data in (
            json.dumps({"name": text}).encode(),
            json.dumps({"name": text}, ensure_ascii=False).encode(),
            json.dumps({text: 1}, ensure_ascii=False).encode(),
            json.dumps({"deep": [{"x": [text]}]}, ensure_ascii=False).encode(),
        ):
            value, reasons = pk.loads(data, 10**6)
            assert value is None and len(reasons) == 1, text
            assert reasons[0].startswith("holds ") and "character" in reasons[0]
            assert PLANTED not in reasons[0] and RLO not in reasons[0] and "txt" not in reasons[0]
    assert pk.hidden("plain text, accents é and a line\nfeed") == 0
    assert pk.hidden(f"a{RLO}b{ZWSP}c\r") == 3 and pk.hidden_reason(1).startswith(
        "holds 1 character that"
    )
    # what does not come from a member (a file's name, an argument) is shown on one line, cleaned and cut
    assert pk.shown(f"file{RLO}txt.exe\n") == f"file{pk.REPLACED}txt.exe{pk.REPLACED}"
    assert pk.shown("x" * 300, 20) == "x" * 19 + pk.CUT and pk.shown(12) == "12"


def test_an_envelope_names_its_member_and_its_schema() -> None:
    good = {"format": pk.FORMATS[pk.RESULTS], "schema": 1}
    assert pk.read_envelope(good, pk.RESULTS) == []
    assert pk.read_envelope(good, pk.PACKAGE) == [pk.NOT_A_MEMBER]  # another member's
    for wrong in ([], "text", {}, {"format": PLANTED, "schema": 1}, {**good, "schema": "1"}):
        assert pk.read_envelope(wrong, pk.RESULTS) == [pk.NOT_A_MEMBER]
    for schema in (0, -1, True, 1.0, None, 10**400):
        assert pk.read_envelope({**good, "schema": schema}, pk.RESULTS) == [pk.NOT_A_MEMBER]
    later = pk.read_envelope({**good, "schema": 2}, pk.RESULTS)
    assert later == ["written by a later version (schema 2; this version reads 1)"]
    with pytest.raises(ValueError):
        pk.read_envelope(good, pk.README)


# ----------------------------------------------------------------------------- the manifest and the hash


def _members() -> dict[str, bytes]:
    return {
        pk.PACKAGE: pk.dumps(pk.PACKAGE, {"tier": pk.SHAREABLE}),
        pk.PROTOCOL: pk.dumps(pk.PROTOCOL, {"body": {}}),
        pk.RESULTS: pk.dumps(pk.RESULTS, {"results": []}),
        pk.README: b"# A study\n",
    }


def test_the_manifest_lists_every_other_member_and_its_hash_is_the_package_s() -> None:
    members = _members()
    manifest = pk.manifest_of(members)
    lines = manifest.decode("ascii").splitlines()
    assert [line.split("  ")[1] for line in lines] == [
        "README.md",
        "package.json",
        "protocol.json",
        "results.json",
    ]
    assert lines[0] == f"{pk.checksum(b'# A study' + bytes([10]))}  README.md"
    assert manifest.endswith(b"\n") and b"\r" not in manifest
    listed, reasons = pk.read_manifest(manifest)
    assert reasons == [] and listed == {name: pk.checksum(data) for name, data in members.items()}
    assert pk.not_matching(members, listed) == []
    assert (
        pk.package_hash(manifest) == pk.checksum(manifest) and len(pk.package_hash(manifest)) == 64
    )
    # one byte of one member: the member is named (the names are the format's own), and the hash is another
    changed = {**members, pk.RESULTS: members[pk.RESULTS] + b" "}
    assert pk.not_matching(changed, listed) == [pk.RESULTS]
    assert pk.package_hash(pk.manifest_of(changed)) != pk.package_hash(manifest)
    # a member more, or one fewer, than the manifest lists
    assert pk.not_matching({**members, pk.LOCK: b"lock"}, listed) == [pk.LOCK]
    assert pk.not_matching({k: v for k, v in members.items() if k != pk.README}, listed) == [
        pk.README
    ]
    # with the lock file, listed in its place
    with_lock = pk.manifest_of({**members, pk.LOCK: b"version = 1\n"})
    assert list(pk.read_manifest(with_lock)[0]) == sorted([*members, pk.LOCK])
    for wrong in ({**members, pk.MANIFEST: b""}, {**members, "notes.txt": b""}, {pk.README: b""}):
        with pytest.raises(ValueError):
            pk.manifest_of(wrong)


def test_a_manifest_is_read_strictly() -> None:
    good = pk.manifest_of(_members())
    lines = good.decode("ascii").splitlines()
    digest = lines[0].split("  ")[0]

    def made(*changed: str) -> bytes:
        return ("\n".join(changed) + "\n").encode()

    refused = [
        good[:-1],  # no final line feed
        good.replace(b"\n", b"\r\n"),
        made(*reversed(lines)),  # not sorted
        made(*lines, lines[-1]),  # a member twice
        made(*lines[1:]),  # a member every package holds is missing
        made(*lines, f"{digest}  notes.txt"),  # not a member
        made(*lines, f"{digest}  MANIFEST.sha256"),  # itself
        made(*lines, f"{digest}  ../package.json"),
        made(lines[0].upper().replace("README.MD", "README.md"), *lines[1:]),  # upper-case hex
        made(lines[0].replace("  ", " "), *lines[1:]),  # one space
        made(lines[0][1:], *lines[1:]),  # 63 hex
        made(*lines, ""),  # an empty line
        good + b"\xe9\n",
        b"",
        b"x" * 5000,
    ]
    for data in refused:
        assert pk.read_manifest(data) == (None, [pk.NOT_A_MANIFEST]), data[:60]


# ----------------------------------------------------------------------------- the policy block


def test_the_policy_block_has_one_closed_shape_with_every_key_written() -> None:
    policy, reasons = pk.Policy.from_dict(_on())
    assert reasons == [] and policy is not None
    assert policy.to_dict() == _on() and list(policy.to_dict()) == list(pk.POLICY_KEYS)
    assert (policy.k, policy.k_asked, policy.k_default, policy.floor) == (10, None, 10, 10)
    assert policy.followed_to_the_event == (("death", False),)
    # "asked none" and "asked 10" read differently, and a later change of the default stays readable
    asked = pk.Policy.from_dict(_on(k_asked=10))[0]
    assert asked is not None and asked.to_dict()["k_asked"] == 10 and asked != policy
    # the k applied is the larger of what was asked (or the default) and the floor
    for k, k_asked, floor in ((20, 20, 10), (10, 5, 10), (12, None, 12), (10, None, 5)):
        assert pk.Policy.from_dict(_on(k=k, k_asked=k_asked, floor=floor))[1] == []
    for k, k_asked, floor in ((10, 20, 10), (5, 5, 10), (20, None, 10), (10, 10, 12)):
        assert pk.Policy.from_dict(_on(k=k, k_asked=k_asked, floor=floor)) == (
            None,
            [pk.K_NOT_THE_LARGER],
        )
    # the two other values of ``rules``, which this version reads and does not write
    # a package that holds the data of a study whose rows are declared not people holds the declaration
    declared = {
        **NONE,
        "rows_not_people": {"reason": "animal_data", "level": "study"},
        "rows_word": "animals",
    }
    for block in (OFF, NONE, declared):
        read, reasons = pk.Policy.from_dict(block)
        assert reasons == [] and read is not None and read.to_dict() == block
    assert pk.Policy.from_dict(OFF)[0].rows_not_people == ("animal_data", "study")
    assert set(NOT_PEOPLE_REASONS) >= {"animal_data"} and "animals" in ROWS_WORDS
    # through the one writer and the strict reader, as package.json will hold it
    data = pk.dumps(pk.PACKAGE, {"policy": policy.to_dict()})
    assert pk.Policy.from_dict(pk.loads(data, pk.CAPS[pk.PACKAGE])[0]["policy"]) == (policy, [])


@pytest.mark.parametrize(
    ("block", "reason"),
    [
        # a key more is a later version's, counted and never named; a key fewer is no policy
        (
            {**_on(), PLANTED: 1},
            "written by a later version: its policy holds 1 key this version does not read",
        ),
        ({**_on(), PLANTED: 1, "retention": 30}, "holds 2 keys this version does not read"),
        ({k: v for k, v in _on().items() if k != "overrides"}, pk.NOT_A_POLICY),
        ([], pk.NOT_A_POLICY),
        (PLANTED, pk.NOT_A_POLICY),
        # a closed value this version does not know
        (_on(rules="paused"), pk.LATER_POLICY_VALUE),
        (_on(floor_set_by=PLANTED), pk.LATER_POLICY_VALUE),
        (_on(cap_set_by="owner"), pk.LATER_POLICY_VALUE),
        (_on(rows_word="patients"), pk.LATER_POLICY_VALUE),
        (
            _on(overrides=[{"rule": "waived", "place": "results.json: results > result 2"}]),
            pk.LATER_POLICY_VALUE,
        ),
        (
            {**NONE, "rows_not_people": {"reason": "my mice", "level": "study"}},
            pk.LATER_POLICY_VALUE,
        ),
        (
            {**OFF, "rows_not_people": {"reason": "my mice", "level": "study"}},
            pk.LATER_POLICY_VALUE,
        ),
        (
            {**OFF, "rows_not_people": {"reason": "animal_data", "level": "ward"}},
            pk.LATER_POLICY_VALUE,
        ),
        # a value of another type, or outside what a policy may hold: never ignored
        (_on(rules=["on"]), pk.NOT_A_POLICY),
        (_on(version="2"), pk.NOT_A_POLICY),
        (_on(version=0), pk.NOT_A_POLICY),
        (_on(k="10"), pk.NOT_A_POLICY),
        (_on(k=10.0), pk.NOT_A_POLICY),
        (_on(k=True), pk.NOT_A_POLICY),
        (_on(k=4, floor=4), pk.NOT_A_POLICY),
        (_on(k_default=None), pk.NOT_A_POLICY),
        (_on(floor=None), pk.NOT_A_POLICY),
        (_on(floor_set_by=None), pk.NOT_A_POLICY),
        (_on(cap=0), pk.NOT_A_POLICY),
        (_on(cap=1.5), pk.NOT_A_POLICY),
        (_on(cap="5%"), pk.NOT_A_POLICY),
        (_on(cap=True), pk.NOT_A_POLICY),
        (_on(cap=1), pk.NOT_A_POLICY),  # a share is written as one: a second writing of 1.0
        (_on(rows_word=3), pk.NOT_A_POLICY),
        (_on(followed_to_the_event={"death": {}}), pk.NOT_A_POLICY),
        (_on(followed_to_the_event=[{"outcome": "death"}]), pk.NOT_A_POLICY),
        (
            _on(followed_to_the_event=[{"outcome": "death", "beyond_the_cap": "yes"}]),
            pk.NOT_A_POLICY,
        ),
        (_on(followed_to_the_event=[{"outcome": "", "beyond_the_cap": False}]), pk.NOT_A_POLICY),
        (_on(followed_to_the_event=[{"outcome": 3, "beyond_the_cap": False}]), pk.NOT_A_POLICY),
        (
            _on(
                followed_to_the_event=[
                    {"outcome": "death", "beyond_the_cap": False, "reason": PLANTED}
                ]
            ),
            pk.NOT_A_POLICY,
        ),
        (  # each outcome once, in the one order
            _on(
                followed_to_the_event=[
                    {"outcome": "visits", "beyond_the_cap": True},
                    {"outcome": "death", "beyond_the_cap": False},
                ]
            ),
            pk.NOT_A_POLICY,
        ),
        (
            _on(
                followed_to_the_event=[
                    {"outcome": "death", "beyond_the_cap": True},
                    {"outcome": "death", "beyond_the_cap": False},
                ]
            ),
            pk.NOT_A_POLICY,
        ),
        (_on(overrides=[{"rule": "scan_hit"}]), pk.NOT_A_POLICY),
        (_on(overrides=[{"rule": "scan_hit", "place": ""}]), pk.NOT_A_POLICY),
        # a place is the format's names and positions, never a text
        (
            _on(overrides=[{"rule": "scan_hit", "place": "criterion age == 97 (QZX Wiltz)"}]),
            pk.NOT_A_POLICY,
        ),
        (
            _on(overrides=[{"rule": "scan_hit", "place": "protocol.json: mask_families > wiltz"}]),
            pk.NOT_A_POLICY,
        ),
        (_on(overrides=[{"rule": "scan_hit", "place": "notes.txt: body"}]), pk.NOT_A_POLICY),
        (  # each override once, in the one order
            _on(
                overrides=[
                    {"rule": "scan_hit", "place": "results.json: results > result 2"},
                    {"rule": "cap", "place": "results.json: results > result 1"},
                ]
            ),
            pk.NOT_A_POLICY,
        ),
        (
            _on(
                overrides=[
                    {"rule": "cap", "place": "results.json"},
                    {"rule": "cap", "place": "results.json"},
                ]
            ),
            pk.NOT_A_POLICY,
        ),
        (_on(overrides=[{"rule": ["scan_hit"], "place": "x"}]), pk.NOT_A_POLICY),
        # a package made under the rules names no declaration that did not apply
        (_on(rows_not_people={"reason": "animal_data", "level": "study"}), pk.NOT_A_POLICY),
        # with the rules off, or for a package that holds the data, nothing was withheld
        ({**OFF, "k": 10}, pk.NOT_A_POLICY),
        ({**OFF, "cap": 0.05}, pk.NOT_A_POLICY),
        ({**OFF, "floor_set_by": "code"}, pk.NOT_A_POLICY),
        (
            {**OFF, "followed_to_the_event": [{"outcome": "death", "beyond_the_cap": False}]},
            pk.NOT_A_POLICY,
        ),
        ({**OFF, "rows_not_people": None}, pk.NOT_A_POLICY),
        ({**OFF, "rows_not_people": {"reason": "animal_data"}}, pk.NOT_A_POLICY),
        ({**OFF, "version": None}, pk.NOT_A_POLICY),
        ({**NONE, "version": 2}, pk.NOT_A_POLICY),
        ({**NONE, "rows_not_people": {"reason": "animal_data"}}, pk.NOT_A_POLICY),
    ],
)
def test_a_policy_this_version_cannot_read_is_refused_in_words_that_quote_nothing(
    block: object, reason: str
) -> None:
    policy, reasons = pk.Policy.from_dict(block)
    assert policy is None and len(reasons) == 1 and reason in reasons[0]
    said = " ".join(reasons)
    for quoted in (
        PLANTED,
        "my mice",
        "patients",
        "paused",
        "waived",
        "retention",
        "ward",
        "Wiltz",
        "wiltz",
        "owner",
    ):
        assert quoted not in said, quoted


def test_a_policy_that_no_package_holds_is_not_built() -> None:
    with pytest.raises(ValueError):
        pk.Policy(rules="on")  # no k, no floor
    with pytest.raises(ValueError):
        pk.Policy(rules="on", version=2, k=FLOOR_K - 1, k_default=10, floor=FLOOR_K - 1)
    with pytest.raises(ValueError):
        pk.Policy(rules="none", k=10)
    with pytest.raises(ValueError):
        pk.Policy(
            rules="off", version=2
        )  # with the rules off, the declaration is what the block is of
    built = pk.Policy(
        rules="on",
        version=2,
        k=20,
        k_asked=20,
        k_default=10,
        floor=10,
        floor_set_by="code",
        cap=0.05,
        cap_set_by="code",
        followed_to_the_event=(("death", False), ("visits", True)),
        rows_word="subjects",
        overrides=(("scan_hit", "protocol.json: body > mask_families > name 2 > description"),),
    )
    assert pk.Policy.from_dict(built.to_dict()) == (built, [])
    assert pk.Policy(rules="none").to_dict()["rows_word"] == "participants"


def test_a_place_is_the_format_s_names_and_positions() -> None:
    """An override names where it holds, never what: "criterion age == 97 (...)" was read as a place."""
    for place in (
        "results.json",
        "protocol.json: body > mask_families > name 3 > description",
        "protocol.json: body > derived_variables > name 1 > formula",
        "results.json: calculations > calculation 2 > source",
        "results.json: results > result 4 > spec > roles > item 1",
    ):
        assert pk.is_place(place), place
    for place in (
        "",
        "README.md",
        "protocol.json:",
        "protocol.json: ",
        "protocol.json:body",
        "protocol.json: body>description",
        "protocol.json: body > ",
        "protocol.json: body > name 0",
        "protocol.json: body > name 03",
        "protocol.json: body > name three",
        "protocol.json: body > site 3",
        "protocol.json: body > Description",
        "protocol.json: body > wiltz",
        "protocol.json: body > name 3 4",
        "protocol.json: body > description" + chr(10) + "Rules: off",
        "protocol.json: body > descripti" + chr(0xF3) + "n",
        "results.json: " + " > ".join(["item 1"] * 40),
        ["results.json"],
        None,
    ):
        assert not pk.is_place(place), place


def test_the_characters_that_show_nothing_are_refused_both_ways() -> None:
    """ "site" + a filler + "A" was written and read back as another name than "siteA": the fillers, the variation
    selectors and the other characters a display ignores are no control and no format characters. A lone half of a
    surrogate pair, which no writer can encode, and the line and paragraph separators are refused too."""
    unseen = [
        0x034F,
        0xFE0F,
        0x115F,
        0x1160,
        0x3164,
        0xFFA0,
        0x17B4,
        0x180B,
        0xE0100,
        0x2800,
        0x2028,
        0x2029,
    ]
    for code in unseen:
        name = "site" + chr(code) + "A"
        assert pk.hidden(name) == 1, hex(code)
        with pytest.raises(ValueError) as refused:
            pk.dumps(pk.RESULTS, {"name": name})
        assert chr(code) not in str(refused.value)
        data = json.dumps({"name": name}, ensure_ascii=False).encode("utf-8")
        value, reasons = pk.loads(data, 10**6)
        assert value is None and reasons == [pk.hidden_reason(1)], hex(code)
        assert pk.shown(name) == "site" + pk.REPLACED + "A"
        assert pk.Policy.from_dict(
            _on(followed_to_the_event=[{"outcome": name, "beyond_the_cap": False}])
        ) == (
            None,
            [pk.NOT_A_POLICY],
        )
    # half of a surrogate pair, from an escape: the reader counts it, and the writer never reaches the encoder
    lone = chr(0xD800)
    escaped = b'{"name": "site' + bytes([92]) + b'ud800A"}'
    assert pk.loads(escaped, 10**6) == (None, [pk.hidden_reason(1)])
    with pytest.raises(ValueError) as refused:
        pk.dumps(pk.RESULTS, {"name": "site" + lone + "A"})
    assert "site" not in str(refused.value)
    # an emoji keeps its picture without the selector, and accents, letters of other scripts and a line feed pass
    for fine in (
        "caf" + chr(0xE9),
        chr(0x4E2D) + chr(0x6587),
        chr(0x1F600),
        "two" + chr(10) + "lines",
    ):
        assert pk.hidden(fine) == 0
    assert (
        pk.loads(b"[" * pk.DEPTH + b"]" * pk.DEPTH, 10**6)[1] == []
    )  # the deepest nest that is read
