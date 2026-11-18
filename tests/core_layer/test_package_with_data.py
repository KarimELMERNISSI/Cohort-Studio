"""
The formats of a study package that holds the data (core.package, the tier ``reproduction``): its members and their
names, the two writings (written, carried), the manifest and the zip's order per tier, the two clear notices, and
the closed index of its package.json. Nothing here writes a zip: the zip's writer and its safe reader come after.
"""

from __future__ import annotations

import copy
import hashlib
import re
import unicodedata
import zipfile
from typing import Any

import pytest

from core import package as pk

BACKSLASH = chr(92)  # built, never typed in a string: the editor would read an escape

PLANTED = "P-0042 Jane"
TAB, SHY, RLO, ZWSP = (
    chr(9),
    chr(0xAD),
    chr(0x202E),
    chr(0x200B),
)  # built, never typed: none stands in this file
BOM = bytes([0xEF, 0xBB, 0xBF])
HASHES = [f"{number:064x}" for number in range(1, 12)]  # made-up step hashes
STEPS = (
    ("initial_load", HASHES[0]),
    ("recode_levels", HASHES[1]),
    ("enrichment", HASHES[2]),
    ("external_write", HASHES[3]),
    ("enrichment", HASHES[4]),
    ("mi_impute", HASHES[5]),
)
FILES = (
    pk.DataFile("source", None, "source.csv", False, 1206, 41),
    pk.DataFile(
        "enrichment", 3, "merge_01.parquet", True, 1011, 5, pk.Cut("rows_taken", 1017, "f" * 64)
    ),
    pk.DataFile("enrichment", 5, "merge_02.parquet", True, 30, 2, pk.Cut("every_row", 30, None)),
    pk.DataFile("snapshot", 4, "snapshot_01.parquet", False, 1206, 43),
)
ANIMALS = pk.Policy(rules=pk.NONE, rows_not_people=("animal_data", "study"), rows_word="animals")
ANCHOR = pk.anchor_of([b"id,age\n1,40\n"])
# the notices of schema 1, by their SHA-256: frozen with the schema (a new wording is a new schema)
NOTICE_A_SHA256 = "cca67eb720e420786a464977b50212e4353a6ff3e8a176173a3664dfe0a13aeb"
NOTICE_B_SHA256 = "05cb35c8a1a68ef0d76d05a7fbab4ba2c8ffe4d62d6c1bc6c2333da6bb22426a"


def _index(**changes: Any) -> pk.DataPackageIndex:
    parts: dict[str, Any] = {
        "study": pk.StudyRef("tutorial_cohort", 3, "a" * 64, "b" * 64, 2),
        "software": pk.Software(
            "2.5.0",
            "c" * 40,
            "d" * 64,
            "e" * 64,
            "3.14.3",
            "Windows",
            "AMD64",
            "little",
            20260101,
            (("lifelines", None), ("numpy", "2.4.0"), ("pandas", "3.1.0")),
        ),
        "trace": pk.TraceSummary(2, "source", STEPS),
        "files": FILES,
        "policy": pk.Policy(rules=pk.NONE),
        "declaration": pk.Declaration(
            "data_sharing_agreement", "A. Martin, University of Lyon", "DSA-2026-014"
        ),
        "anchor": ANCHOR,
    }
    return pk.DataPackageIndex(**{**parts, **changes})


def _written(index: pk.DataPackageIndex) -> dict[str, Any]:
    value, reasons = pk.loads(
        pk.dumps(pk.PACKAGE, index.to_dict(), pk.REPRODUCTION),
        pk.cap_of(pk.PACKAGE, pk.REPRODUCTION),
    )
    assert reasons == []
    return value


def _texts(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [*value, *(text for inner in value.values() for text in _texts(inner))]
    if isinstance(value, list):
        return [text for inner in value for text in _texts(inner)]
    return [value] if isinstance(value, str) else []


# ----------------------------------------------------------------------------- the members and their names


def test_the_members_of_each_tier_and_what_stays_clear() -> None:
    assert pk.members_of(pk.SHAREABLE) == pk.MEMBERS == pk.members_of(pk.RULES_OFF)
    assert pk.members_of(pk.REPRODUCTION) == (
        "NOTICE.txt",
        "package.json",
        "protocol.json",
        "results.json",
        "trace.json",
        "profiles.json",
        "README.md",
        "uv.lock",
        "MANIFEST.sha256",
    )
    # a package that holds the data shows its notice and its manifest only: its hash is read before any password
    assert pk.clear_members(pk.REPRODUCTION) == ("NOTICE.txt", "MANIFEST.sha256")
    assert pk.clear_members(pk.SHAREABLE) == pk.MEMBERS
    assert [pk.holds_data(tier) for tier in pk.TIERS] == [False, False, True]
    # one format name per shape: the index, the revision and the results are not the same in the two tiers
    assert dict(pk.formats_of(pk.REPRODUCTION)) == {
        "package.json": "cohortstudio.study-package-with-data",
        "protocol.json": "cohortstudio.study-definition",
        "results.json": "cohortstudio.exact-results",
        "trace.json": "cohortstudio.trace",
        "profiles.json": "cohortstudio.column-profiles",
    }
    assert pk.formats_of(pk.SHAREABLE) == pk.FORMATS  # the revision as shared, without its notes
    every = [name for tier in pk.TIERS for name in pk.formats_of(tier).values()]
    assert set(pk.FORMAT_SCHEMAS) == set(
        every
    )  # the one table of schemas, and no format without one
    assert not set(pk.formats_of(pk.REPRODUCTION).values()) & set(pk.FORMATS.values())
    carried = [m for m in pk.members_of(pk.REPRODUCTION) if pk.is_carried(m, pk.REPRODUCTION)]
    assert carried == ["protocol.json", "results.json", "trace.json", "profiles.json"]
    assert not any(pk.is_carried(member, pk.SHAREABLE) for member in pk.MEMBERS)
    with pytest.raises(ValueError, match="unknown tier"):
        pk.members_of("public")
    # what each member may hold, before encryption
    assert pk.cap_of(pk.MANIFEST) == 4096 and pk.cap_of(pk.MANIFEST, pk.REPRODUCTION) == 64 * 1024
    assert pk.cap_of("merge_07.parquet", pk.REPRODUCTION) == pk.DATA_MEMBER_CAP
    for name, tier in (
        ("merge_07.parquet", pk.SHAREABLE),
        ("trace.json", pk.SHAREABLE),
        ("notes.txt", pk.REPRODUCTION),
    ):
        with pytest.raises(ValueError):
            pk.cap_of(name, tier)
    fixed = sum(pk.cap_of(member, pk.REPRODUCTION) for member in pk.members_of(pk.REPRODUCTION))
    assert fixed + pk.DATA_MEMBER_CAP <= pk.package_cap(pk.REPRODUCTION)
    assert pk.package_cap(pk.SHAREABLE) == pk.PACKAGE_CAP == pk.package_cap(pk.RULES_OFF)
    # no member needs a ZIP64 field, which the format does not hold, nor the whole a 64-bit offset (a zip writer
    # adds the field above ZIP64_LIMIT, with a margin of 5%)
    assert pk.DATA_MEMBER_CAP * 1.05 < zipfile.ZIP64_LIMIT
    # nor to a member whose header starts beyond that bound: the whole stays below it too
    assert pk.package_cap(pk.REPRODUCTION) * 1.05 < zipfile.ZIP64_LIMIT


def test_the_data_members_are_named_by_the_format_never_by_a_file() -> None:
    from ingest.loaders import SUPPORTED_EXTENSIONS

    # every file the loader reads travels: as its bytes, or (a workbook that can hold code) as the table read
    loaded = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
    assert sorted(pk.SOURCE_EXTENSIONS + pk.TABLE_ONLY_EXTENSIONS) == loaded
    assert not set(pk.SOURCE_EXTENSIONS) & set(pk.TABLE_ONLY_EXTENSIONS)
    assert [pk.source_member(ext) for ext in (".CSV", "xlsx", ".parquet")] == [
        "source.csv",
        "source.xlsx",
        "source.parquet",
    ]
    for workbook in (".xlsm", "XLS"):
        with pytest.raises(ValueError, match="travels as the table the study read"):
            pk.source_member(workbook)
    with pytest.raises(ValueError, match="not a file a study is loaded from"):
        pk.source_member(".exe")
    assert pk.version_member() == "version.parquet"
    assert (pk.merge_member(1), pk.merge_member(12), pk.snapshot_member(99)) == (
        "merge_01.parquet",
        "merge_12.parquet",
        "snapshot_99.parquet",
    )
    for wrong in (0, 100, True, 1.0, -1):
        with pytest.raises(ValueError):
            pk.merge_member(wrong)  # type: ignore[arg-type]
    for name, role in (
        ("source.csv", "source"),
        ("source.xlsx", "source"),
        ("version.parquet", "saved_version"),
        ("merge_01.parquet", "enrichment"),
        ("snapshot_10.parquet", "snapshot"),
    ):
        assert pk.data_role(name) == role
    for name in (
        "source.CSV",
        "Source.csv",
        "source.xlsm",
        "source.xls",
        "source.exe",
        "source.csv.exe",
        "../source.csv",
        "data/source.csv",
        "merge_00.parquet",
        "merge_1.parquet",
        "merge_100.parquet",
        "merge_01.csv",
        "snapshot_01.PARQUET",
        "version.csv",
        "package.json",
        None,
        3,
    ):
        assert pk.data_role(name) is None, name
    # the data of one package: one start, the others numbered from 01 without a gap
    assert (
        pk.data_members_problem(
            ["source.csv", "merge_01.parquet", "merge_02.parquet", "snapshot_01.parquet"]
        )
        == ""
    )
    assert pk.data_members_problem(["version.parquet", "package.json"]) == ""
    for names in (
        [],
        ["merge_01.parquet"],
        ["source.csv", "version.parquet"],
        ["source.csv", "source.xlsx"],
        ["source.csv", "merge_02.parquet"],
        ["source.csv", "snapshot_01.parquet", "snapshot_03.parquet"],
        ["source.csv", "merge_01.parquet", "merge_01.parquet"],
    ):
        assert pk.data_members_problem(names) == pk.NOT_THE_DATA, names


def test_the_roles_of_a_file_are_the_package_s_and_the_engine_reads_them() -> None:
    from core.lineage import TRACE_SCHEMA
    from core.steps import SNAPSHOT
    from core.steps import ids as step_ids
    from manage.replay import engine
    from utils.settings import StatisticsSection

    assert engine.ROLE_WORDS is pk.ROLE_WORDS and engine.FileRole is pk.FileRole
    assert pk.FILE_ROLES == ("source", "saved_version", "replayed_state", "enrichment", "snapshot")
    assert set(pk.ROLE_WORDS) == set(pk.FILE_ROLES)
    assert pk.TRACE_SCHEMA == TRACE_SCHEMA
    # the steps a recorded table stands for: those replayed from their snapshot, but the adopted imputation
    assert set(pk.SNAPSHOT_STEPS) | {pk.ADOPTED_IMPUTATION} == set(step_ids(replay=SNAPSHOT))
    assert pk.ADOPTED_IMPUTATION not in pk.SNAPSHOT_STEPS
    # the seed's bound is the setting's
    bounds = StatisticsSection.model_fields["random_seed"].metadata
    assert pk.SEED_MOST == next(bound.le for bound in bounds if hasattr(bound, "le"))
    # every other kind of step this version records can be named in an index
    every = tuple(name for name in step_ids() if name != pk.ADOPTED_IMPUTATION)
    assert len(every) > 10
    pk.TraceSummary(2, "source", tuple((name, HASHES[0]) for name in every))
    with pytest.raises(ValueError):
        pk.TraceSummary(
            2, "source", (("initial_load", HASHES[0]), (pk.ADOPTED_IMPUTATION, HASHES[1]))
        )


def test_the_tiers_of_a_check_are_the_package_s() -> None:
    from core import verification as v

    assert v.TIERS == pk.TIERS
    assert (v.SHAREABLE, v.RULES_OFF, v.REPRODUCTION) == (
        pk.SHAREABLE,
        pk.RULES_OFF,
        pk.REPRODUCTION,
    )
    # a package made with the rules off holds no data: the checks of a shareable package run on it
    assert v.for_tier(pk.RULES_OFF) == v.for_tier(pk.SHAREABLE)
    assert len(v.for_tier(pk.REPRODUCTION)) > len(v.for_tier(pk.SHAREABLE))


# ----------------------------------------------------------------------------- the two writings


def test_a_carried_member_holds_the_study_s_own_texts_in_ascii() -> None:
    """A sealed trace can hold a tab or a direction mark in a recoded value: the members that carry the study's
    texts are written with escapes, and read back as they were; the written members keep the rule for a text."""
    texts = {
        "site": "Lyon" + ZWSP,
        "reason": "typed" + TAB + "here",
        "name": "x" + SHY + RLO + "y",
        "accents": "Fièvre à l'entrée",
    }
    payload = {"trace": {"steps": [{"params": texts, "Lyon" + ZWSP: 1}]}}
    data = pk.dumps(pk.TRACE, payload, pk.REPRODUCTION)
    assert data.isascii() and (chr(92) + "u200b").encode() in data
    assert pk.dumps(pk.TRACE, dict(reversed(list(payload.items()))), pk.REPRODUCTION) == data
    value, reasons = pk.loads(data, pk.cap_of(pk.TRACE, pk.REPRODUCTION), carried=True)
    assert reasons == [] and value["trace"] == payload["trace"]
    assert pk.read_envelope(value, pk.TRACE, pk.REPRODUCTION) == []
    assert value["format"] == "cohortstudio.trace" and value["schema"] == 1
    # read as a written member, the same bytes are refused: they hold what no written text may
    refused = pk.loads(data, pk.cap_of(pk.TRACE, pk.REPRODUCTION))
    assert refused[0] is None and "that no text of a package may hold" in refused[1][0]
    # a written member keeps the rule, and is not read as a carried one (its accents are UTF-8)
    with pytest.raises(ValueError, match="no text of a package may hold"):
        pk.dumps(pk.PACKAGE, {"study": {"title": "Lyon" + ZWSP}}, pk.REPRODUCTION)
    written = pk.dumps(pk.PACKAGE, {"note": "Fièvre"}, pk.REPRODUCTION)
    assert pk.loads(written, 10**6, carried=True) == (None, [pk.NOT_JSON])
    assert pk.loads(written, 10**6)[1] == []
    # what the strict reader refuses, the carried reader refuses too
    for refused_data, reason in (
        (BOM + b"{}", pk.NOT_JSON),
        (b'{"a": 1, "a": 2}', pk.KEY_TWICE),
        (b'{"a": NaN}', pk.NOT_A_NUMBER),
        (b'{"a": 1e999}', pk.NOT_A_NUMBER),
        (b"[" * 70 + b"]" * 70, pk.NOT_JSON),
        (('{"a": "' + chr(0xE9) + '"}').encode(), pk.NOT_JSON),
    ):
        assert pk.loads(refused_data, 10**6, carried=True) == (None, [reason]), refused_data[:20]
    assert pk.loads(b"{}", 1, carried=True) == (None, [pk.TOO_LARGE])
    # a member of the other tier is no member of this one
    with pytest.raises(ValueError, match="not a JSON member"):
        pk.dumps(pk.TRACE, payload, pk.SHAREABLE)
    # the member's own keys are the format's, in both writings
    with pytest.raises(ValueError, match="no text of a package may hold"):
        pk.dumps(pk.TRACE, {"trace" + ZWSP: 1}, pk.REPRODUCTION)


def test_a_lone_surrogate_is_refused_in_both_writings() -> None:
    """A carried text may hold any character UTF-8 can hold, not a lone surrogate: no sealed trace holds one (its
    hash is taken over UTF-8), and checking the chain of one would fail on it."""
    lone = chr(0xD800)
    for payload in ({"trace": {"value": "a" + lone}}, {"trace": {"a" + lone: 1}}):
        with pytest.raises(ValueError, match="no text of a package may hold"):
            pk.dumps(pk.TRACE, payload, pk.REPRODUCTION)
    escaped = (
        '{"format": "cohortstudio.trace", "schema": 1, "value": "' + BACKSLASH + 'ud800"}'
    ).encode()
    value, reasons = pk.loads(escaped, 10**6, carried=True)
    assert value is None and reasons == [pk.hidden_reason(1)]
    # a pair is one character: written and read back as such
    pair = {"trace": {"value": chr(0x1F600)}}
    data = pk.dumps(pk.TRACE, pair, pk.REPRODUCTION)
    assert pk.loads(data, 10**6, carried=True)[0]["trace"] == pair["trace"]


def test_one_reader_reads_each_member_with_its_writing() -> None:
    texts = {"note": "Fièvre" + TAB + RLO}
    for member in pk.formats_of(pk.REPRODUCTION):
        payload = (
            {"value": texts} if pk.is_carried(member, pk.REPRODUCTION) else {"value": "Fièvre"}
        )
        data = pk.dumps(member, payload, pk.REPRODUCTION)
        value, reasons = pk.read_member(member, data, pk.REPRODUCTION)
        assert reasons == [] and value["value"] == payload["value"], member
    # the same bytes read as the other tier's member: another member
    exact = pk.dumps(pk.RESULTS, {"results": []}, pk.REPRODUCTION)
    assert pk.read_member(pk.RESULTS, exact, pk.SHAREABLE) == (None, [pk.NOT_A_MEMBER])
    # the README as a text, under the rule for a text; the lock file as UTF-8
    assert pk.read_member(pk.README, "# Study\n\nFièvre\n".encode(), pk.REPRODUCTION) == (
        "# Study\n\nFièvre\n",
        [],
    )
    for data, reason in (
        (("# x" + RLO + "\n").encode(), pk.hidden_reason(1)),
        (("# x" + chr(13) + "\n").encode(), pk.hidden_reason(1)),
        (BOM + b"# x\n", pk.NOT_A_TEXT),
        (bytes([0xFF, 0xFE]), pk.NOT_A_TEXT),
        (b"x" * (pk.cap_of(pk.README, pk.REPRODUCTION) + 1), pk.TOO_LARGE),
    ):
        assert pk.read_member(pk.README, data, pk.REPRODUCTION) == (None, [reason])
    assert (
        pk.read_member(pk.LOCK, b'version = 1\n[[package]]\nname = "numpy"\n', pk.REPRODUCTION)[1]
        == []
    )
    assert pk.read_member(pk.LOCK, bytes([0xFF]), pk.REPRODUCTION) == (None, [pk.NOT_A_TEXT])
    for other in (pk.NOTICE, pk.MANIFEST, "source.csv", "merge_01.parquet"):
        with pytest.raises(ValueError):
            pk.read_member(other, b"", pk.REPRODUCTION)
    with pytest.raises(ValueError):
        pk.read_member(pk.TRACE, b"{}", pk.SHAREABLE)


def test_one_tier_s_schema_moves_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A change of a package without the data (its index or its results) moves its own format's schema, never that
    of a package that holds the data, whose notice names its own index's schema."""
    index = _index()
    before = pk.dumps(pk.PACKAGE, index.to_dict(), pk.REPRODUCTION)
    for name in ("cohortstudio.study-package", "cohortstudio.study-results"):
        monkeypatch.setitem(pk.FORMAT_SCHEMAS, name, 2)
    assert pk.dumps(pk.PACKAGE, index.to_dict(), pk.REPRODUCTION) == before
    assert pk.DataPackageIndex.from_dict(_written(index)) == (index, [])
    assert pk.dumps(pk.PACKAGE, {"tier": "shareable"}).count(b'"schema": 2') == 1


def test_a_trace_keeps_its_chain_through_the_carried_writing() -> None:
    from core.lineage import genesis_hash, seal_step, verify_trace

    prev, steps = genesis_hash("session-1"), []
    for params in ({"site": "Lyon" + ZWSP}, {"reason": "typed" + TAB + RLO, "word": "entrée"}):
        step = seal_step({"function": "recode_levels", "params": params}, prev)
        steps.append(step)
        prev = step["hash"]
    trace = {"schema_version": 2, "session_id": "session-1", "steps": steps, "head_hash": prev}
    assert verify_trace(trace).ok
    data = pk.dumps(pk.TRACE, {"trace": trace}, pk.REPRODUCTION)
    read = pk.loads(data, pk.cap_of(pk.TRACE, pk.REPRODUCTION), carried=True)[0]["trace"]
    assert read == trace
    checked = verify_trace(read)
    assert checked.ok and checked.head_hash == prev


def test_the_revision_travels_whole_and_keeps_its_order_and_its_checksum() -> None:
    """The package that holds the data carries the revision as saved, so that its checksum can be checked: the body
    keeps the order of the families and of their rules, at every depth, with its notes."""
    from core.protocol import body_checksum

    body = {
        "mask_families": {"z_family": {"b": 1, "a": 2}, "a_family": {}},
        "follow_up": {},
        "notes": "Fièvre" + TAB + "à l'entrée",
    }
    revision = {
        "schema": 1,
        "meta": {"study_id": "cohort", "revision": 2},
        "body": body,
        "checksum": body_checksum(body),
    }
    data = pk.dumps(pk.PROTOCOL, {"revision": revision}, pk.REPRODUCTION)
    read = pk.loads(data, pk.cap_of(pk.PROTOCOL, pk.REPRODUCTION), carried=True)[0]
    assert list(read["revision"]) == ["schema", "meta", "body", "checksum"]
    assert list(read["revision"]["body"]["mask_families"]) == ["z_family", "a_family"]
    assert list(read["revision"]["body"]["mask_families"]["z_family"]) == ["b", "a"]
    assert body_checksum(read["revision"]["body"]) == revision["checksum"]
    assert pk.read_envelope(read, pk.PROTOCOL, pk.REPRODUCTION) == []
    # the shared definition of a package without the data is another member
    assert pk.read_envelope(read, pk.PROTOCOL) == [pk.NOT_A_MEMBER]
    later = {**read, "schema": 2}
    assert pk.read_envelope(later, pk.PROTOCOL, pk.REPRODUCTION) == [pk.later_schema(2, 1)]


# ----------------------------------------------------------------------------- the manifest and the zip's order

MEMBERS = {
    name: name.encode()
    for name in (
        "NOTICE.txt",
        "package.json",
        "protocol.json",
        "results.json",
        "trace.json",
        "profiles.json",
        "README.md",
        "source.xlsx",
        "merge_01.parquet",
        "merge_02.parquet",
        "snapshot_01.parquet",
    )
}


def test_the_manifest_of_a_package_with_the_data_lists_its_notice_and_its_data() -> None:
    manifest = pk.manifest_of(MEMBERS, pk.REPRODUCTION)
    listed, reasons = pk.read_manifest(manifest, pk.REPRODUCTION)
    assert reasons == [] and listed is not None and list(listed) == sorted(MEMBERS)
    assert listed["merge_02.parquet"] == pk.checksum(b"merge_02.parquet")
    # a writer that streams its members makes the same manifest from their checksums
    sums = {name: pk.checksum(data) for name, data in MEMBERS.items()}
    assert pk.manifest_from_checksums(sums, pk.REPRODUCTION) == manifest
    for wrong_sum in ("A" * 64, "a" * 63, None):
        with pytest.raises(ValueError):
            pk.manifest_from_checksums({**sums, "README.md": wrong_sum}, pk.REPRODUCTION)  # type: ignore[dict-item]
    assert (
        pk.read_manifest(
            pk.manifest_of({**MEMBERS, "uv.lock": b"v"}, pk.REPRODUCTION), pk.REPRODUCTION
        )[1]
        == []
    )
    # one tier's manifest is not the other's
    assert pk.read_manifest(manifest) == (None, [pk.NOT_A_MANIFEST])
    shareable = pk.manifest_of(
        {name: b"x" for name in ("package.json", "protocol.json", "results.json", "README.md")}
    )
    assert pk.read_manifest(shareable, pk.REPRODUCTION) == (None, [pk.NOT_A_MANIFEST])
    for wrong in (
        {name: data for name, data in MEMBERS.items() if name != "source.xlsx"},
        {**MEMBERS, "version.parquet": b"v"},
        {name: data for name, data in MEMBERS.items() if name != "merge_01.parquet"},
        {name: data for name, data in MEMBERS.items() if name != "trace.json"},
        {**MEMBERS, "MERGE_03.parquet": b""},
        {**MEMBERS, "notes.txt": b""},
        {**MEMBERS, "MANIFEST.sha256": b""},
    ):
        with pytest.raises(ValueError):
            pk.manifest_of(wrong, pk.REPRODUCTION)
    # read strictly: a line removed or a name in another case is refused
    lines = manifest.decode().splitlines(keepends=True)
    gap = "".join(line for line in lines if not line.endswith("merge_01.parquet\n")).encode()
    assert pk.read_manifest(gap, pk.REPRODUCTION) == (None, [pk.NOT_A_MANIFEST])
    other_case = manifest.replace(b"merge_02.parquet", b"Merge_02.parquet")
    assert pk.read_manifest(other_case, pk.REPRODUCTION) == (None, [pk.NOT_A_MANIFEST])
    over = manifest + b"0" * (pk.cap_of(pk.MANIFEST, pk.REPRODUCTION) - len(manifest)) + b"\n"
    assert pk.read_manifest(over, pk.REPRODUCTION) == (None, [pk.NOT_A_MANIFEST])


def test_the_zip_holds_its_members_in_one_order() -> None:
    order = pk.in_zip_order([*MEMBERS, "MANIFEST.sha256", "uv.lock"], pk.REPRODUCTION)
    assert order == (
        "NOTICE.txt",
        "package.json",
        "protocol.json",
        "results.json",
        "trace.json",
        "profiles.json",
        "README.md",
        "uv.lock",
        "source.xlsx",
        "merge_01.parquet",
        "merge_02.parquet",
        "snapshot_01.parquet",
        "MANIFEST.sha256",
    )
    assert pk.in_zip_order(reversed(order), pk.REPRODUCTION) == order
    shareable = ["MANIFEST.sha256", "README.md", "results.json", "protocol.json", "package.json"]
    assert pk.in_zip_order(shareable, pk.SHAREABLE) == (
        "package.json",
        "protocol.json",
        "results.json",
        "README.md",
        "MANIFEST.sha256",
    )
    for wrong in (list(MEMBERS), [*MEMBERS, "MANIFEST.sha256", "merge_04.parquet"]):
        with pytest.raises(ValueError):
            pk.in_zip_order(wrong, pk.REPRODUCTION)


# ----------------------------------------------------------------------------- the notice


def test_the_notice_is_fixed_ascii_and_says_what_the_package_holds() -> None:
    forbidden = {
        "validated",
        "verified",
        "confirmed",
        "compliant",
        "guarantee",
        "signed",
        "certificate",
    }
    for kind, text in pk.NOTICES.items():
        assert (
            text.isascii()
            and text.endswith(b"\n")
            and not text.endswith(b"\n\n")
            and b"\r" not in text
        )
        lines = text.decode().split("\n")[:-1]
        assert lines[0] == "CohortStudio study package with the data, schema 1"
        assert max(len(line) for line in lines) <= 72
        assert not forbidden & set(re.findall("[a-z]+", text.decode().lower()))
        assert "    python -m manage.replay inspect <this file>" in lines
        assert pk.read_notice(text) == (kind, [])
        assert len(text) <= pk.cap_of(pk.NOTICE, pk.REPRODUCTION)
    personal = " ".join(pk.NOTICES[pk.NOTICE_PERSONAL_DATA].decode().split())
    assert "personal data" in personal and "AES-256" in personal
    assert (
        "Records about people who can be identified are personal data even without their names"
        in personal
    )
    assert (
        "(GDPR Art. 4(1) and 4(5), Recital 26)" in personal
        and "a special category (Art. 9)" in personal
    )
    assert "delete it and tell its sender" in personal
    assert (
        "a study's records about people, and the trace of every step applied to those records"
        in personal
    )
    for notice in (personal, " ".join(pk.NOTICES[pk.NOTICE_NOT_PEOPLE].decode().split())):
        # what the hash is a hash of, and what to do before typing the password and when it is lost
        assert (
            "It prints the package hash (the SHA-256 of MANIFEST.sha256, not of the zip file)"
            in notice
        )
        assert (
            "Before you type the password, check that this is the package its sender made."
            in notice
        )
        assert "if it is lost, ask the sender for a new package" in notice
        assert "Its sender was asked to send the password apart from the package." in notice
        assert "delete it and tell its sender" in notice
        assert (
            "Help >" not in notice
        )  # no menu of the application: its words are frozen with the schema
    not_people = " ".join(pk.NOTICES[pk.NOTICE_NOT_PEOPLE].decode().split())
    assert "personal data" not in not_people and "author" not in not_people  # either level declares
    assert not_people.startswith(
        "CohortStudio study package with the data, schema 1 The rows of this study were declared not people"
    )


def test_the_notices_are_frozen_with_their_schema() -> None:
    """A reader takes any other bytes for a changed file: a notice is never edited in place (a new wording comes
    with a new schema, and the readers keep the earlier notices' bytes). These are the bytes of schema 1."""
    assert {kind: hashlib.sha256(text).hexdigest() for kind, text in pk.NOTICES.items()} == {
        pk.NOTICE_PERSONAL_DATA: NOTICE_A_SHA256,
        pk.NOTICE_NOT_PEOPLE: NOTICE_B_SHA256,
    }


def test_a_notice_is_read_by_its_exact_bytes() -> None:
    text = pk.NOTICES[pk.NOTICE_PERSONAL_DATA]
    for changed in (
        text.replace(b"AES-256", b"AES-128"),
        text.replace(b"\n", b"\r\n"),
        text + PLANTED.encode(),
        text[:-1],
        BOM + text,
        b"",
    ):
        kind, reasons = pk.read_notice(changed)
        assert kind is None and reasons == [pk.NOT_A_NOTICE]
    # a notice of a later schema says so: made by a later version, not changed
    later = text.replace(b"schema 1\n", b"schema 2\n", 1)
    assert pk.read_notice(later) == (None, [pk.later_schema(2, 1)])
    for head in (
        b"schema 0\n",
        b"schema 01\n",
        b"schema 1000001\n",
        b"schema 99999999\n",
        b"schema 2x\n",
    ):
        assert pk.read_notice(text.replace(b"schema 1\n", head, 1)) == (None, [pk.NOT_A_NOTICE]), (
            head
        )
    assert pk.read_notice(b"x" * (pk.cap_of(pk.NOTICE, pk.REPRODUCTION) + 1)) == (
        None,
        [pk.TOO_LARGE],
    )


def test_the_notice_says_what_the_policy_says() -> None:
    none = pk.Policy(rules=pk.NONE)
    assert pk.notice_for(none) == pk.NOTICES[pk.NOTICE_PERSONAL_DATA]
    assert pk.notice_for(ANIMALS) == pk.NOTICES[pk.NOTICE_NOT_PEOPLE]
    assert pk.notice_problem(pk.NOTICE_PERSONAL_DATA, none) == ""
    assert pk.notice_problem(pk.NOTICE_NOT_PEOPLE, ANIMALS) == ""
    assert pk.notice_problem(pk.NOTICE_PERSONAL_DATA, ANIMALS) == pk.NOTICE_NOT_THE_POLICY
    assert pk.notice_problem(pk.NOTICE_NOT_PEOPLE, none) == pk.NOTICE_NOT_THE_POLICY
    # a declaration of the whole installation reads with the same notice as the study's own
    installation = pk.Policy(
        rules=pk.NONE, rows_not_people=("animal_data", "installation"), rows_word="animals"
    )
    assert pk.notice_problem(pk.NOTICE_NOT_PEOPLE, installation) == ""
    # a kind that is no notice's is refused, never taken for one
    for kind in (None, "", "public_data", PLANTED, ["personal_data"], 1):
        assert pk.notice_problem(kind, none) == pk.NOT_A_NOTICE


# ----------------------------------------------------------------------------- the index


def test_the_index_writes_every_key_and_reads_back() -> None:
    index = _index()
    value = _written(index)
    assert pk.DataPackageIndex.from_dict(value) == (index, [])
    assert set(value) == {"format", "schema", *pk.INDEX_KEYS}
    assert (value["format"], value["schema"], value["tier"]) == (
        "cohortstudio.study-package-with-data",
        1,
        "reproduction",
    )
    # the anchor: the start member's bytes under a fixed prefix, which is not that member's own checksum
    assert value["anchor"] == ANCHOR
    start = b"id,age\n1,40\n"
    assert ANCHOR == hashlib.sha256(pk.ANCHOR_PREFIX + start).hexdigest() != pk.checksum(start)
    assert pk.anchor_of([start[:3], start[3:]]) == ANCHOR
    assert value["trace"]["head"] == HASHES[5]
    assert [step["index"] for step in value["trace"]["steps"]] == [1, 2, 3, 4, 5, 6]
    assert value["files"][1] == {
        "role": "enrichment",
        "step": 3,
        "member": "merge_01.parquet",
        "as_read": True,
        "rows": 1011,
        "columns": 5,
        "cut": {"rule": "rows_taken", "rows_of_original": 1017, "original_sha256": "f" * 64},
    }
    assert value["files"][0]["cut"] is None
    assert value["software"]["libraries"] == {
        "lifelines": None,
        "numpy": "2.4.0",
        "pandas": "3.1.0",
    }
    assert value["declaration"] == {
        "recipient": "A. Martin, University of Lyon",
        "basis": "data_sharing_agreement",
        "reference": "DSA-2026-014",
        "words": 1,
    }
    assert value["policy"]["rules"] == "none"
    assert index.data_members() == (
        "source.csv",
        "merge_01.parquet",
        "merge_02.parquet",
        "snapshot_01.parquet",
    )
    # no text the user or the data gave, but the study's id and the declaration's two: ids, versions, checksums and
    # member names
    declared = value.pop("declaration")
    assert all(re.fullmatch("[A-Za-z0-9_.-]+", text) for text in _texts(value))
    value["declaration"] = declared
    # one writing: the same index, the same bytes; another recipient, another package's hash
    data = pk.dumps(pk.PACKAGE, index.to_dict(), pk.REPRODUCTION)
    assert (
        pk.dumps(pk.PACKAGE, dict(reversed(list(index.to_dict().items()))), pk.REPRODUCTION) == data
    )
    other = _index(
        declaration=pk.Declaration(
            "data_sharing_agreement", "B. Weber, University of Trier", "DSA-2026-015"
        )
    )
    assert pk.dumps(pk.PACKAGE, other.to_dict(), pk.REPRODUCTION) != data


def test_an_index_holds_what_a_study_may_start_from_and_be_declared() -> None:
    # a source read from a workbook's sheets, carried as the table read
    workbook = (pk.DataFile("source", None, "source.parquet", True, 1206, 41), *FILES[1:])
    assert pk.DataPackageIndex.from_dict(_written(_index(files=workbook)))[0] is not None
    # a trace that starts from a saved version, without merged files
    version = _index(
        trace=pk.TraceSummary(2, "saved_version", STEPS[:2]),
        files=(pk.DataFile("saved_version", None, "version.parquet", False, 900, 12),),
    )
    assert pk.DataPackageIndex.from_dict(_written(version)) == (version, [])
    # the exporter's own organisation needs no reference; a study whose rows are declared not people names its
    # declaration and needs neither (M8b)
    own = _index(declaration=pk.Declaration("same_organisation", "Data team, site B"))
    assert pk.DataPackageIndex.from_dict(_written(own)) == (own, [])
    animals = _index(policy=ANIMALS, declaration=pk.Declaration("rows_not_people"))
    assert pk.DataPackageIndex.from_dict(_written(animals)) == (animals, [])
    # alone, the declaration that the rows are not people is one: its index checks it against the policy
    assert pk.Declaration("rows_not_people").recipient is None
    # an index is built only from values a package holds
    on = pk.Policy(
        rules=pk.ON,
        version=2,
        k=10,
        k_default=10,
        floor=10,
        floor_set_by="code",
        cap=0.05,
        cap_set_by="code",
    )
    unsorted = (("pandas", "3"), ("numpy", "2"))
    decomposed = unicodedata.normalize("NFD", "Université de Lyon")
    for build in (
        lambda: pk.Declaration("data_sharing_agreement", None, "DSA-1"),
        lambda: pk.Declaration("other_basis", "A. Martin"),
        lambda: pk.Declaration(
            "data_sharing_agreement", decomposed, "DSA-1"
        ),  # one way of writing each text
        lambda: pk.DataFile("source", 1, "source.csv", False, 1, 1),
        lambda: pk.DataFile("source", None, "source.csv", True, 1, 1),
        lambda: pk.DataFile("source", None, "source.csv", False, 1, 1, pk.Cut("every_row", 1)),
        lambda: pk.DataFile(
            "snapshot", 4, "snapshot_01.parquet", False, 1, 1, pk.Cut("every_row", 1)
        ),
        lambda: pk.DataFile(
            "enrichment", 3, "merge_01.parquet", False, 1, 1, pk.Cut("rows_taken", 1)
        ),
        lambda: pk.DataFile("snapshot", 4, "merge_01.parquet", False, 1, 1),
        lambda: pk.TraceSummary(2, "source", ()),
        lambda: pk.Software(
            "2.5", None, None, None, "3.14.3", "Linux", "x86_64", "little", 1, unsorted
        ),
        lambda: _index(declaration=pk.Declaration("rows_not_people")),
        lambda: _index(policy=on),
        lambda: _index(anchor="0" * 63),
        lambda: _index(files=FILES[:1] + FILES[2:]),  # merge_02 without merge_01
        lambda: _index(
            files=(FILES[0], FILES[2], FILES[1], FILES[3])
        ),  # the merges out of their steps' order
    ):
        with pytest.raises(ValueError) as refused:
            build()
        assert str(refused.value) == pk.NOT_AN_INDEX
    assert (
        pk.Declaration(
            "data_sharing_agreement", unicodedata.normalize("NFC", decomposed), "DSA-1"
        ).recipient
        == "Université de Lyon"
    )


MISSING = object()


def _with(value: dict[str, Any], path: tuple[Any, ...], new: Any) -> dict[str, Any]:
    changed = copy.deepcopy(value)
    where = changed
    for key in path[:-1]:
        where = where[key]
    if new is MISSING:
        del where[path[-1]]
    else:
        where[path[-1]] = new
    return changed


@pytest.mark.parametrize(
    ("path", "new", "reason"),
    [
        (("note",), PLANTED, pk.later_index_keys(1)),
        (("declaration",), MISSING, pk.NOT_AN_INDEX),
        (("tier",), "shareable", pk.NOT_AN_INDEX),
        (("tier",), PLANTED, pk.LATER_INDEX_VALUE),
        (("study", "revision"), 0, pk.NOT_AN_INDEX),
        (("study", "title"), PLANTED, pk.later_index_keys(1)),
        (("study", "id"), "../x", pk.NOT_AN_INDEX),
        (("study", "revision_checksum"), PLANTED, pk.NOT_AN_INDEX),
        (("study", "protocol_hash"), "b" * 63, pk.NOT_AN_INDEX),
        (("study", "rules_version"), 0, pk.NOT_AN_INDEX),
        (("software", "commit"), "abc", pk.NOT_AN_INDEX),
        (("software", "byteorder"), "middle", pk.NOT_AN_INDEX),
        (("software", "libraries"), {"Numpy": "2.4"}, pk.NOT_AN_INDEX),
        (("software", "libraries"), [["numpy", "2.4"]], pk.NOT_AN_INDEX),
        (("software", "libraries"), {"numpy": "2.4 (local)"}, pk.NOT_AN_INDEX),
        (("software", "random_seed"), -1, pk.NOT_AN_INDEX),
        (("software", "random_seed"), True, pk.NOT_AN_INDEX),
        (("software", "random_seed"), 2**32, pk.NOT_AN_INDEX),
        (("software", "python"), "3.14" + ZWSP, pk.NOT_AN_INDEX),
        # the system and the machine are the platform's words: never a host's name, an account or a path
        (("software", "machine"), "host.example/jdupont", pk.NOT_AN_INDEX),
        (("software", "machine"), "JDUPONT-PC", pk.NOT_AN_INDEX),
        (("software", "machine"), "jdupont-laptop.chu-example.local", pk.NOT_AN_INDEX),
        (("software", "machine"), "", pk.NOT_AN_INDEX),
        (("software", "machine"), "aarch64", None),
        (("software", "system"), "Windows 11", pk.NOT_AN_INDEX),
        (("software", "system"), "Linux", None),
        (("software", "cohortstudio"), "", pk.NOT_AN_INDEX),
        (("software", "cohortstudio"), "2.5.0.dev3+g1a2b3c4", None),
        (("software", "code_sha256"), "abc", pk.NOT_AN_INDEX),
        (("software", "uv_lock_sha256"), PLANTED, pk.NOT_AN_INDEX),
        (("anchor",), "A" * 64, pk.NOT_AN_INDEX),
        (("anchor",), MISSING, pk.NOT_AN_INDEX),
        (("trace", "schema_version"), 1, pk.NOT_AN_INDEX),
        (("trace", "schema_version"), 3, pk.LATER_INDEX_VALUE),
        (("trace", "head"), HASHES[0], pk.NOT_AN_INDEX),
        (("trace", "steps", 0, "index"), 2, pk.NOT_AN_INDEX),
        (("trace", "steps", 0, "index"), True, pk.NOT_AN_INDEX),
        (("trace", "steps", 1, "function"), "Recode levels", pk.NOT_AN_INDEX),
        (("trace", "steps"), [], pk.NOT_AN_INDEX),
        (("trace", "starts_from"), "saved_version", pk.NOT_AN_INDEX),
        (("trace", "starts_from"), PLANTED, pk.LATER_INDEX_VALUE),
        # an adopted dataset of a multiple imputation: an imputed table, which never travels
        (("trace", "steps", 1, "function"), "multiple_imputation", pk.NOT_AN_INDEX),
        # a recorded table stands for a step whose table only it gives: never one that documents, nor an imputation
        (("files", 3, "step"), 1, pk.NOT_AN_INDEX),
        (("files", 3, "step"), 6, pk.NOT_AN_INDEX),
        (("files", 3, "step"), 2, pk.NOT_AN_INDEX),
        (("files", 1, "as_read"), False, pk.NOT_AN_INDEX),
        (("files", 1, "as_read"), "yes", pk.NOT_AN_INDEX),
        (("files", 1, "rows"), -1, pk.NOT_AN_INDEX),
        (("files", 1, "rows"), True, pk.NOT_AN_INDEX),
        (("files", 1, "columns"), -1, pk.NOT_AN_INDEX),
        (("files", 1, "cut", "rows_of_original"), -1, pk.NOT_AN_INDEX),
        (
            ("files", 0, "cut"),
            {"rule": "every_row", "rows_of_original": 1206, "original_sha256": None},
            pk.NOT_AN_INDEX,
        ),
        (
            ("files", 3, "cut"),
            {"rule": "every_row", "rows_of_original": 1206, "original_sha256": None},
            pk.NOT_AN_INDEX,
        ),
        (("files", 1, "cut"), None, pk.NOT_AN_INDEX),
        (("files", 1, "cut", "rule"), PLANTED, pk.LATER_INDEX_VALUE),
        (("files", 1, "rows"), 1018, pk.NOT_AN_INDEX),
        (("files", 2, "rows"), 29, pk.NOT_AN_INDEX),
        (("files", 1, "cut", "original_sha256"), PLANTED, pk.NOT_AN_INDEX),
        (("files", 1, "step"), 2, pk.NOT_AN_INDEX),
        (("files", 1, "step"), 5, pk.NOT_AN_INDEX),
        (("files", 3, "step"), 99, pk.NOT_AN_INDEX),
        (("files", 1, "role"), "replayed_state", pk.NOT_AN_INDEX),
        (("files", 1, "role"), PLANTED, pk.LATER_INDEX_VALUE),
        (("files", 0, "as_read"), True, pk.NOT_AN_INDEX),
        (("files", 0, "member"), "../source.csv", pk.NOT_AN_INDEX),
        (("files", 0, "path"), PLANTED, pk.later_index_keys(1)),
        (("files", 1), MISSING, pk.NOT_AN_INDEX),
        (("declaration", "basis"), PLANTED, pk.LATER_INDEX_VALUE),
        (("declaration", "basis"), "rows_not_people", pk.NOT_AN_INDEX),
        (("declaration", "basis"), "other_basis", None),
        (("declaration", "recipient"), None, pk.NOT_AN_INDEX),
        (("declaration", "reference"), None, pk.NOT_AN_INDEX),
        (("declaration", "recipient"), "x" * 201, pk.NOT_AN_INDEX),
        (("declaration", "recipient"), "A. Martin\nUniversity of Lyon", pk.NOT_AN_INDEX),
        (("declaration", "recipient"), "   ", pk.NOT_AN_INDEX),
        (("declaration", "recipient"), "A. Martin" + RLO, pk.NOT_AN_INDEX),
        (
            ("declaration", "recipient"),
            unicodedata.normalize("NFD", "Université de Lyon"),
            pk.NOT_AN_INDEX,
        ),
        (("declaration", "reference"), "x" * 101, pk.NOT_AN_INDEX),
        (("declaration", "basis"), "same_organisation", None),
        # the version of the words is a record: a later one is read (the README carries the words)
        (("declaration", "words"), 2, None),
        (("declaration", "words"), 0, pk.NOT_AN_INDEX),
        (("policy", "rules"), "on", pk.NOT_A_POLICY),
        (("format",), "cohortstudio.trace", pk.NOT_A_MEMBER),
        (("schema",), 2, pk.later_schema(2, 1)),
    ],
)
def test_an_index_this_version_cannot_read_is_refused_in_words_that_quote_nothing(
    path: tuple[Any, ...], new: Any, reason: str | None
) -> None:
    index, reasons = pk.DataPackageIndex.from_dict(_with(_written(_index()), path, new))
    if reason is None:  # a change that leaves an index this version reads
        assert index is not None and reasons == []
        return
    assert index is None and reasons == [reason]
    assert PLANTED not in reasons[0] and "Martin" not in reasons[0]


def test_an_index_of_another_shape_is_refused_whatever_it_holds() -> None:
    for value in (None, [], "index", {"format": "cohortstudio.study-package", "schema": 1}):
        index, reasons = pk.DataPackageIndex.from_dict(value)
        assert index is None and reasons in ([pk.NOT_A_MEMBER], [pk.NOT_AN_INDEX])
    written = _written(_index())
    for path, new in (
        (("study",), [1, 2]),
        (("software",), "2.5.0"),
        (("trace", "steps", 0), ["initial_load"]),
        (("files", 0), None),
        (("files", 1, "cut"), [1017]),
        (
            ("declaration",),
            {"basis": ["data_sharing_agreement"], "recipient": None, "reference": None, "words": 1},
        ),
        (("software", "libraries"), {"numpy": ["2.4"]}),
    ):
        assert pk.DataPackageIndex.from_dict(_with(written, path, new)) == (
            None,
            [pk.NOT_AN_INDEX],
        ), path


# ----------------------------------------------------------------------------- a text on a terminal


def test_a_text_is_shown_on_a_terminal_with_marks_it_can_print() -> None:
    assert pk.shown("a" + RLO + "b\nc", terminal=True) == "a?b?c"
    cut = pk.shown("x" * 200, 20, terminal=True)
    assert cut == "x" * 17 + "..." and len(cut) == 20
    assert pk.shown("x" * 200, 20) == "x" * 19 + pk.CUT
    assert pk.shown("a" + RLO + "b") == "a" + pk.REPLACED + "b"
    # on a terminal, ASCII only: another character outside ASCII is the escape an ASCII stream writes, counted in
    # the width, and "?" means only a character no package may hold
    accented = "Fi" + chr(0xE8) + "vre " + chr(0x416) + " " + chr(0x1F600)
    expected = "Fi" + BACKSLASH + "xe8vre " + BACKSLASH + "u0416 " + BACKSLASH + "U0001f600"
    assert pk.shown(accented, terminal=True) == expected
    assert (
        accented.encode("ascii", "backslashreplace").decode() == expected
    )  # as the stream would write it
    escapes = (chr(0xE8) * 30) + "end"
    for limit in (3, 4, 7, 9, 20, 40, 120):
        text = pk.shown(escapes + RLO + "x" * 200, limit, terminal=True)
        assert text.isascii() and len(text) <= limit and text.endswith("...")
        body = text.removesuffix("...")  # whole escapes only: none is cut in two
        assert re.fullmatch("(" + BACKSLASH * 2 + "xe8)*", body), (limit, text)
    for limit in (0, 1, 2):  # narrower than the mark: what fits of it
        assert pk.shown("abcdef", limit, terminal=True) == "..."[:limit]
    # a backslash is doubled: an escape is never a text's own characters
    looks_like = BACKSLASH + "xe8"
    assert (
        pk.shown(looks_like, terminal=True)
        == BACKSLASH * 2 + "xe8"
        != pk.shown(chr(0xE8), terminal=True)
    )
