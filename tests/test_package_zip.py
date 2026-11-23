"""
The zip of a study package (manage.package_zip): one writer, the password, and the safe reader of a received file,
for a package that holds the data (encrypted) and one that does not (clear). The members are made with
core.package's formats; a data member is bytes the reader never parses.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import os
import re
import struct
import subprocess
import sys
import types
import zipfile
import zlib
from pathlib import Path
from typing import Any

import pytest

from core import package as pk
from manage import package_zip as pz

ROOT = Path(__file__).resolve().parents[1]
# made up, in the one form new_password makes; no package's (gitleaks reads it as a key: this line only is allowed)
PASSWORD = "K7QD-9F2M-XW4R-T8NB-3HVJ"  # gitleaks:allow
WRONG = "AAAA-BBBB-CCCC-DDDD-EEEE"
SOURCE = b"patient_id,age\n" + b"".join(f"{row},{20 + row % 60}\n".encode() for row in range(2000))
CUT = b"made-up bytes standing for a Parquet table" * 10
HASHES = [f"{number:064x}" for number in range(1, 8)]
TAB = chr(9)
CANARY = "canary-7Q4F-never-shown"
RECIPIENT = "A. Martin, University of Lyon"
WITH_DATA_MODE = pz.MODES[pk.REPRODUCTION]
# the SHA-256 of the zip of _shareable(): one study, one byte string, on every system and Python this version runs on
SHAREABLE_ZIP = "234d5c6bbe146c6c08941401ae23a2c8fe4cc858b83d54554cecfd6bb761771a"


def _index(
    anchor: str,
    *,
    files: tuple[pk.DataFile, ...] = (),
    policy: pk.Policy | None = None,
    recipient: str = RECIPIENT,
) -> pk.DataPackageIndex:
    return pk.DataPackageIndex(
        study=pk.StudyRef("tutorial_cohort", 3, "a" * 64, "b" * 64, 2),
        software=pk.Software(
            "2.5.0", None, None, None, "3.14.3", "Windows", "AMD64", "little", 20260101, ()
        ),
        trace=pk.TraceSummary(
            2, "source", (("initial_load", HASHES[0]), ("enrichment", HASHES[1]))
        ),
        files=files
        or (
            pk.DataFile("source", None, "source.csv", False, 2000, 2),
            pk.DataFile(
                "enrichment", 2, "merge_01.parquet", True, 10, 3, pk.Cut("rows_taken", 12, "c" * 64)
            ),
        ),
        policy=policy or pk.Policy(rules=pk.NONE),
        declaration=pk.Declaration("data_sharing_agreement", recipient, "DSA-2026-014"),
        anchor=anchor,
    )


def _members(**changes: bytes) -> dict[str, bytes]:
    """The members of a package that holds the data, every one but the manifest."""
    tier = pk.REPRODUCTION
    members = {
        pk.NOTICE: pk.NOTICES[pk.NOTICE_PERSONAL_DATA],
        pk.PACKAGE: pk.dumps(pk.PACKAGE, _index(pk.anchor_of([SOURCE])).to_dict(), tier),
        pk.PROTOCOL: pk.dumps(
            pk.PROTOCOL, {"revision": {"schema": 1, "body": {"note": "Fièvre" + TAB}}}, tier
        ),
        pk.RESULTS: pk.dumps(pk.RESULTS, {"results": []}, tier),
        pk.TRACE: pk.dumps(pk.TRACE, {"trace": {"steps": []}}, tier),
        pk.PROFILES: pk.dumps(pk.PROFILES, {"profiles": []}, tier),
        pk.README: "# tutorial_cohort, revision 3\n\nFièvre\n".encode(),
        "source.csv": SOURCE,
        "merge_01.parquet": CUT,
    }
    return {**members, **changes}


def _shareable() -> dict[str, bytes]:
    return {
        pk.PACKAGE: pk.dumps(pk.PACKAGE, {"tier": "shareable"}),
        pk.PROTOCOL: pk.dumps(pk.PROTOCOL, {"body": {"mask_families": {}}}),
        pk.RESULTS: pk.dumps(pk.RESULTS, {"results": []}),
        pk.README: b"# tutorial_cohort\n",
    }


def _write(
    members: dict[str, bytes] | None = None, tier: str = pk.REPRODUCTION
) -> tuple[bytes, str]:
    out = io.BytesIO()
    password = PASSWORD if pk.holds_data(tier) else None
    chosen = _members() if members is None else members
    package_hash = pz.write(out, chosen, tier, password=password)
    return out.getvalue(), package_hash


def _crafted(
    monkeypatch: pytest.MonkeyPatch, members: dict[str, bytes], tier: str = pk.REPRODUCTION
) -> bytes:
    """A package written without the writer's checks of its notice, its index and its zip, as another tool could
    write one: what the reader must refuse."""
    with monkeypatch.context() as patched:
        patched.setattr(pz, "_index_given", lambda members, tier: None)
        patched.setattr(pz, "_check_written", lambda *args: None)
        return _write(members, tier)[0]


def _shareable_zip() -> bytes:
    return _write(_shareable(), pk.SHAREABLE)[0]


def _opened(data: bytes) -> pz.Opened:
    opened = pz.open_package(io.BytesIO(data))
    assert isinstance(opened, pz.Opened), opened
    return opened


def _read(data: bytes, password: str | None = PASSWORD, into: Path | None = None) -> Any:
    return pz.read_package(io.BytesIO(data), _opened(data), password, into=into)


def _folder(tmp_path: Path, name: str = "data") -> Path:
    """The folder a caller makes for the data: its user's only (on a system with modes)."""
    into = tmp_path / name
    into.mkdir(mode=0o700)
    return into


def _listing_wrong(monkeypatch: pytest.MonkeyPatch, member: str, checksum: str = "0" * 64) -> None:
    """The writer's manifest lists a checksum that is not ``member``'s (a member altered after it was listed)."""
    real = pk.manifest_from_checksums

    def listing(sums: dict[str, str], tier: str) -> bytes:
        return real({**sums, member: checksum}, tier)

    monkeypatch.setattr(pk, "manifest_from_checksums", listing)


def _raw_member(data: bytes, member: str) -> bytes:
    """A member's bytes as the zip holds them (encrypted or not), read by the test with the standard library."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        info = archive.getinfo(member)
    local = info.header_offset
    name_length, extra_length = struct.unpack("<HH", data[local + 26 : local + 30])
    start = local + 30 + name_length + extra_length
    return data[start : start + info.compress_size]


# ----------------------------------------------------------------------------- the password


def test_the_password_is_made_and_read_in_one_form() -> None:
    shape = re.compile("[0-9A-HJKMNP-TV-Z]{4}(-[0-9A-HJKMNP-TV-Z]{4}){4}")
    drawn = {pz.new_password() for _ in range(200)}
    assert len(drawn) == 200 and all(shape.fullmatch(password) for password in drawn)
    assert all(pz.password_of(password) == password for password in drawn)
    assert len(pz.ALPHABET) == 32 and pz.SYMBOLS == 20  # 100 bits
    # any spelling a person types reads as the one password: lower case, spaces, no hyphens, O for 0, I or L for 1
    for typed in (
        "k7qd-9f2m-xw4r-t8nb-3hvj",
        "K7QD9F2MXW4RT8NB3HVJ",
        " K7QD 9F2M XW4R T8NB 3HVJ ",
        "K7QD--9F2M-XW4R-T8NB-3HVJ",
    ):
        assert pz.password_of(typed) == PASSWORD
    assert pz.password_of("0O1I-L000-0000-0000-0000") == "0011-1000-0000-0000-0000"
    for not_one in (
        "",
        "K7QD-9F2M-XW4R-T8NB-3HV",
        "K7QD-9F2M-XW4R-T8NB-3HVJK",
        "U7QD-9F2M-XW4R-T8NB-3HVJ",
    ):
        assert pz.password_of(not_one) is None


# ----------------------------------------------------------------------------- a package that holds the data


def test_a_package_with_the_data_is_written_and_read_back() -> None:
    data, package_hash = _write()
    opened = _opened(data)
    assert (opened.tier, opened.notice, opened.package_hash) == (
        pk.REPRODUCTION,
        pk.NOTICE_PERSONAL_DATA,
        package_hash,
    )
    names = [entry.name for entry in opened.entries]
    assert tuple(names) == pk.in_zip_order([*_members(), pk.MANIFEST], pk.REPRODUCTION)
    clear = [entry.name for entry in opened.entries if not entry.encrypted]
    assert clear == [pk.NOTICE, pk.MANIFEST]
    assert opened.listed == {name: pk.checksum(member) for name, member in _members().items()}
    assert opened.not_matching == [] and opened.values == {} and opened.index is None
    # the hash is the manifest's, read before any password
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert package_hash == pk.checksum(archive.read(pk.MANIFEST))
        with pytest.raises(RuntimeError):
            archive.read(pk.PACKAGE)  # the standard library reads no encrypted member
    read = _read(data)
    assert isinstance(read, pz.Opened) and read.not_matching == [] and read.exit_code == pz.EXIT_OK
    assert read.index == _index(pk.anchor_of([SOURCE]))
    assert read.values[pk.PROTOCOL]["revision"]["body"]["note"] == "Fièvre" + TAB  # a carried text
    assert read.values[pk.README] == "# tutorial_cohort, revision 3\n\nFièvre\n"
    assert set(read.values) == {
        pk.PACKAGE,
        pk.PROTOCOL,
        pk.RESULTS,
        pk.TRACE,
        pk.PROFILES,
        pk.README,
    }
    assert read.data == {}  # hashed and dropped: no folder given


def test_two_exports_are_other_bytes_and_one_package_hash() -> None:
    (first, hash_one), (second, hash_two) = _write(), _write()
    assert first != second and hash_one == hash_two  # a salt per member; the same plaintexts
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.comment == b""
        for info in archive.infolist():
            assert info.date_time == pz.FIXED_DATE and info.create_system == pz.UNIX
            # mode 0600: a zip program that restores a Unix mode writes it for its owner only
            assert info.external_attr == 0o100600 << 16
            if info.filename in (pk.NOTICE, pk.MANIFEST):
                clear = (info.flag_bits, info.compress_type, info.extra)
                assert clear == (0, zipfile.ZIP_STORED, b"")
            else:  # WinZip AES-256, AE-2 (no CRC of the plaintext), deflate inside
                assert (info.flag_bits, info.compress_type, info.CRC) == (1, 99, 0)
                assert info.extra == bytes.fromhex("0199070002004145030800")


def test_no_plaintext_of_an_encrypted_member_is_in_the_zip() -> None:
    data, _ = _write()
    for text in (
        b"tutorial_cohort",
        RECIPIENT.encode(),
        b"DSA-2026-014",
        b"cohortstudio.trace",
        SOURCE[:40],
        SOURCE[-40:],
        CUT[:40],
    ):
        assert text not in data, text


def test_a_member_decrypts_by_the_winzip_specification() -> None:
    """Read without pyzipper, as 7-Zip and WinZip read it: PBKDF2-HMAC-SHA1 (1,000 rounds) of the password and the
    member's salt gives the AES-256 key, the key of the authentication code and the password's two-byte check; AES
    in CTR mode, a little-endian counter from 1; HMAC-SHA1 of the ciphertext, its first 10 bytes; then deflate."""
    from Cryptodome.Cipher import AES
    from Cryptodome.Util import Counter

    data, _ = _write()
    for member in (pk.PACKAGE, "source.csv"):
        raw = _raw_member(data, member)
        salt, check, body, code = raw[:16], raw[16:18], raw[18:-10], raw[-10:]
        keys = hashlib.pbkdf2_hmac("sha1", PASSWORD.encode("ascii"), salt, 1000, 66)
        assert keys[64:] == check
        assert hmac.new(keys[32:64], body, hashlib.sha1).digest()[:10] == code
        counter = Counter.new(128, initial_value=1, little_endian=True)
        plain = AES.new(keys[:32], AES.MODE_CTR, counter=counter).decrypt(body)
        assert zlib.decompress(plain, -15) == _members()[member]


def test_a_wrong_password_reads_nothing(tmp_path: Path) -> None:
    data, _ = _write()
    into = _folder(tmp_path)
    for password, refusal in (
        (WRONG, pz.WRONG_PASSWORD),
        (None, pz.NOT_A_PASSWORD),
        ("not a password", pz.NOT_A_PASSWORD),  # nothing tried: the caller gives the shape
        ("", pz.NOT_A_PASSWORD),
    ):
        refused = _read(data, password, into)
        assert isinstance(refused, pz.Refused) and refused.id == refusal
        assert refused.exit_code == pz.EXIT_NOT_READ
    assert list(into.iterdir()) == []
    # any spelling of the right one opens it
    assert _read(data, PASSWORD.lower().replace("-", " ")).not_matching == []


def test_a_wrong_password_can_be_typed_again() -> None:
    data, _ = _write()
    opened = _opened(data)
    refused = pz.read_package(io.BytesIO(data), opened, WRONG)
    assert isinstance(refused, pz.Refused) and refused.id == pz.WRONG_PASSWORD
    read = pz.read_package(io.BytesIO(data), opened, PASSWORD)
    assert isinstance(read, pz.Opened) and read is not opened and read.not_matching == []
    assert (opened.values, opened.index, opened.not_matching) == ({}, None, [])


class _Failing(io.RawIOBase):
    """A member's stream that fails as altered bytes make it fail."""

    def __init__(self, error: type[Exception]) -> None:
        super().__init__()
        self.error = error

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        raise self.error("scripted")


def _scripted(monkeypatch: pytest.MonkeyPatch, failures: dict[str, type[Exception]]) -> None:
    """pyzipper's reader, made to fail on the members named: a RuntimeError when it opens one (the password's
    two-byte check), any other error while one is read (a deflate error, the authentication code)."""
    import pyzipper

    class Scripted(pyzipper.AESZipFile):
        def open(self, name: Any, *args: Any, **kwargs: Any) -> Any:
            if failures.get(name) is RuntimeError:
                raise RuntimeError("scripted")
            if name in failures:
                return _Failing(failures[name])
            return super().open(name, *args, **kwargs)

    aes = types.SimpleNamespace(AESZipFile=Scripted, BadZipFile=pyzipper.BadZipFile)
    monkeypatch.setattr(pz, "_aes", lambda: aes)


def test_a_wrong_password_is_told_from_an_altered_member(monkeypatch: pytest.MonkeyPatch) -> None:
    data, _ = _write()
    # one wrong password in 65,536 passes a member's two-byte check: that member then fails as altered bytes do,
    # and the next members' checks fail
    failures = {pk.PACKAGE: zlib.error, pk.PROTOCOL: RuntimeError, pk.RESULTS: RuntimeError}
    _scripted(monkeypatch, failures)
    refused = _read(data)
    assert isinstance(refused, pz.Refused) and refused.id == pz.WRONG_PASSWORD
    # the right password, and the first member's check altered: the member is named, the others are read
    _scripted(monkeypatch, {pk.PACKAGE: RuntimeError})
    read = _read(data)
    assert isinstance(read, pz.Opened) and read.not_matching == [pk.PACKAGE]
    assert read.index is None and pk.RESULTS in read.values
    # a check that fails once a member decrypted is the member's bytes, and so is any error while one is read
    _scripted(monkeypatch, {pk.RESULTS: RuntimeError, pk.TRACE: zlib.error})
    read = _read(data)
    assert isinstance(read, pz.Opened) and read.not_matching == [pk.RESULTS, pk.TRACE]


def test_the_data_go_only_into_the_folder_made_for_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write()
    into = _folder(tmp_path)
    read = _read(data, PASSWORD, into)
    assert read.data == {
        "source.csv": into / "source.csv",
        "merge_01.parquet": into / "merge_01.parquet",
    }
    assert (into / "source.csv").read_bytes() == SOURCE
    assert (into / "merge_01.parquet").read_bytes() == CUT
    assert sorted(path.name for path in into.iterdir()) == ["merge_01.parquet", "source.csv"]
    if os.name == "posix":
        assert (into / "source.csv").stat().st_mode & 0o777 == 0o600
    # a folder that is not empty, or that is not there, is not the one made for them: nothing is written
    again = _read(data, PASSWORD, into)
    assert isinstance(again, pz.Refused) and again.id == pz.FOLDER
    assert (into / "source.csv").read_bytes() == SOURCE
    refused = _read(data, PASSWORD, tmp_path / "not_made")
    assert isinstance(refused, pz.Refused) and refused.id == pz.FOLDER
    if os.name == "posix":  # nor one that others can open
        shared = _folder(tmp_path, "shared")
        shared.chmod(0o755)
        refused = _read(data, PASSWORD, shared)
        assert isinstance(refused, pz.Refused) and refused.id == pz.FOLDER
    # nor one without the room the data declare, found before the first file is written
    monkeypatch.setattr(pz.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=100))
    empty = _folder(tmp_path, "empty")
    refused = _read(data, PASSWORD, empty)
    assert isinstance(refused, pz.Refused) and refused.id == pz.NO_SPACE
    assert list(empty.iterdir()) == []


def _stopping(real: Any, stop: BaseException) -> Any:
    """The member's stream, stopped by ``stop`` when the second data file is written."""
    written: list[int] = []

    def stopping(
        source: Any, declared: int, keep: bool, sink: Any = None, anchor: Any = None
    ) -> tuple[str, bytes]:
        if sink is not None:
            written.append(1)
            if len(written) == 2:
                raise stop
        return real(source, declared, keep, sink, anchor)

    return stopping


def test_nothing_is_left_when_the_reading_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interruption, or an error of the computer (a full disk), while the second data file is written: the
    first is deleted, and the error reaches the caller (never "not what the sender made")."""
    data, _ = _write()
    for stop in (KeyboardInterrupt(), OSError(28, "No space left on device")):
        monkeypatch.setattr(pz, "_stream", _stopping(pz._stream, stop))
        into = _folder(tmp_path, type(stop).__name__)
        with pytest.raises(type(stop)):
            _read(data, PASSWORD, into)
        assert list(into.iterdir()) == []


def test_a_member_that_is_not_what_the_sender_made_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read, but not what the sender made (exit 1): the member is named, the others are read, nothing crashes."""
    # a line of the manifest that is not its member's: an encrypted member, a data member, the clear notice
    for member in (pk.RESULTS, "merge_01.parquet", pk.NOTICE):
        _listing_wrong(monkeypatch, member)
        data = _crafted(monkeypatch, _members())  # the writer reads back no clear member as altered
        monkeypatch.undo()
        read = _read(data)
        assert isinstance(read, pz.Opened), member
        assert read.not_matching == [member] and read.exit_code == pz.EXIT_NOT_WHAT_WAS_SENT
        assert member not in read.values
        # a notice that is not the sender's says nothing of the package
        assert read.notice == (None if member == pk.NOTICE else pk.NOTICE_PERSONAL_DATA)
    # a byte altered inside an encrypted member: its stream fails (a deflate error or the authentication code)
    data, _ = _write()
    altered = bytearray(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        local = archive.getinfo("source.csv").header_offset
    name_length, extra_length = struct.unpack("<HH", altered[local + 26 : local + 30])
    # past the salt and the password's check
    start = local + 30 + name_length + extra_length + 16 + 2
    altered[start + 100] ^= 0x01
    read = _read(bytes(altered))
    assert isinstance(read, pz.Opened) and read.not_matching == ["source.csv"]
    assert read.index is not None and pk.RESULTS in read.values


def test_a_notice_swapped_for_the_other_gives_no_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """Notice B ("declared not people") put into a package of personal data, its manifest line left as notice A's:
    neither the clear part nor the reading says that the rows are not people."""
    _listing_wrong(monkeypatch, pk.NOTICE, pk.checksum(pk.NOTICES[pk.NOTICE_PERSONAL_DATA]))
    data = _crafted(monkeypatch, _members(**{pk.NOTICE: pk.NOTICES[pk.NOTICE_NOT_PEOPLE]}))
    monkeypatch.undo()
    opened = _opened(data)
    assert opened.notice is None and opened.not_matching == [pk.NOTICE]
    read = pz.read_package(io.BytesIO(data), opened, PASSWORD)
    assert isinstance(read, pz.Opened) and read.notice is None and read.not_matching == [pk.NOTICE]


def test_an_altered_data_member_is_not_left_in_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _listing_wrong(monkeypatch, "merge_01.parquet")
    data, _ = _write()
    monkeypatch.undo()
    into = _folder(tmp_path)
    read = _read(data, PASSWORD, into)
    assert read.not_matching == ["merge_01.parquet"] and list(read.data) == ["source.csv"]
    assert sorted(path.name for path in into.iterdir()) == ["source.csv"]


def test_no_data_reach_the_folder_without_their_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index that is not the sender's: the data members are hashed and dropped, never written, since nothing
    could use them."""
    _listing_wrong(monkeypatch, pk.PACKAGE)
    data, _ = _write()
    monkeypatch.undo()
    into = _folder(tmp_path)
    read = _read(data, PASSWORD, into)
    assert isinstance(read, pz.Opened) and read.not_matching == [pk.PACKAGE]
    assert read.index is None and read.data == {} and list(into.iterdir()) == []


# ----------------------------------------------------------------------------- what a package holds must agree


def test_the_writer_puts_in_clear_only_a_notice_of_the_formats(tmp_path: Path) -> None:
    """Before a byte is written: the notice is one of the fixed texts and says what the index's policy says, and the
    index reads and names the data members given. While writing: the anchor is the start member's, and each member
    is what the pre-flight found."""
    tier = pk.REPRODUCTION
    only_the_source = (pk.DataFile("source", None, "source.csv", False, 2000, 2),)
    other_files = _index(pk.anchor_of([SOURCE]), files=only_the_source).to_dict()
    for members in (
        _members(**{pk.NOTICE: _members()[pk.README]}),  # the README's words would be clear
        _members(
            **{pk.NOTICE: pk.NOTICES[pk.NOTICE_NOT_PEOPLE]}
        ),  # a notice its policy does not say
        _members(**{pk.PACKAGE: pk.dumps(pk.PACKAGE, other_files, tier)}),  # other data members
        _members(**{pk.PACKAGE: pk.dumps(pk.PACKAGE, {"tier": "reproduction"}, tier)}),
    ):
        out = io.BytesIO()
        with pytest.raises(ValueError):
            pz.write(out, members, tier, password=PASSWORD)
        assert out.getvalue() == b""
    notice = tmp_path / "NOTICE.txt"
    notice.write_bytes(pk.NOTICES[pk.NOTICE_PERSONAL_DATA])
    with pytest.raises(ValueError):  # the notice and the index are given as their bytes
        pz.write(io.BytesIO(), {**_members(), pk.NOTICE: notice}, tier, password=PASSWORD)
    other = pk.dumps(pk.PACKAGE, _index(pk.anchor_of([b"other"])).to_dict(), tier)
    with pytest.raises(ValueError):
        pz.write(io.BytesIO(), _members(**{pk.PACKAGE: other}), tier, password=PASSWORD)
    with pytest.raises(ValueError):
        changed = {"source.csv": "0" * 64}
        pz.write(io.BytesIO(), _members(), tier, password=PASSWORD, expected=changed)
    # what was checked is what is written
    sums = {name: pk.checksum(member) for name, member in _members().items()}
    out = io.BytesIO()
    written = pz.write(out, _members(), tier, password=PASSWORD, expected=sums)
    assert written == _write()[1]


def test_the_index_the_notice_and_the_anchor_agree_with_what_the_zip_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tier = pk.REPRODUCTION
    anchor = pk.anchor_of([SOURCE])
    # an index that names a merged file the zip does not hold
    lone = _members()
    del lone["merge_01.parquet"]
    refused = _read(_crafted(monkeypatch, lone))
    assert isinstance(refused, pz.Refused)
    assert (refused.id, refused.member) == (pz.NOT_A_PACKAGE, pk.PACKAGE)
    # an anchor that is not the start member's
    other = pk.dumps(pk.PACKAGE, _index(pk.anchor_of([b"other"])).to_dict(), tier)
    refused = _read(_crafted(monkeypatch, _members(**{pk.PACKAGE: other})))
    assert isinstance(refused, pz.Refused) and (refused.id, refused.member) == (
        pz.ANCHOR,
        "source.csv",
    )
    # the notice of a study whose rows are declared not people, with a policy that declares nothing
    notice_b = pk.NOTICES[pk.NOTICE_NOT_PEOPLE]
    refused = _read(_crafted(monkeypatch, _members(**{pk.NOTICE: notice_b})))
    assert isinstance(refused, pz.Refused) and refused.id == pz.NOTICE
    # with a policy that declares it, both agree
    animals = pk.Policy(
        rules=pk.NONE, rows_not_people=("animal_data", "study"), rows_word="animals"
    )
    index = pk.dumps(pk.PACKAGE, _index(anchor, policy=animals).to_dict(), tier)
    read = _read(_write(_members(**{pk.NOTICE: notice_b, pk.PACKAGE: index}))[0])
    assert isinstance(read, pz.Opened) and read.notice == pk.NOTICE_NOT_PEOPLE
    # an index this version cannot read, and one of a later version
    broken = pk.dumps(pk.PACKAGE, {"tier": "reproduction"}, tier)
    refused = _read(_crafted(monkeypatch, _members(**{pk.PACKAGE: broken})))
    assert isinstance(refused, pz.Refused)
    assert (refused.id, refused.member) == (pz.NOT_READ, pk.PACKAGE)
    later = _index(anchor).to_dict() | {"note": "a key of a later version"}
    later_index = pk.dumps(pk.PACKAGE, later, tier)
    refused = _read(_crafted(monkeypatch, _members(**{pk.PACKAGE: later_index})))
    assert isinstance(refused, pz.Refused) and refused.id == pz.LATER_VERSION
    # a carried member that is no JSON, though its bytes are the sender's: the reason quotes nothing of it
    refused = _read(_write(_members(**{pk.TRACE: ('{"' + CANARY + '": ').encode()}))[0])
    assert isinstance(refused, pz.Refused)
    assert (refused.id, refused.member) == (pz.NOT_READ, pk.TRACE)
    assert CANARY not in repr(refused)


def test_nothing_is_left_of_a_reading_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = pk.dumps(pk.PACKAGE, _index(pk.anchor_of([b"other"])).to_dict(), pk.REPRODUCTION)
    into = _folder(tmp_path)
    refused = _read(_crafted(monkeypatch, _members(**{pk.PACKAGE: other})), PASSWORD, into)
    assert isinstance(refused, pz.Refused) and refused.id == pz.ANCHOR
    assert list(into.iterdir()) == []


def test_a_file_the_system_will_not_delete_is_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decrypted file another program holds open cannot be deleted: its count reaches the caller, never its
    path, in place of the refusal or as a note on the error that stopped the reading."""
    other = pk.dumps(pk.PACKAGE, _index(pk.anchor_of([b"other"])).to_dict(), pk.REPRODUCTION)
    refused_package = _crafted(monkeypatch, _members(**{pk.PACKAGE: other}))
    data, _ = _write()
    real = Path.unlink

    def held(self: Path, missing_ok: bool = False) -> None:
        if self.name == "source.csv":
            raise PermissionError(13, "held open by another program", str(self))
        real(self, missing_ok=missing_ok)

    into = _folder(tmp_path)
    monkeypatch.setattr(Path, "unlink", held)
    with pytest.raises(OSError) as raised:
        _read(refused_package, PASSWORD, into)
    assert "1 decrypted file could not be deleted" in str(raised.value)
    assert str(into) not in str(raised.value)
    assert any(pz.ANCHOR in note for note in raised.value.__notes__)
    stopped_in = _folder(tmp_path, "stopped")
    monkeypatch.setattr(pz, "_stream", _stopping(pz._stream, KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt) as stopped:
        _read(data, PASSWORD, stopped_in)
    monkeypatch.undo()
    assert any("could not be deleted" in note for note in stopped.value.__notes__)
    assert [path.name for path in into.iterdir()] == ["source.csv"]
    assert [path.name for path in stopped_in.iterdir()] == ["source.csv"]


def test_a_package_of_a_later_version_reads_as_one(monkeypatch: pytest.MonkeyPatch) -> None:
    text = pk.NOTICES[pk.NOTICE_PERSONAL_DATA]
    later = text.replace(b"schema 1\n", b"schema 2\n", 1)
    data = _crafted(monkeypatch, _members(**{pk.NOTICE: later}))
    refused = pz.open_package(io.BytesIO(data))
    assert isinstance(refused, pz.Refused)
    assert (refused.id, refused.member) == (pz.LATER_VERSION, pk.NOTICE)
    # whatever members a later version adds: its notice is read first
    added = _zip(
        [
            (pk.NOTICE, later, "stored"),
            (pk.PACKAGE, b"{}", "aes"),
            ("graphs.json", b"{}", "aes"),
            (pk.MANIFEST, b"", "stored"),
        ]
    )
    refused = pz.open_package(io.BytesIO(added))
    assert isinstance(refused, pz.Refused) and refused.id == pz.LATER_VERSION
    # a notice that is none of ours, though the manifest lists its bytes: refused before any password
    unknown = text.replace(b"AES-256", b"AES-128")
    refused = pz.open_package(io.BytesIO(_crafted(monkeypatch, _members(**{pk.NOTICE: unknown}))))
    assert isinstance(refused, pz.Refused) and refused.id == pz.NOTICE


def test_every_reason_of_a_later_version_begins_as_the_reader_reads_it() -> None:
    reasons = (
        pk.later_schema(2, 1),
        pk.later_keys(1),
        pk.later_index_keys(2),
        pk.LATER_INDEX_VALUE,
        pk.LATER_POLICY_VALUE,
    )
    assert all(reason.startswith(pz.LATER) for reason in reasons)


def test_the_file_read_is_the_one_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Between the two calls the file may change (three tries of a password); and pyzipper reads the directory with
    its own copy of zipfile: either way, what is decrypted is what open_package checked."""
    import pyzipper

    data, _ = _write()
    opened = _opened(data)
    other = _write(_members(**{pk.README: b"# another README\n"}))[0]
    refused = pz.read_package(io.BytesIO(other), opened, PASSWORD)
    assert isinstance(refused, pz.Refused) and refused.id == pz.LAYOUT

    class Shifted(pyzipper.AESZipFile):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.filelist[2].header_offset += 1

    aes = types.SimpleNamespace(AESZipFile=Shifted, BadZipFile=pyzipper.BadZipFile)
    monkeypatch.setattr(pz, "_aes", lambda: aes)
    refused = _read(data)
    assert isinstance(refused, pz.Refused) and refused.id == pz.LAYOUT


# ----------------------------------------------------------------------------- a package without the data


def test_a_package_without_the_data_is_stored_and_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    data, package_hash = _write(_shareable(), pk.SHAREABLE)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
    assert [info.filename for info in infos] == [
        pk.PACKAGE,
        pk.PROTOCOL,
        pk.RESULTS,
        pk.README,
        pk.MANIFEST,
    ]
    assert all(info.flag_bits == 0 and info.compress_type == zipfile.ZIP_STORED for info in infos)
    assert all(info.external_attr == 0o100644 << 16 for info in infos)
    # one writing: the same members give the same bytes, not only the same hash
    assert _write(_shareable(), pk.SHAREABLE) == (data, package_hash)
    opened = _opened(data)
    assert (
        opened.tier == pk.SHAREABLE
        and opened.package_hash == package_hash
        and opened.notice is None
    )
    read = pz.read_package(io.BytesIO(data), opened)
    assert isinstance(read, pz.Opened) and read.not_matching == []
    assert read.values[pk.README] == "# tutorial_cohort\n" and read.index is None
    # a clear member that is not what the sender made is named, and not read
    _listing_wrong(monkeypatch, pk.README)
    data = _crafted(monkeypatch, _shareable(), pk.SHAREABLE)
    monkeypatch.undo()
    read = pz.read_package(io.BytesIO(data), _opened(data))
    assert read.not_matching == [pk.README] and pk.README not in read.values
    assert pk.RESULTS in read.values


def test_a_package_without_the_data_is_one_byte_string_everywhere() -> None:
    """The zip of fixed members has one SHA-256 on every system and Python (the CI runs on Linux, the owner on
    Windows): the standard library's writing of a stored zip with fixed fields."""
    digest = hashlib.sha256(_shareable_zip()).hexdigest()
    assert digest == SHAREABLE_ZIP, digest


def test_a_package_made_with_the_rules_off_reads_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    members = {
        **_shareable(),
        pk.PACKAGE: pk.dumps(pk.PACKAGE, {"tier": "rules_off"}, pk.RULES_OFF),
    }
    out = io.BytesIO()
    pz.write(out, members, pk.RULES_OFF)
    opened = _opened(out.getvalue())
    assert opened.tier == pk.SHAREABLE  # the zip alone holds the same members
    read = pz.read_package(io.BytesIO(out.getvalue()), opened)
    assert isinstance(read, pz.Opened) and read.tier == pk.RULES_OFF and opened.tier == pk.SHAREABLE
    # a package without the data whose index says it holds them is no package
    claims = {**_shareable(), pk.PACKAGE: pk.dumps(pk.PACKAGE, {"tier": "reproduction"})}
    data = _crafted(monkeypatch, claims, pk.SHAREABLE)  # the writer refuses to write it
    refused = pz.read_package(io.BytesIO(data), _opened(data))
    assert isinstance(refused, pz.Refused)
    assert (refused.id, refused.member) == (pz.NOT_A_PACKAGE, pk.PACKAGE)


def test_a_package_without_the_data_of_a_later_version_reads_as_one() -> None:
    later = json.dumps({"format": pk.FORMATS[pk.PACKAGE], "schema": 2}).encode()
    members = {**_shareable(), pk.PACKAGE: later, "graphs.json": b"{}"}
    refused = pz.open_package(io.BytesIO(_stdlib_zip(members)))
    assert isinstance(refused, pz.Refused)
    assert (refused.id, refused.member) == (pz.LATER_VERSION, pk.PACKAGE)


# ----------------------------------------------------------------------------- the writer


def test_the_writer_refuses_before_a_byte_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    too_long = b"x" * (pk.cap_of(pk.README, pk.REPRODUCTION) + 1)
    without_trace = {name: member for name, member in _members().items() if name != pk.TRACE}
    for members, tier, password in (
        (_members(), pk.REPRODUCTION, None),  # a package with the data is encrypted
        (_shareable(), pk.SHAREABLE, PASSWORD),  # and only it
        # with a password new_password makes, as it makes it
        (_members(), pk.REPRODUCTION, PASSWORD.lower()),
        (_members(), pk.REPRODUCTION, "password"),
        # an index its reader would refuse, or that says another tier than the one written
        ({**_shareable(), pk.PACKAGE: b"{not json"}, pk.SHAREABLE, None),
        (_shareable(), pk.RULES_OFF, None),
        (_members(**{"notes.txt": b"x"}), pk.REPRODUCTION, PASSWORD),  # no member of the format
        (without_trace, pk.REPRODUCTION, PASSWORD),
        (_members(**{pk.README: too_long}), pk.REPRODUCTION, PASSWORD),
        (_members(**{pk.MANIFEST: b"x"}), pk.REPRODUCTION, PASSWORD),  # the writer's own
        ({**_shareable(), pk.README: "a text"}, pk.SHAREABLE, None),  # bytes or a file
    ):
        out = io.BytesIO()
        with pytest.raises(ValueError):
            pz.write(out, members, tier, password=password)  # type: ignore[arg-type]
        assert out.getvalue() == b""
    out = io.BytesIO(b"something")
    with pytest.raises(ValueError):
        pz.write(out, _shareable(), pk.SHAREABLE)
    assert out.getvalue() == b"something"
    # a file the writer could not read back
    path = tmp_path / "tutorial_cohort_r3.cstudio.zip"
    with path.open("wb") as stream, pytest.raises(ValueError):
        pz.write(stream, _shareable(), pk.SHAREABLE)
    assert path.read_bytes() == b""
    # a file that grows past its cap while it is written
    monkeypatch.setattr(pz, "_size", lambda source: 1)
    with pytest.raises(ValueError):
        pz.write(
            io.BytesIO(), _members(**{pk.README: too_long}), pk.REPRODUCTION, password=PASSWORD
        )


def test_a_writer_defect_is_a_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The finished zip goes through the reader's clear part: a zip it would refuse is never returned."""
    monkeypatch.setattr(pz, "_opened", lambda file, size: pz.Refused(pz.LAYOUT))
    with pytest.raises(RuntimeError):
        _write(_shareable(), pk.SHAREABLE)


def test_a_member_given_as_a_file_is_streamed(tmp_path: Path) -> None:
    source = tmp_path / "loaded.csv"
    source.write_bytes(SOURCE)
    out = io.BytesIO()
    members = {**_members(), "source.csv": source}
    package_hash = pz.write(out, members, pk.REPRODUCTION, password=PASSWORD)
    assert package_hash == _write()[1]  # the same plaintexts: the same hash
    read = pz.read_package(io.BytesIO(out.getvalue()), _opened(out.getvalue()), PASSWORD)
    assert read.not_matching == []


# ----------------------------------------------------------------------------- crafted files


def _zip(entries: list[tuple[str, bytes, str]], mode: int = WITH_DATA_MODE) -> bytes:
    """A zip written by hand with pyzipper: each entry stored, deflated or AES-256, with the format's fixed fields."""
    import pyzipper

    out = io.BytesIO()
    with pyzipper.AESZipFile(out, "w") as archive:
        for name, member, kind in entries:
            if kind == "aes":
                archive.setpassword(PASSWORD.encode())
                archive.setencryption(pyzipper.WZ_AES, nbits=256)
            else:
                archive.setpassword(None)
                archive.setencryption(None)
            info = archive.zipinfo_cls(name, date_time=pz.FIXED_DATE)
            info.create_system, info.external_attr = pz.UNIX, mode
            deflated = kind in ("aes", "deflated")
            info.compress_type = zipfile.ZIP_DEFLATED if deflated else zipfile.ZIP_STORED
            archive.writestr(info, member)
    return out.getvalue()


def _stdlib_zip(
    members: dict[str, bytes],
    *,
    gap_after: str = "",
    comment: bytes = b"",
    note_on: str = "",
    date: tuple[int, int, int, int, int, int] = pz.FIXED_DATE,
) -> bytes:
    """A package without the data written by the standard library entry by entry, as the writer writes one, with
    one thing changed."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, member in members.items():
            info = zipfile.ZipInfo(name, date_time=date)
            info.create_system, info.external_attr = pz.UNIX, pz.MODES[pk.SHAREABLE]
            if name == note_on:
                info.comment = b"a note on the entry"
            archive.writestr(info, member)
            if name == gap_after:
                archive.fp.write(b"a few words between two members")  # type: ignore[union-attr]
                archive.start_dir = archive.fp.tell()  # type: ignore[union-attr]
        archive.comment = comment
    return out.getvalue()


def _complete(members: dict[str, bytes]) -> dict[str, bytes]:
    return {**members, pk.MANIFEST: pk.manifest_of(members)}


def _central(data: bytes) -> list[int]:
    """The offsets of the directory's entries, walked from its end record."""
    count, _, offset = struct.unpack("<HLL", data[-12:-2])
    entries, at = [], offset
    for _ in range(count):
        entries.append(at)
        name, extra, comment = struct.unpack("<HHH", data[at + 28 : at + 34])
        at += 46 + name + extra + comment
    return entries


def _local(data: bytes, member: str) -> int:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return archive.getinfo(member).header_offset


def _overlapping() -> bytes:
    data = bytearray(_write()[0])
    entries = _central(bytes(data))
    # the second entry points at the first's local header
    data[entries[1] + 42 : entries[1] + 46] = data[entries[0] + 42 : entries[0] + 46]
    return bytes(data)


def _renamed_in_the_directory() -> bytes:
    data = bytearray(_shareable_zip())
    # README.md, renamed in the directory only, at the same length
    at = _central(bytes(data))[3] + 46
    data[at : at + len(pk.README)] = b"README.mx"
    return bytes(data)


def _local_method() -> bytes:
    """A local header that says deflate where the directory says stored: a tool that reads the local headers in
    turn would inflate the member."""
    data = bytearray(_shareable_zip())
    local = _local(bytes(data), pk.README)
    data[local + 8 : local + 10] = struct.pack("<H", zipfile.ZIP_DEFLATED)
    return bytes(data)


def _declared(member: str, size: int) -> bytes:
    """A package whose headers declare another size for one member (its local header and the directory agree)."""
    data = bytearray(_write()[0])
    local = _local(bytes(data), member)
    data[local + 22 : local + 26] = struct.pack("<I", size)
    for entry in _central(bytes(data)):
        name_length = struct.unpack("<H", data[entry + 28 : entry + 30])[0]
        if data[entry + 46 : entry + 46 + name_length] == member.encode():
            data[entry + 24 : entry + 28] = struct.pack("<I", size)
    return bytes(data)


def _zip64(cover: bool) -> bytes:
    """A ZIP64 end record and its locator between the directory and the end record (the format never holds them),
    its extensible sector holding rows of the source in clear; the end record's directory made to cover them, or
    not. The standard library follows the locator; pyzipper's older copy of zipfile reads the record elsewhere."""
    data = _write()[0]
    body, end = data[:-22], bytearray(data[-22:])
    count, length, start = struct.unpack("<HLL", bytes(end[10:20]))
    hidden = SOURCE[:200]
    record = (
        struct.pack(
            "<4sQ2H2L4Q",
            bytes.fromhex("504b0606"),
            44 + len(hidden),
            45,
            45,
            0,
            0,
            count,
            count,
            length,
            start,
        )
        + hidden
    )
    locator = struct.pack("<4sLQL", bytes.fromhex("504b0607"), 0, len(body), 1)
    if cover:
        end[12:16] = struct.pack("<L", length + len(record) + len(locator))
    return body + record + locator + bytes(end)


def _end_record(entries: int | None = None, length: int | None = None) -> bytes:
    """A package whose end record declares other entries, or another size of its directory."""
    data = bytearray(_shareable_zip())
    if entries is not None:
        data[-14:-10] = struct.pack("<HH", entries, entries)
    if length is not None:
        data[-10:-6] = struct.pack("<L", length)
    return bytes(data)


def _clear(members: dict[str, bytes]) -> list[tuple[str, bytes, str]]:
    return [(name, member, "stored") for name, member in members.items()]


SHAREABLE_MODE = pz.MODES[pk.SHAREABLE]


@pytest.mark.parametrize(
    ("label", "build", "refusal"),
    [
        ("not a zip", lambda: b"PK" + b"x" * 100, pz.NOT_A_ZIP),
        ("empty", lambda: b"", pz.NOT_A_ZIP),
        (
            "a folder",
            lambda: _zip([("data/", b"", "stored"), (pk.MANIFEST, b"", "stored")]),
            pz.NOT_A_PACKAGE,
        ),
        (
            "a name off the list",
            lambda: _zip(_clear({**_shareable(), "notes.txt": b"x"}), SHAREABLE_MODE),
            pz.NOT_A_PACKAGE,
        ),
        (
            "a name in another case",
            lambda: _zip(_clear({"readme.md": b"x", pk.PACKAGE: b"x"}), SHAREABLE_MODE),
            pz.NOT_A_PACKAGE,
        ),
        ("a path", lambda: _zip(_clear({"../README.md": b"x"})), pz.NOT_A_PACKAGE),
        (
            "a name outside ASCII",
            lambda: _zip(_clear({"R" + chr(0xC9) + "ADME.md": b"x"})),
            pz.NOT_A_PACKAGE,
        ),
        (
            "a name too long",
            lambda: _zip(_clear({"x" * (pz.NAME_MOST + 1): b"x"})),
            pz.NOT_A_PACKAGE,
        ),
        (
            "a name twice",
            lambda: _zip(_clear({pk.README: b"x", "Readme.md": b"y"})),
            pz.NAME_TWICE,
        ),
        ("more entries than a package has", lambda: _end_record(entries=2000), pz.NOT_A_PACKAGE),
        ("a directory larger than its entries", lambda: _end_record(length=10**8), pz.LAYOUT),
        ("overlapping entries", _overlapping, pz.LAYOUT),
        ("a directory that renames a member", _renamed_in_the_directory, pz.LAYOUT),
        ("a local header that is not the directory's", _local_method, pz.LAYOUT),
        ("bytes before the members", lambda: b"MZ" + bytes(62) + _shareable_zip(), pz.LAYOUT),
        (
            "bytes between two members",
            lambda: _stdlib_zip(_complete(_shareable()), gap_after=pk.PROTOCOL),
            pz.LAYOUT,
        ),
        ("a ZIP64 record the directory covers", lambda: _zip64(cover=True), pz.LAYOUT),
        ("a ZIP64 record after the directory", lambda: _zip64(cover=False), pz.LAYOUT),
        (
            "a comment",
            lambda: _stdlib_zip(_complete(_shareable()), comment=b"words"),
            pz.LAYOUT,
        ),
        (
            "a comment on an entry",
            lambda: _stdlib_zip(_complete(_shareable()), note_on=pk.RESULTS),
            pz.LAYOUT,
        ),
        (
            "another date",
            lambda: _stdlib_zip(_complete(_shareable()), date=(2026, 11, 18, 9, 30, 0)),
            pz.NOT_A_PACKAGE,
        ),
        (
            "a member that should be encrypted is clear",
            lambda: _zip(_clear(_members())),
            pz.ENCRYPTION,
        ),
        (
            "the notice encrypted",
            lambda: _zip([(name, member, "aes") for name, member in _members().items()]),
            pz.ENCRYPTION,
        ),
        (
            "a clear member deflated",
            lambda: _zip(
                [(name, member, "deflated") for name, member in _shareable().items()],
                SHAREABLE_MODE,
            ),
            pz.COMPRESSION,
        ),
        (
            "a member over its cap",
            lambda: _declared("merge_01.parquet", pk.DATA_MEMBER_CAP + 1),
            pz.MEMBER_TOO_LARGE,
        ),
        ("a ratio no data has", lambda: _declared("merge_01.parquet", 900 << 20), pz.RATIO),
    ],
)
def test_a_crafted_file_is_refused_before_any_password(
    label: str, build: Any, refusal: str
) -> None:
    refused = pz.open_package(io.BytesIO(build()))
    assert isinstance(refused, pz.Refused), label
    assert refused.id == refusal and refused.exit_code == pz.EXIT_NOT_READ, (label, refused)


def test_bytes_after_the_end_are_refused() -> None:
    """The standard library reads such a file without a word, and says it has no comment."""
    tailed = _shareable_zip() + b"words after the end"
    with zipfile.ZipFile(io.BytesIO(tailed)) as archive:
        assert archive.comment == b""
    refused = pz.open_package(io.BytesIO(tailed))
    assert isinstance(refused, pz.Refused) and refused.id == pz.LAYOUT


def test_no_zip_library_reads_a_received_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clear part walks the directory itself, bounded before a byte of it is read."""
    data, package_hash = _write()

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a zip library read the received file")

    monkeypatch.setattr(zipfile, "ZipFile", refused)
    opened = pz.open_package(io.BytesIO(data))
    assert isinstance(opened, pz.Opened) and opened.package_hash == package_hash


def test_the_standard_library_writes_what_the_writer_writes() -> None:
    """The writer's package without the data is the standard library's zip of its members, entry by entry: the
    crafted files above change one thing of a file the reader reads."""
    assert _stdlib_zip(_complete(_shareable())) == _shareable_zip()
    assert isinstance(pz.open_package(io.BytesIO(_shareable_zip())), pz.Opened)


def test_a_file_too_large_and_a_manifest_that_lists_another_zip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, _ = _write()
    refused = pz.open_package(io.BytesIO(data), max_size=len(data) - 1)
    assert isinstance(refused, pz.Refused) and refused.id == pz.TOO_LARGE
    # the members declare more together than a package may hold
    monkeypatch.setitem(pk._PACKAGE_CAPS, pk.REPRODUCTION, 1000)
    refused = pz.open_package(io.BytesIO(data))
    assert isinstance(refused, pz.Refused) and refused.id == pz.TOO_LARGE
    monkeypatch.undo()
    # a manifest that lists a member the zip does not hold
    _listing_wrong(monkeypatch, pk.LOCK)
    data = _crafted(monkeypatch, _members())
    monkeypatch.undo()
    refused = pz.open_package(io.BytesIO(data))
    assert isinstance(refused, pz.Refused) and refused.id == pz.MANIFEST


def test_the_container_has_room_for_every_package_of_the_format() -> None:
    """The container is frozen across schemas: its bounds hold every package the format can make."""
    assert pz.MOST_ENTRIES >= len(pk.members_of(pk.REPRODUCTION)) + 1 + 2 * pk.DATA_MEMBERS_MOST
    names = (*pk.members_of(pk.REPRODUCTION), *pk.members_of(pk.SHAREABLE), *pz._DATA_NAMES)
    assert pz.NAME_MOST >= max(len(name) for name in names)
    assert pz.MOST_FILE_BYTES > pk.package_cap(pk.REPRODUCTION) > pk.package_cap(pk.SHAREABLE)
    # a data file is named by the format's own string: exactly the names data_role reads as a data member's
    extensions = (*pk.SOURCE_EXTENSIONS, *pk.TABLE_ONLY_EXTENSIONS, "xyz", "CSV")
    candidates = {f"source.{extension}" for extension in extensions} | {
        "version.parquet",
        "Version.parquet",
        "merge_1.parquet",
    }
    candidates |= {
        f"{kind}_{number:02d}.parquet"
        for kind in ("merge", "snapshot", "other")
        for number in range(0, 101)
    }
    assert {name for name in candidates if pk.data_role(name)} == set(pz._DATA_NAMES)
    assert all(name == value for name, value in pz._DATA_NAMES.items())


# ----------------------------------------------------------------------------- what is never shown, never loaded


def test_nothing_read_is_in_a_log_a_reason_or_a_repr(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """The password, the recipient's name, a carried text and the folder of the data appear nowhere a log or a
    message could take them from."""
    caplog.set_level(logging.DEBUG)
    tier = pk.REPRODUCTION
    planted = _members(
        **{
            pk.PACKAGE: pk.dumps(
                pk.PACKAGE, _index(pk.anchor_of([SOURCE]), recipient=CANARY).to_dict(), tier
            ),
            pk.PROTOCOL: pk.dumps(pk.PROTOCOL, {"revision": {"body": {"note": CANARY}}}, tier),
        }
    )
    data = _write(planted)[0]
    into = _folder(tmp_path)
    read = _read(data, PASSWORD, into)
    assert read.index is not None and read.not_matching == []
    assert read.index.declaration.recipient == CANARY
    refused = _read(data, WRONG)
    for text in (caplog.text, repr(read), repr(refused), repr(_opened(data))):
        for secret in (PASSWORD, PASSWORD.replace("-", ""), WRONG, CANARY, str(into)):
            assert secret not in text


def test_a_package_is_opened_without_pyzipper_until_its_password(tmp_path: Path) -> None:
    """Start-up, a package read without its password, and a package without the data written and read never load
    the encryption."""
    path = tmp_path / "tutorial_cohort_r3_with_data.cstudio.zip"
    path.write_bytes(_write()[0])
    shareable = "{pk.PACKAGE: pk.dumps(pk.PACKAGE, {}), pk.PROTOCOL: pk.dumps(pk.PROTOCOL, {}), "
    shareable += "pk.RESULTS: pk.dumps(pk.RESULTS, {}), pk.README: b'# r'}"
    code = (
        "import io, sys; from pathlib import Path; from core import package as pk; "
        "from manage import package_zip as pz; "
        "assert 'pyzipper' not in sys.modules; "
        f"opened = pz.open_package(Path({str(path)!r}).open('rb')); "
        f"out = io.BytesIO(); pz.write(out, {shareable}, pk.SHAREABLE); "
        "read = pz.read_package(out, pz.open_package(out)); "
        "print(type(opened).__name__, type(read).__name__, 'pyzipper' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-800:]
    assert done.stdout.strip() == "Opened Opened False"
