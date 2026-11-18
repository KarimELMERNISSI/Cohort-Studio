"""
The zip of a study package (M8): one writer, and the safe reader of a received file. The formats are
``core.package``'s: this module puts them in a zip, and takes them out of one.

**The writer** (:func:`write`) puts a package's members in the format's order, one after the other, with fixed
fields only (the date 1980-01-01, the system, the version, the mode of its tier; no comment, no extra field but
AES's), and makes the manifest itself from the members' checksums, taken as it streams them: the package's hash is
the SHA-256 of that manifest. A package without the data is stored, clear. One that holds the data is encrypted
member by member (WinZip AES-256, AE-2, deflate inside), except its notice and its manifest, which stay clear and
stored, so that its hash is read before any password; its notice must be one of the fixed texts and say what its
index's policy says, and its index's anchor must be that of the member the trace starts from. The finished zip is
read back by the reader's own clear part before :func:`write` returns. The password is made here
(:func:`new_password`): 20 symbols of Crockford's base32, 100 bits, in 5 groups of 4 joined by hyphens; the string
shown, hyphens included, is the password, and :func:`password_of` reads any spelling of it a person types.

**The reader** (:func:`open_package`, then :func:`read_package`) never gives a received file to a zip library before
it has read the file's directory itself, and stops at the first refusal (:class:`Refused`, a closed id whose words
are the caller's):

1. its size, against the caller's limit (``MOST_FILE_BYTES`` by default);
2. its end record, the file's last 22 bytes (no comment), and its directory, bounded before a byte of it is read (at
   most ``MOST_ENTRIES`` entries of names in ASCII of at most ``NAME_MOST`` characters), each record walked in turn
   until they fill the directory exactly: no folder, no name twice (case folded);
3. its layout, which must be the writer's: the members one after the other from the file's first byte, in the
   directory's order, each local header saying what the directory says of its entry, the directory right after the
   last member. No byte lies before, between or after the members (no ZIP64 record, no comment), and a tool that
   reads the local headers in turn finds what the directory lists;
4. whether a later version made it: the notice of a package that holds the data, the envelope of the clear
   ``package.json`` of one that does not, read first, whatever else the file holds;
5. the names: exactly the format's, in their case; the encryption, the compression, the extra field, the version
   and the fixed fields each member's tier says; the declared sizes within their caps, and their ratio;
6. the clear members: the manifest, strictly, listing exactly the zip's members, which are in the format's order
   (``layout`` otherwise); **the package's hash is known here, before any password**; each clear member's checksum
   against its line.

Steps 1 to 4 are the container, frozen across the format's schemas (``core.package`` says so): a later version may
change the members and what they hold, never these, nor the member that tells a later version beyond this version's
cap of it (and, for ``package.json``, beyond JSON this version's strict reader reads).

Then every member is read (:func:`read_package`, which leaves what :func:`open_package` gave as it was, so that a
password can be typed again). The file must still be the one opened (its size and its directory), and pyzipper's
reading of the directory the one checked. Each member is streamed in the zip's order through a counter and a
SHA-256 compared with its line of the manifest, ``package.json`` first. The password is wrong when two members fail
its check before any member decrypted: a wrong password passes a member's two-byte check once in 65,536 tries (that
member then fails as altered bytes do), and one member's altered check is not taken for a wrong password. A member
whose bytes are not those the manifest lists does not stop the reading: it is named (``Opened.not_matching``, exit
1, "read, but not what the sender made"). Only what altered bytes raise counts so: an error of the computer (a
full disk, a file that cannot be read) is raised to the caller. The JSON members are read by
``core.package.read_member`` and the index by its closed reader; a data member is hashed and dropped, or, once the
index is read, written into a folder the caller made (``into``: ``verify`` only), each file created there
exclusively under the format's own name. Each is deleted again when its bytes are not the sender's, when the
reading is refused or when anything stops it; a file the system will not delete is said (an ``OSError``, or a note
on the error that stopped the reading), never left in silence.

pyzipper is imported inside the functions that encrypt or decrypt: start-up, and a package read without its
password, never load it.
"""

from __future__ import annotations

import hashlib
import io
import os
import secrets
import shutil
import struct
import zipfile
import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO

from core import package as pk

FIXED_DATE = (1980, 1, 1, 0, 0, 0)
UNIX = 3  # the system each member says it was made on
VERSION = 20  # the version of the zip format each header names
# a member's mode: read and written by its owner and read by others; in a package that holds the data, its owner's
# only. A zip program that restores a Unix mode (Info-ZIP's unzip, the usual tools of Linux and macOS) writes the
# files it extracts so; on Windows, or through Python's zipfile, they take the rights of the folder they go into
# (the package's README, written with the package, says that files taken out of it are personal data on that disk)
MODES = {
    pk.SHAREABLE: 0o100644 << 16,
    pk.RULES_OFF: 0o100644 << 16,
    pk.REPRODUCTION: 0o100600 << 16,
}
CHUNK = 1 << 20
# the container's bounds (frozen across schemas): the format holds at most 208 members, of names of at most 19
MOST_ENTRIES, NAME_MOST = 1024, 64
# a member above 1 MB declares at most deflate's own bound times its compressed size: no deflated stream holds more
RATIO_FROM, RATIO_MOST = 1 << 20, 1032
AES = 99  # the compression method a WinZip AES member declares
# the extra field of a WinZip AES-256 member: id 0x9901, 7 bytes, AE-2, vendor "AE", strength 3, deflate inside
_AES_EXTRA = bytes.fromhex("0199070002004145030800")
_AES_COST = 16 + 2 + 10  # a member's salt, the password's check and the authentication code
# a local header: signature, version, flags, method, time, date, CRC, the two sizes, the name's and the extra's
# lengths; a record of the directory: signature, the versions made by and needed, flags, method, time, date, CRC,
# the two sizes, the name's, the extra's and the comment's lengths, disk, the internal and external attributes, the
# local header's offset; the end of the directory: signature, the two disks, its entries (on this disk, in all),
# its size and offset, the comment's length
_LOCAL = struct.Struct("<4s5H3L2H")
_CENTRAL = struct.Struct("<4s6H3L5H2L")
_END = struct.Struct("<4s4H2LH")
_LOCAL_SIGNATURE, _CENTRAL_SIGNATURE = b"PK\x03\x04", b"PK\x01\x02"
_END_SIGNATURE = b"PK\x05\x06"
_RECORD_MOST = _CENTRAL.size + NAME_MOST + len(_AES_EXTRA)
# the most bytes a package of this version can be: its members with the data, deflate's growth of bytes that do not
# compress (well under 1 in 1,024), and each entry's headers
_WHOLE = pk.package_cap(pk.REPRODUCTION)
MOST_FILE_BYTES = (
    _WHOLE
    + _WHOLE // 1024
    + MOST_ENTRIES * (_LOCAL.size + _RECORD_MOST + NAME_MOST + len(_AES_EXTRA) + _AES_COST)
    + _END.size
)
# core.package's reasons for a package of a later version all begin so (a test reads them)
LATER = "written by a later version"
# every name a data member may have: a data file's path is made of the format's own string, never of the zip's
_DATA_NAMES = {
    name: name
    for name in (
        *(pk.source_member(extension) for extension in pk.SOURCE_EXTENSIONS),
        pk.version_member(),
        *(pk.merge_member(number) for number in range(1, pk.DATA_MEMBERS_MOST + 1)),
        *(pk.snapshot_member(number) for number in range(1, pk.DATA_MEMBERS_MOST + 1)),
    )
}

# ----------------------------------------------------------------------------- the password

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford's base32: no I, L, O, U
SYMBOLS, GROUP = 20, 4  # 100 bits, shown in 5 groups of 4
_READ_AS = str.maketrans("OIL", "011")  # how a person may mistype it


def _grouped(bare: str) -> str:
    return "-".join(bare[start : start + GROUP] for start in range(0, len(bare), GROUP))


def new_password() -> str:
    """A package's password: 20 symbols drawn with ``secrets`` from Crockford's base32 (100 bits), in 5 groups of 4
    joined by hyphens. The string, hyphens included, is the password: what is shown opens the package."""
    return _grouped("".join(secrets.choice(ALPHABET) for _ in range(SYMBOLS)))


def password_of(typed: str) -> str | None:
    """
    The password a person typed or pasted, in the one form it was made in: spaces and hyphens dropped, upper case,
    O read as 0, I and L as 1, then the hyphens put back. None when what is left is not 20 symbols of the alphabet:
    the caller says what a password looks like (the format is public, so that hints nothing), and nothing is tried.
    """
    bare = "".join(str(typed).split()).replace("-", "").upper().translate(_READ_AS)
    if len(bare) != SYMBOLS or not set(bare) <= set(ALPHABET):
        return None
    return _grouped(bare)


# ----------------------------------------------------------------------------- the writer

Member = bytes | Path


def _size(source: Member) -> int:
    return len(source) if isinstance(source, bytes) else source.stat().st_size


def _chunks(source: Member) -> Iterator[bytes]:
    if isinstance(source, bytes):
        for start in range(0, len(source), CHUNK):
            yield source[start : start + CHUNK]
        return
    with source.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            yield chunk


def _info(
    cls: Callable[..., zipfile.ZipInfo], name: str, compress: int, size: int, mode: int
) -> zipfile.ZipInfo:
    info = cls(name, date_time=FIXED_DATE)
    info.create_system = UNIX
    info.external_attr = mode
    info.compress_type = compress
    info.file_size = size  # known in advance: no ZIP64 field for a member under its cap
    return info


def _manifest_size(names: tuple[str, ...]) -> int:
    """The manifest's bytes: a line of 64 hexadecimal characters, two spaces, the name and a line feed per member."""
    return sum(64 + 2 + len(name) + 1 for name in names if name != pk.MANIFEST)


def _tier_said(value: Any) -> str | None:
    """The tier a package without the data says in its index: its ``tier`` when it is one of the two, ``shareable``
    when it says none (the index's closed reader comes with the package without the data); None for another."""
    tier = value.get("tier", pk.SHAREABLE) if isinstance(value, dict) else pk.SHAREABLE
    return tier if tier in (pk.SHAREABLE, pk.RULES_OFF) else None


def _index_given(members: Mapping[str, Member], tier: str) -> pk.DataPackageIndex | None:
    """The index read from the members given, as its reader will read it: of a package that holds the data, with its
    notice; None for a package without the data, whose index must say the tier it is written in. ValueError for an
    index its reader refuses, or that says another tier; with the data, for a notice that is not one of the fixed
    texts (no other bytes go in clear), data members that are not those the index names, or a notice that does not
    say what its policy says."""
    given = members.get(pk.PACKAGE)
    if not isinstance(given, bytes):
        raise ValueError("a package's index is given as its bytes")
    value, _ = pk.read_member(pk.PACKAGE, given, tier)
    if value is None:
        raise ValueError("a package's index is one its reader reads")
    if not pk.holds_data(tier):
        if _tier_said(value) != tier:
            raise ValueError("a package's index says the tier it is written in")
        return None
    notice = members.get(pk.NOTICE)
    if not isinstance(notice, bytes):
        raise ValueError("a package's notice is given as its bytes")
    kind, _ = pk.read_notice(notice)
    if kind is None:
        raise ValueError("a package's notice is one of the fixed texts")
    index, _ = pk.DataPackageIndex.from_dict(value)
    if index is None:
        raise ValueError("a package's index is one its reader reads")
    names = pk.in_zip_order([*members, pk.MANIFEST], tier)
    if index.data_members() != tuple(name for name in names if pk.data_role(name)):
        raise ValueError("a package holds the data members its index names")
    if pk.notice_problem(kind, index.policy):
        raise ValueError("a package's notice says what its index's policy says")
    return index


def write(
    out: BinaryIO,
    members: Mapping[str, Member],
    tier: str,
    *,
    password: str | None = None,
    expected: Mapping[str, str] | None = None,
) -> str:
    """
    Write a package of ``tier`` into ``out``, an empty binary file that can be read and sought (an in-memory buffer,
    or a file opened for writing and reading): ``members`` (every member but the manifest, which the writer makes),
    each given as its bytes or as a file whose bytes are streamed. A package that holds the data is encrypted with
    ``password`` (one :func:`new_password` made), and only it. ``expected`` gives the SHA-256 a member was found to
    have when it was checked (the pre-flight's): a member streamed with other bytes stops the writing. Returns the
    package's hash.

    ValueError, before a byte is written, for a name that is no member of the tier, a member given otherwise or
    over its cap, a whole over the package's, a password that is missing, not needed or not one this module makes,
    an ``out`` that is not empty or cannot be read and sought, an index its reader would refuse or that says another
    tier, or (with the data) a notice that is not the format's; and while writing, for a file that grew past a cap,
    a member that is not what ``expected`` says,
    or an anchor that is not that of the member the trace starts from. RuntimeError when the finished zip is not one
    its reader opens: a defect of this writer, never of the data. After any of these while writing, ``out`` holds a
    zip to discard.
    """
    holds = pk.holds_data(tier)
    if holds != (password is not None):
        raise ValueError("a package that holds the data, and only it, is encrypted")
    if password is not None and password_of(password) != password:
        raise ValueError("a package's password is one new_password makes")
    if pk.MANIFEST in members:
        raise ValueError("a package's manifest is the writer's")
    if not all(isinstance(source, (bytes, Path)) for source in members.values()):
        raise ValueError("a member is given as its bytes or as a file")
    if expected is not None and not set(expected) <= set(members):
        raise ValueError("a checksum is expected of a member the package holds")
    names = pk.in_zip_order([*members, pk.MANIFEST], tier)
    sizes = {name: _size(source) for name, source in members.items()}
    if any(size > pk.cap_of(name, tier) for name, size in sizes.items()):
        raise ValueError("a member is larger than a member of a package may be")
    whole = pk.package_cap(tier)
    if sum(sizes.values()) + _manifest_size(names) > whole:
        raise ValueError("the package is larger than a package may be")
    if not (out.readable() and out.seekable()):
        raise ValueError("a package is written into a file that can be read and sought")
    out.seek(0, os.SEEK_END)
    if out.tell() != 0:
        raise ValueError("a package is written into an empty file")
    index = _index_given(members, tier)
    clear = set(pk.clear_members(tier))
    sums: dict[str, str] = {}
    manifest, written = b"", 0
    aes = _aes() if holds else None  # the encryption, and only when a package holds the data
    if aes is not None:
        archive: Any = aes.AESZipFile(out, "w", compression=aes.ZIP_DEFLATED)
    else:
        archive = zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED)
    with archive:
        for name in names:
            if name == pk.MANIFEST:
                manifest = pk.manifest_from_checksums(sums, tier)
            source = manifest if name == pk.MANIFEST else members[name]
            encrypted = name not in clear
            if aes is not None and password is not None:
                if encrypted:
                    archive.setpassword(password.encode("ascii"))
                    archive.setencryption(aes.WZ_AES, nbits=256)
                else:  # both, or pyzipper writes the member encrypted all the same
                    archive.setpassword(None)
                    archive.setencryption(None)
            compress = zipfile.ZIP_DEFLATED if encrypted else zipfile.ZIP_STORED
            kind = archive.zipinfo_cls if aes is not None else zipfile.ZipInfo
            info = _info(kind, name, compress, _size(source), MODES[tier])
            digest, total, cap = hashlib.sha256(), 0, pk.cap_of(name, tier)
            starts = index is not None and pk.data_role(name) in pk.STARTS_FROM
            anchor = hashlib.sha256(pk.ANCHOR_PREFIX) if starts else None
            with archive.open(info, "w") as stream:
                for chunk in _chunks(source):
                    total += len(chunk)
                    written += len(chunk)
                    if total > cap or written > whole:
                        raise ValueError("a file grew larger than a package may hold")
                    digest.update(chunk)
                    if anchor is not None:
                        anchor.update(chunk)
                    stream.write(chunk)
            sums[name] = digest.hexdigest()
            if expected is not None and name in expected and sums[name] != expected[name]:
                raise ValueError("a member's bytes changed since they were checked")
            if anchor is not None and index is not None and anchor.hexdigest() != index.anchor:
                raise ValueError("the index's anchor is not that of the data the trace starts from")
    _check_written(out, names, manifest)
    return pk.package_hash(manifest)


def _aes() -> Any:
    """pyzipper, imported where a package that holds the data is written or read: nowhere else."""
    import pyzipper

    return pyzipper


def _check_written(out: BinaryIO, names: tuple[str, ...], manifest: bytes) -> None:
    """The finished zip, read back by the reader's own clear part (steps 1 to 6): opened, with its members in their
    order, each as its manifest lists it, and the package's hash the manifest's."""
    out.seek(0, os.SEEK_END)
    opened = _opened(out, out.tell())
    if (
        not isinstance(opened, Opened)
        or tuple(entry.name for entry in opened.entries) != names
        or opened.not_matching
        or opened.package_hash != pk.package_hash(manifest)
    ):
        raise RuntimeError("the package's zip is not one its reader opens")


# ----------------------------------------------------------------------------- the reader

# Why a received file is refused (closed ids; the words are the caller's). Each is "not read" (exit 2).
TOO_LARGE = "too_large"  # the file, or the sizes its members declare together
NOT_A_ZIP = "not_a_zip"
NOT_A_PACKAGE = "not_a_package"  # names, entries or fields no package of this version holds
LATER_VERSION = "later_version"
NAME_TWICE = "name_twice"
# not laid out as the writer lays a zip out: overlapping entries, bytes before, between or after them, a record or a
# local header that does not say what the other says, a file changed since it was opened
LAYOUT = "layout"
ENCRYPTION = "encryption"
COMPRESSION = "compression"
MEMBER_TOO_LARGE = "member_too_large"
RATIO = "ratio"
NOTICE = "notice"
MANIFEST = "manifest"
NOT_A_PASSWORD = "not_a_password"  # not 20 symbols of the alphabet: nothing was tried (the caller gives the shape)
WRONG_PASSWORD = "wrong_password"
NOT_READ = "not_read"  # a member whose bytes are the sender's does not read: its JSON, or the index
ANCHOR = "anchor"  # the index's anchor is not that of the data the trace starts from
NO_SPACE = "no_space"  # the folder the data are written into has not the room their sizes need
# the folder the data would be written into is not an existing, empty folder of this user's, closed to others
FOLDER = "folder"
REFUSALS = (
    TOO_LARGE,
    NOT_A_ZIP,
    NOT_A_PACKAGE,
    LATER_VERSION,
    NAME_TWICE,
    LAYOUT,
    ENCRYPTION,
    COMPRESSION,
    MEMBER_TOO_LARGE,
    RATIO,
    NOTICE,
    MANIFEST,
    NOT_A_PASSWORD,
    WRONG_PASSWORD,
    NOT_READ,
    ANCHOR,
    NO_SPACE,
    FOLDER,
)
EXIT_OK, EXIT_NOT_WHAT_WAS_SENT, EXIT_NOT_READ = 0, 1, 2


@dataclass(frozen=True)
class Refused:
    """A received file refused: why (a closed id), the member concerned (its name, the format's own, once the names
    passed the list; "" before), and ``core.package``'s reason when it gave one (words that quote nothing)."""

    id: str
    member: str = ""
    reason: str = ""
    exit_code: int = EXIT_NOT_READ


@dataclass(frozen=True)
class Entry:
    name: str
    size: int  # as declared, before encryption
    encrypted: bool


@dataclass(frozen=True)
class _Record:
    """A record of a received file's directory, as its bytes say."""

    name: str
    made_by: int
    needed: int
    flags: int
    method: int
    time: int
    date: int
    crc: int
    compressed: int
    size: int
    extra: bytes
    internal: int
    external: int
    offset: int

    @property
    def data_at(self) -> int:
        """Where its bytes start: after its local header, which holds its name and its extra field again."""
        return self.offset + _LOCAL.size + len(self.name) + len(self.extra)


@dataclass
class Opened:
    """What a received package holds, as far as it was read: its tier (``shareable`` for a package without the
    data until :func:`read_package` reads in its index whether it was made with the rules or without them), its
    file's size, its members in the zip's order, the checksums its manifest lists and the package's hash; after
    :func:`read_package`, the JSON members and the README by name, the index of a package that holds the data, and
    the data members written into ``into``. A member whose bytes are not those its manifest lists is in
    ``not_matching``. Its representation holds nothing a member holds, nor where the data were written."""

    tier: str
    size: int
    entries: tuple[Entry, ...]
    listed: dict[str, str]
    package_hash: str
    notice: str | None = None
    not_matching: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict, repr=False)
    index: pk.DataPackageIndex | None = field(default=None, repr=False)
    data: dict[str, Path] = field(default_factory=dict, repr=False)
    # the clear members' bytes and the directory as open_package read them, for read_package
    clear: dict[str, bytes] = field(default_factory=dict, repr=False)
    directory: tuple[_Record, ...] = field(default=(), repr=False)

    @property
    def exit_code(self) -> int:
        return EXIT_NOT_WHAT_WAS_SENT if self.not_matching else EXIT_OK


def _later(reasons: list[str]) -> bool:
    return bool(reasons) and reasons[0].startswith(LATER)


def _dos(date_time: tuple[int, ...]) -> tuple[int, int]:
    """A date as a header writes it: its time, then its date (MS-DOS)."""
    year, month, day, hour, minute, second = date_time
    return hour << 11 | minute << 5 | second // 2, (year - 1980) << 9 | month << 5 | day


def _directory(file: BinaryIO, size: int) -> tuple[tuple[_Record, ...], int] | str:
    """The directory of a file of ``size`` bytes, read by this module, and where it starts; or why the file is
    refused (an id). Its end record must end the file without a comment, on one disk, for at most ``MOST_ENTRIES``
    entries and a directory of the size so many records can have, ending where the end record starts; then each
    record is walked in turn (no comment, on the first disk, a name in ASCII of at most ``NAME_MOST`` characters, an
    extra field no longer than AES's), and the records fill the directory exactly. Nothing else of the file is read
    before these bounds hold."""
    if size < _END.size:
        return NOT_A_ZIP
    file.seek(size - _END.size)
    signature, disk, its_disk, on_disk, entries, length, start, comment = _END.unpack(
        file.read(_END.size)
    )
    if signature != _END_SIGNATURE:
        # a zip whose end record does not end the file (a comment, bytes after it), or no zip at all
        file.seek(max(0, size - _END.size - 0xFFFF))
        return LAYOUT if _END_SIGNATURE in file.read() else NOT_A_ZIP
    if (disk, its_disk, comment) != (0, 0, 0) or on_disk != entries:
        return LAYOUT
    if not 0 < entries <= MOST_ENTRIES:
        return NOT_A_PACKAGE
    if length > entries * _RECORD_MOST or start + length != size - _END.size:
        return LAYOUT
    file.seek(start)
    data = file.read(length)
    records: list[_Record] = []
    at = 0
    while at < length and len(records) < entries:
        if at + _CENTRAL.size > length:
            return LAYOUT
        (sign, made_by, needed, flags, method, time, date, crc, compressed, plain, *lengths) = (
            _CENTRAL.unpack_from(data, at)
        )
        name_length, extra_length, comment_length, first_disk, internal, external, offset = lengths
        named_at = at + _CENTRAL.size
        at = named_at + name_length + extra_length
        if sign != _CENTRAL_SIGNATURE or comment_length or first_disk or at > length:
            return LAYOUT
        name = data[named_at : named_at + name_length]
        if not 0 < name_length <= NAME_MOST or not name.isascii() or extra_length > len(_AES_EXTRA):
            return NOT_A_PACKAGE
        extra = data[named_at + name_length : at]
        records.append(
            _Record(
                name.decode("ascii"),
                made_by,
                needed,
                flags,
                method,
                time,
                date,
                crc,
                compressed,
                plain,
                extra,
                internal,
                external,
                offset,
            )
        )
    if at != length or len(records) != entries:
        return LAYOUT
    return tuple(records), start


def _layout_problem(file: BinaryIO, records: tuple[_Record, ...], start: int) -> bool:
    """Whether the members are not laid out as the writer lays them out: one after the other from the file's first
    byte, in the directory's order, each local header saying what the directory's record says (its version, flags,
    method, date, CRC, sizes, name and extra field), the last one ending where the directory starts."""
    at = 0
    for record in records:
        if record.offset != at:
            return True
        file.seek(at)
        header = file.read(_LOCAL.size)
        if len(header) != _LOCAL.size:
            return True
        name = record.name.encode("ascii")
        said = (
            _LOCAL_SIGNATURE,
            record.needed,
            record.flags,
            record.method,
            record.time,
            record.date,
            record.crc,
            record.compressed,
            record.size,
            len(name),
            len(record.extra),
        )
        if _LOCAL.unpack(header) != said:
            return True
        if file.read(len(name)) != name or file.read(len(record.extra)) != record.extra:
            return True
        at = record.data_at + record.compressed
    return at != start


def _member_problem(record: _Record, tier: str, clear: set[str]) -> str:
    """Why a member's entry is not the format's ("" when it is): its encryption, its fixed fields, its compression
    and its extra field, its declared size, its ratio."""
    encrypted = record.name not in clear
    if record.flags != (1 if encrypted else 0):
        return ENCRYPTION if bool(record.flags & 1) != encrypted else NOT_A_PACKAGE
    fixed = (record.made_by, record.needed, (record.time, record.date), record.internal)
    if (
        fixed != (UNIX << 8 | VERSION, VERSION, _dos(FIXED_DATE), 0)
        or record.external != MODES[tier]
    ):
        return NOT_A_PACKAGE
    if encrypted:
        # AE-2 (no CRC of the plaintext), 256 bits, deflate inside: what the writer writes
        if record.method != AES or record.extra != _AES_EXTRA or record.crc != 0:
            return COMPRESSION
    elif record.method != zipfile.ZIP_STORED or record.extra or record.compressed != record.size:
        return COMPRESSION
    if record.size > pk.cap_of(record.name, tier):
        return MEMBER_TOO_LARGE
    if record.size > RATIO_FROM and record.size > RATIO_MOST * max(1, record.compressed):
        return RATIO
    return ""


def _clear(file: BinaryIO, record: _Record | None, tier: str) -> bytes:
    """A clear member's bytes, read where its record says; b"" when it has no record, is not stored clear, is over
    its cap, or its bytes do not have the CRC-32 its record gives (its checksum then names it)."""
    if record is None or record.flags or record.method != zipfile.ZIP_STORED:
        return b""
    if record.compressed != record.size or record.size > pk.cap_of(record.name, tier):
        return b""
    file.seek(record.data_at)
    data = file.read(record.size)
    return data if len(data) == record.size and zlib.crc32(data) == record.crc else b""


class _TooLong(Exception):
    """A member's stream longer than its entry declares."""


def _stream(
    source: BinaryIO,
    declared: int,
    keep: bool,
    sink: BinaryIO | None = None,
    anchor: Any = None,
) -> tuple[str, bytes]:
    """A member's stream read through a counter that stops one byte past what its entry declares: its SHA-256, and
    its bytes when ``keep`` (written to ``sink`` instead when one is given); ``anchor``, a digest begun with
    ``core.package.ANCHOR_PREFIX``, is fed the same bytes (one pass over the member the trace starts from)."""
    digest, total, kept = hashlib.sha256(), 0, io.BytesIO()
    while chunk := source.read(CHUNK):
        total += len(chunk)
        if total > declared:
            raise _TooLong
        digest.update(chunk)
        if anchor is not None:
            anchor.update(chunk)
        if sink is not None:
            sink.write(chunk)
        elif keep:
            kept.write(chunk)
    return digest.hexdigest(), kept.getvalue()


def open_package(file: BinaryIO, *, max_size: int = MOST_FILE_BYTES) -> Opened | Refused:
    """
    The clear part of a received file (a binary file that can be sought; see the module's words, steps 1 to 6): the
    package's tier, its members, the checksums its manifest lists and its hash, or why it is refused. Nothing is
    decrypted, no zip library reads the file, and pyzipper is not loaded. A clear member whose bytes are not those
    the manifest lists is named in ``not_matching``; a notice so named gives no kind.
    """
    file.seek(0, os.SEEK_END)
    size = file.tell()
    if size > max_size:
        return Refused(TOO_LARGE)
    return _opened(file, size)


def _opened(file: BinaryIO, size: int) -> Opened | Refused:
    found = _directory(file, size)
    if isinstance(found, str):
        return Refused(found)
    records, start = found
    names = [record.name for record in records]
    if any(name.endswith("/") for name in names):
        return Refused(NOT_A_PACKAGE)
    if len({name.casefold() for name in names}) != len(names):
        return Refused(NAME_TWICE)
    if _layout_problem(file, records, start):
        return Refused(LAYOUT)
    tier = pk.REPRODUCTION if pk.NOTICE in names else pk.SHAREABLE
    by_name = {record.name: record for record in records}
    clear = set(pk.clear_members(tier))
    # whether a later version made it, read before anything else the file holds
    notice: str | None = None
    notice_refusal: list[str] = []
    if tier == pk.REPRODUCTION:
        notice, notice_refusal = pk.read_notice(_clear(file, by_name[pk.NOTICE], tier))
        if _later(notice_refusal):
            return Refused(LATER_VERSION, pk.NOTICE, notice_refusal[0])
    else:
        index = _clear(file, by_name.get(pk.PACKAGE), tier)
        value, reasons = pk.read_member(pk.PACKAGE, index, tier)
        if value is None and _later(reasons):
            return Refused(LATER_VERSION, pk.PACKAGE, reasons[0])
    fixed = set(pk.members_of(tier))
    with_data = tier == pk.REPRODUCTION
    if any(name not in fixed and not (with_data and pk.data_role(name)) for name in names):
        return Refused(NOT_A_PACKAGE)
    for record in records:
        problem = _member_problem(record, tier, clear)
        if problem:
            return Refused(problem, record.name)
    if sum(record.size for record in records) > pk.package_cap(tier):
        return Refused(TOO_LARGE)
    read_clear = {name: _clear(file, by_name[name], tier) for name in names if name in clear}
    manifest = read_clear.get(pk.MANIFEST, b"")
    listed, refused = pk.read_manifest(manifest, tier)
    if listed is None or set(listed) != set(names) - {pk.MANIFEST}:
        return Refused(MANIFEST, pk.MANIFEST, refused[0] if refused else "")
    if tuple(names) != pk.in_zip_order(names, tier):
        return Refused(LAYOUT)
    opened = Opened(
        tier=tier,
        size=size,
        entries=tuple(Entry(r.name, r.size, r.name not in clear) for r in records),
        listed=listed,
        package_hash=pk.package_hash(manifest),
        notice=notice,
        clear=read_clear,
        directory=records,
    )
    opened.not_matching.extend(
        name
        for name, data in read_clear.items()
        if name != pk.MANIFEST and pk.checksum(data) != listed[name]
    )
    if pk.NOTICE in opened.not_matching:
        opened.notice = None  # a notice that is not the sender's says nothing of the package
    elif notice_refusal:
        # the sender's bytes, but no notice of this version's: nothing would say what the package holds
        return Refused(NOTICE, pk.NOTICE, notice_refusal[0])
    return opened


def read_package(
    file: BinaryIO, opened: Opened, password: str | None = None, *, into: Path | None = None
) -> Opened | Refused:
    """
    Every member of a package :func:`open_package` opened from ``file``, streamed in the zip's order (see the
    module's words): a copy of ``opened`` completed with the members' values, the index of a package that holds the
    data and the data written into ``into``, or why it is refused. ``opened`` is left as it was.

    A package that holds the data needs its ``password``, in any spelling :func:`password_of` reads (none, or one of
    another shape, is ``not_a_password``: nothing is tried). ``into`` is a folder the caller made for the data
    (``verify`` only): it must exist and be empty, and on a system with modes belong to this user and give the group
    and others nothing (``tempfile.mkdtemp`` makes such a folder; on Windows the files take the folder's access
    list, and ``mkdtemp`` makes it the user's only from Python 3.13). The data are written there once the index is
    read, each file only if it does not exist, with the mode 0600 where modes apply. An error of the computer
    (``OSError``: a full disk, a file that cannot be read) is raised, after every file written was deleted. A file
    the system will not delete is said: on a refusal, an ``OSError`` that counts the files left (it names no path)
    instead of the refusal; on an error, a note added to it.
    """
    holds = pk.holds_data(opened.tier)
    read = replace(opened, not_matching=list(opened.not_matching), values={}, index=None, data={})
    # the clear members open_package read and found as the manifest lists them
    for name, data in opened.clear.items():
        if name not in (pk.NOTICE, pk.MANIFEST) and name not in read.not_matching:
            refused = _value_of(read, name, data)
            if refused is not None:
                return refused
    if not holds:
        return read
    canonical = None if password is None else password_of(password)
    if canonical is None:
        return Refused(NOT_A_PASSWORD)
    if into is not None and _folder_problem(into):
        return Refused(FOLDER)
    if not _same_file(file, opened):
        return Refused(LAYOUT)
    created: list[Path] = []
    try:
        refused = _decrypted(file, read, canonical, into, created)
    except BaseException as error:  # nothing is left of a reading that something stopped
        left = _drop_all(created)
        if left:
            error.add_note(_left_behind(left))
        raise
    if refused is not None:  # nor of a reading refused
        left = _drop_all(created)
        if left:
            stuck = OSError(_left_behind(left))
            stuck.add_note(f"the reading was refused first: {refused.id}")
            raise stuck
        return refused
    return read


def _folder_problem(into: Path) -> bool:
    """Whether ``into`` is not a folder the data may be written into: an existing, empty folder, which on a system
    with modes belongs to this process's user and gives the group and others nothing."""
    if not into.is_dir() or any(into.iterdir()):
        return True
    if hasattr(os, "getuid"):
        status = into.stat()
        return status.st_uid != os.getuid() or bool(status.st_mode & 0o077)
    return False


def _same_file(file: BinaryIO, opened: Opened) -> bool:
    """Whether ``file`` is still the one :func:`open_package` opened: its size, and its directory record by record."""
    file.seek(0, os.SEEK_END)
    size = file.tell()
    found = _directory(file, size) if size == opened.size else None
    return isinstance(found, tuple) and found[0] == opened.directory


def _decrypted(
    file: BinaryIO, read: Opened, password: str, into: Path | None, created: list[Path]
) -> Refused | None:
    """The encrypted members streamed into ``read``, ``package.json`` first, then in the zip's order; each file
    created in ``into`` is added to ``created``, which the caller deletes on a refusal or an exception."""
    aes = _aes()
    altered = (zlib.error, zipfile.BadZipFile, aes.BadZipFile, EOFError, _TooLong)
    file.seek(0)
    try:
        archive = aes.AESZipFile(file)
    except altered:  # pyzipper does not read what open_package read
        return Refused(LAYOUT)
    with archive:  # the decryption, and only with a password
        # pyzipper reads the directory with its own copy of zipfile: its reading must be the one checked
        seen = [
            (info.filename, info.header_offset, info.compress_size, info.file_size, info.flag_bits)
            for info in archive.infolist()
        ]
        if seen != [(r.name, r.offset, r.compressed, r.size, r.flags) for r in read.directory]:
            return Refused(LAYOUT)
        archive.setpassword(password.encode("ascii"))
        needed = sum(entry.size for entry in read.entries if pk.data_role(entry.name))
        # a stable sort: the index first, the zip's order otherwise
        order = sorted(
            (entry for entry in read.entries if entry.encrypted),
            key=lambda entry: entry.name != pk.PACKAGE,
        )
        # once a member decrypted, the password is the right one
        decrypted, failed_checks = False, 0
        for entry in order:
            role = pk.data_role(entry.name)
            try:
                stream = archive.open(entry.name)  # pyzipper checks the password's two bytes here
            except RuntimeError as error:
                if type(error) is RuntimeError and not decrypted:
                    failed_checks += 1
                    if failed_checks == 2:
                        return Refused(WRONG_PASSWORD)
                read.not_matching.append(entry.name)
                continue
            except altered:
                read.not_matching.append(entry.name)
                continue
            # the data go to disk only once the index is read: without it, nothing can use them
            path = None
            if role is not None and into is not None and read.index is not None:
                path = into / _DATA_NAMES[entry.name]
            with stream:
                if path is not None and into is not None and not created:
                    if shutil.disk_usage(into).free < needed:  # before the first file is written
                        return Refused(NO_SPACE)
                sink = None
                if path is not None:
                    try:
                        sink = _created(path)
                    except FileExistsError:  # the folder changed since it was checked
                        return Refused(FOLDER, entry.name)
                    created.append(path)
                starts = role in pk.STARTS_FROM and read.index is not None
                # as core.package.anchor_of, in the same pass as the member's checksum
                anchor = hashlib.sha256(pk.ANCHOR_PREFIX) if starts else None
                result = None
                try:
                    result = _stream(stream, entry.size, role is None, sink, anchor)
                except altered:  # a deflate error, the authentication code, a stream cut short
                    pass
                finally:
                    if sink is not None:
                        sink.close()
            if result is None:
                _drop(path)
                read.not_matching.append(entry.name)
                continue
            # its stream and its authentication code read: the password is the sender's
            decrypted = True
            digest, data = result
            if digest != read.listed.get(entry.name):
                _drop(path)
                read.not_matching.append(entry.name)
                continue
            if path is not None:
                read.data[entry.name] = path
            if role is None:
                refused = _value_of(read, entry.name, data)
                if refused is not None:
                    return refused
            elif anchor is not None and read.index is not None:
                if anchor.hexdigest() != read.index.anchor:
                    return Refused(ANCHOR, entry.name)
    return None


def _created(path: Path) -> BinaryIO:
    """A data member's file, created only if it does not exist, readable by its owner only where modes apply."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.fdopen(os.open(path, flags, 0o600), "wb")


def _drop(path: Path | None) -> None:
    """A data member's file whose bytes are not the sender's: deleted at once."""
    if path is not None:
        path.unlink(missing_ok=True)


def _drop_all(paths: list[Path]) -> int:
    """Every file a reading wrote, deleted, each tried whatever another one does: how many the system would not
    delete (a file another program holds open)."""
    left = 0
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            left += 1
    return left


def _left_behind(left: int) -> str:
    """What a reader says of decrypted files it could not delete: their count, never their path."""
    files = "file" if left == 1 else "files"
    return f"{left} decrypted {files} could not be deleted from the folder given for the data: delete them"


def _value_of(read: Opened, name: str, data: bytes) -> Refused | None:
    """A member's value read by its one reader (``core.package.read_member``); for ``package.json``, the index's
    checks of a package that holds the data, or the tier a package without the data says it is."""
    value, reasons = pk.read_member(name, data, read.tier)
    if value is None:
        later = LATER_VERSION if _later(reasons) else NOT_READ
        return Refused(later, name, reasons[0] if reasons else "")
    read.values[name] = value
    if name != pk.PACKAGE:
        return None
    if pk.holds_data(read.tier):
        return _index_of(read, value)
    # its index's closed reader comes with the package without the data (13c): here, its tier only
    tier = _tier_said(value)
    if tier is None:
        return Refused(NOT_A_PACKAGE, pk.PACKAGE)
    read.tier = tier
    return None


def _index_of(read: Opened, value: Any) -> Refused | None:
    """The index of a package that holds the data, read and checked against what the zip holds: its data members,
    and the notice against its policy (a notice whose bytes are not the sender's is named, not compared)."""
    index, reasons = pk.DataPackageIndex.from_dict(value)
    if index is None:
        later = LATER_VERSION if _later(reasons) else NOT_READ
        return Refused(later, pk.PACKAGE, reasons[0] if reasons else "")
    present = tuple(entry.name for entry in read.entries if pk.data_role(entry.name))
    if index.data_members() != present:
        return Refused(NOT_A_PACKAGE, pk.PACKAGE)
    if pk.NOTICE not in read.not_matching:
        problem = pk.notice_problem(read.notice, index.policy)
        if problem:
            return Refused(NOTICE, pk.NOTICE, problem)
    read.index = index
    return None


__all__ = [
    "AES",
    "ALPHABET",
    "ANCHOR",
    "CHUNK",
    "COMPRESSION",
    "ENCRYPTION",
    "EXIT_NOT_READ",
    "EXIT_NOT_WHAT_WAS_SENT",
    "EXIT_OK",
    "FIXED_DATE",
    "FOLDER",
    "GROUP",
    "LATER",
    "LATER_VERSION",
    "LAYOUT",
    "MANIFEST",
    "MEMBER_TOO_LARGE",
    "MODES",
    "MOST_ENTRIES",
    "MOST_FILE_BYTES",
    "NAME_MOST",
    "NAME_TWICE",
    "NOTICE",
    "NOT_A_PACKAGE",
    "NOT_A_PASSWORD",
    "NOT_A_ZIP",
    "NOT_READ",
    "NO_SPACE",
    "RATIO",
    "RATIO_FROM",
    "RATIO_MOST",
    "REFUSALS",
    "SYMBOLS",
    "TOO_LARGE",
    "UNIX",
    "VERSION",
    "WRONG_PASSWORD",
    "Entry",
    "Member",
    "Opened",
    "Refused",
    "new_password",
    "open_package",
    "password_of",
    "read_package",
    "write",
]
