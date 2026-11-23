"""
``python -m manage.replay inspect PACKAGE``: what a received study package holds, without replaying it. (The module
is not named ``inspect``: a file run from this folder would then import it in place of the standard library's.)

It reads the file with the safe reader (``manage.package_zip``) and writes, for a terminal of 80 columns and of any
code page: the file, the kind of package, **the package's hash before any password**, and its files with their
sizes; then, with the password of a package that holds the data, that each file is the sender's, and what the
package's index says of it: the software that made it, the study, to whom and under what it was sent, the data by
role, the trace and the privacy rules. Nothing of a file that holds the study's own texts is shown: the numbers and
the closed values of the index only, and the two texts the exporter typed for the export, through
``core.package.shown``, as the file's name is. It writes nothing to disk.

Its exit code says what happened:

- ``0``: read, and nothing failed: the clear part when no password is given, every file with it, and the hash given,
  if any, is the package's;
- ``1``: read, but not the package its sender made: a file does not match its checksum, or the package's hash is not
  the one given;
- ``2``: not read: not a package, a later version's, a zip that is not laid out as the writer's, the wrong password,
  a password file that cannot be read;
- ``64``: the command was not understood (an option, or the value given to one), or cannot run here (a password
  asked for where nobody can type one unseen);
- ``70``: an error of the program itself (``manage.replay.__main__`` names it by its kind).

The password is never an argument (a command line is kept by the shell and shown to every user of the computer):
``--password-file FILE`` reads it from a file's first line, ``--ask-password`` asks for it on a terminal that can
hide what is typed, three times at most. Without either, the clear part is all it reads, and it says so: it never
waits for an input nobody is there to give. Nothing typed on the command line is written back: a file that is not
found is not named.

Standard library, ``core.package`` (with ``core.privacy``), ``manage.package_zip``, ``manage.replay.commandline``
and ``utils.environment`` only: never numpy, pandas, pydantic, Streamlit or the replay's engine; no ``.env`` and no
setting; pyzipper only once a password is given (a test reads the modules a run has loaded). It configures no log
and writes none: its report is its output.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from core import package as pk
from manage import package_zip as pz
from manage.replay.commandline import (
    EXIT_NOT_READ,
    EXIT_NOT_WHAT_WAS_SENT,
    EXIT_OK,
    EXIT_USAGE,
    PROMPT,
    CannotAsk,
    no_console_words,
    password_file_note,
    password_from_file,
    typed_unseen,
    wrapped,
)
from utils.environment import app_version, git_commit

TRIES = 3
SAID_ALOUD = 16  # the first characters of the hash, enough when it is read aloud
HASH_MOST = 64

# what each member is, in the format's own words (the same for every package)
_MEMBER_WORDS = {
    pk.NOTICE: "what this file is",
    pk.PACKAGE: "what the package is",
    pk.PROTOCOL: "the study definition",
    pk.RESULTS: "the results, with their settings",
    pk.TRACE: "the trace of every step",
    pk.PROFILES: "each column's profile at each step",
    pk.README: "how to read and check it",
    pk.LOCK: "the exact libraries",
    pk.MANIFEST: "a checksum (SHA-256) per file",
}
_DATA_WORDS = {
    "source": "the source file",
    "saved_version": "the saved version the trace starts from",
    "enrichment": "a merged file: the rows its merge took",
    "snapshot": "a step's recorded table",
}
_NOTICE_WORDS = {
    pk.NOTICE_PERSONAL_DATA: (
        "It holds personal data, encrypted. Open it only if an agreement or another legal basis covers you as its "
        "recipient."
    ),
    pk.NOTICE_NOT_PEOPLE: (
        "Its rows were declared not people. It is encrypted: open it only if its sender meant it for you."
    ),
}
_BASIS_WORDS = {
    "data_sharing_agreement": "a data sharing agreement",
    "data_processing_agreement": "a data processing agreement",
    "joint_controllers": "an arrangement between joint controllers",
    "same_organisation": "the sender's own organisation (one of its sites or computers)",
    "other_basis": "another legal basis",
    "rows_not_people": "the declaration that its rows are not people",
}
_SHAPE = (
    "A package's password is 20 letters and digits in 5 groups of 4, as in XXXX-XXXX-XXXX-XXXX-XXXX (the hyphens "
    "and the case do not matter)."
)
_NOTHING_READ = "Nothing was read from it."
_ASK_AGAIN = "Ask the sender for it again."
_NOT_OPENED = (
    "Not opened: {count} are encrypted and were not checked. To read them, run it again with --ask-password, or "
    "with --password-file FILE."
)
_NOT_THE_RECIPIENT = (
    "If you are not this recipient, do not use the data: delete the package and tell its sender."
)


def hash_given(text: str) -> str | None:
    """The package hash a sender gave, as it is compared: its 16 to 64 hexadecimal characters in lower case, the
    hyphens and the spaces between them dropped; None for anything else."""
    kept = "".join(char for char in text.lower() if char not in "- \t")
    if SAID_ALOUD <= len(kept) <= HASH_MOST and all(char in "0123456789abcdef" for char in kept):
        return kept
    return None


def differs_at(given: str, package_hash: str) -> int | None:
    """The first character (counted from 1) at which the hash given is not the package's; None when it is."""
    for position, (typed, own) in enumerate(zip(given, package_hash, strict=False), start=1):
        if typed != own:
            return position
    return None


def _grouped(digits: str) -> str:
    return "-".join(digits[at : at + 4] for at in range(0, len(digits), 4))


def _size(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{-(-count // 1024):,} KB"
    if count < 1024**3:
        return f"{count / 1024**2:,.1f} MB"
    return f"{count / 1024**3:,.2f} GB"


def _plural(count: int, one: str, many: str | None = None) -> str:
    return f"{count:,} {one if count == 1 else (many or one + 's')}"


# ----------------------------------------------------------------------------- the words of a refusal


def refusal_words(
    refused: pz.Refused, name: str, size: int, *, from_file: bool = False, opened: bool = False
) -> str:
    """
    Why a file was refused, for whoever received it: what happened, what was not done, what to do. Nothing of the
    file is quoted: a member is named only by the format's own name (``Refused.member``, set once the names passed
    the format's list), ``core.package``'s reason quotes nothing, and ``name`` is the file's name as shown.
    ``from_file``: the password came from ``--password-file``; ``opened``: the clear part was read and shown.
    """
    member, reason = refused.member, refused.reason
    changed = "it was changed after it was made, or made by another program"
    if refused.id == pz.NOT_A_ZIP:
        return (
            f"{name} is not a complete zip file: it may have been cut short while it was downloaded. Download it "
            "again."
        )
    if refused.id == pz.TOO_LARGE:
        if size > pz.MOST_FILE_BYTES:
            return (
                f"This file ({_size(size)}) is larger than a CohortStudio study package can be "
                f"({_size(pz.MOST_FILE_BYTES)}). {_NOTHING_READ}"
            )
        return (
            f"The files of this zip would unpack to more than a CohortStudio study package holds: {changed}. "
            f"{_NOTHING_READ} {_ASK_AGAIN}"
        )
    if refused.id == pz.LATER_VERSION:
        if opened and member:
            return (
                f"{member} was written by a later version of CohortStudio than this one. Install the latest "
                "version to read it. Nothing was shown from it."
            )
        return (
            "This package was made by a later version of CohortStudio, in a format this version does not read. "
            f"Install the latest version of CohortStudio to read it. {_NOTHING_READ}"
        )
    if refused.id == pz.NOT_A_PACKAGE:
        return (
            "This zip is not a CohortStudio study package: it holds a file that a package does not hold, or lacks "
            f"one. {_NOTHING_READ} If it was sent to you as a package, ask the sender for it again."
        )
    if refused.id == pz.NAME_TWICE:
        return (
            "This zip holds two files of one name, or of names that differ by their case only, as CohortStudio "
            f"never writes them. {_NOTHING_READ} {_ASK_AGAIN}"
        )
    if refused.id == pz.LAYOUT:
        return f"This zip is not laid out as CohortStudio writes one: {changed}. {_NOTHING_READ} {_ASK_AGAIN}"
    if refused.id in (pz.ENCRYPTION, pz.COMPRESSION):
        how = "encrypted" if refused.id == pz.ENCRYPTION else "stored"
        return (
            f"A file of this package is not {how} as CohortStudio writes it: the package was changed after it was "
            f"made. {_NOTHING_READ} {_ASK_AGAIN}"
        )
    if refused.id in (pz.MEMBER_TOO_LARGE, pz.RATIO):
        return f"A file of this package unpacks to more than CohortStudio reads. {_NOTHING_READ} {_ASK_AGAIN}"
    if refused.id == pz.NOTICE:
        why = (
            "its notice does not say what its index says of its rows"
            if reason == pk.NOTICE_NOT_THE_POLICY
            else "its notice is not one CohortStudio writes"
        )
        return (
            f"This package was changed after it was made, or made by another program: {why}. {_NOTHING_READ} "
            f"{_ASK_AGAIN}"
        )
    if refused.id == pz.MANIFEST:
        return (
            f"{pk.MANIFEST} does not list this zip's files as CohortStudio lists them: the package was changed "
            f"after it was made. {_NOTHING_READ} {_ASK_AGAIN}"
        )
    if refused.id == pz.NOT_A_PASSWORD:
        if from_file:
            return f"--password-file: its first line is not a package's password (nothing was tried). {_SHAPE}"
        return f"That is not a password of a CohortStudio package: nothing was tried. {_SHAPE}"
    if refused.id == pz.WRONG_PASSWORD:
        return (
            "The password does not open this package; nothing was read from it. Look for a slip in it, then ask "
            "the sender to read it to you again: by telephone or in person, never in the message that carries "
            "the file."
        )
    if refused.id == pz.NOT_READ:
        if member and reason:
            # core.package's reasons either follow the member's name ("... cannot be read: ...") or begin "its ..."
            said = f"{member}: {reason}" if reason.startswith("its ") else f"{member} {reason}"
        else:
            said = "A file of this package cannot be read"
        return (
            f"{said}. CohortStudio never writes such a file: it was made or changed by another program. Nothing "
            f"was shown from it. {_ASK_AGAIN}"
        )
    if refused.id == pz.ANCHOR:
        return (
            f"{pk.PACKAGE} is not the index of the data this package holds: the package was put together from "
            f"two. Nothing was shown from it. {_ASK_AGAIN}"
        )
    return f"This package could not be read ({refused.id}). {_NOTHING_READ}"


def not_matching_words(members: list[str]) -> list[str]:
    """What is said of members whose bytes are not those the manifest lists (their names are the format's own)."""
    said = []
    for member in members:
        said.append(
            f"{member} does not match its checksum in {pk.MANIFEST}: the package was changed after it was made, or "
            f"damaged on the way. {_ASK_AGAIN}"
        )
    said.append("Nothing more is shown from it.")
    return said


# ----------------------------------------------------------------------------- the report's parts


def _member_line(entry: pz.Entry, tier: str) -> str:
    role = pk.data_role(entry.name)
    words = _DATA_WORDS[role] if role else _MEMBER_WORDS.get(entry.name, "")
    if pk.holds_data(tier) and not entry.encrypted:
        words = f"clear: {words}"
    return f"  {entry.name:<21}{_size(entry.size):>9}  {words}"


def clear_part(
    opened: pz.Opened, name: str, given: str | None, this_version: tuple[str, str | None]
) -> list[str]:
    """What is read before any password: the file, the kind of package, its hash and its files."""
    tier, holds = opened.tier, pk.holds_data(opened.tier)
    schema = pk.FORMAT_SCHEMAS[pk.formats_of(tier)[pk.PACKAGE]]
    lines = [f"{name}, {_size(opened.size)}"]
    kind = "with the data" if holds else "without the data"
    lines.append(f"CohortStudio study package, format {schema}: {kind}.")
    if holds and opened.notice in _NOTICE_WORDS:
        lines += wrapped(_NOTICE_WORDS[opened.notice], 2)
    lines.append(f"Package hash (SHA-256 of {pk.MANIFEST}, not of the zip file):")
    lines.append(f"  {opened.package_hash}")
    first = _grouped(opened.package_hash[:SAID_ALOUD])
    at = None if given is None else differs_at(given, opened.package_hash)
    if given is None:
        lines += wrapped(
            f"Compare its first {SAID_ALOUD} characters, {first}, with those the sender gave you apart from the "
            "file, or give them with --package-hash.",
            2,
        )
    elif at is None:
        whole = len(given) == HASH_MOST
        lines.append(
            "  It is the hash you gave."
            if whole
            else f"  Its first {len(given)} characters are those you gave."
        )
    else:
        # the hash given under the package's own, so that the two are read one above the other
        lines.append(f"The hash you gave differs from it at character {at}:")
        lines.append(f"  {given}")
        lines += wrapped(
            "Check it against what the sender sent you: the package hash, not a checksum of the zip file. If it "
            "is theirs, this is not the package the sender made, or it was changed on the way: do not open it, "
            "and ask the sender for it again.",
            2,
        )
    version, commit = this_version
    built = f" (commit {commit[:7]})" if commit else ""
    lines.append(f"This is CohortStudio {version}{built}: it reads format {schema}.")
    lines.append("")
    clear = [entry.name for entry in opened.entries if not entry.encrypted]
    if holds:
        lines.append(f"Files: {len(opened.entries):,}, all encrypted but {' and '.join(clear)}.")
    else:
        lines.append(f"Files: {len(opened.entries):,}, none encrypted.")
    lines += [_member_line(entry, tier) for entry in opened.entries]
    return lines


def _file_line(entry: pk.DataFile) -> str:
    table = f"{_plural(entry.rows, 'row')} and {_plural(entry.columns, 'column')}"
    if entry.role == "enrichment":
        cut = entry.cut
        assert cut is not None  # an index's merged file has its cut (core.package.DataFile)
        kept = (
            "every row of the file"
            if cut.rule == "every_row"
            else f"the rows its merge took, of {cut.rows_of_original:,}"
        )
        return f"a merged file read at step {entry.step}: {table} ({kept})"
    if entry.role == "snapshot":
        return f"the table recorded for step {entry.step}: {table}"
    how = "as the table the study read" if entry.as_read else "as the file was loaded"
    what = (
        "the source file" if entry.role == "source" else "the saved version the trace starts from"
    )
    return f"{what}: {table}, {how}"


def opened_part(read: pz.Opened, utf8: bool) -> list[str]:
    """What the index of a package that holds the data says of it, once every file was read with the password."""
    index = read.index
    assert (
        index is not None
    )  # read_package gives the index of a package that holds the data, or refuses
    made = index.software
    commit = f" (commit {made.commit[:7]})" if made.commit else ""
    lines = [f"With the password: each file matches its checksum in {pk.MANIFEST}."]
    lines += wrapped(
        f"Made with CohortStudio {made.cohortstudio}{commit}, Python {made.python}, on {made.system} "
        f"({made.machine})."
    )
    lines.append("")
    lines.append(f"Study: {index.study.id}, revision {index.study.revision}")
    declared = index.declaration
    # the two texts the exporter typed, each on lines of its own (never side by side: a "?" is shown's mark), cut
    # between two characters where one is wider than a line
    if declared.recipient is not None:
        recipient = pk.shown(declared.recipient, pk.RECIPIENT_MAX, terminal=True, utf8=utf8)
        lines += wrapped(f"Sent to: {recipient}", 2, breakable=True)
    lines += wrapped(f"Covered by: {_BASIS_WORDS[declared.basis]}", 2)
    if declared.reference is not None:
        reference = pk.shown(declared.reference, pk.REFERENCE_MAX, terminal=True, utf8=utf8)
        lines += wrapped(f"Reference: {reference}", 2, breakable=True)
    if declared.recipient is not None:
        lines += wrapped(_NOT_THE_RECIPIENT, 2)
    lines.append("  Data:")
    for entry in index.files:
        lines += wrapped(_file_line(entry), 4)
    start = "the source file" if index.trace.starts_from == "source" else "a saved version"
    lines += wrapped(f"Trace: {_plural(len(index.trace.steps), 'step')}, from {start}.", 2)
    lines += wrapped(_policy_words(index.policy))
    return lines


def _policy_words(policy: pk.Policy) -> str:
    said = "Privacy rules: none applied: the package holds the data themselves."
    if policy.rows_not_people is not None:
        reason, level = policy.rows_not_people
        whose = "this study" if level == "study" else "its whole installation"
        said += (
            f" Its rows were declared not people, for {whose} ({pk.NOT_PEOPLE_REASONS[reason]})."
        )
    return said


def without_data_part(read: pz.Opened) -> list[str]:
    """A package without the data, read whole: it is not encrypted."""
    lines = [f"Each file matches its checksum in {pk.MANIFEST}."]
    if read.tier == pk.RULES_OFF:
        lines += wrapped(
            "It was made with the privacy rules off, for a study whose rows were declared not people: its counts "
            "are exact."
        )
    else:
        lines.append("It was made under the privacy rules: it holds no row of the data.")
    return lines


# ----------------------------------------------------------------------------- the command


class _Report:
    """Where the report goes: lines as they come, or one JSON object at the end."""

    def __init__(self, out: TextIO, err: TextIO, as_json: bool) -> None:
        self.out, self.err, self.as_json = out, err, as_json
        self.facts: dict[str, Any] = {}

    def say(self, lines: list[str]) -> None:
        if not self.as_json:
            self.out.write("".join(line + "\n" for line in lines))
            self.out.flush()

    def note(self, text: str) -> None:
        """Words for whoever runs the command, on the error stream, wrapped."""
        self.err.write("".join(line + "\n" for line in wrapped(text)))
        self.err.flush()

    def refuse(self, text: str) -> None:
        """A refusal: on the error stream, and kept among the facts."""
        self.facts.setdefault("said", []).append(text)
        self.note(text)

    def end(self, code: int) -> int:
        self.facts["exit"] = code
        if self.as_json:
            self.out.write(
                json.dumps(self.facts, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
            )
            self.out.flush()
        return code


def _is_utf8(stream: TextIO) -> bool:
    return (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace(
        "_", ""
    ) == "utf8"


def run(
    options: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    is_terminal: Callable[[], bool],
    typed: Callable[[str, TextIO], str] = typed_unseen,
) -> int:
    """
    ``inspect`` for the options ``manage.replay.__main__`` defines: the report on ``out``, the refusals and the prompts on
    ``err``, and the exit code. ``is_terminal``: whether a person is there to type a password; ``typed``: how one
    is read without being shown (``CannotAsk`` when it cannot be).
    """
    report = _Report(out, err, bool(options.json))
    given = None
    if options.package_hash is not None:
        given = hash_given(options.package_hash)
        if given is None:
            report.refuse(
                "--package-hash takes the 16 to 64 hexadecimal characters (0-9, a-f) the sender gave you (in "
                "quotes if they hold spaces)."
            )
            return report.end(EXIT_USAGE)
    if options.ask_password and not is_terminal():
        report.refuse(no_console_words())
        return report.end(EXIT_USAGE)
    path = Path(options.package)
    if path.is_dir():  # a folder that exists is nobody's password: it is named
        folder = pk.shown(path.name, 60, terminal=True, utf8=_is_utf8(err))
        report.refuse(
            f"{folder} is a folder: give the .cstudio.zip file itself, as the sender sent it."
        )
        return report.end(EXIT_NOT_READ)
    try:
        file = open(path, "rb")  # closed below, after the last reading
    except FileNotFoundError:
        # what was typed is not written back: a password pasted in the file's place would be
        report.refuse(
            "The package file was not found. What you typed is not repeated here: it could be a password. Give "
            "the file's whole path, in quotes: dragging the file onto this window writes it."
        )
        return report.end(EXIT_NOT_READ)
    except OSError as error:
        report.refuse(f"The package file cannot be opened ({type(error).__name__}).")
        return report.end(EXIT_NOT_READ)
    with file:
        return _inspected(file, report, given, options, typed)


def _inspected(
    file: BinaryIO,
    report: _Report,
    given: str | None,
    options: argparse.Namespace,
    typed: Callable[[str, TextIO], str],
) -> int:
    utf8 = _is_utf8(report.out)
    shown_name = Path(options.package).name
    name = pk.shown(shown_name, 60, terminal=True, utf8=utf8)
    # in a refusal, as the stream the refusals go to writes it
    refused_name = pk.shown(shown_name, 60, terminal=True, utf8=_is_utf8(report.err))
    size = file.seek(0, os.SEEK_END)
    # the file's name one way in the JSON, whatever the stream it is written to
    report.facts["file"] = {"name": pk.shown(Path(options.package).name, 255), "bytes": size}
    this_version = (app_version(), git_commit())
    report.facts["this_version"] = {"cohortstudio": this_version[0], "commit": this_version[1]}
    opened = pz.open_package(file)
    if isinstance(opened, pz.Refused):
        report.facts["refused"] = {
            "id": opened.id,
            "member": opened.member,
            "reason": opened.reason,
        }
        report.refuse(refusal_words(opened, refused_name, size))
        return report.end(opened.exit_code)
    holds = pk.holds_data(opened.tier)
    report.facts["package"] = {
        "tier": opened.tier,
        "schema": pk.FORMAT_SCHEMAS[pk.formats_of(opened.tier)[pk.PACKAGE]],
        "notice": opened.notice,
        "hash": opened.package_hash,
        "files": [
            {"name": entry.name, "bytes": entry.size, "encrypted": entry.encrypted}
            for entry in opened.entries
        ],
    }
    at = None if given is None else differs_at(given, opened.package_hash)
    report.facts["hash_given"] = (
        None
        if given is None
        else {"characters": len(given), "matches": at is None, "differs_at": at}
    )
    report.facts["opened"] = False
    report.say(clear_part(opened, name, given, this_version))
    if (
        at is not None
    ):  # said under the hash, and again last: no password is read, asked for or tried
        report.refuse(
            "Stopped before the password: the package hash is not the one you gave. Check the hash you gave; if "
            "it is the sender's, ask them for the package again."
        )
        return report.end(EXIT_NOT_WHAT_WAS_SENT)
    if opened.not_matching:
        report.facts["not_matching"] = list(opened.not_matching)
        for text in not_matching_words(opened.not_matching):
            report.refuse(text)
        return report.end(EXIT_NOT_WHAT_WAS_SENT)
    if not holds:
        return _read(report, pz.read_package(file, opened), refused_name, size, utf8)
    encrypted = _plural(sum(1 for entry in opened.entries if entry.encrypted), "file")
    if options.password_file is not None:
        # read here, after the clear part: the hash's verdict never waits on a password's file
        password, why = password_from_file(options.password_file)
        if password is None:
            report.refuse(why)
            return report.end(EXIT_NOT_READ)
        note = password_file_note(options.password_file)
        if note:
            report.note(note)
        read = pz.read_package(file, opened, password)
        return _read(report, read, refused_name, size, utf8, from_file=True)
    if not options.ask_password:
        report.say(wrapped(_NOT_OPENED.format(count=encrypted)))
        return report.end(EXIT_OK)
    for left in range(TRIES - 1, -1, -1):
        try:
            answer = typed(PROMPT, report.err)
        except CannotAsk:
            report.refuse(
                "This terminal cannot hide what is typed: the password was not asked for. Use --password-file "
                "FILE."
            )
            return report.end(EXIT_USAGE)
        if not answer.strip():
            report.say(wrapped(_NOT_OPENED.format(count=encrypted)))
            return report.end(EXIT_OK)
        read = pz.read_package(file, opened, answer)
        again = isinstance(read, pz.Refused) and read.id in (pz.NOT_A_PASSWORD, pz.WRONG_PASSWORD)
        if not again or left == 0:
            return _read(report, read, refused_name, size, utf8)
        assert isinstance(read, pz.Refused)  # for the type checker
        tries = "1 try left" if left == 1 else f"{left} tries left"
        if read.id == pz.NOT_A_PASSWORD:
            report.note(f"That is not a password of a CohortStudio package: {tries}. {_SHAPE}")
        else:
            report.note(f"That password does not open this package: {tries}.")
    raise AssertionError("the last try returns")  # pragma: no cover


def _read(
    report: _Report,
    read: pz.Opened | pz.Refused,
    name: str,
    size: int,
    utf8: bool,
    *,
    from_file: bool = False,
) -> int:
    """The end of the report once every file was read, or the reading refused."""
    if isinstance(read, pz.Refused):
        report.facts["refused"] = {"id": read.id, "member": read.member, "reason": read.reason}
        report.refuse(refusal_words(read, name, size, from_file=from_file, opened=True))
        return report.end(read.exit_code)
    report.facts["package"]["tier"] = read.tier
    if read.not_matching:
        report.facts["not_matching"] = list(read.not_matching)
        for text in not_matching_words(read.not_matching):
            report.refuse(text)
        return report.end(EXIT_NOT_WHAT_WAS_SENT)
    report.facts["opened"] = True
    if read.index is not None:
        report.facts["index"] = read.index.to_dict()
        report.say(opened_part(read, utf8))
    else:
        report.say(without_data_part(read))
    report.say(["", "Nothing was written to disk."])
    return report.end(EXIT_OK)


__all__ = [
    "clear_part",
    "differs_at",
    "hash_given",
    "not_matching_words",
    "opened_part",
    "refusal_words",
    "run",
    "without_data_part",
]
