"""
``python -m manage.replay inspect`` (manage.replay.inspection, on manage.replay.commandline): what a received study
package holds, without replaying it. Its hash before any password; with the password, what its index says; nothing
of the study's own texts or of the data; lines of 78 columns; exit codes 0, 1, 2, 64 and 70; nothing typed on the
command line written back; and never a wait for a password nobody types. The packages are made up with the zip
tests' own helpers.
"""

from __future__ import annotations

import codecs
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import package as pk
from manage import package_zip as pz
from manage.replay import commandline, inspection
from manage.replay.__main__ import main, parser
from tests import test_package_zip as made

ROOT = Path(__file__).resolve().parents[1]
CANARY = "canary-7Q4F-never-shown"
OK, NOT_SENT, NOT_READ = (
    commandline.EXIT_OK,
    commandline.EXIT_NOT_WHAT_WAS_SENT,
    commandline.EXIT_NOT_READ,
)
USAGE, FAILED = commandline.EXIT_USAGE, commandline.EXIT_FAILED


def _package(
    tmp_path: Path, data: bytes | None = None, name: str = "study_r3_with_data.cstudio.zip"
) -> Path:
    path = tmp_path / name
    path.write_bytes(made._write()[0] if data is None else data)
    return path


def _password_file(tmp_path: Path, password: str = made.PASSWORD) -> Path:
    path = tmp_path / "password.txt"
    path.write_bytes((password + "\n").encode("ascii"))
    if hasattr(os, "getuid"):
        path.chmod(0o600)
    return path


def _run(*argv: str, terminal: bool = False, typed: Any = None) -> tuple[int, str, str]:
    """The command run in this process: its exit code, what it wrote and what it refused."""
    out, err = io.StringIO(), io.StringIO()
    kwargs: dict[str, Any] = {} if typed is None else {"typed": typed}
    code = inspection.run(
        parser().parse_args(argv), out=out, err=err, is_terminal=lambda: terminal, **kwargs
    )
    return code, out.getvalue(), err.getvalue()


def _narrow(text: str) -> bool:
    return all(commandline.columns(line) <= commandline.WIDTH for line in text.splitlines())


def _narrow_ascii(text: str) -> bool:
    return text.isascii() and _narrow(text)


def _said(text: str) -> str:
    """Words as the command writes them on the error stream."""
    return "".join(line + "\n" for line in commandline.wrapped(text))


# ----------------------------------------------------------------------------- the clear part


def test_the_hash_is_read_before_any_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, package_hash = made._write()
    path = _package(tmp_path, data)

    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("nothing is decrypted without a password")

    monkeypatch.setattr(pz, "read_package", never)
    code, out, err = _run("inspect", str(path))
    assert code == OK and err == ""
    lines = out.splitlines()
    assert lines[0] == f"{path.name}, {inspection._size(len(data))}"
    assert lines[1] == "CohortStudio study package, format 1: with the data."
    assert f"  {package_hash}" in lines
    grouped = "-".join(package_hash[at : at + 4] for at in range(0, 16, 4))
    assert f"Compare its first 16 characters, {grouped}," in out
    assert "Files: 10, all encrypted but NOTICE.txt and MANIFEST.sha256." in lines
    assert "  NOTICE.txt                2 KB  clear: what this file is" in lines
    assert "  merge_01.parquet         420 B  a merged file: the rows its merge took" in lines
    # what it could not check is said, with how to go on; nothing of the index is shown
    assert "Not opened: 8 files are encrypted and were not checked." in out
    assert "--ask-password" in out and "--password-file FILE" in out
    assert "Study:" not in out and made.RECIPIENT not in out
    assert _narrow_ascii(out)


def test_a_package_without_the_data_is_read_whole(tmp_path: Path) -> None:
    path = _package(tmp_path, made._shareable_zip(), "study_r3.cstudio.zip")
    code, out, err = _run("inspect", str(path))
    assert code == OK and err == ""
    assert "CohortStudio study package, format 1: without the data." in out
    assert "Files: 5, none encrypted." in out and "clear:" not in out
    assert "Each file matches its checksum in MANIFEST.sha256." in out
    assert "It was made under the privacy rules: it holds no row of the data." in out
    assert out.rstrip().endswith("Nothing was written to disk.") and _narrow_ascii(out)


# ----------------------------------------------------------------------------- with the password


def test_with_the_password_it_says_what_the_index_says(tmp_path: Path) -> None:
    path = _package(tmp_path)
    code, out, err = _run("inspect", str(path), "--password-file", str(_password_file(tmp_path)))
    assert code == OK and err == ""
    lines = out.splitlines()
    assert "With the password: each file matches its checksum in MANIFEST.sha256." in lines
    assert "Made with CohortStudio 2.5.0, Python 3.14.3, on Windows (AMD64)." in lines
    assert "Study: tutorial_cohort, revision 3" in lines
    assert f"  Sent to: {made.RECIPIENT}" in lines
    assert (
        "  Covered by: a data sharing agreement" in lines and "  Reference: DSA-2026-014" in lines
    )
    # whoever reads it and is not the recipient is told what to do
    assert "  If you are not this recipient, do not use the data: delete the package and" in lines
    assert "    the source file: 2,000 rows and 2 columns, as the file was loaded" in lines
    assert "    a merged file read at step 2: 10 rows and 3 columns (the rows its merge" in lines
    assert "  Trace: 2 steps, from the source file." in lines
    assert "Privacy rules: none applied: the package holds the data themselves." in lines
    assert lines[-1] == "Nothing was written to disk." and _narrow_ascii(out)
    # the spelling a person types is read: lower case, spaces for hyphens
    typed = made.PASSWORD.lower().replace("-", " ")
    again, _, _ = _run(
        "inspect", str(path), "--password-file", str(_password_file(tmp_path, typed))
    )
    assert again == OK


def test_nothing_of_the_studys_texts_or_of_the_data_is_shown(tmp_path: Path) -> None:
    """The members that carry the study's own texts, the README and the data: read, checked, never written out."""
    tier = pk.REPRODUCTION
    members = made._members(
        **{
            pk.PROTOCOL: pk.dumps(
                pk.PROTOCOL, {"revision": {"schema": 1, "body": {"note": CANARY}}}, tier
            ),
            pk.RESULTS: pk.dumps(pk.RESULTS, {"results": [{"title": CANARY}]}, tier),
            pk.TRACE: pk.dumps(pk.TRACE, {"trace": {"steps": [{"description": CANARY}]}}, tier),
            pk.PROFILES: pk.dumps(pk.PROFILES, {"profiles": [{"column": CANARY}]}, tier),
            pk.README: f"# tutorial_cohort\n\n{CANARY}\n".encode(),
        }
    )
    path = _package(tmp_path, made._write(members)[0])
    for extra in ((), ("--json",)):
        code, out, err = _run(
            "inspect", str(path), "--password-file", str(_password_file(tmp_path)), *extra
        )
        assert code == OK
        assert CANARY not in out + err and "patient_id" not in out + err


def _with_recipient(tmp_path: Path, recipient: str) -> Path:
    index = made._index(pk.anchor_of([made.SOURCE]), recipient=recipient)
    members = made._members(**{pk.PACKAGE: pk.dumps(pk.PACKAGE, index.to_dict(), pk.REPRODUCTION)})
    return _package(tmp_path, made._write(members)[0])


def _run_utf8(*argv: str) -> tuple[int, str]:
    raw = io.BytesIO()
    utf8 = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
    code = inspection.run(
        parser().parse_args(argv), out=utf8, err=io.StringIO(), is_terminal=lambda: False
    )
    utf8.flush()
    return code, raw.getvalue().decode("utf-8")


def test_the_texts_the_exporter_typed_are_shown_for_the_terminal_at_hand(tmp_path: Path) -> None:
    recipient = "Élise Müller, Université du Luxembourg"
    path = _with_recipient(tmp_path, recipient)
    argv = ("inspect", str(path), "--password-file", str(_password_file(tmp_path)))
    # a console of any code page: ASCII, each other character as its escape
    code, out, _ = _run(*argv)
    assert code == OK and _narrow_ascii(out)
    assert "  Sent to: \\xc9lise M\\xfcller, Universit\\xe9 du Luxembourg" in out.splitlines()
    # a terminal that writes UTF-8: as itself
    code, shown = _run_utf8(*argv)
    assert code == OK and f"  Sent to: {recipient}" in shown.splitlines()


def test_a_text_wider_than_a_line_is_cut_between_two_characters(tmp_path: Path) -> None:
    """A recipient written in a script without spaces: as escapes or as itself, no line is wider than 78 columns,
    and no escape is cut in two."""
    recipient = chr(0x4E2D) * 45  # 45 ideographs: 90 columns as themselves, 270 as escapes
    path = _with_recipient(tmp_path, recipient)
    argv = ("inspect", str(path), "--password-file", str(_password_file(tmp_path)))
    code, out, _ = _run(*argv)
    assert code == OK and _narrow_ascii(out)
    lines = [line for line in out.splitlines() if "\\u4e2d" in line]
    assert (
        lines[0].startswith("  Sent to: \\u4e2d") and len(lines) == 3
    )  # it goes on from the label's line
    escaped = "".join(line.strip() for line in lines)
    # whole escapes only, up to the 200 columns a text may take as escapes (whole on a UTF-8 terminal, below)
    assert escaped == "Sent to: " + "\\u4e2d" * 32 + "..."
    code, shown = _run_utf8(*argv)
    assert code == OK and _narrow(shown)
    assert sum(line.count(chr(0x4E2D)) for line in shown.splitlines()) == 45
    # a word that is not a typed text is never cut: the hash stays whole on its line
    assert commandline.wrapped("x " + "a" * 100) == ["x", "  " + "a" * 100]


def test_a_file_s_name_is_shown_as_any_text_that_is_not_the_formats(tmp_path: Path) -> None:
    hidden = (
        "study" + chr(0x202E) + "piz.cstudio.zip"
    )  # a direction mark: the name would read "study.zip..."
    path = _package(tmp_path, name=hidden)
    code, out, _ = _run("inspect", str(path))
    assert code == OK and out.splitlines()[0].startswith("study?piz.cstudio.zip, ")
    # the JSON names the file one way, whatever the stream: the page's form, escaped by JSON
    code, out, _ = _run("inspect", str(path), "--json")
    assert json.loads(out)["file"]["name"] == "study" + pk.REPLACED + "piz.cstudio.zip"


# ----------------------------------------------------------------------------- the hash the sender gave


def test_the_hash_given_is_compared_before_the_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, package_hash = made._write()
    path = _package(tmp_path, data)
    secret = str(_password_file(tmp_path))
    spoken = package_hash[:16]
    code, out, _ = _run(
        "inspect", str(path), "--package-hash", spoken.upper(), "--password-file", secret
    )
    assert code == OK and "  Its first 16 characters are those you gave." in out
    grouped = " ".join(package_hash[at : at + 4] for at in range(0, 24, 4))
    code, out, _ = _run("inspect", str(path), "--package-hash", grouped)
    assert code == OK and "  Its first 24 characters are those you gave." in out
    code, out, _ = _run("inspect", str(path), "--package-hash", package_hash)
    assert code == OK and "  It is the hash you gave." in out

    # another hash: no password is read, typed or tried, not even a password file that does not exist
    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("no password is tried on a package that is not the sender's")

    monkeypatch.setattr(pz, "read_package", never)
    monkeypatch.setattr(inspection, "password_from_file", never)
    slip = package_hash[:40] + ("0" if package_hash[40] != "0" else "1") + package_hash[41:]
    for how in (("--password-file", str(tmp_path / "nothing.txt")), ("--ask-password",)):
        code, out, err = _run(
            "inspect", str(path), "--package-hash", slip, *how, terminal=True, typed=never
        )
        assert code == NOT_SENT
        lines = out.splitlines()
        # the hash given under the package's own, with where they part and the likeliest cause first
        at = lines.index(f"  {package_hash}")
        assert lines[at + 1] == "The hash you gave differs from it at character 41:"
        assert lines[at + 2] == f"  {slip}"
        assert "Check it against what the sender sent you" in out
        assert err.startswith("Stopped before the password: the package hash is not the one you")
        assert "Study:" not in out and "--password-file" not in err and _narrow_ascii(out + err)


@pytest.mark.parametrize(
    ("typed", "read"),
    [
        ("3f2a-9c1e-4b7d-0a6c", "3f2a9c1e4b7d0a6c"),
        ("3F2A 9C1E 4B7D 0A6C 5E8F", "3f2a9c1e4b7d0a6c5e8f"),
        ("a" * 64, "a" * 64),
        ("3f2a-9c1e-4b7d-0a6", None),  # 15 characters
        ("a" * 65, None),
        ("3f2a-9c1e-4b7d-0a6g", None),
        (
            made.PASSWORD,
            None,
        ),  # a password pasted in the hash's place is no hash: never written back
        ("", None),
    ],
)
def test_a_hash_is_16_to_64_hexadecimal_characters(typed: str, read: str | None) -> None:
    assert inspection.hash_given(typed) == read


def test_where_two_hashes_part() -> None:
    assert inspection.differs_at("abcd", "abcdef") is None
    assert inspection.differs_at("abxd", "abcdef") == 3
    assert inspection.differs_at("0bcd", "abcdef") == 1


# ----------------------------------------------------------------------------- the password


def test_a_password_that_does_not_open_the_package_reads_nothing(tmp_path: Path) -> None:
    path = _package(tmp_path)
    code, out, err = _run(
        "inspect", str(path), "--password-file", str(_password_file(tmp_path, made.WRONG))
    )
    assert code == NOT_READ and "Study:" not in out
    assert err.startswith("The password does not open this package; nothing was read from it.")
    assert "never in the message that carries the file" in " ".join(err.split())
    # not the shape of a password: nothing is tried, and the words point at the file's line
    code, out, err = _run(
        "inspect", str(path), "--password-file", str(_password_file(tmp_path, "hunter2"))
    )
    said = " ".join(err.split())
    assert code == NOT_READ
    assert said.startswith(
        "--password-file: its first line is not a package's password (nothing was tried)."
    )
    assert "20 letters and digits in 5 groups of 4" in said and "hunter2" not in out + err
    assert _narrow_ascii(err)


def test_a_password_file_that_cannot_be_read_is_said_without_its_path(tmp_path: Path) -> None:
    path = _package(tmp_path)
    missing = tmp_path / "zq-account-55" / "password.txt"
    code, out, err = _run("inspect", str(path), "--password-file", str(missing))
    # after the clear part: the hash is printed whatever the password's file is
    assert code == NOT_READ and "Package hash" in out and "Study:" not in out
    assert err == "--password-file: the file cannot be read (FileNotFoundError).\n"
    empty = tmp_path / "empty.txt"
    empty.write_text("\n" + made.PASSWORD + "\n", encoding="ascii")
    code, _, err = _run("inspect", str(path), "--password-file", str(empty))
    assert code == NOT_READ and err.startswith("--password-file: its first line is empty.\n")


@pytest.mark.parametrize(
    "written",
    [
        made.PASSWORD.encode("ascii"),  # no line end
        (made.PASSWORD + "\r\nkeep this file to yourself\r\n").encode("ascii"),
        codecs.BOM_UTF8 + (made.PASSWORD + "\r\n").encode("utf-8"),  # Notepad's "UTF-8 with BOM"
        codecs.BOM_UTF16_LE
        + (made.PASSWORD + "\r\n").encode("utf-16-le"),  # Windows PowerShell's ">"
        codecs.BOM_UTF16_BE + (made.PASSWORD + "\n").encode("utf-16-be"),
    ],
)
def test_a_password_file_is_read_as_the_tool_that_wrote_it_wrote_it(
    tmp_path: Path, written: bytes
) -> None:
    """Refused, such a file would read "not a password", and blame the password the sender gave."""
    path = _package(tmp_path)
    secret = tmp_path / "password.txt"
    secret.write_bytes(written)
    assert commandline.password_from_file(str(secret)) == (made.PASSWORD, "")
    code, out, _ = _run("inspect", str(path), "--password-file", str(secret))
    assert code == OK and "Study: tutorial_cohort, revision 3" in out


@pytest.mark.skipif(
    not hasattr(os, "getuid"), reason="Windows: a file's rights are its folder's list"
)
def test_a_password_file_others_can_read_is_said(tmp_path: Path) -> None:
    path = _package(tmp_path)
    secret = _password_file(tmp_path)
    secret.chmod(0o644)
    code, out, err = _run("inspect", str(path), "--password-file", str(secret))
    assert code == OK and "Study:" in out
    assert err.startswith("Note: other users of this computer can read the password's file.")
    assert str(tmp_path) not in err and _narrow_ascii(err)
    secret.chmod(0o600)
    assert _run("inspect", str(path), "--password-file", str(secret))[2] == ""


def _answers(*given: str) -> Any:
    left = list(given)
    prompts: list[str] = []

    def typed(prompt: str, err: Any) -> str:
        prompts.append(prompt)
        return left.pop(0)

    typed.prompts = prompts  # type: ignore[attr-defined]
    return typed


def test_the_password_is_asked_for_three_times_at_most(tmp_path: Path) -> None:
    path = _package(tmp_path)
    # a wrong one, one of another shape, then the right one
    typed = _answers(made.WRONG, "hunter2", made.PASSWORD)
    code, out, err = _run("inspect", str(path), "--ask-password", terminal=True, typed=typed)
    assert code == OK and "Study: tutorial_cohort, revision 3" in out
    assert typed.prompts == [commandline.PROMPT] * 3 and "each key shows as *" in commandline.PROMPT
    assert "That password does not open this package: 2 tries left." in err
    assert "That is not a password of a CohortStudio package: 1 try left." in err
    assert "hunter2" not in out + err
    # three that do not open it: not read
    typed = _answers(made.WRONG, made.WRONG, made.WRONG)
    code, out, err = _run("inspect", str(path), "--ask-password", terminal=True, typed=typed)
    assert code == NOT_READ and len(typed.prompts) == 3 and "Study:" not in out
    assert "The password does not open this package; nothing was read from it." in err
    # Enter: stops with the clear part, which read
    typed = _answers("  ")
    code, out, err = _run("inspect", str(path), "--ask-password", terminal=True, typed=typed)
    assert code == OK and "Not opened: 8 files are encrypted" in out and err == ""
    assert _narrow_ascii(out + err)


def test_no_password_is_asked_where_nobody_can_type_one_unseen(tmp_path: Path) -> None:
    path = _package(tmp_path)

    def never(prompt: str, err: Any) -> str:
        raise AssertionError("asked without a terminal")

    code, out, err = _run("inspect", str(path), "--ask-password", terminal=False, typed=never)
    assert code == USAGE and out == ""
    assert err == _said(commandline.no_console_words()) and "--password-file FILE" in err

    def shown(prompt: str, err: Any) -> str:  # a terminal that cannot hide what is typed
        raise commandline.CannotAsk

    code, out, err = _run("inspect", str(path), "--ask-password", terminal=True, typed=shown)
    assert code == USAGE and "Study:" not in out
    assert "This terminal cannot hide what is typed: the password was not asked for." in err


def test_a_terminal_is_one_somebody_can_type_on() -> None:
    assert not commandline.at_a_terminal(None)
    assert not commandline.at_a_terminal(io.StringIO("typed"))

    class Says:  # says it is a terminal, as the null device does on Windows, and is no console
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            raise OSError("no file")

    assert commandline.at_a_terminal(Says()) == (os.name != "nt")
    with open(os.devnull) as nobody:
        assert not commandline.at_a_terminal(nobody) or os.name != "nt"


# ----------------------------------------------------------------------------- what is refused


def test_a_file_that_is_no_package_is_refused_in_words_that_quote_nothing(tmp_path: Path) -> None:
    cut = _package(tmp_path, made._write()[0][:400], "cut.cstudio.zip")
    code, out, err = _run("inspect", str(cut))
    assert code == NOT_READ and out == ""
    assert err.startswith("cut.cstudio.zip is not a complete zip file")
    # a zip that holds a file no package holds: counted in no word, named in none
    whole = made._complete(made._shareable())
    listed = {name: member for name, member in whole.items() if name != pk.MANIFEST}
    foreign = made._stdlib_zip({**listed, CANARY + ".txt": b"x", pk.MANIFEST: whole[pk.MANIFEST]})
    code, out, err = _run("inspect", str(_package(tmp_path, foreign, "foreign.cstudio.zip")))
    assert code == NOT_READ and out == ""
    assert "is not a CohortStudio study package" in err and CANARY not in err
    assert _narrow_ascii(err)


def test_what_was_typed_in_the_files_place_is_not_written_back(tmp_path: Path) -> None:
    """A file that is not found is not named: a password pasted where the file goes would be on the terminal, and
    in the message a recipient sends the sender. A folder that exists is nobody's password: it is named."""
    for typed in (
        made.PASSWORD,
        str(tmp_path / made.PASSWORD),
        str(tmp_path / "nothing.cstudio.zip"),
    ):
        code, out, err = _run("inspect", typed)
        assert code == NOT_READ and out == ""
        assert err.startswith("The package file was not found. What you typed is not repeated here")
        assert made.PASSWORD not in err and "nothing.cstudio.zip" not in err
        assert "whole path, in quotes" in " ".join(err.split()) and _narrow_ascii(err)
        facts = json.loads(_run("inspect", typed, "--json")[1])
        assert (
            facts["exit"] == NOT_READ
            and "file" not in facts
            and made.PASSWORD not in json.dumps(facts)
        )
    folder = tmp_path / "study_r3_with_data.cstudio"
    folder.mkdir()
    code, out, err = _run("inspect", str(folder))
    assert code == NOT_READ and out == ""
    assert err.startswith(
        "study_r3_with_data.cstudio is a folder: give the .cstudio.zip file itself"
    )


def test_a_member_that_is_not_the_senders_is_named_and_nothing_more_is_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made._listing_wrong(monkeypatch, pk.RESULTS)
    data = made._crafted(monkeypatch, made._members())
    path = _package(tmp_path, data)
    code, out, err = _run("inspect", str(path), "--password-file", str(_password_file(tmp_path)))
    assert code == NOT_SENT
    assert "results.json does not match its checksum in MANIFEST.sha256" in err
    assert "Nothing more is shown from it." in err and "Study:" not in out
    assert _narrow_ascii(out + err)
    # without the password, the clear part still reads: the member is encrypted, and was not checked
    code, out, err = _run("inspect", str(path))
    assert code == OK and err == ""


def test_every_refusal_of_the_reader_has_its_words() -> None:
    for refusal in pz.REFUSALS:
        for reason in (pk.NOT_JSON, pk.NOT_A_POLICY, ""):
            refused = pz.Refused(refusal, pk.RESULTS, reason)
            for opened in (False, True):
                said = inspection.refusal_words(refused, "a.cstudio.zip", 10, opened=opened)
                assert said.isascii() and said.endswith(".") and "  " not in said
                assert "((" not in said and "))" not in said
                if refusal not in (
                    pz.NO_SPACE,
                    pz.FOLDER,
                ):  # of a reading that writes the data (verify)
                    assert f"({refusal})" not in said, refusal
    # a later version's package: no version this reader cannot name, the same word as the clear part ("format")
    later = pz.Refused(pz.LATER_VERSION, pk.NOTICE, pk.later_schema(2, 1))
    said = inspection.refusal_words(later, "a.cstudio.zip", 10)
    assert said.startswith(
        "This package was made by a later version of CohortStudio, in a format this version"
    )
    assert "Install the latest version of CohortStudio" in said and "schema" not in said
    inside = pz.Refused(pz.LATER_VERSION, pk.RESULTS, pk.later_keys(2))
    assert inspection.refusal_words(inside, "a.cstudio.zip", 10, opened=True).startswith(
        "results.json was written by a later version of CohortStudio than this one."
    )
    # a reason that begins "its ..." follows the member's name and a colon
    policy = pz.Refused(pz.NOT_READ, pk.PACKAGE, pk.NOT_A_POLICY)
    assert inspection.refusal_words(policy, "a.cstudio.zip", 10, opened=True).startswith(
        "package.json: its policy cannot be read by this version."
    )
    # too large: the file itself, or what its files declare
    large = pz.Refused(pz.TOO_LARGE)
    assert "is larger than a CohortStudio study package can be" in inspection.refusal_words(
        large, "a.cstudio.zip", pz.MOST_FILE_BYTES + 1
    )
    assert inspection.refusal_words(large, "a.cstudio.zip", 10).startswith(
        "The files of this zip would unpack to more than a CohortStudio study package holds"
    )
    # a password of another shape, by where it came from
    shape = pz.Refused(pz.NOT_A_PASSWORD)
    assert inspection.refusal_words(shape, "a", 1, from_file=True).startswith(
        "--password-file: its first line"
    )
    assert inspection.refusal_words(shape, "a", 1).startswith(
        "That is not a password of a CohortStudio"
    )


def test_the_words_know_every_closed_value_of_the_format() -> None:
    """A basis, a notice, a role or a member the format gains must get its words here: a missing one would stop
    the command on a package that is sound."""
    assert set(inspection._BASIS_WORDS) == set(pk.BASES)
    assert set(inspection._NOTICE_WORDS) == set(pk.NOTICES)
    travelling = set(pk.FILE_ROLES) - {
        "replayed_state"
    }  # a state replayed from another trace never travels
    assert set(inspection._DATA_WORDS) == travelling
    members = {name for tier in pk.TIERS for name in pk.members_of(tier)}
    assert set(inspection._MEMBER_WORDS) == members
    index = made._index(pk.anchor_of([made.SOURCE]))
    for reason in pk.NOT_PEOPLE_REASONS:
        for level in pk.LEVELS:
            said = inspection._policy_words(
                pk.Policy(rules=pk.NONE, rows_not_people=(reason, level))
            )
            assert "declared not people" in said and said.isascii()
    assert inspection._policy_words(index.policy).endswith("the package holds the data themselves.")


# ----------------------------------------------------------------------------- the same facts as JSON


def test_the_same_facts_as_json(tmp_path: Path) -> None:
    data, package_hash = made._write()
    path = _package(tmp_path, data)
    code, out, err = _run(
        "inspect", str(path), "--json", "--password-file", str(_password_file(tmp_path))
    )
    facts = json.loads(out)
    assert code == OK == facts["exit"] and err == "" and out.isascii()
    assert facts["file"] == {"name": path.name, "bytes": len(data)}
    assert facts["package"]["hash"] == package_hash and facts["package"]["tier"] == pk.REPRODUCTION
    assert facts["package"]["notice"] == pk.NOTICE_PERSONAL_DATA and facts["opened"] is True
    assert [entry["name"] for entry in facts["package"]["files"]][:2] == [pk.NOTICE, pk.PACKAGE]
    assert facts["index"]["declaration"]["recipient"] == made.RECIPIENT
    assert facts["index"]["study"]["id"] == "tutorial_cohort"
    # the clear part alone, a hash that differs, a refusal: the same keys say so
    code, out, _ = _run("inspect", str(path), "--json")
    facts = json.loads(out)
    assert code == 0 and facts["opened"] is False and "index" not in facts
    assert facts["hash_given"] is None
    code, out, _ = _run("inspect", str(path), "--json", "--package-hash", "0" * 16)
    facts = json.loads(out)
    assert code == facts["exit"] == 1
    assert facts["hash_given"] == {"characters": 16, "matches": False, "differs_at": 1}
    code, out, _ = _run(
        "inspect", str(path), "--json", "--password-file", str(_password_file(tmp_path, made.WRONG))
    )
    facts = json.loads(out)
    assert code == facts["exit"] == 2
    assert facts["refused"] == {"id": pz.WRONG_PASSWORD, "member": "", "reason": ""}


# ----------------------------------------------------------------------------- the command line itself


def test_nothing_typed_on_the_command_line_is_written_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _package(tmp_path)
    for argv in (["inspect"], []):  # argparse's own words where they name no value
        with pytest.raises(SystemExit) as stopped:
            main(argv)
        said = capsys.readouterr().err
        assert stopped.value.code == USAGE and "the following arguments are required" in said
        assert _narrow_ascii(said)
    # a password, wherever it is typed: no option takes one (no abbreviation of --password-file either), and
    # argparse's "invalid choice: '...'" and "unrecognized arguments: ..." never reach the terminal
    for argv in (
        ["inspect", str(path), "--password", made.PASSWORD],
        ["inspect", str(path), "--pass", made.PASSWORD],
        ["inspect", str(path), made.PASSWORD],
        [made.PASSWORD, "inspect", str(path)],
        ["--password", made.PASSWORD, "inspect", str(path)],
        [made.PASSWORD],
        ["inspect", str(path), f"--json={made.PASSWORD}"],
    ):
        with pytest.raises(SystemExit) as stopped:
            main(argv)
        said = capsys.readouterr().err
        assert stopped.value.code == USAGE, argv
        assert made.PASSWORD not in said, argv
        assert "The password is never an argument" in " ".join(said.split())
        assert "goes in quotes" in " ".join(said.split()) and _narrow_ascii(said)
    code, out, err = _run("inspect", str(path), "--package-hash", made.PASSWORD)
    assert code == USAGE and out == "" and made.PASSWORD not in err
    assert err.startswith("--package-hash takes the 16 to 64")
    with pytest.raises(SystemExit) as stopped:
        main(["inspect", "--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "--password-file FILE" in shown and "Exit codes: 0 read" in shown
    assert "showing * for each key" in " ".join(shown.split()) and _narrow_ascii(shown)


def test_a_command_that_was_not_understood_ends_with_the_way_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A misspelt command or option is not written back, and argparse's "(choose from ...)" goes with it: the usage
    above the error and the help command under it, on a line of its own, are the way back to what the command
    takes."""
    path = _package(tmp_path)
    way_back = "  Its options are listed by:\n    python -m manage.replay inspect -h\n"
    # a misspelt option: refused by the command's own parser, whose usage names the option as it is spelt
    with pytest.raises(SystemExit) as stopped:
        main(["inspect", str(path), "--package-has", "3f2a9c1e4b7d0a6c"])
    said = capsys.readouterr().err
    assert stopped.value.code == USAGE
    assert said.startswith("usage: python -m manage.replay inspect [-h]")
    assert "[--package-hash HASH]" in said and said.endswith(way_back)
    assert "--package-has" not in said.replace("--package-hash", "")
    assert "3f2a9c1e4b7d0a6c" not in said and str(tmp_path) not in said and _narrow_ascii(said)
    # a misspelt command: the usage is the command line's own, and the help command names the command as it is spelt
    with pytest.raises(SystemExit) as stopped:
        main(["inpsect", str(path)])
    said = capsys.readouterr().err
    assert stopped.value.code == USAGE
    assert said.startswith("usage: python -m manage.replay [-h] COMMAND ...\n")
    assert said.endswith(way_back) and "inpsect" not in said and _narrow_ascii(said)
    # argparse's own words, where they name no value, end the same way
    with pytest.raises(SystemExit):
        main(["inspect"])
    said = capsys.readouterr().err
    assert said.endswith("error: the following arguments are required: PACKAGE\n" + way_back)
    # one line for each command once there are several; none for a parser that names no command
    assert commandline.way_back(["one -h", "two -h"]) == [
        "  The options of each command are listed by:",
        "    one -h",
        "    two -h",
    ]
    assert commandline.way_back([]) == []


def test_only_argparse_messages_that_name_no_value_are_written() -> None:
    for safe in (
        "the following arguments are required: PACKAGE",
        "the following arguments are required: COMMAND",
        "argument --package-hash: expected one argument",
        "argument --password-file: not allowed with argument --ask-password",
    ):
        assert commandline.usage_words(safe) == safe
    for typed in (
        f"unrecognized arguments: {made.PASSWORD}",
        f"argument COMMAND: invalid choice: '{made.PASSWORD}' (choose from inspect)",
        f"argument --json: ignored explicit argument '{made.PASSWORD}'",
        f"the following arguments are required: {made.PASSWORD}",
        "a message of a later argparse",
    ):
        assert commandline.usage_words(typed) == commandline.NOT_UNDERSTOOD


@pytest.mark.parametrize(
    "error",
    [KeyError(CANARY), UnicodeDecodeError("utf-8", CANARY.encode("ascii"), 0, 1, CANARY)],
    ids=["KeyError", "UnicodeDecodeError"],
)
def test_an_error_of_the_program_is_named_by_its_kind_and_exits_70(
    error: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not 2: a script would read "not read", or "the wrong password"."""
    path = _package(tmp_path)
    kind = type(error).__name__

    def fails(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(pz, "open_package", fails)
    out, err = io.StringIO(), io.StringIO()
    assert main(["inspect", str(path)], out=out, err=err) == FAILED
    assert err.getvalue() == _said(
        f"The reading stopped ({kind}): an error of the program, not of the package. Nothing was changed. "
        "Please report it with the lines above."
    )
    assert len(err.getvalue().splitlines()) == 2 and _narrow_ascii(err.getvalue())
    assert CANARY not in out.getvalue() + err.getvalue() and out.getvalue() == ""
    # a program that reads the facts gets an object on this path too
    out = io.StringIO()
    assert main(["inspect", str(path), "--json"], out=out, err=io.StringIO()) == FAILED
    assert json.loads(out.getvalue()) == {"exit": FAILED, "failed": kind}


def _fresh(argv: list[str], *, stdin: Any = subprocess.DEVNULL) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    return subprocess.run(
        [sys.executable, "-B", *argv],
        cwd=ROOT,
        env=env,
        stdin=stdin,
        capture_output=True,
        text=True,
        timeout=120,
    )


LOADED = """
import sys
from manage.replay.__main__ import main
code = main(sys.argv[1:])
ours = ("app_pages", "biostats", "core", "enrich", "explore", "graph", "ingest", "manage", "monitor", "prompts",
        "ui", "utils", "main")
heavy = ("numpy", "pandas", "pydantic", "pydantic_settings", "streamlit", "scipy", "sklearn", "pyzipper", "dotenv")
first = sorted(name for name in sys.modules if name.split(".")[0] in ours)
third = sorted(name for name in heavy if name in sys.modules)
print("LOADED", code, ",".join(first), "|", ",".join(third))
"""
# every module of the application a run may load: a new one is added here on purpose, never by an import's chain
ALLOWED = (
    "core,core.package,core.privacy,manage,manage.package_zip,manage.replay,manage.replay.__main__,"
    "manage.replay.commandline,manage.replay.inspection,utils,utils.environment"
)


def test_the_command_loads_no_engine_and_no_setting(tmp_path: Path) -> None:
    """A recipient checks a package's hash on a computer where only this command has run: neither the replay's
    engine nor its libraries are loaded, no setting is read, and the library that decrypts is loaded only once a
    password is given."""
    path = _package(tmp_path)
    clear = _fresh(["-c", LOADED, "inspect", str(path)])
    assert clear.returncode == 0, clear.stderr[-2000:]
    assert clear.stdout.splitlines()[-1] == f"LOADED 0 {ALLOWED} | "
    opened = _fresh(
        ["-c", LOADED, "inspect", str(path), "--password-file", str(_password_file(tmp_path))]
    )
    assert opened.returncode == 0, opened.stderr[-2000:]
    assert opened.stdout.splitlines()[-1] == f"LOADED 0 {ALLOWED} | pyzipper"


def test_the_command_never_waits_for_a_password_nobody_types(tmp_path: Path) -> None:
    """Run as ``python -m manage.replay`` with no input (a script, a scheduled task): on Windows the null device
    says it is a terminal, and the console's own keys were read whatever the input was."""
    path = _package(tmp_path)
    asked = _fresh(["-m", "manage.replay", "inspect", str(path), "--ask-password"])
    assert asked.returncode == USAGE and asked.stdout == ""
    assert asked.stderr == _said(commandline.no_console_words())
    plain = _fresh(["-m", "manage.replay", "inspect", str(path)])
    assert plain.returncode == 0 and "Not opened: 8 files are encrypted" in plain.stdout
    assert _narrow_ascii(plain.stdout) and plain.stderr == ""
    # a password where the command goes, in the real command: exit 64, and not a character of it
    typed = _fresh(["-m", "manage.replay", made.PASSWORD, "inspect", str(path)])
    assert typed.returncode == USAGE and made.PASSWORD not in typed.stdout + typed.stderr
