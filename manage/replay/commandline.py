"""
What every command of ``python -m manage.replay`` shares: the exit codes, the parser that never writes back what was
typed, the password read from a file or typed without being shown, and the lines of 78 columns.

Standard library and ``manage.package_zip``'s exit codes only: a command's own module (``manage.replay.inspection``;
later the one that replays a package) is imported when that command runs, so that reading a package's hash never
loads what a replay needs.

**The password is never an argument**, and nothing typed on the command line is written back: a command line is
kept by the shell, and an error message is what a recipient pastes into an e-mail to the sender. argparse's own
messages repeat what they refuse ("invalid choice: '...'", "unrecognized arguments: ..."): only those of a known
form, which name options and never values, are written; any other becomes fixed words.
"""

from __future__ import annotations

import argparse
import codecs
import getpass
import os
import re
import sys
import unicodedata
import warnings
from collections.abc import Sequence
from typing import Any, NoReturn, TextIO

from manage import package_zip as pz

EXIT_OK, EXIT_NOT_WHAT_WAS_SENT, EXIT_NOT_READ = (
    pz.EXIT_OK,
    pz.EXIT_NOT_WHAT_WAS_SENT,
    pz.EXIT_NOT_READ,
)
# sysexits' numbers for what is not about the package: the command was not understood or cannot run here
# (argparse's own 2 would read "not read"), and an error of the program itself (2 would read "the wrong password"
# to a script)
EXIT_USAGE, EXIT_FAILED = 64, 70
EXIT_STOPPED = 130  # Ctrl-C, as a shell reports it
WIDTH = 78
PROGRAM = "python -m manage.replay"

# ----------------------------------------------------------------------------- lines of 78 columns

# what is never cut in two: an escape as core.package.shown writes one, a doubled backslash, else one character
_TOKEN = re.compile(r"\\\\|\\x[0-9a-f]{2}|\\u[0-9a-f]{4}|\\U[0-9a-f]{8}|.", re.DOTALL)


def columns(text: str) -> int:
    """The columns ``text`` takes on a terminal: two for a wide character (an ideograph, a kana)."""
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def wrapped(text: str, indent: int = 0, *, breakable: bool = False) -> list[str]:
    """
    ``text`` in lines of ``WIDTH`` columns at most, the first indented by ``indent`` and the others by two more.
    A word longer than a line stays whole on a line of its own (the hash, a command to copy), unless ``breakable``:
    a text someone typed (a recipient's name in a script without spaces, or shown as escapes) then goes on from
    the line begun, cut between two characters or two escapes, never inside one.
    """
    lines: list[str] = []
    follow = " " * (indent + 2)
    prefix, current = " " * indent, ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if columns(prefix + candidate) <= WIDTH:
            current = candidate
            continue
        if breakable and columns(word) > WIDTH - len(follow):
            line = f"{current} " if current else ""
            for token in _TOKEN.findall(word):
                if line.strip() and columns(prefix + line + token) > WIDTH:
                    lines.append(prefix + line.rstrip())
                    prefix, line = follow, ""
                line += token
            current = line
            continue
        if current:
            lines.append(prefix + current)
            prefix = follow
        current = word
    if current:
        lines.append(prefix + current)
    return lines


# ----------------------------------------------------------------------------- the parser

NOT_UNDERSTOOD = (
    "the command was not understood. What was typed is not repeated here: it could be a password. A path or a hash "
    "that holds a space goes in quotes. The password is never an argument: use --ask-password, or --password-file "
    "FILE."
)
# argparse's messages that name options and arguments by their own names, never by what was typed
_OPTION = r"-{1,2}[a-z][a-z-]*"
_SAFE = (
    re.compile(r"the following arguments are required: [A-Z][A-Z_]*(?:, [A-Z][A-Z_]*)*"),
    re.compile(rf"argument {_OPTION}: expected one argument"),
    re.compile(rf"argument {_OPTION}: not allowed with argument {_OPTION}"),
)


def usage_words(message: str) -> str:
    """What is written of an argparse error: itself when its form names no value, else fixed words."""
    return message if any(form.fullmatch(message) for form in _SAFE) else NOT_UNDERSTOOD


def way_back(listed_by: Sequence[str]) -> list[str]:
    """
    The lines that end a usage error: the command that lists the options, on a line of its own so that a copy keeps
    it whole. A misspelt command or option is not written back, and argparse's "(choose from ...)" goes with it:
    these lines are the way back to what a command takes.
    """
    if not listed_by:
        return []
    first = (
        "Its options are listed by:"
        if len(listed_by) == 1
        else "The options of each command are listed by:"
    )
    return [f"  {first}", *(f"    {command}" for command in listed_by)]


class Parser(argparse.ArgumentParser):
    """
    A parser whose errors exit 64, never repeat what was typed, and end with the command that lists the options
    (``listed_by``: its own for a command's parser, every command's for the parser of the command line); its help
    is 78 columns wide. No abbreviation: ``--password X`` would read as ``--password-file``.
    """

    listed_by: tuple[str, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        kwargs.setdefault("formatter_class", _help)
        super().__init__(*args, **kwargs)

    def parse_known_args(
        self, args: Sequence[str] | None = None, namespace: Any = None
    ) -> tuple[Any, list[str]]:
        """What this parser does not take, it refuses itself. argparse hands what a command did not take back to
        the parser of the command line, whose usage names none of the command's options: refused here, the usage
        above the error is the command's own, where a misspelt option is found again."""
        namespace, rest = super().parse_known_args(args, namespace)
        if rest:
            self.error("unrecognized arguments")  # never the arguments themselves
        return namespace, rest

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        said = [*wrapped(f"error: {usage_words(message)}"), *way_back(self.listed_by)]
        self.exit(EXIT_USAGE, "".join(line + "\n" for line in said))


def _help(prog: str) -> argparse.HelpFormatter:
    return argparse.HelpFormatter(prog, width=WIDTH)  # whatever the terminal's own width


# ----------------------------------------------------------------------------- the password

PROMPT = "Password (each key shows as *; Enter alone stops here): "


def at_a_terminal(stream: Any) -> bool:
    """
    Whether somebody can type on ``stream`` (the standard input): a terminal, and on Windows a console. There the
    null device says it is a terminal too, and ``getpass`` reads the console's keys whatever the input is: asked
    without this check, a command run by a script with no input waits for a password nobody types.
    """
    if stream is None:
        return False
    try:
        if not stream.isatty():
            return False
        if os.name != "nt":
            return True
        import ctypes
        import msvcrt

        mode = ctypes.c_ulong()
        handle = msvcrt.get_osfhandle(stream.fileno())
        return bool(
            ctypes.windll.kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode))
        )
    except OSError, ValueError, AttributeError:
        return False


def no_console_words() -> str:
    """What ``--ask-password`` says where nobody can type a password unseen."""
    if os.name == "nt":
        return (
            "--ask-password needs a console that can hide what is typed, and this window is not one (Git Bash's is "
            "not): run it in PowerShell or the Command Prompt, or use --password-file FILE."
        )
    return (
        "--ask-password needs a terminal that can hide what is typed, and this is not one: run it in a terminal, "
        "or use --password-file FILE."
    )


class CannotAsk(Exception):
    """The terminal cannot hide what is typed, or there is none."""


def typed_unseen(prompt: str, err: TextIO) -> str:
    """
    A password typed on the terminal, each key shown as ``*`` (a prompt that shows nothing reads as a window that
    froze; the length it shows is the format's own, which is public). ``CannotAsk`` where what is typed would be
    shown (``getpass`` then only warns, and echoes).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass(prompt, stream=err, echo_char="*")
        except (getpass.GetPassWarning, EOFError) as cannot:
            raise CannotAsk from cannot


_MARKS = (
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)
_FILE_MOST = 4096  # a password is 24 characters: nothing of a larger file is read


def password_from_file(path: str) -> tuple[str | None, str]:
    """
    The password a file's first line holds, or None with why (in words that name no path and quote nothing). The
    file is read as its byte order mark says: Notepad's "UTF-8 with BOM" and the UTF-16 that Windows PowerShell's
    ``>`` writes are a password as much as plain text is, and refusing them would blame the password itself.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read(_FILE_MOST)
    except OSError as error:
        return None, f"--password-file: the file cannot be read ({type(error).__name__})."
    encoding = "utf-8"
    for mark, named in _MARKS:
        if data.startswith(mark):
            data, encoding = data[len(mark) :], named
            break
    lines = data.decode(encoding, errors="replace").splitlines()
    first = lines[0].strip() if lines else ""
    if not first:
        return None, "--password-file: its first line is empty."
    return first, ""


def password_file_note(path: str) -> str:
    """What is said of a password file that other users of the computer can read ("" when they cannot, and on a
    system without such rights): its path is not named."""
    if not hasattr(os, "getuid"):
        return ""
    try:
        open_to_others = bool(os.stat(path).st_mode & 0o044)  # read by its group, or by everyone
    except OSError:
        return ""
    if not open_to_others:
        return ""
    return (
        "Note: other users of this computer can read the password's file. Keep it to yourself (chmod 600), away "
        "from the package, and delete it when you are done."
    )


__all__ = [
    "EXIT_FAILED",
    "EXIT_NOT_READ",
    "EXIT_NOT_WHAT_WAS_SENT",
    "EXIT_OK",
    "EXIT_STOPPED",
    "EXIT_USAGE",
    "NOT_UNDERSTOOD",
    "PROGRAM",
    "PROMPT",
    "WIDTH",
    "CannotAsk",
    "Parser",
    "at_a_terminal",
    "columns",
    "no_console_words",
    "password_file_note",
    "password_from_file",
    "typed_unseen",
    "usage_words",
    "way_back",
    "wrapped",
]
