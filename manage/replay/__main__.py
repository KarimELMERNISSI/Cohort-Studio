"""
``python -m manage.replay``: the command line of a study package.

- ``inspect PACKAGE`` says what a received package holds, without replaying it (:mod:`manage.replay.inspection`):
  its hash before any password, its files, and with its password what its index says of it.
- ``verify`` (the replay of a package against what it recorded) comes with the package's own replay.

Each command names its arguments here and its module in ``_COMMANDS``; the module is imported when the command runs,
so that reading a package's hash loads nothing a replay needs (the engine, numpy, pandas, pydantic, Streamlit, a
setting or ``.env``): a recipient checks a hash on a computer where only this command has run. What every command
shares is :mod:`manage.replay.commandline`: a command that was not understood exits 64 (not argparse's 2, which a
reader gives a file it did not read), and nothing typed is written back.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from typing import TextIO

from manage.replay.commandline import (
    EXIT_FAILED,
    EXIT_STOPPED,
    PROGRAM,
    Parser,
    at_a_terminal,
    wrapped,
)

_INSPECT_EPILOG = (
    "Exit codes: 0 read, and nothing failed; 1 read, but not the package its sender made; 2 not read; 64 the "
    "command was not understood, or cannot run here; 70 an error of the program. The password is never an "
    "argument. Nothing is written to disk."
)


def _inspect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "package", metavar="PACKAGE", help="the .cstudio.zip file (its whole path, in quotes)"
    )
    parser.add_argument(
        "--package-hash",
        metavar="HASH",
        help="the package hash the sender gave you apart from the file (its first 16 characters are enough)",
    )
    how = parser.add_mutually_exclusive_group()
    how.add_argument(
        "--ask-password",
        action="store_true",
        help="ask for the password on this terminal, showing * for each key (three tries)",
    )
    how.add_argument(
        "--password-file",
        metavar="FILE",
        help="read the password from the first line of FILE",
    )
    parser.add_argument("--json", action="store_true", help="print the same facts as JSON")


# each command: what -h says of it, its arguments, the module that runs it (imported only then), and whether it
# writes a log (a command that does configures it; one that does not turns the process's logging off)
_COMMANDS: dict[str, dict[str, object]] = {
    "inspect": {
        "help": "print what a study package holds, without replaying it",
        "description": "Print what a CohortStudio study package holds, without replaying it.",
        "epilog": _INSPECT_EPILOG,
        "arguments": _inspect_arguments,
        "module": "manage.replay.inspection",
        "logs": False,
    },
}


def _lists_its_options(name: str) -> str:
    return f"{PROGRAM} {name} -h"


def parser() -> argparse.ArgumentParser:
    top = Parser(prog=PROGRAM, description="The command line of a CohortStudio study package.")
    commands = top.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for name, command in _COMMANDS.items():
        added = commands.add_parser(
            name,
            help=str(command["help"]),
            description=str(command["description"]),
            epilog=str(command["epilog"]),
        )
        added.listed_by = (_lists_its_options(name),)
        arguments = command["arguments"]
        assert callable(arguments)
        arguments(added)
    # a command that was not understood: the help of each, which names it as it is spelt
    top.listed_by = tuple(_lists_its_options(name) for name in _COMMANDS)
    return top


def _as_a_command(command: str) -> None:
    """What the process is given when it is the command itself, not a call from a test or another program: streams
    that write any character on any console, and its log. A command that writes one has it configured here (no
    record then holds an exception's text: ``utils.logging_config``), so that no command can forget to; one that
    writes none has the process's logging turned off, so that no library's line comes among its report."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    if _COMMANDS[command]["logs"]:
        # the settings are read only for a command that logs
        from utils.logging_config import configure_logging

        configure_logging()
    else:
        import logging

        logging.disable(logging.CRITICAL)


def main(
    argv: list[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    is_terminal: Callable[[], bool] | None = None,
    as_a_command: bool = False,
) -> int:
    """The command's exit code (see :mod:`manage.replay.inspection`); a command that was not understood exits
    through ``SystemExit(64)``, as ``--help`` does through ``SystemExit(0)``."""
    options = parser().parse_args(argv)
    if as_a_command:
        _as_a_command(options.command)
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    try:
        run = importlib.import_module(str(_COMMANDS[options.command]["module"])).run
        return run(
            options,
            out=out,
            err=err,
            is_terminal=is_terminal or (lambda: at_a_terminal(sys.stdin)),
        )
    except KeyboardInterrupt:
        err.write("\nStopped.\n")
        return EXIT_STOPPED
    except Exception as error:  # by its kind: an error's own text can quote what the file holds
        kind = type(error).__name__
        said = wrapped(
            f"The reading stopped ({kind}): an error of the program, not of the package. Nothing was changed. "
            "Please report it with the lines above."
        )
        err.write("".join(line + "\n" for line in said))
        if getattr(options, "json", False):
            # a program that reads the facts gets an object on this path too
            facts = {"exit": EXIT_FAILED, "failed": kind}
            out.write(json.dumps(facts, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main(as_a_command=True))
