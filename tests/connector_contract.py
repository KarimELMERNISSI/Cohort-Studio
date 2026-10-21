"""
The contract of every connector family (M5b, connectors on one kit: core/registry.py). Each family's tests call these
checks on its registry:

- the stored names never change, and each registration names a module of this version;
- every connector's availability is an answer, never an exception or a connection, and a refusal says why;
- listing the connectors and their availability, in a fresh interpreter, opens no connection, starts no process and
  loads none of the family's heavy packages (its SDKs, Streamlit);
- a connector whose package is missing names it, and the optional group that installs it;
- each class carries the name it is registered under;
- the choice is the session's, then the saved preference, then the server's setting, then the family's own order;
- an unknown name is never repeated: the server's setting is refused with the valid names only, and a saved name is
  shown as code.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from core.registry import Availability, ConnectorRegistry, UnknownConnectorError, as_code

ROOT = Path(__file__).resolve().parents[1]
# a mistyped value, which can be a pasted key or address: never repeated by an error, and shown as inert code when it
# is the user's own saved name (a backtick and a link that must not become formatting)
PLANTED = "zz9`[x](http://e.x)"


def _refuse(*args: object, **kwargs: object) -> None:
    raise AssertionError("an availability check opened a connection")


def refuse_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any connection attempt fails the test."""
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)


def check_names(registry: ConnectorRegistry[Any], names: tuple[str, ...]) -> None:
    """The names are stored in the users' preferences and in the settings: a retired one stays registered. Each
    registration names a module of this version (a typo would read as a package to install)."""
    assert registry.names() == names
    for name in names:
        assert registry.canonical(f" {name.upper()} ") == name
        module = registry.target(name).partition(":")[0]
        assert importlib.util.find_spec(module) is not None, f"{name}: no module {module}"


def check_availability(registry: ConnectorRegistry[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Every connector's availability is an answer, never an exception or a connection; a refusal says why."""
    refuse_connections(monkeypatch)
    for name in registry.names():
        found = registry.availability(name)
        assert isinstance(found, Availability), name
        assert found.ok or found.reason, f"{name} is unavailable without a reason"


# Run in a fresh interpreter: what earlier tests imported cannot hide an import, and the audit hook sees connections
# and processes started from any thread or native code that goes through Python.
_LIGHT = """
import importlib, json, sys

WATCHED = {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "subprocess.Popen", "os.system",
           "os.exec", "os.posix_spawn", "os.spawn", "os.startfile", "urllib.Request"}
attempts = []


def watch(event, args):
    if event in WATCHED:
        attempts.append(event)
        raise RuntimeError(f"an availability check tried {event}")


sys.addaudithook(watch)
module, _, attribute = sys.argv[1].partition(":")
registry = getattr(importlib.import_module(module), attribute)
answers = {name: registry.availability(name) for name in registry.names()}
print(json.dumps({
    "attempts": attempts,
    "loaded": sorted(m for m in sys.argv[2:] if m in sys.modules),
    "unanswered": sorted(name for name, found in answers.items() if not (found.ok or found.reason)),
}))
"""


def check_light(target: str, *, never_loaded: tuple[str, ...]) -> None:
    """Listing the family's connectors and their availability (``target`` = ``"module:REGISTRY"``), in a fresh
    interpreter, opens no connection, starts no process and loads none of ``never_loaded``."""
    done = subprocess.run(
        [sys.executable, "-c", _LIGHT, target, *never_loaded],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report == {"attempts": [], "loaded": [], "unanswered": []}, report


def check_missing_package(
    registry: ConnectorRegistry[Any], name: str, module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``module`` missing, ``name`` cannot run: the reason names the module, and the group that installs it."""
    monkeypatch.setitem(sys.modules, module, None)  # missing for find_spec and for import alike
    found = registry.availability(name)
    assert not found.ok and module in found.reason, found
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert found.extra in extras, found
    assert found.install_hint == f"uv sync --extra {found.extra}"


def check_classes(registry: ConnectorRegistry[Any]) -> None:
    """Each class that loads here carries its registered name and a name to display."""
    for name in registry.names():
        try:
            cls = registry.get(name)
        except ImportError:
            continue  # its module needs a package this install lacks, which its availability says
        assert cls.name == name and cls.display_name, name
        assert registry.display_name(name) == cls.display_name


def check_precedence(registry: ConnectorRegistry[Any]) -> None:
    """The session's choice, then the saved preference, then the server's setting; "auto" at a level leaves the
    choice to the next; a stale name says why, as code."""
    names = registry.names()
    first, last = names[0], names[-1]

    def pick(**choices: str) -> tuple[str | None, str]:
        chosen = registry.select(check=lambda name: Availability(True), **choices)
        return chosen.chosen, chosen.source

    assert pick(session=first, preference=last, setting=last) == (first, "session")
    assert pick(preference=last, setting=first) == (last, "preference")
    assert pick(setting=last) == (last, "setting")
    assert pick(session="auto", preference="auto", setting=last) == (last, "setting")
    stale = registry.select(
        preference=PLANTED, setting=first, check=lambda name: Availability(True)
    )
    assert (stale.chosen, stale.source) == (first, "setting")
    assert stale.warnings[0].endswith(f"{as_code(PLANTED)} is not offered here"), stale.warnings
    assert "](http" not in stale.warnings[0].replace(as_code(PLANTED), "")
    down = registry.select(session=last, check=lambda name: Availability(name != last, "a test"))
    assert down.chosen != last and f"`{last}` is not available here (a test)" in down.warnings[0]
    broken = registry.select(
        session=last, check=lambda name: 1 / 0
    )  # a family's own rule that fails
    assert broken.chosen is None and "its check failed (ZeroDivisionError)" in broken.warnings[0]


def check_unknown_setting(registry: ConnectorRegistry[Any]) -> None:
    """An unknown name in the server's setting is refused with the valid names, and never shown."""
    problem = registry.setting_problem(PLANTED)
    assert problem is not None and problem.startswith(registry.setting) and "zz9" not in problem
    assert all(name in problem for name in registry.valid())
    with pytest.raises(ValueError) as refused:
        registry.select(setting=PLANTED)
    assert "zz9" not in str(refused.value)
    assert all(registry.setting_problem(name.upper()) is None for name in registry.valid())


def check_unknown_names(registry: ConnectorRegistry[Any]) -> None:
    """An unknown name asked of the registry is never repeated by its error (an uncaught error reaches the page), and
    is displayed as code."""
    for ask in (registry.get, registry.target):
        with pytest.raises(UnknownConnectorError) as refused:
            ask(PLANTED)
        assert "zz9" not in str(refused.value) and refused.value.unknown is None
        assert all(name in str(refused.value) for name in registry.names())
    assert registry.display_name(PLANTED) == as_code(PLANTED)
