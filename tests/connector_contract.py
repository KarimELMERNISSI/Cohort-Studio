"""
The contract of every connector family (M5b, connectors on one kit: core/registry.py). Each family's tests call these
checks on its registry:

- the stored names never change;
- every connector's availability is an answer, never an exception or a connection, and a refusal says why;
- a connector whose package is missing names it, and the optional group that installs it;
- each class carries the name it is registered under;
- the choice is the session's, then the saved preference, then the server's setting, then the family's own order;
- an unknown name in the server's setting is refused with the valid names, never with the value.
"""

from __future__ import annotations

import socket
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from core.registry import Availability, ConnectorRegistry

ROOT = Path(__file__).resolve().parents[1]
PLANTED = "zz9-planted"  # a mistyped value, which can be a pasted key or address: it is never shown


def _refuse(*args: object, **kwargs: object) -> None:
    raise AssertionError("an availability check opened a connection")


def refuse_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any connection attempt fails the test."""
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)


def check_names(registry: ConnectorRegistry[Any], names: tuple[str, ...]) -> None:
    """The names are stored in the users' preferences and in the settings: a retired one stays registered."""
    assert registry.names() == names
    for name in names:
        assert registry.canonical(f" {name.upper()} ") == name


def check_availability(registry: ConnectorRegistry[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Every connector's availability is an answer, never an exception or a connection; a refusal says why."""
    refuse_connections(monkeypatch)
    for name in registry.names():
        found = registry.availability(name)
        assert isinstance(found, Availability), name
        assert found.ok or found.reason, f"{name} is unavailable without a reason"


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


def check_precedence(registry: ConnectorRegistry[Any]) -> None:
    """The session's choice, then the saved preference, then the server's setting; a stale name says why."""
    names = registry.names()
    first, last = names[0], names[-1]

    def pick(**choices: str) -> tuple[str | None, str]:
        chosen = registry.select(check=lambda name: Availability(True), **choices)
        return chosen.chosen, chosen.source

    assert pick(session=first, preference=last, setting=last) == (first, "session")
    assert pick(preference=last, setting=first) == (last, "preference")
    assert pick(setting=last) == (last, "setting")
    stale = registry.select(
        preference=PLANTED, setting=first, check=lambda name: Availability(True)
    )
    assert (stale.chosen, stale.source) == (first, "setting")
    assert f"`{PLANTED}` is not offered here" in stale.warnings[0]
    down = registry.select(session=last, check=lambda name: Availability(name != last, "a test"))
    assert down.chosen != last and f"`{last}` is not available here (a test)" in down.warnings[0]


def check_unknown_setting(registry: ConnectorRegistry[Any]) -> None:
    """An unknown name in the server's setting is refused with the valid names, and never shown."""
    problem = registry.setting_problem(PLANTED)
    assert problem is not None and problem.startswith(registry.setting) and PLANTED not in problem
    assert all(name in problem for name in registry.valid())
    with pytest.raises(ValueError) as refused:
        registry.select(setting=PLANTED)
    assert PLANTED not in str(refused.value)
    assert all(registry.setting_problem(name.upper()) is None for name in registry.valid())
