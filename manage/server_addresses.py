"""
The addresses of the servers a user may choose in Settings, by kind (``OLLAMA_ADDRESSES``, the Ollama services;
``QDRANT_ADDRESSES``, the Qdrant servers): the
deployment's and those an administrator added in Settings, kept for the whole server in
``<DATA_ROOT>/server/<kind>_addresses.json``.

A user chooses among them and never types one: the server then sends no request to an address someone wrote, which on
a shared server could reach a machine of its network (CodeQL ``py/full-ssrf``). An administrator adds and removes
addresses; the deployment's are changed in the deployment. Every address follows
:func:`utils.settings.server_address` (http or https, ASCII, a host, no user name or password, no query), and a request
goes only to an address :meth:`AddressList.offered` holds (an engine checks it again before it starts, and stops when
its address is removed).

Each file holds the addresses added, the account that added each one and when, and its optional name (free text every
user sees: no "@" nor "://", and no two alike); it is read whole and written whole,
atomically, under a lock. A missing, unreadable or oversized file, or an entry that is not an address, adds nothing:
the deployment's addresses are still offered. Who added, renamed or removed an address is logged too (never a name's
text), since the file forgets an address once removed. The file outlives a deleted account: it is one of the server's stores of account names.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from core.workspace import UserScope, data_root, path_lock, write_atomic
from utils.logging_config import single_line
from utils.settings import get_settings, server_address

logger = logging.getLogger(__name__)

VERSION = 1
# the end of an AI engine's stop reason for an address an administrator removed: Settings asks for another choice
ADDRESS_REMOVED = "removed from CohortStudio's list"
MAX_ADDED = 50  # an administrator's list, not a directory
MAX_BYTES = 64 * 1024  # a list of 50 addresses holds a few kilobytes: a larger file is not read


class NotAllowed(PermissionError):
    """Only an administrator changes a list."""


@dataclass(frozen=True)
class Added:
    """An address an administrator added: the address as offered, the account that added it, when (UTC), and the
    short name it may be given ("GPU server, Biostatistics")."""

    address: str
    added_by: str
    added_at: str
    name: str = ""


NAME_MAX_LENGTH = (
    40  # the list shows the name, then the address: a longer name hides the address at 1280 px
)

# printable, yet blank: the Hangul fillers, the halfwidth filler, the empty Braille pattern
_BLANK = frozenset("\u115f\u1160\u3164\uffa0\u2800")


def address_name(value: str | None) -> str:
    """A name for an address, on one line: its white space collapsed, its invisible and blank characters dropped, at
    most ``NAME_MAX_LENGTH`` characters (it is shown as text, never as a format); "" when no letter or digit is
    left."""
    text = "".join(
        c for c in str(value or "") if (c.isprintable() or c.isspace()) and c not in _BLANK
    )
    named = " ".join(text.split())[:NAME_MAX_LENGTH]
    return named if any(c.isalnum() for c in named) else ""


def _names_a_place(name: str) -> bool:
    """A name that reads as another address ("://"), or names a person's e-mail ("@"): every user sees it."""
    return "://" in name or "@" in name


def _shown_name(value: object) -> str:
    """A name read back from the file: cleaned again, and dropped when it reads as an address or a person's e-mail."""
    named = address_name(value if isinstance(value, str) else "")
    return "" if _names_a_place(named) else named


def _checked_name(name: str | None, items: list[Added], address: str) -> str:
    """``name`` as an address takes it ("" for none). ValueError when it reads as another address or names a person,
    or when another address of the list already has it (whatever its case): a user tells the addresses apart by it."""
    named = address_name(name)
    if named and _names_a_place(named):
        raise ValueError(
            'a name may not hold "@" or "://" (name the service, never a person or another address)'
        )
    if named and any(
        item.name.casefold() == named.casefold() for item in items if item.address != address
    ):
        raise ValueError("another address in the list already has that name")
    return named


@dataclass(frozen=True)
class AddressList:
    """The addresses offered for one kind of server: the deployment's (``deployment_addresses``), then those added."""

    kind: str  # the file's name: "ollama"
    label: str  # in the log: "Ollama"
    deployment_addresses: Callable[[], list[str]]
    # an address added must say http or https: one written without would read http, and its key cross in clear
    scheme_required: bool = False

    @property
    def path(self) -> Path:
        return data_root() / "server" / f"{self.kind}_addresses.json"

    def deployment(self) -> list[str]:
        """The deployment's addresses, each once, in their order."""
        return self.deployment_addresses()

    def added(self) -> list[Added]:
        """The addresses administrators added, in the order they were added; none when the file is missing or cannot
        be read (logged without its content)."""
        return self._read()[0]

    def unreadable(self) -> bool:
        """Whether the file exists and cannot be read as a list: Settings says so, and nothing is added until it is
        repaired or removed (an Add would otherwise write a list holding that address alone)."""
        return not self._read()[1]

    def _read(self) -> tuple[list[Added], bool]:
        """The addresses added, and whether the file could be read (a missing file holds none, and can)."""
        try:
            if self.path.stat().st_size > MAX_BYTES:
                logger.warning(
                    "The list of %s addresses is too large to be one: only the deployment's are offered",
                    self.label,
                )
                return [], False
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return [], True
        except (OSError, ValueError, RecursionError) as exc:  # RecursionError: nested too deeply
            logger.warning(
                "The list of %s addresses cannot be read (%s): only the deployment's are offered",
                self.label,
                type(exc).__name__,
            )
            return [], False
        entries = raw.get("added") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            logger.warning(
                "The list of %s addresses holds no list: only the deployment's are offered",
                self.label,
            )
            return [], False
        found: list[Added] = []
        for entry in entries:
            try:
                address = server_address(entry["address"])
            except KeyError, TypeError, ValueError:
                logger.warning(
                    "An entry of the list of %s addresses is not an address and is not offered",
                    self.label,
                )
                continue
            if all(item.address != address for item in found):
                found.append(
                    Added(
                        address,
                        str(entry.get("added_by") or "")[:200],
                        str(entry.get("added_at") or "")[:40],
                        _shown_name(entry.get("name")),
                    )
                )
        return found[:MAX_ADDED], True

    def listed(self) -> list[str]:
        """Every address offered, each once: the deployment's first, then those administrators added."""
        return list(dict.fromkeys([*self.deployment(), *(item.address for item in self.added())]))

    def names(self) -> dict[str, str]:
        """The names administrators gave the addresses they added, by address."""
        return {item.address: item.name for item in self.added() if item.name}

    def offered(self, address: str | None) -> bool:
        """Whether ``address`` (as offered, or written otherwise) is one of the addresses offered now."""
        try:
            return server_address(address) in self.listed()
        except ValueError:
            return False

    def add(
        self, address: str, *, by: UserScope, account: str | None = None, name: str | None = None
    ) -> str:
        """
        Add ``address`` to the addresses offered, for every user of the server, with an optional short ``name``;
        returns it as offered. An address already added is not added twice: a name given then becomes its name (one
        the deployment sets takes none, and keeps none). ``account`` is the administrator's sign-in name (``by``'s
        folder name otherwise). Raises NotAllowed when ``by`` is not an administrator, and ValueError, whose message
        never holds the address, when it is not an address, the list is full, or the name may not be one
        (``_checked_name``).
        """
        if not by.is_admin:
            raise NotAllowed("only an administrator adds an address")
        if self.scheme_required and "://" not in str(address or ""):
            raise ValueError(
                "write https:// (a server outside your network, such as Qdrant Cloud) or http:// first"
            )
        offered = server_address(address)
        who = (account or by.username)[:200]
        with path_lock(self.path):
            current, readable = self._read()
            if not readable:
                raise ValueError(
                    "the list of added addresses cannot be read (see the server log): repair or remove its file first"
                )
            if offered in self.deployment():
                return offered
            named = _checked_name(name, current, offered)
            held = [n for n, item in enumerate(current) if item.address == offered]
            if held:
                if named and current[held[0]].name != named:  # named again: its new name
                    current[held[0]] = replace(current[held[0]], name=named)
                    self._write(current)
                    self._log_rename(who, offered)
                return offered
            if len(current) >= MAX_ADDED:
                raise ValueError(f"the list holds {MAX_ADDED} addresses: remove one first")
            current.append(
                Added(offered, who, datetime.now(UTC).isoformat(timespec="seconds"), named)
            )
            self._write(current)
        logger.info(
            "%s address added to the list by the administrator %s: %s",
            self.label,
            single_line(who),
            single_line(offered),
        )
        return offered

    def rename(
        self, address: str, name: str | None, *, by: UserScope, account: str | None = None
    ) -> bool:
        """
        Give an address an administrator added a new ``name``, for every user ("" removes its name); False when the
        list holds no such address (one the deployment sets takes no name). Raises NotAllowed when ``by`` is not an
        administrator, and ValueError when the name may not be one (``_checked_name``) or the list cannot be read.
        """
        if not by.is_admin:
            raise NotAllowed("only an administrator renames an address")
        who = (account or by.username)[:200]
        with path_lock(self.path):
            current, readable = self._read()
            if not readable:
                raise ValueError(
                    "the list of added addresses cannot be read (see the server log): repair or remove its file first"
                )
            held = [n for n, item in enumerate(current) if item.address == address]
            if not held:
                return False
            named = _checked_name(name, current, address)
            if current[held[0]].name == named:
                return True
            current[held[0]] = replace(current[held[0]], name=named)
            self._write(current)
        self._log_rename(who, address)
        return True

    def _log_rename(self, who: str, address: str) -> None:
        logger.info(  # the event and the account, never the name's text
            "%s address renamed by the administrator %s: %s",
            self.label,
            single_line(who),
            single_line(address),
        )

    def remove(self, address: str, *, by: UserScope, account: str | None = None) -> bool:
        """Remove an address an administrator added; False when the list holds no such address (one the deployment
        sets is changed in the deployment). Raises NotAllowed when ``by`` is not an administrator."""
        if not by.is_admin:
            raise NotAllowed("only an administrator removes an address")
        with path_lock(self.path):
            current = self.added()
            kept = [item for item in current if item.address != address]
            if len(kept) == len(current):
                return False
            self._write(kept)
        logger.info(
            "%s address removed from the list by the administrator %s: %s",
            self.label,
            single_line((account or by.username)[:200]),
            single_line(address),
        )
        return True

    def _write(self, items: list[Added]) -> None:
        data = {"version": VERSION, "added": [asdict(item) for item in items]}
        write_atomic(
            self.path,
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            prefix=f".{self.kind}_addresses.",
        )


def _ollama_deployment() -> list[str]:
    return get_settings().ai.ollama_address_list()


def _qdrant_deployment() -> list[str]:
    return get_settings().vector_store.qdrant_address_list()


# The Ollama services (chat, embeddings): OLLAMA_BASE_URL first, then CMP_OLLAMA_ADDRESSES, then those added
OLLAMA_ADDRESSES = AddressList("ollama", "Ollama", _ollama_deployment)
# The Qdrant servers that keep the index (no server: the embedded mode): CMP_QDRANT_URL, then those added
QDRANT_ADDRESSES = AddressList("qdrant", "Qdrant", _qdrant_deployment, scheme_required=True)

__all__ = [
    "ADDRESS_REMOVED",
    "MAX_ADDED",
    "MAX_BYTES",
    "NAME_MAX_LENGTH",
    "OLLAMA_ADDRESSES",
    "QDRANT_ADDRESSES",
    "Added",
    "AddressList",
    "NotAllowed",
    "address_name",
]
