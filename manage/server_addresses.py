"""
The addresses of the servers a user may choose in Settings, by kind (``OLLAMA_ADDRESSES``: the Ollama services): the
deployment's and those an administrator added in Settings, kept for the whole server in
``<DATA_ROOT>/server/<kind>_addresses.json``.

A user chooses among them and never types one: the server then sends no request to an address someone wrote, which on
a shared server could reach a machine of its network (CodeQL ``py/full-ssrf``). An administrator adds and removes
addresses; the deployment's are changed in the deployment. Every address follows
:func:`utils.settings.server_address` (http or https, ASCII, a host, no user name or password, no query), and a request
goes only to an address :meth:`AddressList.offered` holds (an engine checks it again before it starts, and stops when
its address is removed).

Each file holds the addresses added, the account that added each one and when; it is read whole and written whole,
atomically, under a lock. A missing, unreadable or oversized file, or an entry that is not an address, adds nothing:
the deployment's addresses are still offered. Who added or removed an address is logged too, since the file forgets
an address once removed. The file outlives a deleted account: it is one of the server's stores of account names.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.workspace import UserScope, data_root, path_lock, write_atomic
from utils.logging_config import single_line
from utils.settings import get_settings, server_address

logger = logging.getLogger(__name__)

VERSION = 1
MAX_ADDED = 50  # an administrator's list, not a directory
MAX_BYTES = 64 * 1024  # a list of 50 addresses holds a few kilobytes: a larger file is not read


class NotAllowed(PermissionError):
    """Only an administrator changes a list."""


@dataclass(frozen=True)
class Added:
    """An address an administrator added: the address as offered, the account that added it, when (UTC)."""

    address: str
    added_by: str
    added_at: str


@dataclass(frozen=True)
class AddressList:
    """The addresses offered for one kind of server: the deployment's (``deployment_addresses``), then those added."""

    kind: str  # the file's name: "ollama"
    label: str  # in the log: "Ollama"
    deployment_addresses: Callable[[], list[str]]

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
                    )
                )
        return found[:MAX_ADDED], True

    def listed(self) -> list[str]:
        """Every address offered, each once: the deployment's first, then those administrators added."""
        return list(dict.fromkeys([*self.deployment(), *(item.address for item in self.added())]))

    def offered(self, address: str | None) -> bool:
        """Whether ``address`` (as offered, or written otherwise) is one of the addresses offered now."""
        try:
            return server_address(address) in self.listed()
        except ValueError:
            return False

    def add(self, address: str, *, by: UserScope, account: str | None = None) -> str:
        """
        Add ``address`` to the addresses offered, for every user of the server; returns it as offered (an address
        already offered is not added twice). ``account`` is the administrator's sign-in name (``by``'s folder name
        otherwise). Raises NotAllowed when ``by`` is not an administrator, and ValueError, whose message never holds
        the address, when it is not an address or the list is full.
        """
        if not by.is_admin:
            raise NotAllowed("only an administrator adds an address")
        offered = server_address(address)
        who = (account or by.username)[:200]
        with path_lock(self.path):
            current, readable = self._read()
            if not readable:
                raise ValueError(
                    "the list of added addresses cannot be read (see the server log): repair or remove its file first"
                )
            if offered in self.deployment() or any(item.address == offered for item in current):
                return offered
            if len(current) >= MAX_ADDED:
                raise ValueError(f"the list holds {MAX_ADDED} addresses: remove one first")
            current.append(Added(offered, who, datetime.now(UTC).isoformat(timespec="seconds")))
            self._write(current)
        logger.info(
            "%s address added to the list by the administrator %s: %s",
            self.label,
            single_line(who),
            single_line(offered),
        )
        return offered

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


# The Ollama services (chat, embeddings): OLLAMA_BASE_URL first, then CMP_OLLAMA_ADDRESSES, then those added
OLLAMA_ADDRESSES = AddressList("ollama", "Ollama", _ollama_deployment)

__all__ = ["MAX_ADDED", "MAX_BYTES", "OLLAMA_ADDRESSES", "Added", "AddressList", "NotAllowed"]
