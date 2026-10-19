"""
The shared kit of the extension points (architecture review, 2026-10-19). Standard library only.

A catalogue holds the kinds of one extension point side by side (trace steps; later criterion kinds, date formats,
analysis kinds...), addressed by ids that traces, protocols and results store. Hence its rules:

- an id is never renamed: a kind that nothing records any more stays registered, marked legacy, so what was stored
  still reads;
- an unknown id raises :class:`UnknownIdError`, whose message lists the valid ids;
- an alias leads to its kind, for a name that was spelt another way.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol


class Kind(Protocol):
    """What a catalogue holds: anything with an ``id`` (a frozen dataclass per kind)."""

    @property
    def id(self) -> str: ...


class UnknownIdError(LookupError):
    """An id that the catalogue does not hold; the message lists the valid ids."""

    def __init__(self, catalogue: str, unknown: str, valid: Iterable[str]) -> None:
        self.catalogue = catalogue
        self.unknown = unknown
        self.valid = tuple(valid)
        super().__init__(
            f"unknown {catalogue} {unknown!r}: the valid ids are {', '.join(self.valid)}"
        )

    def __str__(self) -> str:
        return str(self.args[0])


class Catalogue[K: Kind]:
    """The kinds of one extension point, by id, in the order they were registered."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._kinds: dict[str, K] = {}
        self._legacy: set[str] = set()
        self._aliases: dict[str, str] = {}

    def register(self, kind: K, *, legacy: bool = False, aliases: Iterable[str] = ()) -> K:
        """Add a kind; ``legacy`` for one that nothing records any more but stored ids still name."""
        names = (kind.id, *aliases)
        taken = [n for n in names if n in self._kinds or n in self._aliases]
        if taken or len(set(names)) != len(names):
            raise ValueError(f"{self.name}: {', '.join(taken) or kind.id} is already registered")
        self._kinds[kind.id] = kind
        if legacy:
            self._legacy.add(kind.id)
        for alias in aliases:
            self._aliases[alias] = kind.id
        return kind

    def find(self, id: str) -> K | None:
        """The kind with this id or alias, None when there is none."""
        return self._kinds.get(self._aliases.get(id, id))

    def get(self, id: str) -> K:
        """The kind with this id or alias; :class:`UnknownIdError` when there is none."""
        kind = self.find(id)
        if kind is None:
            raise UnknownIdError(self.name, id, self.ids(legacy=False))
        return kind

    def is_legacy(self, id: str) -> bool:
        return self._aliases.get(id, id) in self._legacy

    def ids(self, *, legacy: bool = True) -> tuple[str, ...]:
        """The ids in registration order; ``legacy=False`` leaves out the legacy ones."""
        return tuple(i for i in self._kinds if legacy or i not in self._legacy)

    def __contains__(self, id: object) -> bool:
        return isinstance(id, str) and self.find(id) is not None

    def __iter__(self) -> Iterator[K]:
        return iter(self._kinds.values())

    def __len__(self) -> int:
        return len(self._kinds)


__all__ = ["Catalogue", "Kind", "UnknownIdError"]
