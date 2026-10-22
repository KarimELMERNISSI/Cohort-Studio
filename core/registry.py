"""
The shared kit of the extension points (architecture review, 2026-10-19). Standard library only.

Two mechanisms, one shape each.

A **catalogue** holds the kinds of one extension point side by side (trace steps; later criterion kinds, date
formats, analysis kinds...), addressed by ids that traces, protocols and results store. Hence its rules:

- an id is never renamed: a kind that nothing records any more stays registered, marked legacy, so what was stored
  still reads;
- an unknown id raises :class:`UnknownIdError`, whose message lists the valid ids;
- an alias leads to its kind, for a name that was spelt another way.

A **connector registry** holds the implementations of one family of which one is used at a time (graph renderers,
vector databases, AI providers, identity providers; M5b step 4). Each is registered as ``"module:Class"`` and
imported only when used; its availability names the optional group to install when it is missing, and never raises.
The choice is the session's, then the user's saved preference, then the server's setting, then the family's own
order (``auto``); a stale name falls through with a warning that says why. An unknown name in the server's setting
stops start-up, with the valid names and never the value.
"""

from __future__ import annotations

import importlib
import importlib.util
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol


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


# ----------------------------------------------------------------------------------------------- connector families

AUTO = "auto"
Source = Literal["session", "preference", "setting", "auto"]


def as_code(name: object, *, limit: int = 64) -> str:
    """
    A name in a notice, as code: a saved name is the user's text, and must not become a link or a format, nor reorder
    or hide part of the sentence. Backticks, control and invisible characters (bidirectional overrides, zero-width
    spaces) are dropped, white space is collapsed, and a long name is cut at ``limit`` characters.
    """
    text = " ".join(str(name).replace("`", "").split())
    text = " ".join("".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cf")).split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return f"`{text or '?'}`"


@dataclass(frozen=True)
class Availability:
    """Whether a connector can run here, why not, and the optional group that provides it."""

    ok: bool
    reason: str = ""
    extra: str | None = None

    @property
    def install_hint(self) -> str | None:
        return f"uv sync --extra {self.extra}" if self.extra and not self.ok else None

    @classmethod
    def of_module(cls, module: str, *, extra: str | None = None) -> Availability:
        """
        Whether ``module`` is installed, found without importing it (and without a cache: a test may block it). For a
        dotted name, ``find_spec`` imports the parent packages: they are looked up first, and imported only when the
        top-level package is installed (``google`` for ``google.genai``, a namespace package).
        """
        top = module.partition(".")[0]
        try:
            found = importlib.util.find_spec(top) is not None and (
                top == module or importlib.util.find_spec(module) is not None
            )
        except ImportError, ValueError:  # a parent package missing, or blocked
            found = False
        return cls(True) if found else cls(False, f"{module} is not installed", extra)


class Connector(Protocol):
    """What a registry holds: a class with its registered ``name``, a ``display_name`` and ``availability()``."""

    name: ClassVar[str]
    display_name: ClassVar[str]

    @classmethod
    def availability(cls) -> Availability: ...


@dataclass(frozen=True)
class _Registration:
    id: str
    target: str  # "module:Class"
    extra: str | None = None


@dataclass(frozen=True)
class Selection:
    """The connector chosen, where the choice came from, why, what was set aside, and every connector's state."""

    family: str
    chosen: str | None  # None: none is available here
    source: Source
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()  # whole sentences, names as code
    alternatives: tuple[tuple[str, Availability], ...] = field(default=())


class UnknownConnectorError(UnknownIdError):
    """
    A connector the registry does not hold; the message lists the valid names. Unlike a catalogue's error, it never
    holds the name asked for: the name can come from a setting, where a mistyped value can be a pasted key or address,
    and an uncaught error is shown on the page.
    """

    def __init__(self, family: str, valid: Iterable[str]) -> None:
        self.catalogue = family
        self.unknown = None
        self.valid = tuple(valid)
        LookupError.__init__(
            self, f"not {_article(family)} {family} of this version: use {_listed(self.valid)}"
        )


class ConnectorRegistry[C: Connector]:
    """The implementations of one family, imported when used, chosen in a fixed order (see the module docstring)."""

    def __init__(
        self,
        family: str,
        *,
        setting: str,
        auto: Iterable[str] | None = (),
        setting_only: bool = False,
    ) -> None:
        self.family = family  # "graph engine"
        self.setting = setting  # "CMP_GRAPH_RENDERER"
        self._auto = (
            None if auto is None else tuple(auto)
        )  # None: the family has no "auto" (identity providers)
        # the deployment's setting alone chooses (identity providers): a session or a saved preference never does
        self.setting_only = setting_only
        self._catalogue: Catalogue[_Registration] = Catalogue(family)
        self._classes: dict[str, type[C]] = {}

    def register(
        self, name: str, target: str, *, extra: str | None = None, aliases: Iterable[str] = ()
    ) -> None:
        """Add ``name`` (lower case), implemented by ``target`` = ``"module:Class"``, provided by the group ``extra``."""
        module, _, attr = target.partition(":")
        if not module or not attr:
            raise ValueError(f"{self.family}: {target!r} is not 'module:Class'")
        names = (name, *aliases)
        if any(n != n.lower() or n == AUTO for n in names):
            raise ValueError(
                f"{self.family}: names are in lower case, and {AUTO!r} is the family's own order"
            )
        self._catalogue.register(_Registration(name, target, extra), aliases=aliases)

    def names(self) -> tuple[str, ...]:
        """The registered names, in registration order: nothing is imported."""
        return self._catalogue.ids()

    def valid(self) -> tuple[str, ...]:
        """What the setting accepts: ``auto`` first when the family has it, then the names."""
        return ((AUTO,) if self._auto is not None else ()) + self.names()

    def canonical(self, name: str | None) -> str | None:
        """The registered name for a name or an alias in any case, ``auto``, or None when it is unknown."""
        if not name:
            return None
        key = name.strip().lower()
        if key == AUTO and self._auto is not None:
            return AUTO
        found = self._catalogue.find(key)
        return found.id if found else None

    def _registration(self, name: str | None) -> _Registration | None:
        canonical = self.canonical(name)
        return self._catalogue.find(canonical) if canonical and canonical != AUTO else None

    def target(self, name: str) -> str:
        """Where ``name`` is implemented, ``"module:Class"``; UnknownConnectorError when unknown."""
        registration = self._registration(name)
        if registration is None:
            raise UnknownConnectorError(self.family, self.names())
        return registration.target

    def get(self, name: str) -> type[C]:
        """The class of ``name`` (or an alias), imported the first time; UnknownConnectorError when unknown."""
        registration = self._registration(name)
        if registration is None:
            raise UnknownConnectorError(self.family, self.names())
        if registration.id not in self._classes:
            module, _, attr = registration.target.partition(":")
            self._classes[registration.id] = getattr(importlib.import_module(module), attr)
        return self._classes[registration.id]

    def availability(self, name: str) -> Availability:
        """
        Whether ``name`` can run here. Never raises: a missing package or a broken check is a reason. A package missing
        for the connector's module names the optional group to install; the class's own refusals name it themselves,
        since installing may not help (a server that does not answer).
        """
        registration = self._registration(name)
        if registration is None:
            return Availability(False, "not offered by this version")
        module = registration.target.partition(":")[0]
        try:
            cls = self.get(registration.id)
        except ImportError as exc:
            missing = getattr(exc, "name", None) or "a package"
            if missing == module or module.startswith(
                f"{missing}."
            ):  # the registration names no module
                return Availability(
                    False, "this version does not include it (a registration error)"
                )
            return Availability(False, f"{missing} is not installed", registration.extra)
        except Exception as exc:  # a broken module: reported, never raised
            return Availability(
                False, f"it cannot be loaded ({type(exc).__name__})", registration.extra
            )
        try:
            return cls.availability()
        except Exception as exc:
            return Availability(False, f"its check failed ({type(exc).__name__})")

    def _checked(self, check: Callable[[str], Availability] | None, name: str) -> Availability:
        """``check`` (a family's own rule, such as the offline mode) guarded as :meth:`availability` is."""
        if check is None:
            return self.availability(name)
        try:
            return check(name)
        except Exception as exc:
            return Availability(False, f"its check failed ({type(exc).__name__})")

    def display_name(self, name: str) -> str:
        """The connector's name to display: its registered name when its class cannot be loaded here, and an unknown
        name as code (it can come from a saved preference)."""
        canonical = self.canonical(name)
        if canonical is None or canonical == AUTO:
            return as_code(name)
        try:
            return self.get(canonical).display_name
        except Exception:  # a module that needs a package this install lacks
            return canonical

    def setting_problem(self, value: str | None) -> str | None:
        """The start-up message for a setting that names nothing: the variable and the valid names, never the value
        (a mistyped value can be a pasted key or address, and the page shows it before sign-in)."""
        if not value or not value.strip() or self.canonical(value):
            return None
        return f"{self.setting} is not {_article(self.family)} {self.family}: use {_listed(self.valid())}"

    def select(
        self,
        *,
        session: str | None = None,
        preference: str | None = None,
        setting: str | None = None,
        auto: str | None = None,
        auto_reasons: Iterable[str] = (),
        check: Callable[[str], Availability] | None = None,
    ) -> Selection:
        """
        The connector to use: the session's choice, then the saved preference, then the server's setting, then
        ``auto`` (a recommendation, then the family's own order). A name that is unknown or cannot run here falls
        through with a warning; ``auto`` at a level is no choice there, so the next level decides (the session's
        "automatic" is the server's default). ``check`` replaces :meth:`availability` (a family may add a rule of its
        own, such as the offline mode), guarded the same way.
        """
        if self.setting_only and (session is not None or preference is not None):
            raise ValueError(f"the {self.family} is chosen by {self.setting} only")
        if self.setting_problem(setting):
            raise ValueError(self.setting_problem(setting))  # start-up stops before this
        alternatives = tuple((name, self._checked(check, name)) for name in self.names())
        status = dict(alternatives)
        warnings: list[str] = []
        for source, raw in (("session", session), ("preference", preference), ("setting", setting)):
            if not raw or not raw.strip():
                continue
            name = self.canonical(raw)
            if name == AUTO:
                continue
            if name is None:
                warnings.append(
                    f"{_WHOSE[source]} {self.family} {as_code(raw)} is not offered here"
                )
                continue
            if status[name].ok:
                why = (_WHY[source].format(setting=self.setting),)
                return Selection(self.family, name, source, why, tuple(warnings), alternatives)  # type: ignore[arg-type]
            warnings.append(
                f"{_WHOSE[source]} {self.family} {as_code(name)} is not available here ({status[name].reason})"
            )
        if auto and auto in status and not status[auto].ok:
            warnings.append(
                f"The recommended {self.family} {as_code(auto)} is not available here ({status[auto].reason})"
            )
        order = ((auto,) if auto else ()) + (self._auto or ())
        for name in order:
            if name in status and status[name].ok:
                reasons = (
                    tuple(auto_reasons)
                    if name == auto
                    else (f"the first available of {', '.join(self._auto or ())}",)
                )
                return Selection(self.family, name, "auto", reasons, tuple(warnings), alternatives)
        return Selection(self.family, None, "auto", (), tuple(warnings), alternatives)


_WHOSE = {"session": "The chosen", "preference": "Your saved", "setting": "The server's"}
_WHY = {
    "session": "chosen for this session",
    "preference": "your saved choice",
    "setting": "the server's setting ({setting})",
}


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _listed(names: Iterable[str]) -> str:
    """``a, b or c``."""
    items = list(names)
    return ", ".join(items[:-1]) + f" or {items[-1]}" if len(items) > 1 else "".join(items)


__all__ = [
    "AUTO",
    "Availability",
    "Catalogue",
    "Connector",
    "ConnectorRegistry",
    "Kind",
    "Selection",
    "UnknownConnectorError",
    "UnknownIdError",
    "as_code",
]
