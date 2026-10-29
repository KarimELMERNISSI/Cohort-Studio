"""
The user's preferences: the choices made in Settings, kept for the next session (M5b, step 3).

A preference is not a setting. A setting is the server's (utils.settings, read from the environment); a preference is
the user's choice among what the server allows: the day-first reading of dates on import, the AI engine and its
models, the graph renderer, the vector database. No key, password, consent or address is ever stored: a cloud provider
saved here starts nothing until the user confirms again in the session (audit E6), and the address of a model
server (Ollama, Qdrant), which decides where the data go, is entered again in each session.

The file is ``<workspace>/preferences.json``, read whole and written whole, atomically, under a lock. A missing or
unreadable file gives no preference; a field that is not valid is left out, and the others are kept.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from core.workspace import UserScope, path_lock, write_atomic

logger = logging.getLogger(__name__)

FILE_NAME = "preferences.json"
VERSION = 1
# A preferences file holds a few hundred bytes: a larger one, or a longer name, is not read (a hostile file must not
# hold up the page or be carried in the session)
MAX_BYTES = 64 * 1024
Name = Annotated[str, StringConstraints(max_length=200)]


class Preferences(BaseModel):
    """The choices kept for the next session. Every field is optional: unset, the setting or the default applies."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    version: int = VERSION
    import_dayfirst: bool | None = None
    ai_provider: Name | None = None
    ai_chat_model: dict[Name, Name] = Field(default_factory=dict, max_length=50)  # by provider
    ai_embedding_model: dict[Name, Name] = Field(default_factory=dict, max_length=50)  # by provider
    # Claude answers; another provider embeds the documents
    ai_claude_embedding_provider: Name | None = None
    graph_renderer: Name | None = None
    vector_store: Name | None = None


# the fields kept by provider: a change to one provider's model keeps the others'
_BY_PROVIDER = ("ai_chat_model", "ai_embedding_model")


def preferences_path(scope: UserScope) -> Path:
    return scope.root / FILE_NAME


def _valid_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """The fields of ``raw`` that are valid on their own: one wrong field does not lose the others."""
    clean: dict[str, Any] = {}
    for name in Preferences.model_fields:
        if name not in raw:
            continue
        try:
            Preferences.model_validate({name: raw[name]})
        except ValidationError:
            logger.warning("A saved preference is not valid and is ignored: %s", name)
            continue
        clean[name] = raw[name]
    return clean


def load_preferences(scope: UserScope) -> Preferences:
    """The user's preferences; none when the file is missing or unreadable (logged without its content)."""
    path = preferences_path(scope)
    try:
        if path.stat().st_size > MAX_BYTES:
            logger.warning(
                "The preferences file is too large to be preferences: no preference is used"
            )
            return Preferences()
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Preferences()
    except (OSError, ValueError, RecursionError) as exc:  # RecursionError: nested too deeply
        logger.warning(
            "The preferences file cannot be read (%s): no preference is used", type(exc).__name__
        )
        return Preferences()
    if not isinstance(raw, dict):
        logger.warning("The preferences file holds no preferences: no preference is used")
        return Preferences()
    return Preferences.model_validate(_valid_fields(raw))


def update_preferences(scope: UserScope, **changes: Any) -> Preferences | None:
    """
    Keep these choices for the next session. ``None`` forgets a choice; for the models, a mapping changes the
    providers it names (``None`` for a provider forgets its model). Never raises for the file: a preference that
    cannot be saved is logged, the session goes on, and None is returned. An unknown name is a programming error.
    """
    unknown = sorted(set(changes) - set(Preferences.model_fields) - {"version"})
    if unknown:
        raise TypeError(f"not a preference: {', '.join(unknown)}")
    path = preferences_path(scope)
    with path_lock(path):
        current = load_preferences(scope).model_dump()
        for name, value in changes.items():
            if name in _BY_PROVIDER and value is not None:
                merged = {**current.get(name, {}), **value}
                current[name] = {
                    provider: model for provider, model in merged.items() if model is not None
                }
            else:
                current[name] = value
        try:
            preferences = Preferences.model_validate(current)
        except ValidationError as exc:
            logger.warning(
                "Preferences not saved: %s", ", ".join(str(e["loc"][0]) for e in exc.errors())
            )
            return None
        data = {
            name: value
            for name, value in preferences.model_dump().items()
            if value is not None and value != {}
        }
        try:
            write_atomic(
                path,
                json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                prefix=".preferences.",
            )
        except OSError as exc:
            logger.warning("Preferences not saved (%s)", type(exc).__name__)
            return None
        return preferences


__all__ = ["FILE_NAME", "Preferences", "load_preferences", "preferences_path", "update_preferences"]
