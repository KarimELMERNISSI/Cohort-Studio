"""
The application's settings: one validated object, read from the environment (M5b, step 3).

Every setting the application reads is a field here, under the name of its environment variable, which does not
change. ``load_settings()`` runs first in ``main.py`` and ``manage_users.py``: it loads ``.env`` into the environment
(a variable already set wins), so that every part of the process sees the same values, the log redaction and the
worker process of a parallel imputation included; then it validates every setting, and stops start-up with the name
of each wrong one, never its value.

``get_settings()`` returns the settings of the current environment. It validates again only when one of their
variables has changed, so that a test that sets one sees it at once. Secrets are ``SecretStr``: printed, they show
``**********``. ``core`` and ``biostats`` never read settings: they receive values as parameters.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# The variables whose values the log redaction removes (utils.logging_config): API keys, secrets and passwords
SECRET_VARIABLE = re.compile(r"(?i)(API_KEY|_SECRET|_PASSWORD)$")


class SettingsError(ValueError):
    """Settings that cannot be used. The message names each wrong variable, never its value."""


class _Section(BaseSettings):
    """One group of settings, each field read from the environment variable named by its alias."""

    model_config = SettingsConfigDict(
        case_sensitive=True, env_ignore_empty=True, frozen=True, extra="ignore"
    )

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return (
            value.strip() if isinstance(value, str) else value
        )  # " 1 " is refused as a boolean otherwise


class _Names(_Section):
    """A section whose engine and provider names are compared in lower case, as their readers did."""

    @field_validator("*", mode="before")
    @classmethod
    def _lower(cls, value: object, info) -> object:
        if info.field_name in ("provider", "engine", "renderer") and isinstance(value, str):
            return value.strip().lower()
        return value


class PathsSection(_Section):
    data_root: str = Field("./data", validation_alias="DATA_ROOT")


class LoggingSection(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"] = Field(
        "INFO", validation_alias="CMP_LOG_LEVEL"
    )

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class IAMSection(_Names):
    # the provider's name is checked by manage.iam (an unknown one stops start-up)
    provider: str = Field("local", validation_alias="CMP_IAM_PROVIDER")
    better_auth_url: str = Field(
        "http://localhost:3000/api/auth", validation_alias="CMP_BETTER_AUTH_URL"
    )
    better_auth_api_key: SecretStr | None = Field(None, validation_alias="CMP_BETTER_AUTH_API_KEY")
    cache_ttl: int = Field(300, ge=0, validation_alias="CMP_IAM_CACHE_TTL")
    min_password_length: int = Field(12, ge=1, validation_alias="CMP_MIN_PASSWORD_LENGTH")
    max_failed_logins: int = Field(5, ge=1, validation_alias="CMP_MAX_FAILED_LOGINS")
    lockout_minutes: int = Field(15, ge=1, validation_alias="CMP_LOCKOUT_MINUTES")
    admin_username: str = Field("admin", validation_alias="CMP_ADMIN_USERNAME")
    admin_password: SecretStr | None = Field(None, validation_alias="CMP_ADMIN_PASSWORD")
    admin_password_file: str | None = Field(None, validation_alias="CMP_ADMIN_PASSWORD_FILE")


# the server's key of each cloud provider (manage.rag_manager), by the field that holds it
_PROVIDER_KEYS = {
    "gemini": "google_api_key",
    "openai": "openai_api_key",
    "mistral": "mistral_api_key",
    "anthropic": "anthropic_api_key",
}


class AISection(_Section):
    ollama_base_url: str = Field("http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    google_api_key: SecretStr | None = Field(None, validation_alias="GOOGLE_API_KEY")
    openai_api_key: SecretStr | None = Field(None, validation_alias="OPENAI_API_KEY")
    mistral_api_key: SecretStr | None = Field(None, validation_alias="MISTRAL_API_KEY")
    anthropic_api_key: SecretStr | None = Field(None, validation_alias="ANTHROPIC_API_KEY")

    def api_key(self, provider: str | None) -> str | None:
        """The server's key for a cloud provider (gemini, openai, mistral, anthropic), or None."""
        field = _PROVIDER_KEYS.get((provider or "").strip().lower())
        secret = getattr(self, field) if field else None
        return secret.get_secret_value() if secret is not None else None


class VectorStoreSection(_Names):
    engine: str = Field("auto", validation_alias="CMP_VECTOR_STORE")
    qdrant_url: str | None = Field(None, validation_alias="CMP_QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(None, validation_alias="CMP_QDRANT_API_KEY")


class GraphSection(_Names):
    renderer: str = Field("auto", validation_alias="CMP_GRAPH_RENDERER")
    offline: bool = Field(False, validation_alias="CMP_OFFLINE")
    no_fallback: bool = Field(False, validation_alias="CMP_GRAPH_NO_FALLBACK")


class StatisticsSection(_Section):
    # the seeds of numpy's legacy generators and of scikit-learn are below 2**32
    random_seed: int = Field(20260101, ge=0, le=2**32 - 1, validation_alias="CMP_RANDOM_SEED")
    # the worker processes of the multiple imputation shared by the server (manage.compute); None: processors - 1
    mi_max_workers: int | None = Field(None, ge=1, validation_alias="CMP_MI_MAX_WORKERS")


class DevelopmentSection(_Section):
    # a change of the data that bypasses the trace raises instead of logging (the test suite sets it)
    strict_state: bool = Field(False, validation_alias="CMP_STRICT_STATE")


@dataclass(frozen=True)
class Settings:
    paths: PathsSection
    logging: LoggingSection
    iam: IAMSection
    ai: AISection
    vector_store: VectorStoreSection
    graph: GraphSection
    statistics: StatisticsSection
    development: DevelopmentSection


_SECTIONS: dict[str, type[_Section]] = {
    "paths": PathsSection,
    "logging": LoggingSection,
    "iam": IAMSection,
    "ai": AISection,
    "vector_store": VectorStoreSection,
    "graph": GraphSection,
    "statistics": StatisticsSection,
    "development": DevelopmentSection,
}
# The environment variables of the settings, section by section
NAMES: tuple[str, ...] = tuple(
    str(field.validation_alias)
    for section in _SECTIONS.values()
    for field in section.model_fields.values()
)


def _validate() -> Settings:
    sections: dict[str, _Section] = {}
    problems: list[str] = []
    for attribute, section in _SECTIONS.items():  # each on its own: every wrong variable is named
        try:
            sections[attribute] = section()
        except ValidationError as exc:
            problems += [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
    if problems:
        raise SettingsError("These settings cannot be used: " + "; ".join(problems) + ".")
    return Settings(**sections)  # type: ignore[arg-type]


_cache: tuple[tuple[str | None, ...], Settings] | None = None


def get_settings() -> Settings:
    """The settings of the current environment, validated again only when one of their variables has changed."""
    global _cache
    raw = tuple(os.environ.get(name) for name in NAMES)
    cached = _cache  # one immutable pair: two sessions racing only validate twice
    if cached is not None and cached[0] == raw:
        return cached[1]
    settings = _validate()
    _cache = (raw, settings)
    return settings


_loaded: set[str] = set()
_load_lock = (
    threading.Lock()
)  # the first runs of several sessions: .env is loaded before any of them validates
_warned = False


def _secrets_toml(app_folder: Path | None) -> list[str]:
    """Where a Streamlit secrets file exists, named without its path (the path names the server's account)."""
    places = {"the working folder": Path.cwd()}
    try:
        places["the home folder"] = Path.home()
    except RuntimeError, OSError:  # no home folder (a service account): nothing to look at there
        pass
    if app_folder is not None:
        places["the application folder"] = app_folder
    found: dict[Path, str] = {}
    for name, folder in places.items():
        path = folder / ".streamlit" / "secrets.toml"
        if path.is_file():
            found.setdefault(path.resolve(), name)
    return sorted(found.values())


def load_settings(env_file: str | Path | None = None) -> Settings:
    """
    Start-up: load ``env_file`` (a ``.env``) into the environment, once per process, a variable already set winning;
    warn, once, about a ``.streamlit/secrets.toml``, which is not read (the settings have one source); then validate
    every setting. Raises SettingsError naming each wrong variable, never its value.
    """
    global _warned
    app_folder = None
    with _load_lock:
        if env_file is not None:
            path = Path(env_file).resolve()
            app_folder = path.parent
            if str(path) not in _loaded:
                import dotenv  # through the module: the test suite turns load_dotenv into a no-op

                dotenv.load_dotenv(path, override=False)
                _loaded.add(str(path))
        if not _warned:
            _warned = True
            for place in _secrets_toml(app_folder):
                logger.warning(
                    "A .streamlit/secrets.toml exists in %s and is not read ([secrets] files = [] in "
                    ".streamlit/config.toml): the settings come from the environment and .env only. Move its "
                    "values to .env, then delete it.",
                    place,
                )
    return get_settings()


def secret_values(min_length: int = 8) -> list[str]:
    """
    The values of the secret variables of the environment (names ending in API_KEY, _SECRET or _PASSWORD), for the
    log redaction: read as they are, never validated, so that a wrong setting cannot stop the redaction.
    """
    return [
        value
        for name, value in os.environ.items()
        if len(value) >= min_length and SECRET_VARIABLE.search(name)
    ]


def reset_settings() -> None:
    """Forget the settings and the .env files already loaded (tests)."""
    global _cache, _warned
    _cache = None
    _warned = False
    _loaded.clear()


__all__ = [
    "NAMES",
    "SECRET_VARIABLE",
    "Settings",
    "SettingsError",
    "get_settings",
    "load_settings",
    "reset_settings",
    "secret_values",
]
