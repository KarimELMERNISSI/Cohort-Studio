"""One validated settings object, read from the environment (utils/settings.py, M5b step 3)."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from utils import settings as cfg

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """No setting set: the defaults (the suite sets CMP_STRICT_STATE and CMP_ADMIN_PASSWORD for its own needs)."""
    for name in cfg.NAMES:
        monkeypatch.delenv(name, raising=False)
    cfg.reset_settings()
    yield monkeypatch
    cfg.reset_settings()


def test_the_defaults(env: pytest.MonkeyPatch) -> None:
    s = cfg.get_settings()
    assert s.paths.data_root == "./data" and s.logging.level == "INFO"
    assert (s.iam.provider, s.iam.cache_ttl, s.iam.min_password_length) == ("local", 300, 12)
    assert (s.iam.max_failed_logins, s.iam.lockout_minutes, s.iam.admin_username) == (
        5,
        15,
        "admin",
    )
    assert (
        s.iam.admin_password is None and s.iam.better_auth_url == "http://localhost:3000/api/auth"
    )
    assert s.ai.ollama_base_url == "http://localhost:11434" and s.ai.api_key("gemini") is None
    assert (s.vector_store.engine, s.vector_store.qdrant_url) == ("auto", None)
    assert (s.graph.renderer, s.graph.offline, s.graph.no_fallback) == ("auto", False, False)
    assert (s.statistics.random_seed, s.statistics.mi_max_workers) == (20260101, None)
    assert s.development.strict_state is False


def test_values_are_read_as_their_readers_read_them(env: pytest.MonkeyPatch) -> None:
    env.setenv("CMP_IAM_CACHE_TTL", " 12 ")  # spaces around
    env.setenv("CMP_OFFLINE", "yes")
    env.setenv("CMP_GRAPH_RENDERER", " Plotly ")  # names compared in lower case
    env.setenv("CMP_LOG_LEVEL", "debug")
    env.setenv("CMP_QDRANT_URL", "")  # empty: unset
    s = cfg.get_settings()
    assert (s.iam.cache_ttl, s.graph.offline, s.graph.renderer) == (12, True, "plotly")
    assert s.logging.level == "DEBUG" and s.vector_store.qdrant_url is None


def test_every_wrong_value_is_named_without_its_value(env: pytest.MonkeyPatch) -> None:
    wrong = {
        "CMP_IAM_CACHE_TTL": "-7771",
        "CMP_OFFLINE": "enabled-zz",
        "CMP_LOG_LEVEL": "verbose-zz",
        "CMP_MI_MAX_WORKERS": "0",
        "CMP_RANDOM_SEED": "-2",
        "CMP_MIN_PASSWORD_LENGTH": "twelve-zz",
    }
    for name, value in wrong.items():
        env.setenv(name, value)
    with pytest.raises(cfg.SettingsError) as raised:
        cfg.get_settings()
    message = str(raised.value)
    assert all(name in message for name in wrong)
    assert not any(value in message for value in ("-7771", "enabled-zz", "verbose-zz", "twelve-zz"))


def test_keys_are_hidden_when_printed(env: pytest.MonkeyPatch) -> None:
    env.setenv("GOOGLE_API_KEY", "AIza-planted-secret-value")
    env.setenv("CMP_ADMIN_PASSWORD", "planted-password-value")
    s = cfg.get_settings()
    assert "planted" not in repr(s) and "planted" not in str(s)
    assert s.ai.api_key("gemini") == "AIza-planted-secret-value" and s.ai.api_key("ollama") is None


def test_a_changed_variable_is_read_again(env: pytest.MonkeyPatch) -> None:
    env.setenv("CMP_LOG_LEVEL", "DEBUG")
    first = cfg.get_settings()
    assert cfg.get_settings() is first  # kept while nothing changes
    env.setenv("CMP_LOG_LEVEL", "WARNING")
    assert cfg.get_settings().logging.level == "WARNING"


def test_the_log_redaction_covers_every_secret(env: pytest.MonkeyPatch) -> None:
    secrets = [
        str(field.validation_alias)
        for section in cfg._SECTIONS.values()
        for field in section.model_fields.values()
        if "SecretStr" in str(field.annotation)
    ]
    assert len(secrets) == 7
    assert all(cfg.SECRET_VARIABLE.search(name) for name in secrets)
    env.setenv("OPENAI_API_KEY", "sk-planted-1234")
    env.setenv("CMP_LOG_LEVEL", "not-a-level")  # a wrong setting does not stop the redaction
    assert "sk-planted-1234" in cfg.secret_values()


def test_load_settings_reads_the_env_file_once_and_never_over_a_set_variable(
    env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import dotenv

    calls: list[tuple] = []
    env.setattr(dotenv, "load_dotenv", lambda *a, **k: calls.append((a, k)) or True)
    env_file = tmp_path / ".env"
    env_file.write_text("CMP_LOG_LEVEL=ERROR\n", encoding="utf-8")
    cfg.load_settings(env_file)
    cfg.load_settings(env_file)  # Streamlit runs main.py again on every click
    assert len(calls) == 1 and calls[0][1] == {"override": False}


def test_a_streamlit_secrets_file_is_named_not_read(
    env: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / ".streamlit").mkdir()
    (tmp_path / ".streamlit" / "secrets.toml").write_text(
        'GOOGLE_API_KEY = "x"\n', encoding="utf-8"
    )
    env.chdir(tmp_path)
    with caplog.at_level(logging.WARNING, logger="utils.settings"):
        cfg.load_settings()
        cfg.load_settings()
    warnings = [r.getMessage() for r in caplog.records if "secrets.toml" in r.getMessage()]
    assert len(warnings) == 1 and "the working folder" in warnings[0]
    assert str(tmp_path) not in warnings[0]  # a path names the server's account
    assert cfg.get_settings().ai.api_key("gemini") is None


def test_the_env_example_lists_exactly_the_settings() -> None:
    """Both ways: a setting missing from .env.example, or a line naming no setting, fails with its name."""
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assigned = [re.match(r"^#?\s*([A-Z][A-Z0-9_]*)=(.*)$", line) for line in lines]
    listed = {m.group(1): m for m in assigned if m}
    assert set(listed) - set(cfg.NAMES) == set(), "lines naming no setting"
    assert set(cfg.NAMES) - set(listed) == set(), "settings missing from .env.example"
    defaults = {
        str(field.validation_alias): (section, name)
        for section in cfg._SECTIONS.values()
        for name, field in section.model_fields.items()
    }
    for variable, match in listed.items():
        commented, value = match.group(0).startswith("#"), match.group(2)
        assert not value.startswith(("'", '"')), (
            f"{variable}: not quoted (docker --env-file keeps quotes)"
        )
        section, name = defaults[variable]
        field = section.model_fields[name]
        if cfg.SECRET_VARIABLE.search(variable) and "SecretStr" in str(field.annotation):
            assert value == "" or commented, f"{variable}: a secret line stays empty"
            continue
        if commented or value == "":
            continue
        # an uncommented value is valid and equal to the default
        parsed = section.model_validate({variable: value})
        assert getattr(parsed, name) == field.default, variable


def test_settings_are_frozen(env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        setattr(cfg.get_settings().iam, "cache_ttl", 1)


def test_the_statistics_seed_is_read_as_the_settings_read_it(env: pytest.MonkeyPatch) -> None:
    """biostats reads its seed itself (it imports no first-party package): its parsing must agree."""
    from biostats._types import seed_from

    for value in (None, "", " 7 ", "123", "4294967295", "20260101.0"):
        if value is None:
            env.delenv("CMP_RANDOM_SEED", raising=False)
        else:
            env.setenv("CMP_RANDOM_SEED", value)
        assert seed_from(value) == cfg.get_settings().statistics.random_seed, value
    # refused by both: start-up stops on the setting, and the engine never reads a seed the settings refused
    for value in ("-1", "abc", "4294967296", "1.5"):
        env.setenv("CMP_RANDOM_SEED", value)
        with pytest.raises(cfg.SettingsError):
            cfg.get_settings()
        with pytest.raises(ValueError):  # pydantic's ValidationError is a ValueError
            seed_from(value)


def test_streamlit_reads_no_secrets_file() -> None:
    """Streamlit copied every value of .streamlit/secrets.toml into the environment when the server started, over the
    environment and .env, before the settings were validated: the committed configuration turns that off."""
    import subprocess
    import sys
    import tomllib

    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["secrets"]["files"] == []
    probe = "import streamlit.config as c; c.get_config_options(force_reparse=True); print(c.get_option('secrets.files'))"
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_a_server_without_a_home_folder_starts(env: pytest.MonkeyPatch) -> None:
    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory.")

    env.setattr(Path, "home", staticmethod(no_home))
    assert cfg.load_settings().paths.data_root == "./data"


def test_a_password_is_used_as_it_is(env: pytest.MonkeyPatch) -> None:
    env.setenv("CMP_ADMIN_PASSWORD", " pass phrase ")  # spaces are part of a password
    password = cfg.get_settings().iam.admin_password
    assert password is not None and password.get_secret_value() == " pass phrase "


@pytest.mark.parametrize("variable", ["OLLAMA_BASE_URL", "CMP_QDRANT_URL"])
def test_an_address_holding_credentials_is_refused(env: pytest.MonkeyPatch, variable: str) -> None:
    """It would be shown to every user in Settings, and logged (data-protection review, M5b)."""
    env.setenv(variable, "http://admin:S3cret-zz9@10.0.0.5:11434")
    with pytest.raises(cfg.SettingsError) as raised:
        cfg.get_settings()
    assert variable in str(raised.value) and "zz9" not in str(raised.value)


def test_a_wrong_boolean_names_the_spellings(env: pytest.MonkeyPatch) -> None:
    env.setenv("CMP_OFFLINE", "enabled")
    with pytest.raises(cfg.SettingsError, match="1/0, true/false, yes/no, on/off"):
        cfg.get_settings()


def test_the_redaction_covers_a_key_as_set_and_as_used(env: pytest.MonkeyPatch) -> None:
    """The settings strip a key: the stripped form is the one sent, and logged by a provider's error."""
    from utils.logging_config import redact_secrets

    env.setenv("MISTRAL_API_KEY", "  mistral-planted-key-0123  ")
    assert "mistral-planted-key-0123" in cfg.secret_values()
    assert "mistral-planted-key-0123" not in redact_secrets("refused key mistral-planted-key-0123")
    assert "s3cret" not in redact_secrets("connecting to http://user:s3cret@10.0.0.5:11434/api")


def test_the_qdrant_key_is_never_printed() -> None:
    from manage.vectorstores import VectorStoreSettings

    assert "planted" not in repr(VectorStoreSettings(qdrant_api_key="planted-qdrant-key"))
