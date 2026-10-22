"""The identity providers as a connector family (M5b): the registry keeps the connector contract; the deployment's
setting alone chooses one, today's spellings still accepted."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from manage.iam import IAMConfig, IAMConfigurationError, IAMManager, IAMProviderType
from manage.iam.providers import IAM_PROVIDERS
from tests import connector_contract as contract

ROOT = Path(__file__).resolve().parents[1]


def test_the_stored_names_never_change() -> None:
    # kept in deployments' CMP_IAM_PROVIDER, with the spellings already accepted; no automatic choice
    contract.check_names(IAM_PROVIDERS, ("local", "better_auth", "disabled"))
    assert IAM_PROVIDERS.valid() == ("local", "better_auth", "disabled")
    aliases = {
        "betterauth": "better_auth",
        "better-auth": "better_auth",
        "remote": "better_auth",
        "none": "disabled",
        "off": "disabled",
        "bypass": "disabled",
    }
    assert {alias: IAM_PROVIDERS.canonical(alias) for alias in aliases} == aliases
    # a custom provider is installed with set_provider, never from the setting
    assert IAM_PROVIDERS.canonical("custom") is None


def test_the_registry_keeps_the_connector_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    contract.check_availability(IAM_PROVIDERS, monkeypatch)
    contract.check_classes(IAM_PROVIDERS)
    contract.check_unknown_setting(IAM_PROVIDERS)
    contract.check_unknown_names(IAM_PROVIDERS)
    # no connection or process at import, in a fresh interpreter. DuckDB, bcrypt and requests are loaded by the
    # availability (it imports the classes): core dependencies, not optional ones. check_precedence does not apply:
    # neither a session nor a preference chooses the identity provider
    contract.check_light("manage.iam.providers:IAM_PROVIDERS", never_loaded=("streamlit",))


def test_neither_a_session_nor_a_preference_chooses_the_identity_provider() -> None:
    """The kit's generic choice would let them; the deployment's setting alone chooses, so a selector wired by
    mistake fails at once."""
    for choice in ({"session": "disabled"}, {"preference": "disabled"}, {"session": "auto"}):
        with pytest.raises(ValueError, match="chosen by CMP_IAM_PROVIDER only"):
            IAM_PROVIDERS.select(setting="local", **choice)
    assert IAM_PROVIDERS.select(setting="better-auth").chosen == "better_auth"


def test_each_provider_is_built_from_the_configuration(tmp_path: Path) -> None:
    local = IAMManager(
        config=IAMConfig(
            provider_type=IAMProviderType.LOCAL, duckdb_path=str(tmp_path / "u.duckdb")
        )
    )
    assert local.provider.name == "local" and local.provider.user_db_path == str(
        tmp_path / "u.duckdb"
    )
    remote = IAMManager(
        config=IAMConfig(
            provider_type=IAMProviderType.BETTER_AUTH,
            better_auth_url="http://auth.example:3000/api/auth",
            session_cache_ttl_seconds=42,
        )
    )
    assert remote.provider.name == "better_auth" and remote.provider.cache_ttl_seconds == 42
    off = IAMManager(config=IAMConfig(provider_type=IAMProviderType.DISABLED))
    assert off.provider.name == "disabled" and not off.is_enabled
    with pytest.raises(IAMConfigurationError, match="set_provider"):
        IAMManager(config=IAMConfig(provider_type=IAMProviderType.CUSTOM))


def test_the_message_of_an_unknown_provider_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMP_IAM_PROVIDER", "zz9-ldap")
    with pytest.raises(IAMConfigurationError) as refused:
        IAMManager()
    assert str(refused.value) == (
        "CMP_IAM_PROVIDER is not an identity provider: use local, better_auth or disabled."
    )


def test_the_registry_loads_none_of_the_providers() -> None:
    """The providers are loaded when used: the registry, in a fresh interpreter, loads neither DuckDB, bcrypt nor
    requests."""
    code = (
        "import json, sys\n"
        "import manage.iam.providers\n"
        "print(json.dumps(sorted(m for m in ('duckdb', 'bcrypt', 'requests') if m in sys.modules)))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=180
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []
