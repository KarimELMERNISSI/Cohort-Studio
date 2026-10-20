"""
The configuration at start-up (M5): .env is read before anything reads the configuration, and a misspelt identity
provider stops start-up instead of giving local accounts.

Before, .env was read only when the Settings page's AI section was first imported: outside Docker, the identity
provider, the administrator bootstrap, DATA_ROOT and the log level never saw it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from manage.iam import IAMConfig, IAMConfigurationError, IAMManager, IAMProviderType

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("app_main", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # runs the top of main.py: .env, then the imports
from manage.iam import get_iam_manager
import utils.data_paths as dp
print(get_iam_manager().provider_name, dp.DATA_ROOT)
"""


def test_the_env_file_is_read_before_the_configuration(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    shutil.copy(ROOT / "main.py", app / "main.py")
    data = tmp_path / "data"
    (app / ".env").write_text(f"CMP_IAM_PROVIDER=disabled\nDATA_ROOT={data.as_posix()}\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("CMP_IAM_PROVIDER", "DATA_ROOT")}
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(app / "main.py")],
        capture_output=True, text=True, env=env, cwd=tmp_path, check=True, timeout=300,
    )
    provider, data_root = out.stdout.strip().splitlines()[-1].split(" ", 1)
    assert provider == "disabled" and Path(data_root) == data


def test_a_variable_set_in_the_environment_wins_over_the_env_file(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    shutil.copy(ROOT / "main.py", app / "main.py")
    (app / ".env").write_text("CMP_IAM_PROVIDER=disabled\n", encoding="utf-8")
    env = {**os.environ, "CMP_IAM_PROVIDER": "local", "PYTHONPATH": str(ROOT), "DATA_ROOT": str(tmp_path / "data")}
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(app / "main.py")],
        capture_output=True, text=True, env=env, cwd=tmp_path, check=True, timeout=300,
    )
    assert out.stdout.strip().splitlines()[-1].startswith("local_duckdb")  # Docker passes its variables this way


@pytest.mark.parametrize("value", ["ldap", "better_authh", "custom", "oidc"])
def test_an_unknown_identity_provider_stops_start_up(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("CMP_IAM_PROVIDER", value)
    with pytest.raises(IAMConfigurationError, match="use local, better_auth or disabled"):
        IAMManager()


def test_the_known_names_are_read_without_regard_to_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMP_IAM_PROVIDER", " Disabled ")
    assert IAMManager().provider_name == "disabled"
    monkeypatch.setenv("CMP_IAM_PROVIDER", "Better-Auth")
    assert IAMManager().provider_type == IAMProviderType.BETTER_AUTH


def test_a_custom_provider_is_not_built_from_the_configuration() -> None:
    # CUSTOM gave local accounts without a word; a custom provider is installed with set_provider
    with pytest.raises(IAMConfigurationError, match="set_provider"):
        IAMManager(config=IAMConfig(provider_type=IAMProviderType.CUSTOM))


def test_a_wrong_setting_stops_the_app_named_without_its_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every setting is validated first (utils.settings): before, a wrong value failed later, or silently counted as
    another (CMP_LOG_LEVEL=verbose as INFO, CMP_OFFLINE=enabled as off, CMP_MI_MAX_WORKERS=0 at the first imputation)."""
    from streamlit.testing.v1 import AppTest

    from utils.settings import reset_settings

    monkeypatch.setenv("CMP_LOG_LEVEL", "verbose-zz9")
    monkeypatch.setenv("CMP_OFFLINE", "enabled-zz9")
    monkeypatch.setenv("CMP_MI_MAX_WORKERS", "0")
    reset_settings()
    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=120).run()
    shown = " ".join(e.value for e in at.error)
    assert "cannot start" in shown
    assert all(name in shown for name in ("CMP_LOG_LEVEL", "CMP_OFFLINE", "CMP_MI_MAX_WORKERS"))
    assert "zz9" not in shown and not at.text_input  # no sign-in form
    reset_settings()


def test_the_command_line_stops_on_a_wrong_setting(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    shutil.copy(ROOT / "manage_users.py", app / "manage_users.py")  # no .env next to it: the developer's is not read
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CMP_IAM_CACHE_TTL": "-55zz9", "DATA_ROOT": str(tmp_path / "data")}
    out = subprocess.run(
        [sys.executable, str(app / "manage_users.py"), "list"],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300,
    )
    assert out.returncode != 0
    assert "CMP_IAM_CACHE_TTL" in out.stderr and "zz9" not in out.stderr


def test_an_empty_setting_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """CMP_IAM_CACHE_TTL= (a line left empty in .env) stopped the identity manager with int('')."""
    monkeypatch.setenv("CMP_IAM_PROVIDER", "")
    monkeypatch.setenv("CMP_IAM_CACHE_TTL", "")
    config = IAMManager._resolve_config_from_env()
    assert config.provider_type == IAMProviderType.LOCAL and config.session_cache_ttl_seconds == 300
