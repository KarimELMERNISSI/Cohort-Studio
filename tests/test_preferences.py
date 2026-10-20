"""The user's preferences, kept for the next session, never a secret (manage/preferences.py, M5b step 3)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

import utils.data_paths as dp
from core.workspace import UserScope
from manage.preferences import (
    Preferences,
    load_preferences,
    preferences_path,
    update_preferences,
)


@pytest.fixture
def scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UserScope:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return UserScope.for_user("alice")


def test_no_file_gives_no_preference(scope: UserScope) -> None:
    assert load_preferences(scope) == Preferences()


def test_choices_are_kept_and_merged_by_provider(scope: UserScope) -> None:
    update_preferences(scope, import_dayfirst=True, ai_provider="gemini")
    update_preferences(scope, ai_chat_model={"gemini": "gemini-2.5-flash"})
    update_preferences(
        scope, ai_chat_model={"ollama": "llama3.2"}
    )  # keeps the other provider's model
    prefs = load_preferences(scope)
    assert prefs.import_dayfirst is True and prefs.ai_provider == "gemini"
    assert prefs.ai_chat_model == {"gemini": "gemini-2.5-flash", "ollama": "llama3.2"}
    update_preferences(scope, ai_provider=None, ai_chat_model={"gemini": None})  # None forgets
    prefs = load_preferences(scope)
    assert prefs.ai_provider is None and prefs.ai_chat_model == {"ollama": "llama3.2"}
    text = preferences_path(scope).read_text(encoding="utf-8")
    assert list(json.loads(text)) == sorted(json.loads(text))  # sorted keys, unset fields left out
    assert "ai_provider" not in json.loads(text)
    assert [p.name for p in preferences_path(scope).parent.iterdir() if p.suffix == ".tmp"] == []


def test_an_unreadable_file_gives_no_preference_and_is_not_logged(
    scope: UserScope, caplog: pytest.LogCaptureFixture
) -> None:
    preferences_path(scope).parent.mkdir(parents=True, exist_ok=True)
    preferences_path(scope).write_text("{not json PLANTED-CONTENT", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert load_preferences(scope) == Preferences()
    assert "PLANTED-CONTENT" not in caplog.text


def test_a_wrong_field_does_not_lose_the_others(scope: UserScope) -> None:
    path = preferences_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"import_dayfirst": "maybe", "graph_renderer": "plotly"}), encoding="utf-8"
    )
    prefs = load_preferences(scope)
    assert prefs.import_dayfirst is None and prefs.graph_renderer == "plotly"


def test_no_secret_is_ever_a_preference(scope: UserScope) -> None:
    """Keys, passwords and consent stay in the session (audit E6): a preference holds none."""
    words = ("key", "password", "secret", "token", "consent")
    assert not [name for name in Preferences.model_fields if any(w in name for w in words)]
    assert not [f for f in Preferences.model_fields.values() if "Secret" in str(f.annotation)]
    path = preferences_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"api_key": "AIza-PLANTED", "rag_consent": True}), encoding="utf-8")
    assert "PLANTED" not in repr(load_preferences(scope))
    update_preferences(scope, vector_store="lancedb")
    text = path.read_text(encoding="utf-8")
    assert "PLANTED" not in text and "consent" not in text  # dropped, never written back


def test_an_unknown_name_is_refused(scope: UserScope) -> None:
    with pytest.raises(TypeError, match="api_key"):
        update_preferences(scope, api_key="x")


def test_concurrent_changes_are_all_kept(scope: UserScope) -> None:
    """Two tabs of one user saving at once: the file is read, changed and written under a lock."""
    providers = [f"provider{i}" for i in range(8)]
    threads = [
        threading.Thread(
            target=update_preferences, args=(scope,), kwargs={"ai_chat_model": {p: f"model-{p}"}}
        )
        for p in providers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert load_preferences(scope).ai_chat_model == {p: f"model-{p}" for p in providers}


@pytest.mark.parametrize(
    "content",
    [
        "[" * 100_000 + "]" * 100_000,  # nested too deeply: json raises RecursionError
        '{"ai_provider": "' + "x" * 70_000 + '"}',  # larger than any preferences file
    ],
    ids=["nested-too-deeply", "too-large"],  # an id holding the content is too long for Windows
)
def test_a_hostile_file_gives_no_preference(scope: UserScope, content: str) -> None:
    path = preferences_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    assert load_preferences(scope) == Preferences()
    assert update_preferences(scope, vector_store="lancedb") is not None  # and saving starts afresh


def test_names_and_maps_are_bounded(scope: UserScope) -> None:
    path = preferences_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    many = {f"p{i}": "m" for i in range(60)}
    path.write_text(
        json.dumps({"ai_provider": "y" * 300, "ai_chat_model": many, "graph_renderer": "plotly"}),
        encoding="utf-8",
    )
    prefs = load_preferences(scope)
    assert (
        prefs.ai_provider is None and prefs.ai_chat_model == {} and prefs.graph_renderer == "plotly"
    )
