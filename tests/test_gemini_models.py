"""The Gemini chat models offered in Settings: the ones that fit the app's uses, Flash-Lite included (cheap and fast),
and no specialised model (image, speech, robotics) when a typed key lists more."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core", reason="the AI stack is in the optional 'ai' extra")

from manage.rag.providers.gemini import GEMINI_CHAT_MODELS, GeminiChatConnector, is_chat_model
from manage.rag.schemas import ProviderConfig, ProviderType


def test_the_offered_models_include_flash_lite() -> None:
    assert GEMINI_CHAT_MODELS[0] == "gemini-3.8-flash"  # the default stays first
    for model in ("gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"):
        assert model in GEMINI_CHAT_MODELS
    assert len(GEMINI_CHAT_MODELS) <= 10  # a short list of the models that fit the app's uses
    assert not any(m.startswith(("gemini-1.5", "gemini-2.0")) for m in GEMINI_CHAT_MODELS)  # no longer served
    assert len(set(GEMINI_CHAT_MODELS)) == len(GEMINI_CHAT_MODELS)


def test_the_settings_page_offers_the_provider_list() -> None:
    from manage.rag.providers import declared

    assert declared("gemini").chat_models == GEMINI_CHAT_MODELS  # Settings reads the declaration


def test_the_model_list_keeps_the_chat_models_only() -> None:
    chat = ["generateContent", "countTokens"]
    assert is_chat_model("gemini-3.5-flash-lite", chat) and is_chat_model("gemma-4-31b-it", chat)
    for name in ("gemini-3.8-flash-tts", "gemini-3.1-flash-image", "gemini-robotics-er-2-preview", "lyria-3.5"):
        assert not is_chat_model(name, chat)
    assert not is_chat_model("gemini-embedding-2", ["embedContent"])


def test_the_listed_models_come_from_the_api_when_it_answers() -> None:
    listed = [
        SimpleNamespace(name="models/gemini-flash-lite-latest", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-3.8-flash-tts", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-embedding-2", supported_actions=["embedContent"]),
    ]
    connector = GeminiChatConnector(config=ProviderConfig(provider=ProviderType.GEMINI))
    connector.client = SimpleNamespace(models=SimpleNamespace(list=lambda: listed))
    assert connector.get_available_models() == ["gemini-flash-lite-latest"]
