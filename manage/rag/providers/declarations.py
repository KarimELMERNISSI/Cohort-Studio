"""
The AI providers, declared (M5b connectors on one kit): what Settings, the factory, the consent notices and the settings
need to know about each provider, read without importing langchain or a provider's SDK. The connectors themselves
(manage/rag/providers/<provider>.py) are imported when an engine starts, from ``chat_connector`` and
``embedding_connector``.

The names are stored in the users' preferences: they never change. The first model of each list is the default.
"""

from __future__ import annotations

from typing import ClassVar

from core.registry import Availability

# a custom endpoint (ProviderType.CUSTOM) speaks OpenAI's protocol: OpenAI's declaration serves it
CUSTOM_PROTOCOL = "openai"


class BaseAIProvider:
    """One AI provider, a connector of :data:`manage.rag.providers.AI_PROVIDERS`."""

    name: ClassVar[str]  # stored in the preferences: never renamed
    display_name: ClassVar[str]  # in Settings' list
    label: ClassVar[str]  # in sentences: the consent notices, the recipients
    cloud: ClassVar[bool]  # data leave the server: the confirmation of each session applies
    sdk: ClassVar[str]  # the package its connectors need, beside langchain_core
    key_variable: ClassVar[str | None] = None  # the setting that holds its API key
    key_label: ClassVar[str | None] = None
    chat_models: ClassVar[tuple[str, ...]]
    embedding_models: ClassVar[
        tuple[str, ...]
    ] = ()  # none: it answers, and another provider embeds
    lists_models: ClassVar[bool] = False  # its server or a key can list more chat models
    # Document insight sends it the PDF file itself (its chat connector's read_document), which the consent notice
    # says; the others are sent the PDF's text, as with every other use
    sends_files: ClassVar[bool] = False
    chat_connector: ClassVar[str]  # "module:Class"
    embedding_connector: ClassVar[str | None] = None

    @classmethod
    def availability(cls) -> Availability:
        """langchain_core and the provider's SDK, found without importing them; every missing one is named."""
        missing = [m for m in ("langchain_core", cls.sdk) if not Availability.of_module(m).ok]
        if missing:
            verb = "is" if len(missing) == 1 else "are"
            return Availability(False, f"{' and '.join(missing)} {verb} not installed", "ai")
        return Availability(True)

    @classmethod
    def embeds(cls) -> bool:
        return bool(cls.embedding_models)


class Ollama(BaseAIProvider):
    name = "ollama"
    display_name = "Ollama (open-source, at the address you give)"
    label = "Ollama"
    cloud = False
    sdk = "httpx"
    chat_models = (
        "llama3.2:latest",
        "llama3.1:8b",
        "mistral:latest",
        "deepseek-r1:8b",
        "qwen2.5:7b",
    )
    embedding_models = (
        "nomic-embed-text:latest",
        "bge-m3:latest",
        "all-minilm:latest",
        "mxbai-embed-large:latest",
    )
    lists_models = True  # the models pulled on its server
    chat_connector = "manage.rag.providers.ollama:OllamaChatConnector"
    embedding_connector = "manage.rag.providers.ollama:OllamaEmbeddingConnector"


class Gemini(BaseAIProvider):
    name = "gemini"
    display_name = "Google Gemini (cloud)"
    label = "Google Gemini"
    cloud = True
    sdk = "google.genai"
    key_variable = "GOOGLE_API_KEY"
    key_label = "Google GenAI API key"
    # the models that fit the application's uses (question answering, taxonomies, variable suggestions), with the
    # Flash-Lite versions for cheap and fast answers; checked against the Gemini API's model list on 2026-10-19 (2.0 and
    # 1.5 are no longer served). A key typed in Settings can add the other chat models it may use, after these.
    chat_models = (
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-pro-latest",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    )
    embedding_models = (
        "models/gemini-embedding-2",
        "models/text-embedding-005",
        "models/gemini-embedding-001",
    )
    lists_models = True
    sends_files = True  # its file API, then a request to delete the file
    chat_connector = "manage.rag.providers.gemini:GeminiChatConnector"
    embedding_connector = "manage.rag.providers.gemini:GeminiEmbeddingConnector"


class OpenAI(BaseAIProvider):
    name = "openai"
    display_name = "OpenAI (cloud)"
    label = "OpenAI"
    cloud = True
    sdk = "openai"
    key_variable = "OPENAI_API_KEY"
    key_label = "OpenAI API key"
    chat_models = ("gpt-4o-mini", "gpt-4o", "o3-mini")
    embedding_models = ("text-embedding-3-small", "text-embedding-3-large")
    chat_connector = "manage.rag.providers.openai:OpenAIChatConnector"
    embedding_connector = "manage.rag.providers.openai:OpenAIEmbeddingConnector"


class Mistral(BaseAIProvider):
    name = "mistral"
    display_name = "Mistral AI (cloud)"
    label = "Mistral AI"
    cloud = True
    sdk = "openai"  # its API speaks OpenAI's protocol
    key_variable = "MISTRAL_API_KEY"
    key_label = "Mistral API key"
    chat_models = ("mistral-small-latest", "mistral-large-latest", "codestral-latest")
    embedding_models = ("mistral-embed",)
    chat_connector = "manage.rag.providers.mistral:MistralChatConnector"
    embedding_connector = "manage.rag.providers.mistral:MistralEmbeddingConnector"


class Claude(BaseAIProvider):
    name = "anthropic"
    display_name = "Anthropic Claude (cloud)"
    label = "Anthropic Claude"
    cloud = True
    sdk = "anthropic"
    key_variable = "ANTHROPIC_API_KEY"
    key_label = "Anthropic API key"
    chat_models = (
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
    )  # most capable first
    lists_models = True  # a key lists the models it may use
    chat_connector = "manage.rag.providers.claude:ClaudeChatConnector"  # no embedding API


__all__ = ["CUSTOM_PROTOCOL", "BaseAIProvider", "Claude", "Gemini", "Mistral", "Ollama", "OpenAI"]
