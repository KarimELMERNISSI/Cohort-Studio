"""
Anthropic Claude Provider Connector.

Chat connector for Claude models through the official ``anthropic`` SDK (Messages API). Anthropic has no
embedding API: when Claude answers, documents are embedded by another provider chosen in Settings (local
Ollama by default, so the document library never leaves the server for indexing). See
``ConnectorFactory.create_connectors(embedding_provider=...)``.

Claude models choose their own sampling: the temperature setting of the app is not sent.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from utils.logging_config import redact_secrets

from ..base import BaseLLMConnector
from ..schemas import ProviderConfig

logger = logging.getLogger(__name__)

# Most capable first (used when the model list cannot be fetched).
CLAUDE_MODELS: tuple[str, ...] = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
)
DEFAULT_MAX_TOKENS = 4096

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency of the project
    anthropic = None
    ANTHROPIC_AVAILABLE = False


def _text(content: Any) -> str:
    """LangChain message content (string or list of parts) as plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return str(content)


def to_anthropic_messages(messages: list[BaseMessage]) -> tuple[str, list[dict[str, str]]]:
    """
    LangChain messages -> (system prompt, Messages API turns).

    System messages are joined into the ``system`` parameter. The turns alternate user / assistant and start
    with a user turn, as the API requires: consecutive turns of the same role are merged.
    """
    system_parts: list[str] = []
    turns: list[dict[str, str]] = []
    for message in messages:
        text = _text(message.content)
        if isinstance(message, SystemMessage):
            if text.strip():
                system_parts.append(text)
            continue
        role = "assistant" if isinstance(message, AIMessage) else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] = f"{turns[-1]['content']}\n\n{text}"
        else:
            turns.append({"role": role, "content": text})
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "Please continue."})
    return "\n\n".join(system_parts), turns


class ClaudeChatConnector(BaseLLMConnector):
    """Chat connector for Anthropic Claude models (cloud; requires an API key and the data-sharing consent)."""

    client: Any = None

    def __init__(self, config: ProviderConfig, **kwargs: Any):
        super().__init__(config=config, **kwargs)
        if ANTHROPIC_AVAILABLE and config.api_key:
            client_kwargs: dict[str, Any] = {
                "api_key": config.api_key,
                "timeout": config.timeout_seconds,
                "max_retries": config.max_retries,
            }
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            if config.extra_headers:
                client_kwargs["default_headers"] = dict(config.extra_headers)
            self.client = anthropic.Anthropic(**client_kwargs)

    @property
    def _llm_type(self) -> str:
        return "anthropic-claude"

    def get_available_models(self) -> list[str]:
        if not self.client:
            return list(CLAUDE_MODELS)
        try:
            ids = [m.id for m in self.client.models.list(limit=100).data]
            return ids or list(CLAUDE_MODELS)
        except Exception as exc:
            logger.warning("Claude model list unavailable: %s", exc)
            return list(CLAUDE_MODELS)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not ANTHROPIC_AVAILABLE:
            return self._error("the 'anthropic' package is required for Claude.")
        if not self.client:
            return self._error("Anthropic API key is missing.")
        system, turns = to_anthropic_messages(messages)
        request: dict[str, Any] = {
            "model": self.config.model_name,
            "max_tokens": int(kwargs.get("max_tokens") or DEFAULT_MAX_TOKENS),
            "messages": turns,
        }
        if system:
            request["system"] = system
        if stop:
            request["stop_sequences"] = list(stop)
        try:
            response = self.client.messages.create(**request)
        except Exception as exc:
            logger.error("Claude API error: %s", exc)
            return self._error(f"Claude API Error: {exc!s}")
        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        info = {
            "model": getattr(response, "model", self.config.model_name),
            "stop_reason": getattr(response, "stop_reason", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text, response_metadata=info))]
        )

    @staticmethod
    def _error(message: str) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=f"Error: {redact_secrets(message)}"))]
        )


__all__ = ["CLAUDE_MODELS", "ClaudeChatConnector", "to_anthropic_messages"]
