"""
Mistral AI Provider Connectors.

Integrates European sovereign models (Mistral Small, Mistral Large, Codestral)
and Mistral Embeddings via Mistral OpenAI-compatible API.
"""

import logging
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..base import BaseEmbeddingConnector, BaseLLMConnector
from ..schemas import ProviderConfig

logger = logging.getLogger(__name__)

MISTRAL_API_BASE = "https://api.mistral.ai/v1"

try:
    from openai import OpenAI
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    OpenAI = None


class MistralChatConnector(BaseLLMConnector):
    """
    Chat connector for Mistral AI sovereign models.
    """
    client: Any = None

    def __init__(self, config: ProviderConfig, **kwargs: Any):
        super().__init__(config=config, **kwargs)
        if MISTRAL_AVAILABLE and config.api_key:
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url or MISTRAL_API_BASE,
                timeout=config.timeout_seconds,
            )

    @property
    def _llm_type(self) -> str:
        return "mistral-chat"

    def get_available_models(self) -> list[str]:
        if not self.client or not MISTRAL_AVAILABLE:
            return ["mistral-small-latest", "mistral-large-latest", "codestral-latest"]
        try:
            models_data = self.client.models.list()
            chat_models = [m.id for m in models_data.data if "embed" not in m.id]
            return sorted(chat_models) or ["mistral-small-latest", "mistral-large-latest"]
        except Exception:
            return ["mistral-small-latest", "mistral-large-latest", "codestral-latest"]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not MISTRAL_AVAILABLE:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="Error: 'openai' client package required for Mistral API.")
            )])
        if not self.client:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="Error: Mistral API key is missing.")
            )])

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": str(msg.content)})
            else:
                formatted_messages.append({"role": "user", "content": str(msg.content)})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=formatted_messages,
                temperature=self.config.temperature,
                stop=stop,
            )
            content = response.choices[0].message.content or ""
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        except Exception as e:
            err = f"Mistral API Error: {e!s}"
            logger.error(err)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {err}"))])


class MistralEmbeddingConnector(BaseEmbeddingConnector):
    """
    Embedding connector for Mistral AI embeddings (mistral-embed).
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None
        if MISTRAL_AVAILABLE and config.api_key:
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url or MISTRAL_API_BASE,
                timeout=config.timeout_seconds,
            )

    def get_available_models(self) -> list[str]:
        return ["mistral-embed"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            raise ValueError("Mistral client not initialized. Check API key.")
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model or "mistral-embed",
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Mistral embedding error: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        results = self.embed_documents([text])
        return results[0] if results else []
