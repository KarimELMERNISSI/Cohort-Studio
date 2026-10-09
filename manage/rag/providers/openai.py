"""
OpenAI Provider Connectors.

Integrates OpenAI chat completion models (GPT-4o, GPT-4o-mini) and embeddings
(text-embedding-3-small, text-embedding-3-large) using official OpenAI client.
"""

import logging
from typing import List, Optional, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from ..base import BaseLLMConnector, BaseEmbeddingConnector
from ..schemas import ProviderConfig

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAIChatConnector(BaseLLMConnector):
    """
    Chat connector for OpenAI API models (GPT-4o, GPT-4o-mini, o3-mini).
    """
    client: Any = None

    def __init__(self, config: ProviderConfig, **kwargs: Any):
        super().__init__(config=config, **kwargs)
        if OPENAI_AVAILABLE and config.api_key:
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url or None,
                timeout=config.timeout_seconds,
            )

    @property
    def _llm_type(self) -> str:
        return "openai-chat"

    def get_available_models(self) -> List[str]:
        if not self.client or not OPENAI_AVAILABLE:
            return ["gpt-4o-mini", "gpt-4o", "o3-mini"]
        try:
            models_data = self.client.models.list()
            chat_models = [
                m.id for m in models_data.data
                if ("gpt" in m.id or "o1" in m.id or "o3" in m.id) and "realtime" not in m.id
            ]
            return sorted(chat_models) or ["gpt-4o-mini", "gpt-4o"]
        except Exception as e:
            logger.warning(f"Could not list OpenAI models: {e}")
            return ["gpt-4o-mini", "gpt-4o", "o3-mini"]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not OPENAI_AVAILABLE:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="Error: 'openai' package is not installed.")
            )])
        if not self.client:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="Error: OpenAI API key is missing.")
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
            err = f"OpenAI API Error: {str(e)}"
            logger.error(err)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {err}"))])


class OpenAIEmbeddingConnector(BaseEmbeddingConnector):
    """
    Embedding connector for OpenAI embeddings (text-embedding-3-small, text-embedding-3-large).
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None
        if OPENAI_AVAILABLE and config.api_key:
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url or None,
                timeout=config.timeout_seconds,
            )

    def get_available_models(self) -> List[str]:
        return ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise ValueError("OpenAI client not initialized. Check API key.")
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        results = self.embed_documents([text])
        return results[0] if results else []
