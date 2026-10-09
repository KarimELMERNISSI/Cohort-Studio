"""
Ollama Local & Open-Source Provider Connectors.

Enables 100% offline, privacy-preserving, sovereign execution using locally hosted
open-source models (Llama 3.2, Mistral, DeepSeek, Qwen) and local embeddings
(Nomic Embed Text, BGE-M3, etc.) via Ollama REST API.
"""

import logging
from typing import Any

import httpx
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..base import BaseEmbeddingConnector, BaseLLMConnector

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaChatConnector(BaseLLMConnector):
    """
    Chat connector for local Ollama instances via native REST API.
    Zero cloud dependencies, completely offline and private.
    """

    @property
    def _llm_type(self) -> str:
        return "ollama-local"

    @property
    def base_url(self) -> str:
        return self.config.base_url or DEFAULT_OLLAMA_HOST

    def get_available_models(self) -> list[str]:
        """
        Query the local Ollama server for currently downloaded models.
        Falls back to recommended open-source models if Ollama is unreachable.
        """
        endpoint = f"{self.base_url}/api/tags"
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(endpoint)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                    if models:
                        return models
        except Exception as e:
            logger.debug(f"Ollama local ping failed ({endpoint}): {e}")

        # Recommended open-weights models
        return [
            "llama3.2:latest",
            "llama3.1:8b",
            "mistral:latest",
            "deepseek-r1:8b",
            "qwen2.5:7b",
            "phi3:latest",
        ]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        endpoint = f"{self.base_url}/api/chat"
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

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            }
        }
        if stop:
            payload["options"]["stop"] = stop

        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.post(endpoint, json=payload)
                if response.status_code != 200:
                    err_msg = f"Ollama HTTP {response.status_code}: {response.text}"
                    logger.error(err_msg)
                    return ChatResult(generations=[ChatGeneration(
                        message=AIMessage(content=f"Error from Ollama: {err_msg}")
                    )])

                result = response.json()
                content = result.get("message", {}).get("content", "")
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

        except httpx.ConnectError:
            err = (
                f"Connection Error: Cannot reach Ollama at '{self.base_url}'. "
                "Ensure Ollama is installed and running locally ('ollama serve')."
            )
            logger.warning(err)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {err}"))])

        except httpx.TimeoutException:
            err = f"Timeout Error: Ollama did not respond within {self.config.timeout_seconds} seconds."
            logger.error(err)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {err}"))])

        except Exception as e:
            logger.error(f"Ollama execution error: {e}")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {e!s}"))])


class OllamaEmbeddingConnector(BaseEmbeddingConnector):
    """
    Embedding connector for local Ollama embedding models.
    Supports high-performance embeddings like nomic-embed-text, bge-m3, all-minilm.
    """

    @property
    def base_url(self) -> str:
        return self.config.base_url or DEFAULT_OLLAMA_HOST

    def get_available_models(self) -> list[str]:
        return [
            "nomic-embed-text:latest",
            "bge-m3:latest",
            "all-minilm:latest",
            "mxbai-embed-large:latest",
        ]

    def _call_embed(self, input_data: Any) -> Any:
        """Call either /api/embed or fallback to /api/embeddings."""
        # Ollama v0.1.34+ supports /api/embed
        url_embed = f"{self.base_url}/api/embed"
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            resp = client.post(
                url_embed,
                json={"model": self.config.embedding_model, "input": input_data}
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("embeddings", [])

            # Fallback to legacy /api/embeddings
            url_legacy = f"{self.base_url}/api/embeddings"
            if isinstance(input_data, str):
                resp_legacy = client.post(
                    url_legacy,
                    json={"model": self.config.embedding_model, "prompt": input_data}
                )
                if resp_legacy.status_code == 200:
                    return [resp_legacy.json().get("embedding", [])]

            raise RuntimeError(f"Ollama embedding failed ({resp.status_code}): {resp.text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            embeddings = self._call_embed(texts)
            if embeddings and len(embeddings) == len(texts):
                return embeddings
            # Individual fallback if batch endpoint returned single
            results = []
            for t in texts:
                res = self._call_embed(t)
                results.append(res[0] if res else [])
            return results
        except Exception as e:
            logger.error(f"Error computing local Ollama embeddings: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        try:
            res = self._call_embed(text)
            return res[0] if res else []
        except Exception as e:
            logger.error(f"Error computing local Ollama query embedding: {e}")
            raise
