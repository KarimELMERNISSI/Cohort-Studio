"""
Google Gemini Provider Connectors.

Utilizes the modern `google-genai` SDK (v2.x) with retry logic,
temperature controls, and embedding integration.
"""

import logging
import random
import time
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..base import BaseEmbeddingConnector, BaseLLMConnector
from ..schemas import ProviderConfig

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None


class GeminiChatConnector(BaseLLMConnector):
    """
    Chat connector for Google Gemini models via google-genai SDK.
    """
    client: Any = None

    def __init__(self, config: ProviderConfig, **kwargs: Any):
        super().__init__(config=config, **kwargs)
        if GEMINI_AVAILABLE and config.api_key:
            self.client = genai.Client(api_key=config.api_key)

    @property
    def _llm_type(self) -> str:
        return "gemini-genai-v2"

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def get_available_models(self) -> list[str]:
        """Query Gemini API for available text models."""
        if not self.client or not GEMINI_AVAILABLE:
            return ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        try:
            models = []
            for m in self.client.models.list():
                name_lower = m.name.lower()
                if "gemini" in name_lower or "gemma" in name_lower:
                    if "imagen" not in name_lower and "veo" not in name_lower and "embedding" not in name_lower:
                        models.append(m.name)
            return models or ["gemini-2.5-flash", "gemini-1.5-flash"]
        except Exception as e:
            logger.warning(f"Could not fetch Gemini models list: {e}")
            return ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not GEMINI_AVAILABLE or not self.client:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="Error: Google GenAI SDK not installed or API key missing.")
            )])

        google_contents = []
        system_instruction = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = msg.content
            elif isinstance(msg, HumanMessage):
                google_contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg.content)]
                ))
            elif isinstance(msg, AIMessage):
                google_contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=msg.content)]
                ))
            else:
                google_contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=str(msg.content))]
                ))

        gen_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            candidate_count=1,
            stop_sequences=stop,
            system_instruction=system_instruction
        )

        clean_model = self.config.model_name
        clean_model = clean_model.removeprefix("models/")

        max_retries = self.config.max_retries
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=clean_model,
                    contents=google_contents,
                    config=gen_config
                )
                generated_text = response.text or ""
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=generated_text))])
            except Exception as e:
                error_str = str(e)
                if attempt == max_retries - 1:
                    logger.error(f"Gemini API failure after {max_retries} attempts: {error_str}")
                    return ChatResult(generations=[ChatGeneration(
                        message=AIMessage(content=f"API Error (Gemini): {error_str}")
                    )])
                delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                logger.info(f"Gemini API retry {attempt+1}/{max_retries} in {delay:.2f}s: {error_str}")
                time.sleep(delay)


class GeminiEmbeddingConnector(BaseEmbeddingConnector):
    """
    Embedding connector for Google Gemini models.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None
        if GEMINI_AVAILABLE and config.api_key:
            self.client = genai.Client(api_key=config.api_key)

    def get_available_models(self) -> list[str]:
        if not self.client or not GEMINI_AVAILABLE:
            return ["models/gemini-embedding-001", "models/text-embedding-004"]
        try:
            embed_models = []
            for m in self.client.models.list():
                if "embedding" in m.name.lower():
                    embed_models.append(m.name)
            return embed_models or ["models/gemini-embedding-001", "models/text-embedding-004"]
        except Exception:
            return ["models/gemini-embedding-001", "models/text-embedding-004"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            raise ValueError("Google GenAI client not initialized. Check API key.")
        
        clean_model = self.config.embedding_model
        results: list[list[float]] = []
        for text in texts:
            try:
                response = self.client.models.embed_content(
                    model=clean_model,
                    contents=text
                )
                results.append(response.embeddings[0].values)
            except Exception as e:
                logger.error(f"Gemini embedding error: {e}")
                raise
        return results

    def embed_query(self, text: str) -> list[float]:
        if not self.client:
            raise ValueError("Google GenAI client not initialized. Check API key.")
        
        clean_model = self.config.embedding_model
        response = self.client.models.embed_content(
            model=clean_model,
            contents=text
        )
        return response.embeddings[0].values
