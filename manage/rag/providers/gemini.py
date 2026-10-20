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

from utils.logging_config import redact_secrets

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

# The chat models offered in Settings, the default first: the models that fit the app's uses (question answering,
# taxonomies, variable suggestions), with the Flash-Lite versions for cheap and fast answers. Checked against the
# Gemini API's model list on 2026-10-19 (2.0 and 1.5 are no longer served). A key typed in Settings can add the other
# chat models it may use, after these.
GEMINI_CHAT_MODELS = (
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
# Models the API lists for generateContent that are not chat models: images, speech, robotics, agents...
NOT_CHAT_MODELS = (
    "image",
    "tts",
    "transcribe",
    "robotics",
    "computer-use",
    "customtools",
    "embedding",
    "imagen",
    "veo",
    "lyria",
    "nano-banana",
    "antigravity",
    "deep-research",
)


def is_chat_model(name: str, actions: list[str] | None) -> bool:
    """A Gemini or Gemma model that answers text (generateContent) and is not one of the specialised models."""
    lowered = name.lower()
    return (
        lowered.startswith(("gemini", "gemma"))
        and (actions is None or "generateContent" in actions)
        and not any(word in lowered for word in NOT_CHAT_MODELS)
    )


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
        """The chat models the key may use (GEMINI_CHAT_MODELS when the API cannot be asked)."""
        fallback = list(GEMINI_CHAT_MODELS)
        if not self.client or not GEMINI_AVAILABLE:
            return fallback
        try:
            models = []
            for m in self.client.models.list():
                clean_name = m.name.removeprefix("models/")
                if is_chat_model(clean_name, getattr(m, "supported_actions", None)):
                    models.append(clean_name)
            return models or fallback
        except Exception as e:
            logger.warning(f"Could not fetch Gemini models list: {e}")
            return fallback

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

        error_str = "no attempt was made (max_retries is 0)"
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
                    break
                delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                logger.info(f"Gemini API retry {attempt+1}/{max_retries} in {delay:.2f}s: {error_str}")
                time.sleep(delay)
        logger.error(f"Gemini API failure after {max_retries} attempts: {error_str}")
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=f"API Error (Gemini): {redact_secrets(error_str)}")
        )])


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
        fallback = [
            "models/gemini-embedding-2",
            "models/text-embedding-005",
            "models/gemini-embedding-001",
        ]
        if not self.client or not GEMINI_AVAILABLE:
            return fallback
        try:
            embed_models = []
            for m in self.client.models.list():
                if "embedding" in m.name.lower():
                    embed_models.append(m.name)
            return embed_models or fallback
        except Exception:
            return fallback

    def _embed_single(self, text: str) -> list[float]:
        """Embeds single text with automatic fallback across supported Gemini models."""
        models_to_try = [
            self.config.embedding_model,
            "models/gemini-embedding-2",
            "models/text-embedding-005",
        ]
        seen = set()
        deduped = [m for m in models_to_try if not (m in seen or seen.add(m))]

        last_err = None
        for m in deduped:
            try:
                response = self.client.models.embed_content(
                    model=m,
                    contents=text
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    return response.embeddings[0].values
            except Exception as e:
                last_err = e
                logger.warning(f"Embedding attempt with {m} failed: {e}. Trying fallback model...")
                continue
        logger.error(f"All Gemini embedding attempts failed: {last_err}")
        if last_err:
            raise last_err
        return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            raise ValueError("Google GenAI client not initialized. Check API key.")
        
        return [self._embed_single(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not self.client:
            raise ValueError("Google GenAI client not initialized. Check API key.")
        
        return self._embed_single(text)
