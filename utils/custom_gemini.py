import random
import time
from typing import Any

from google import genai
from google.genai import types
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class CustomGeminiEmbeddings(Embeddings):
    """
    Custom embedding class that uses the modern Google GenAI SDK (v1.0+).
    """
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-2"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _embed_single(self, text: str) -> list[float]:
        models_to_try = [
            self.model,
            "models/gemini-embedding-2",
            "models/text-embedding-005",
            "models/text-embedding-004",
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
                continue
        if last_err:
            raise last_err
        return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents with fallback support."""
        return [self._embed_single(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with fallback support."""
        return self._embed_single(text)


class CustomGeminiChat(BaseChatModel):
    """
    A custom LangChain wrapper for Google's new 'google-genai' SDK (v1.0+).
    Bypasses 'langchain-google-genai' to avoid dependency hell.
    """
    client: Any = None
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.3
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model
        self.temperature = temperature
        # Initialize the NEW SDK client
        self.client = genai.Client(api_key=api_key)

    @property
    def _llm_type(self) -> str:
        return "custom-google-genai-v1"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        # 1. Convert LangChain Messages to Google GenAI Format
        google_contents = []
        system_instruction = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # The new SDK handles system prompts separately in config
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

        # 2. Prepare Configuration
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            candidate_count=1,
            stop_sequences=stop,
            system_instruction=system_instruction
        )

        # 3. Call the Google API with Retry Logic
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=google_contents,
                    config=config
                )
                
                # 4. Extract Text
                generated_text = response.text
                
                # 5. Return as LangChain Result
                generation = ChatGeneration(message=AIMessage(content=generated_text))
                return ChatResult(generations=[generation])
                
            except Exception as e:
                error_str = str(e)
                is_last_attempt = attempt == max_retries - 1
                
                if is_last_attempt:
                    # Handle API errors gracefully on final failure
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {error_str}"))])
                else:
                    # Exponential backoff with jitter
                    delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                    print(f"Gemini API Error (Attempt {attempt+1}/{max_retries}): {error_str}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)

    # Required Pydantic configuration to allow arbitrary types (the client)
    model_config = {
        "arbitrary_types_allowed": True
    }