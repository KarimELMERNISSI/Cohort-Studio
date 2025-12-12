from typing import Any, List, Optional, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings 


class CustomGeminiEmbeddings(Embeddings):
    """
    Custom embedding class that uses the modern Google GenAI SDK (v1.0+).
    """
    def __init__(self, api_key: str, model: str = "models/text-embedding-005"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Batch processing might be needed for large lists, 
        # but here is the simple implementation.
        results = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            # The new SDK returns an object with an 'embedding' attribute
            results.append(response.embeddings[0].values)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return response.embeddings[0].values


class CustomGeminiChat(BaseChatModel):
    """
    A custom LangChain wrapper for Google's new 'google-genai' SDK (v1.0+).
    Bypasses 'langchain-google-genai' to avoid dependency hell.
    """
    client: Any = None
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.3
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.3, **kwargs):
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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

        # 3. Call the Google API
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
            # Handle API errors gracefully
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {str(e)}"))])

    # Required Pydantic configuration to allow arbitrary types (the client)
    model_config = {
        "arbitrary_types_allowed": True
    }