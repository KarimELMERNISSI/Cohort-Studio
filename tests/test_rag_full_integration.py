import os

import pytest

# the real loader: tests/conftest.py turns dotenv.load_dotenv into a no-op
from dotenv.main import load_dotenv

from manage.rag_manager import RAGManager


@pytest.mark.integration
def test_rag_integration():
    """Integration test verifying RAG functionality against Gemini API if key is present."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not configured for integration test")

    rag = RAGManager(api_key=api_key, provider="gemini")  # the server's is local Ollama
    assert rag.is_available()

    models = rag.get_available_models()
    assert isinstance(models, list)
