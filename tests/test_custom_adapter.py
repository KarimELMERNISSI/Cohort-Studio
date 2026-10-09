import os

from langchain_core.prompts import ChatPromptTemplate

from utils.custom_gemini import CustomGeminiChat, CustomGeminiEmbeddings

API_KEY = os.environ.get("GOOGLE_API_KEY", "dummy_key")


def test_embeddings_structure():
    """Verify CustomGeminiEmbeddings method signatures."""
    embeddings = CustomGeminiEmbeddings(api_key=API_KEY)
    assert hasattr(embeddings, "embed_query")
    assert hasattr(embeddings, "embed_documents")


def test_chat_structure():
    """Verify CustomGeminiChat invocation capabilities."""
    chat = CustomGeminiChat(api_key=API_KEY)
    assert hasattr(chat, "invoke")
    assert hasattr(chat, "_generate")


def test_chain_integration():
    """Verify compatibility with LangChain LCEL chain creation."""
    chat = CustomGeminiChat(api_key=API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Context: {context}"),
        ("human", "{input}"),
    ])
    chain = prompt | chat
    assert chain is not None
