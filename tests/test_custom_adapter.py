import os
import sys

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.custom_gemini import CustomGeminiChat, CustomGeminiEmbeddings

# Mock API Key for testing structure (won't make real calls unless key is valid)
API_KEY = os.environ.get("GOOGLE_API_KEY", "dummy_key")

def test_embeddings_structure():
    print("\n--- Testing Embeddings Structure ---")
    try:
        embeddings = CustomGeminiEmbeddings(api_key=API_KEY)
        print("Successfully initialized CustomGeminiEmbeddings")
        
        # We can't easily test actual API calls without a valid key and network,
        # but we can check if the method exists and has correct signature
        if hasattr(embeddings, 'embed_query') and hasattr(embeddings, 'embed_documents'):
            print("Embeddings methods present.")
        else:
            print("FAIL: Embeddings methods missing.")
            
    except Exception as e:
        print(f"FAIL: Embeddings initialization failed: {e}")

def test_chat_structure():
    print("\n--- Testing Chat Model Structure ---")
    try:
        chat = CustomGeminiChat(api_key=API_KEY)
        print("Successfully initialized CustomGeminiChat")
        
        if hasattr(chat, 'invoke'):
            print("Chat model has invoke method.")
        else:
            print("FAIL: Chat model missing invoke method.")
            
    except Exception as e:
        print(f"FAIL: Chat initialization failed: {e}")

def test_chain_integration():
    print("\n--- Testing Chain Integration ---")
    try:
        chat = CustomGeminiChat(api_key=API_KEY)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Context: {context}"),
            ("human", "{input}"),
        ])
        
        # Test chain creation (this checks if the LLM is compatible with LangChain's expected interface)
        # We use a simple chain that doesn't require retrieval to test the LLM part
        chain = prompt | chat
        print("Successfully created simple LCEL chain (prompt | llm)")
        
        # Test stuff documents chain creation (used in RAG)
        doc_prompt = ChatPromptTemplate.from_template("Content: {page_content}")
        stuff_chain = create_stuff_documents_chain(chat, prompt)
        print("Successfully created create_stuff_documents_chain")

    except Exception as e:
        print(f"FAIL: Chain integration failed: {e}")
        # Print detailed error to help debug
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embeddings_structure()
    test_chat_structure()
    test_chain_integration()
