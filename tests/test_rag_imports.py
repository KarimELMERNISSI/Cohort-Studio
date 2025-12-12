import sys
import os

print("Python executable:", sys.executable)
print("Python version:", sys.version)

print("\n--- Testing Imports ---")

try:
    import google.genai as genai
    print("✅ google.genai imported successfully. Version:", genai.__version__)
except ImportError as e:
    print("❌ Failed to import google.genai:", e)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    print("✅ langchain_google_genai imported successfully.")
except ImportError as e:
    print("❌ Failed to import langchain_google_genai:", e)

try:
    from langchain_chroma import Chroma
    print("✅ langchain_chroma imported successfully.")
except ImportError as e:
    print("❌ Failed to import langchain_chroma:", e)

try:
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    print("✅ langchain_community.document_loaders imported successfully.")
except ImportError as e:
    print("❌ Failed to import langchain_community.document_loaders:", e)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✅ langchain.text_splitter imported successfully.")
except ImportError as e:
    print("❌ Failed to import langchain.text_splitter:", e)

print("\n--- Testing Initialization (Mock) ---")
try:
    # Test if we can instantiate the classes (without API key for now, just class existence)
    # Note: Some might fail without API key, so we wrap in try/except
    print("Attempting to instantiate ChatGoogleGenerativeAI (dry run)...")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="TEST_KEY")
        print("✅ ChatGoogleGenerativeAI instantiated.")
    except Exception as e:
        print("⚠️ ChatGoogleGenerativeAI instantiation warning (expected if no valid key):", e)

except Exception as e:
    print("❌ Initialization test failed:", e)
