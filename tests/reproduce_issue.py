import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from manage.rag_manager import RAGManager
    print("Successfully imported RAGManager")
    
    rag = RAGManager()
    print("Successfully instantiated RAGManager")
    
    # Check if we can call is_available
    print(f"RAG Available: {rag.is_available()}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
