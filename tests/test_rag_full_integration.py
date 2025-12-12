import sys
import os
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from manage.rag_manager import RAGManager
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_integration():
    logger.info("Starting RAG Integration Test...")
    
    # Load API Key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in .env")
        return

    # Initialize Manager
    logger.info("Initializing RAGManager...")
    rag = RAGManager(api_key=api_key)
    
    # Check Availability
    if not rag.is_available():
        logger.error("RAG dependencies missing!")
        return

    # List Available Models
    logger.info("Listing available models...")
    models = rag.get_available_models()
    logger.info(f"Available Models: {models}")

    # Initialize System (Mocking documents by pointing to an empty or existing dir)
    # We'll use the actual DOCUMENTS folder if it exists, otherwise create a dummy one
    if not os.path.exists("DOCUMENTS"):
        os.makedirs("DOCUMENTS")
    
    # Use the first available gemini model if default fails, or just stick to default and see
    model_to_use = "models/gemini-1.5-flash"
    if models:
        # Prefer gemini-1.5-flash if available, else take first
        if "models/gemini-1.5-flash" in models:
             model_to_use = "models/gemini-1.5-flash"
        elif "gemini-1.5-flash" in models:
             model_to_use = "gemini-1.5-flash"
        else:
             model_to_use = models[0]
    
    logger.info(f"Calling initialize_system with model={model_to_use}...")
    success, msg = rag.initialize_system(model_name=model_to_use, use_existing_db=True)
    logger.info(f"Initialization Result: {success} | Message: {msg}")
    
    # Even if init fails due to no docs, the LLM might still be set up? 
    # Looking at code: "self.llm = ..." happens at step 4. 
    # If step 1 (Load Documents) fails, it returns False early.
    # So we MUST have documents.
    
    # Mock columns
    columns = ["age", "sex", "weight_kg", "height_cm", "sys_bp", "dias_bp", "cholesterol"]
    
    if success or rag.llm: # If we managed to set up LLM
        # Test 1: Suggest Computed Variables
        logger.info("Testing suggest_computed_variables...")
        result, error = rag.suggest_computed_variables(columns, search_hint="BMI", num_suggestions=1)
        if error:
            logger.error(f"suggest_computed_variables failed: {error}")
        else:
            logger.info(f"suggest_computed_variables result: {result[:100]}...")

        # Test 2: Suggest Column Renaming
        logger.info("Testing suggest_column_renaming...")
        result, error = rag.suggest_column_renaming(columns)
        if error:
            logger.error(f"suggest_column_renaming failed: {error}")
        else:
            logger.info(f"suggest_column_renaming result: {str(result)[:100]}...")

        # Test 3: Suggest Proxy Variable
        logger.info("Testing suggest_proxy_variable...")
        result = rag.suggest_proxy_variable("BMI", columns)
        logger.info(f"suggest_proxy_variable result: {str(result)[:100]}...")
        
    else:
        logger.warning("Skipping functional tests because RAG initialization failed (likely no docs).")

if __name__ == "__main__":
    test_rag_integration()
