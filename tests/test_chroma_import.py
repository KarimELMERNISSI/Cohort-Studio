try:
    from langchain_chroma import Chroma
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
