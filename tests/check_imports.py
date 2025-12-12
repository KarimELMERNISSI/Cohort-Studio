
import sys
import traceback

def check_import(module_name):
    try:
        mod = __import__(module_name)
        print(f"{module_name} found: {getattr(mod, '__version__', 'unknown version')}")
    except Exception:
        print(f"{module_name} failed to import")
        traceback.print_exc()

check_import("langchain")
check_import("langchain_community")
check_import("langchain_chroma")
check_import("langchain_google_genai")
