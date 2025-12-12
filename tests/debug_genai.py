
try:
    import google.genai as genai
    print("Successfully imported google.genai")
    print(f"Attributes: {dir(genai)}")
    try:
        client = genai.Client(api_key="TEST")
        print("Successfully created Client")
    except Exception as e:
        print(f"Failed to create Client: {e}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
