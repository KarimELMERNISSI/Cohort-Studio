import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Try to grab from arguments or just fail
    print("No API Key found")
else:
    try:
        client = genai.Client(api_key=api_key)
        print("Listing first model to inspect its attributes...")
        models = list(client.models.list())
        if models:
            m = models[0]
            print(f"Type: {type(m)}")
            print(f"Dir: {dir(m)}")
            print(f"Name: {getattr(m, 'name', 'N/A')}")
            # print(f"Supported methods? {getattr(m, 'supported_generation_methods', 'N/A')}") 
        else:
            print("No models found.")
            
        print("-" * 20)
        print("Listing all names:")
        for m in models:
             print(f"Name: {m.name}")
    except Exception as e:
        print(f"Error: {e}")
