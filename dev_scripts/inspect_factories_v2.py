import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    import yfiles_graphs_for_streamlit.layout.factories as factories_module
    
    if hasattr(factories_module, 'factories'):
        print("--- KEYS START ---")
        for k in factories_module.factories.keys():
            print(f"KEY: '{k}'")
        print("--- KEYS END ---")
    else:
        print("No 'factories' dict found in module.")
        
except Exception as e:
    print(f"Error: {e}")
