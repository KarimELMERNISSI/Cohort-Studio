from yfiles_graphs_for_streamlit.layout import factories
print("Factories keys:")
try:
    # It seems factories might be a module or a dict in a module
    # The traceback said: from yfiles_graphs_for_streamlit.layout.factories import _get_layout_factory
    # And _get_layout_factory uses 'factories[name]'
    
    # Let's try to inspect the factories module
    import yfiles_graphs_for_streamlit.layout.factories as factories_module
    print(dir(factories_module))
    
    if hasattr(factories_module, 'factories'):
        print("Keys:", list(factories_module.factories.keys()))
        
except Exception as e:
    print(f"Error: {e}")
