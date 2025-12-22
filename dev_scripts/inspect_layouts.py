try:
    from yfiles_graphs_for_streamlit.layout.factories import factories
    print("Available layout factories:")
    for key in factories.keys():
        print(f"'{key}'")
except ImportError:
    print("Could not import factories directly.")
    try:
        import yfiles_graphs_for_streamlit.layout as layout
        print("Layout module dir:")
        print(dir(layout))
    except Exception as e:
        print(f"Error: {e}")
