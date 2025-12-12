from yfiles_graphs_for_streamlit import GraphWidget

print("Searching for show/render in dir(GraphWidget):")
for name in dir(GraphWidget):
    if "show" in name.lower() or "render" in name.lower():
        print(f"Found: {name}")
