from yfiles_graphs_for_streamlit import GraphWidget
import inspect

print("GraphWidget methods:")
for name, member in inspect.getmembers(GraphWidget):
    if inspect.isfunction(member) or inspect.ismethod(member):
        print(f"- {name}")
