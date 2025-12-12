from yfiles_graphs_for_streamlit import GraphWidget
import inspect

print("GraphWidget init signature:")
try:
    print(inspect.signature(GraphWidget.__init__))
except Exception as e:
    print(f"Could not get signature: {e}")

print("\nGraphWidget docstring:")
print(GraphWidget.__doc__)

print("\nGraphWidget attributes:")
print(dir(GraphWidget))
