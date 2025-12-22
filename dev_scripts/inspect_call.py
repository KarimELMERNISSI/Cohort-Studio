from yfiles_graphs_for_streamlit import GraphWidget
import inspect

try:
    print("GraphWidget.__call__ signature:")
    print(inspect.signature(GraphWidget.__call__))
except Exception as e:
    print(f"Error: {e}")
