import yfiles_graphs_for_streamlit as yfiles
import inspect

print("Module dir:")
print(dir(yfiles))

print("\nGraphWidget members:")
try:
    w = yfiles.GraphWidget()
    print(dir(w))
except Exception as e:
    print(f"Instantiation failed: {e}")
