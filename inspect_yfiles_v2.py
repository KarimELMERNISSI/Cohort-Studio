from yfiles_graphs_for_streamlit import GraphWidget
import inspect
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("--- START INSPECTION ---")
try:
    print(f"Type: {type(GraphWidget)}")
    print(f"Doc: {GraphWidget.__doc__}")
    
    if inspect.isclass(GraphWidget):
        print("Signature:")
        try:
            print(inspect.signature(GraphWidget))
        except ValueError:
            print("No signature found (maybe C extension or built-in?)")
            
        print("\nInit Signature:")
        try:
            print(inspect.signature(GraphWidget.__init__))
        except ValueError:
            print("No init signature found")
            
        print("\nMembers:")
        for name, member in inspect.getmembers(GraphWidget):
            if not name.startswith("__"):
                print(f"- {name}")
                
    elif inspect.isfunction(GraphWidget):
        print("Signature:")
        print(inspect.signature(GraphWidget))

except Exception as e:
    print(f"Error: {e}")
print("--- END INSPECTION ---")
