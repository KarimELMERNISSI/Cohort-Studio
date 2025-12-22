path = r"C:\Users\208452\AppData\Local\anaconda3\envs\cardiateam_v4\Lib\site-packages\yfiles_graphs_for_streamlit\widget.py"

try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "class GraphWidget" in line:
            print(f"Found class at line {i+1}")
            # Print next 200 lines to see methods
            for j in range(i, min(i+200, len(lines))):
                print(f"{j+1}: {lines[j].strip()}")
            break
            
except Exception as e:
    print(f"Error: {e}")
