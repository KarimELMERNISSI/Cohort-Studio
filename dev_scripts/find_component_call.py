path = r"C:\Users\208452\AppData\Local\anaconda3\envs\cardiateam_v4\Lib\site-packages\yfiles_graphs_for_streamlit\widget.py"

try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "component_value =" in line or "return _component_func" in line or "_component_func(" in line:
            print(f"Found call at line {i+1}: {line.strip()}")
            # Print context
            for j in range(max(0, i-5), min(len(lines), i+5)):
                print(f"{j+1}: {lines[j].strip()}")
            
except Exception as e:
    print(f"Error: {e}")
