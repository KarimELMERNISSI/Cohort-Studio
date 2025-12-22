path = r"C:\Users\208452\AppData\Local\anaconda3\envs\cardiateam_v4\Lib\site-packages\yfiles_graphs_for_streamlit\widget.py"

try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start = max(0, len(lines) - 200)
    for i in range(start, len(lines)):
        print(f"{i+1}: {lines[i].strip()}")
            
except Exception as e:
    print(f"Error: {e}")
