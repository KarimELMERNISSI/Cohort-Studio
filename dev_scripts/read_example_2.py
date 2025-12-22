path = r"C:\Users\208452\AppData\Local\anaconda3\envs\cardiateam_v4\Lib\site-packages\yfiles_graphs_for_streamlit\widget.py"

try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start = 2853
    end = 2870
    for i in range(start, min(end, len(lines))):
        print(f"{i+1}: {lines[i].strip()}")
            
except Exception as e:
    print(f"Error: {e}")
