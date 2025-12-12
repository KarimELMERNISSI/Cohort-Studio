path = r"C:\Users\208452\AppData\Local\anaconda3\envs\cardiateam_v4\Lib\site-packages\yfiles_graphs_for_streamlit\widget.py"

try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines[:50]):
        if "import" in line:
            print(f"{i+1}: {line.strip()}")
            
except Exception as e:
    print(f"Error: {e}")
