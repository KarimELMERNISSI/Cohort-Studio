path = r"C:\Users\208452\AppData\Local\anaconda3\envs\cardiateam_v4\Lib\site-packages\yfiles_graphs_for_streamlit\__init__.py"

try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(f"{i+1}: {line.strip()}")
except Exception as e:
    print(f"Error reading file: {e}")
