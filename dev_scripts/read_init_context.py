path = r"C:\Users\208452\AppData\Local\anaconda3\envs\cardiateam_v4\Lib\site-packages\yfiles_graphs_for_streamlit\__init__.py"

try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(10, 40):
            print(f"{i+1}: {lines[i].strip()}")
except Exception as e:
    print(f"Error reading file: {e}")
