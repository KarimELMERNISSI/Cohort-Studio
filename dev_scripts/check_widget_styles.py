path = r"C:\Users\208452\AppData\Local\anaconda3\envs\cardiateam_v4\Lib\site-packages\yfiles_graphs_for_streamlit\widget.py"

try:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    print("Searching for 'selection', 'return', 'value', 'logo', 'about':")
    keywords = ['selection', 'return', 'value', 'logo', 'about', 'copyright']
    for i, line in enumerate(content.splitlines()):
        for kw in keywords:
            if kw in line.lower():
                print(f"{i+1}: {line.strip()}")

except Exception as e:
    print(f"Error: {e}")
